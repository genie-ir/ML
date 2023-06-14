import os, math
import torch
from loguru import logger
import torch.nn.functional as F
import pytorch_lightning as pl
from sys import getsizeof

from apps.VQGAN.modules.configuration import Config
instantiate_from_config = Config.instantiate_from_config

from apps.VQGAN.modules.util import SOSProvider


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


from torch.optim.adamw import adamw
class MyCustomOptimizer(torch.optim.SGD):
    pass
class MyCustomOptimizer2(torch.optim.AdamW): # TODO: this can be delete becuse I wrote this function
    # @torch.no_grad()
    # def step(self, closure):
    #     logger.critical('run optimizer')
    #     t = super().step(closure)
    #     logger.critical('done')
    #     return t

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        logger.warning(loss.item())
        
        for group in self.param_groups:
            params_with_grad = group['params'] # []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            logger.info(getsizeof(group['params']) / (1024))
            
            for p in group['params']:
                torch.cuda.empty_cache()
                if p.grad is None:
                    continue
                # params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state_steps.append(state['step'])

            logger.warning('uuuuuuuuuuuuuuuuuuuuuuuuu')
            adamw(params_with_grad,
                  grads,
                  exp_avgs,
                  exp_avg_sqs,
                  max_exp_avg_sqs,
                  state_steps,
                  amsgrad=amsgrad,
                  beta1=beta1,
                  beta2=beta2,
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  eps=group['eps'],
                  maximize=group['maximize'],
                  foreach=group['foreach'],
                  capturable=group['capturable'])

        return loss


class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 downsample_cond_size=-1,
                 pkeep=1.0,
                 sos_token=0,
                 unconditional=False,
                 **unused_kwargs
                 ):
        super().__init__()
        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        if permuter_config is None:
            permuter_config = {"target": "apps.VQGAN.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep

    def init_from_ckpt(self, path, ignore_keys=list()):
        # sd = torch.load(path, map_location="cpu")["state_dict"] # asli
        if '.state_dict' in path:
            sd = torch.load(path, map_location='cuda')
        else:
            sd = torch.load(path)["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        del sd
        logger.info(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__" or self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key
            self.cond_stage_model = SOSProvider(self.sos_token)
        else:
            print('$$$$$$$$$$$$$$$$$$$$$$$$')
            model = instantiate_from_config(config)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model

    def forward(self, x, c):
        # one step to produce the logits
        # Notic: we peresent each point By its index that is a scaler number in range(number of clusters=1024)
        _, z_indices = self.encode_to_z(x) # z_indices.shape: [B, 16x16] == [2, 256] -> totally has 512 point in latent space(d=256) | here we present each point By its index which is a scaler number in range(1024) that is the number of clusters! # Note that if we wont present each point By its complete values thus z_indices.shape: [B, number of points, latent dim]==[2,256,256]
        _, c_indices = self.encode_to_c(c) # c_indices.shape: [B, 1] == [2,1] its column vector with dtype=int64 of label classes.
        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size) # eyepacs.transformer.transformer_config.params.vocab_size
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        cz_indices = torch.cat((c_indices, a_indices), dim=1) # cz_indices.shape: [B, 257] == [2, 257] It is class scaler label folowed By 256= 2^4 x 2^4 point. (per box!)

        # print('hhhhhhhhhhhhhhhhh', cz_indices.shape)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logger.critical('@'*30)
        logits, _ = self.transformer(cz_indices[:, :-1])
        logger.critical('&'*30)
        # logits.shape: torch.Size([B, (number of points)256, (number of clusters)16384]) -> each value is logit of belonging to certain cluster
        # Notic: now logits is not probibility distribution and needs to apply softmax on it. softmax applied in F.crossentropy loss function
        # print('LLLLLLLLLLLLLLL', logits[0, 0], logits[0, 0].sum().item(), logits.shape)

        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1]-1:]

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape)==2
            noise_shape = (x.shape[0], steps-1)
            #noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
            x = torch.cat((x,noise),dim=1)
            logits, _ = self.transformer(x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1]-1:]
        else:
            for k in range(steps):
                callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _ = self.transformer(x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            # cut off conditioning
            x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x) # quant_z.shape -> torch.Size([2, 256, 16, 16])
        indices = info[2].view(quant_z.shape[0], -1) # info[2] is (512 -> number of points B2xH16xW16) dim vector corspand with nearst cluster
        # logger.critical(info[2].shape) # torch.Size([512])
        # logger.warning(indices.shape) # torch.Size([2, 256]) ->becuse each index is scaller not a real vector
        # print('kkkkkkkkkkkkkk', 
        #     quant_z.shape, # torch.Size([2, 256, 16, 16])
        #     indices.shape # torch.Size([2, 256]) # totally 512
        # )
        indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        # print(c, c.dtype, c.shape)
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
        # print('(NOTE: indices is [column and long] versian of c) JJJJJJJJJJJJJJJJJ', c, indices, indices.shape, len(indices.shape) > 2)
        # logger.warning(indices.shape) # torch.Size([2, 1])
        # logger.critical(len(indices.shape)) # 2
        
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True) # [B, number of columns]==[2, 256] each value is inddex refer to corespanding row of codebook matrix
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        
        # print(
        #     '++++++++++++++',
        #     index, index.shape
        # )
        # print(
        #     '++++++++++++++ 2',
        #     index.reshape(-1), index.reshape(-1).shape
        # )
        
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z) # x.shape: torch.Size([2, 3, 256, 256])
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@ decode', x.shape)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        logger.warning(kwargs.keys())
        log = dict()

        N = 4
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(x)
        quant_c, c_indices = self.encode_to_c(c)
        
        # logger.warning('{} | {} | {} | {} | {} | {}'.format(
        #     x.shape, c.shape, quant_z.shape, quant_c.shape, z_indices.shape, c_indices.shape 
        # )) 
        # torch.Size([2, 3, 256, 256]) | torch.Size([2]) | torch.Size([2, 256, 16, 16]) | torch.Size([2, 1]) | torch.Size([2, 256]) | torch.Size([2, 1])

        # create a "half"" sample
        z_start_indices = z_indices[:,:z_indices.shape[1]//2]

        # logger.critical(z_start_indices.shape) # torch.Size([2, 128])

        for xsmp_half_i in range(10):
            index_sample = self.sample(z_start_indices, c_indices,
                                    steps=z_indices.shape[1]-z_start_indices.shape[1],
                                    temperature=temperature if temperature is not None else 1.0,
                                    sample=True,
                                    top_k=top_k if top_k is not None else 100,
                                    callback=callback if callback is not None else lambda k: None)
            
            # logger.warning(index_sample.shape) # torch.Size([2, 256])
            x_sample = self.decode_to_img(index_sample, quant_z.shape)
            # logger.error(x_sample.shape) # torch.Size([2, 3, 256, 256])
            log["samples_half_{}".format(xsmp_half_i)] = x_sample

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)

        # logger.critical(z_start_indices.shape) # torch.Size([2, 0])
        # logger.warning(index_sample.shape) # torch.Size([2, 256])
        # logger.error(x_sample_nopix.shape) # torch.Size([2, 3, 256, 256])
        
        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # logger.critical(z_start_indices.shape) # torch.Size([2, 0])
        # logger.warning(index_sample.shape) # torch.Size([2, 256])
        # logger.error(x_sample_det.shape) # torch.Size([2, 3, 256, 256])

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)
        
        # logger.critical(z_indices.shape) # torch.Size([2, 256])
        # logger.warning(quant_z.shape) # torch.Size([2, 256, 16, 16])
        # logger.error(x_rec.shape) # torch.Size([2, 3, 256, 256])

        log['_'] = dict()
        log["inputs"] = x
        log["reconstructions"] = x_rec

        if self.cond_stage_key in ["objects_bbox", "objects_center_points"]:
            figure_size = (x_rec.shape[2], x_rec.shape[3])
            dataset = kwargs["pl_module"].trainer.datamodule.datasets["validation"]
            label_for_category_no = dataset.get_textual_label_for_category_no
            plotter = dataset.conditional_builders[self.cond_stage_key].plot
            log["conditioning"] = torch.zeros_like(log["reconstructions"])
            for i in range(quant_c.shape[0]):
                log["conditioning"][i] = plotter(quant_c[i], label_for_category_no, figure_size)
            log["conditioning_rec"] = log["conditioning"]
        elif self.cond_stage_key != "image":
            # logger.critical(self.cond_stage_key) # class_label
            # logger.critical(quant_c) # tensor([[1],[0]], device='cuda:0')
            # logger.warning(quant_c.shape) # torch.Size([2, 1])
            cond_rec = getattr(self.cond_stage_model, 'decode', lambda I: I)(quant_c)
            if self.cond_stage_key == "segmentation":
                # get image from segmentation mask
                num_classes = cond_rec.shape[1]

                c = torch.argmax(c, dim=1, keepdim=True)
                c = F.one_hot(c, num_classes=num_classes)
                c = c.squeeze(1).permute(0, 3, 1, 2).float()
                c = self.cond_stage_model.to_rgb(c)

                cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
                cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
                cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
                cond_rec = self.cond_stage_model.to_rgb(cond_rec)
            log['_']["conditioning_rec"] = cond_rec
            log['_']["conditioning"] = c

        # log["samples_half"] = x_sample
        # log["samples_nopix"] = x_sample_nopix
        # log["samples_det"] = x_sample_det
        return log

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        logger.critical('1'*30)
        logits, target = self(x, c) # logits is now not a probibility and softmax in F.crossentropy applied to it
        logger.critical('2'*30)
        # logits is [2, 256, NC] # target==indecies [2, 16x16=256]
        # logger.critical('{} | {}'.format(logits.shape, target.shape))
        # torch.Size([2, 256, 16384]) | torch.Size([2, 256])
        print(getsizeof(logits)/(1024*1024))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        logger.critical('3'*30)
        # logger.critical(loss.shape)
        # torch.Size([])
        return loss

    def training_step(self, batch, batch_idx):
        logger.critical('0'*30)
        loss = self.shared_step(batch, batch_idx)
        logger.critical('4'*30, loss.item())
        # self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        logger.critical('5'*30)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        print('hooooooooooooooooooooooooo!!')
        
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        optimizer = MyCustomOptimizer(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer