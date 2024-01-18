import torch
from loguru import logger
import pytorch_lightning as pl
from libs.basicIO import signal_save
from apps.VQGAN.modules.configuration import Config
instantiate_from_config = Config.instantiate_from_config
from apps.VQGAN.modules.diffusionmodules.model import Encoder, Decoder
from apps.VQGAN.modules.vqvae.quantize import VectorQuantizer #2 as VectorQuantizer
import torchvision
from apps.VQGAN.models.metrics import SQLiteLogger
from time import time

class PLModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.automatic_optimization = False
        self.__start()

    def __start(self):
        self.IDX = 1
        self.OPT = []
        setattr(self, 'forward', self.pipline)
        self.metrics = SQLiteLogger(db='/content/metrics.db') # TODO configurable

    def setbatch(self, batch, idx=-1):
        if idx == -1:
            return batch
        return batch

    def reset(self):
        """call at start of each opt step loop"""
        self._loss = None # current loss value
        self._log = dict(metrics=dict(), data=dict()) # current log dict

    def freeze(self, *M):
        """every where recursivly..."""
        if len(M) == 0:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for m in M:
                for param in m.parameters():
                    param.requires_grad = False

    def loss(self, loss_value):
        if self._loss == None:
            self._loss = loss_value
        else:
            self._loss = self._loss + loss_value
    
    def log(self, lDict, lType='metrics'):
        # print(lDict)
        self._log[lType] = {**self._log[lType], **lDict} # TODO better performance can be achived by replacing unpack aproch with other methods:)

    def save(self, tag):
        self.metrics.save(tag)
        # TODO logdata -> save

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        logger.critical(f"Restored from {path}")
    
    def acceptance(self):
        """All Batches will be accept in default"""
        return True
    
    def step(self, batch, batch_idx, **kwargs):
        self.TAG = kwargs['tag']
        optFlag = self.TAG == 'train' or kwargs.get('force_train', False)
        
        stime = time()
        self.bidx = batch_idx

        if optFlag:
            opt_ae, opt_disc = self.optimizers()
        
        self.batch = self.setbatch(batch)

        if not self.acceptance():
            return

        for idx in range(self.IDX):
            self.idx = idx # current index
            self.batch = self.setbatch(batch, self.idx) # current batch

            for optimizer_idx, optimizer_params in self.OPT:
                # print('START)', optimizer_idx, optimizer_params)

                self.reset()
                self.opt = optimizer_params # current opt params
                self.optidx = optimizer_idx # current opt index
                self.tag = f'{self.TAG}_opt{self.optidx}_' # current tag

                if optFlag:
                    opt_ae.zero_grad()
                    opt_disc.zero_grad()
                
                self.pipline()

                self.log({
                    'epoch': self.current_epoch
                })
                
                if optFlag: # TODO
                    self.manual_backward(self._loss.mean())
                    if self.optidx == 0:
                        opt_ae.step()
                    else:
                        opt_disc.step()
                
                self.metrics.log(self.TAG, self._log['metrics'])
                
                # print(f'after optidx={optimizer_idx}',optimizer_params, self.decoder.up[4].attn[1].k.weight.requires_grad, self.decoder.up[4].attn[1].k.weight.sum().item())

        dtime = (time() - stime)




class VQModel(PLModule):
    def acceptance(self,
            # N=[254, 110, 3],
            F=[
                'prototype/fundus/1/10192_left.jpg', # NOTE OK
                
                'prototype/fundus/2/14651_right_clahe.jpg', # NOTE OK
                
                'prototype/fundus/4/2800_left_clahe.jpg', # NOTE OK
            ]
    ):
        if self.imglogger[self.batch['x_class']] == None and self.TAG == 'train':
            # self.counter[self.batch['x_class']] = self.counter[self.batch['x_class']] + 1

            # if self.counter[self.batch['x_class']] == N[self.batch['x_class']]:
            if self.batch['names'][0] == F[self.batch['x_class']]:
                # print('*******************************************')
                # print(f"class={self.batch['x_class']} | N={N[self.batch['x_class']]}", self.batch['names'])
                # print('*******************************************')
                self.pack_logdata = dict()
                self.pack_logdata['xs'] = self.batch['x']
                self.pack_logdata['y_edit'] = self.batch['x_class']
                self.imglogger[self.batch['x_class']] = self.pack_logdata
                return True
            else:
                return False
        else:
            return False

    def H(self, phi1, phi2, tag): # NOTE: kernel regressor
        h = self.encoder.kernel_regressor(phi1, phi2)
        # h.register_hook(lambda grad: (self.HGrad.get(tag, 1) * grad) * self.HGrad2.get(tag, 1))
        # h.register_hook(lambda grad: print(f'--------> h.grad | {tag}', grad.mean().item()))
        return h

    def phi(self, t): # NOTE: kernel function
        """t.shape=Bx3x256x256"""
        h, h_ilevel1, h_endDownSampling = self.encoder(t)
        h = self.quant_conv(h)
        quant, diff = self.quantize(h)
        quant.register_hook(lambda grad: 1e5*grad)
        # quant.register_hook(lambda grad: print('phi.grad', self.opt, grad))
        return self.post_quant_conv(quant)

    def netA(self, Cmue, mue, tag):
        alpha = self.batch['x'] * Cmue
        phi_alpha = self.phi(alpha)
        phi_null = self.phi(self.Z.detach())
        Qn = self.H(phi_alpha.detach(), phi_null.detach(), tag)
        xn = self.batch['x'] * Cmue + mue * self.decoder(Qn)
        xn.register_hook(lambda grad: print(f'xn.grad | {tag}', grad.mean().item()))

        # signal_save(torch.cat([
        #     (self.batch['x']+1)*127.5,
        #     (xn+1)*127.5,
        #     torch.cat([Cmue, Cmue, Cmue], dim=1)*255,
        #     torch.cat([mue, mue, mue], dim=1)*255,
        # ], dim=0), f'/content/export/netA_{tag}.png', stype='img', sparams={'chw2hwc': True, 'nrow': 2})

        return xn

    def netB(self, x, xc, xnf, Cmue, Cmuel, muel, tag):
        alhpa = x * Cmue
        betta = xc * muel
        phi_alpha = self.phi(alhpa)
        phi_betta = self.phi(betta)
        Qp = self.H(phi_alpha.detach(), phi_betta.detach(), tag)
        xp = xnf * Cmuel + muel * self.decoder(Qp)
        # xp.register_hook(lambda grad: print(f'xp.grad | {tag}', grad.mean().item()))

        # signal_save(torch.cat([
        #     (x+1)*127.5,
        #     (xc+1)*127.5,
        #     (xnf+1)*127.5,
        #     (xp+1)*127.5,
        #     torch.cat([Cmue, Cmue, Cmue], dim=1)*255,
        #     torch.cat([Cmuel, Cmuel, Cmuel], dim=1)*255,
        #     torch.cat([muel, muel, muel], dim=1)*255,
        # ], dim=0), f'/content/export/netB_{tag}.png', stype='img', sparams={'chw2hwc': True, 'nrow': 4})

        return xp
    
    def sec0(self):
        x1 = self.batch['x']
        x2 = self.batch['x'] * self.batch['M']
        x3 = self.batch['x'] * self.batch['Mbar']

        A_loss1, A_loss_logdict1 = self.Loss.geometry(x1, self.decoder(self.phi(x1)), split=self.tag + 'S0_GeoX')
        A_loss2, A_loss_logdict2 = self.Loss.geometry(x2, self.decoder(self.phi(x2)), split=self.tag + 'S0_GeoXM')
        A_loss3, A_loss_logdict3 = self.Loss.geometry(x3, self.decoder(self.phi(x3)), split=self.tag + 'S0_GeoXMbar')
        self.loss(A_loss1+A_loss2+A_loss3)
        self.log({
            **A_loss_logdict1,
            **A_loss_logdict2,
            **A_loss_logdict3
        })
    
    def secA(self):
        if self.batch['x_class'] == 0: # NOTE: x has no any lesions
            xnf = self.batch['x']
            xn = self.netA(self.batch['Mcbar'], self.batch['Mc'], tag=self.tag + 'A_IF')
            
            if self.optidx == 0:
                A_loss, A_loss_logdict = self.Loss.geometry(self.batch['x'], xn, split=self.tag + 'A_Geo', λ1=10, λ2=10)
                self.loss(A_loss)
                self.log(A_loss_logdict)
            else:
                A_loss1, A_d1 = self.Loss.D12(self.batch['x'], l1=1, l2=1, split=self.tag + 'A_if1_Rxs')
                A_loss3, A_d3 = self.Loss.D12(xn.detach(), l1=1, l2=1, flag=True, split=self.tag + 'A_if1_Fpsistm')
                self.loss(A_loss1 + A_loss3)
                self.log({
                    **A_d1,
                    **A_d3,
                })
        else: # NOTE: x has lesions
            xn = self.netA(self.batch['Mbar'], self.batch['M'], tag=self.tag + 'A_ELSE')
            xnf = xn

            if self.optidx == 0:
                A_loss1, A_d1 = self.Loss.D12(xn, l1=self.acc[f'{self.TAG}_']['d1'], l2=self.acc[f'{self.TAG}_']['d2'], split=self.tag + 'A_el0_Rpsistm')
                self.loss(A_loss1)
                self.log({
                    **A_d1
                })
            else:
                A_loss1, A_d1 = self.Loss.D12(self.batch['x'], l1=1, l2=1, split=self.tag + 'A_el1_Rxs')
                A_loss3, A_d3 = self.Loss.D12(xn.detach(), l1=1, l2=1, flag=True, split=self.tag + 'A_el1_Fpsistm')
                self.loss(A_loss1 + A_loss3)
                self.log({
                    **A_d1,
                    **A_d3,
                })

        self.batch['xnf'] = xnf
    
    def secB(self):
        xnf = self.batch['xnf'].detach()
        if self.batch['xc_class'] == '[01]': # NOTE: xc has no any lesions
            xpf = xnf
            # BUG: can not be trained with geometry loss.
            # xp = self.netB(self.batch['x'], self.batch['x'], self.batch['x']*self.batch['Mbar'], self.batch['Mbar'], self.batch['Mbar'], self.batch['M'], tag=self.tag + 'B_IF')

            # if self.optidx == 0:
            #     B_loss, B_loss_logdict = self.Loss.geometry(self.batch['x'], xp, split=self.tag + 'B_Geo')
            #     self.loss(B_loss)
            #     self.log(B_loss_logdict)
            # else:
            #     B_loss2, B_d2 = self.Loss.D12(self.batch['x'], l1=1, l2=1, split=self.tag + 'B_if1_Rxs')
            #     B_loss5, B_d5 = self.Loss.D12(xp, l1=1, l2=1, flag=True, split=self.tag + 'B_if1_Fpsistp')
            #     self.loss(B_loss2 + B_loss5)
            #     self.log({
            #         **B_d2,
            #         **B_d5,
            #     })
        else:
            xp = self.netB(self.batch['x'], self.batch['xc'], xnf, self.batch['Mbar'], self.batch['Mcbar'], self.batch['Mc'], tag=self.tag + 'B_ELSE')
            xpf = xp

            if self.optidx == 0:
                B_loss1, B_d1 = self.Loss.D12(xp, l1=self.acc[f'{self.TAG}_']['d1'], l2=self.acc[f'{self.TAG}_']['d2'], split=self.tag + 'B_el0_Rpsistp')
                self.loss(B_loss1)
                self.log({
                    **B_d1,
                })
            else:
                B_loss1, B_d1 = self.Loss.D12(self.batch['x'], l1=1, l2=1, split=self.tag + 'B_el1_Rxs')
                B_loss3, B_d3 = self.Loss.D12(xp.detach(), l1=1, l2=1, flag=True, split=self.tag + 'B_el1_Fpsistp')
                self.loss(B_loss1 + B_loss3)
                self.log({
                    **B_d1,
                    **B_d3,
                })

        self.batch['xpf'] = xpf

    def pipline(self):
        if self.opt.get('condstep', False):
            self.sec0()
        else:
            self.secA()
            self.secB()

            if self.opt.get('flag_logdata', False):
                self.pack_logdata[f'xc{self.idx}'] = self.batch['xc']
                self.pack_logdata[f'c{self.idx}_optidx{self.optidx}_pipline'] = {'𝝍s_tp_final': self.batch['xpf'], '𝝍s_tm_final': self.batch['xnf']}

    def save(self, tag):
        super().save(tag)

        # self.imglogger --> save!! # TODO
        signal_save((torch.cat([
            self.imglogger[0]['xc0'], self.imglogger[0]['xc1'], self.bb(self.imglogger[0]['xs']), self.bb(self.imglogger[0]['c0_optidx0_pipline']['𝝍s_tp_final'], 'green'), self.bb(self.imglogger[0]['c1_optidx0_pipline']['𝝍s_tp_final'], 'green'),
            self.imglogger[1]['xc0'], self.imglogger[1]['xc1'], self.bb(self.imglogger[1]['c0_optidx0_pipline']['𝝍s_tp_final'], 'blue'), self.bb(self.imglogger[1]['xs']), self.bb(self.imglogger[1]['c1_optidx0_pipline']['𝝍s_tp_final'], 'green'),
            self.imglogger[2]['xc0'], self.imglogger[2]['xc1'], self.bb(self.imglogger[2]['c0_optidx0_pipline']['𝝍s_tp_final'], 'blue'), self.bb(self.imglogger[2]['c1_optidx0_pipline']['𝝍s_tp_final'], 'blue'), self.bb(self.imglogger[2]['xs'])
        ], dim=0)+1)*127.5, f'/content/export/E{self.current_epoch}.png', stype='img', sparams={'chw2hwc': True, 'nrow': 5})
        
        # last_d1_acc = self.metrics.inference(tag, self.regexp_d1_acc)
        # self.acc[f'{tag}_'] = {'d1': last_d1_acc, 'd2': 0, 'O': 0}
        # print(f'{tag}_', self.acc[f'{tag}_'])
        self.imglogger = [None, None, None]
        self.counter = [0,0,0]

    def setbatch(self, batch, idx=-1):
        if idx == -1:
            ######################################## [changing name:)]
            batch['names'] = batch['x']
            batch['Xc'] = batch['xc']
            batch['xc'] = dict()
            ########################################

            batch['FM'] = batch['xsf']
            batch['x'] = batch['xs'] * batch['FM']
            batch['M'] = batch['xslmask']
            batch['Mbar'] = (1-batch['M']) * batch['FM']
            batch['x_class'] = batch['y_edit'].item()
            return batch

        batch['FMc'] = batch['xcf'][idx] # ROT
        batch['xc'] = batch['Xc'][idx] * batch['FMc'] # ROT
        batch['Mc'] = batch['xclmask'][idx] * batch['FM'] # ROT
        batch['Mcbar'] = (1-batch['Mc']) * batch['FM']
        batch['xc_class'] = batch['ynl'][idx][0]

        # signal_save(torch.cat([
        #     (batch['x']+1)*127.5, 
        #     (batch['xc']+1)*127.5,
        #     torch.cat([batch['M'],batch['M'],batch['M']], dim=1)*255,
        #     torch.cat([batch['Mbar'],batch['Mbar'],batch['Mbar']], dim=1)*255,
        #     torch.cat([batch['Mc'],batch['Mc'],batch['Mc']], dim=1)*255,
        #     torch.cat([batch['Mcbar'],batch['Mcbar'],batch['Mcbar']], dim=1)*255,
        # ], dim=0), f'/content/export/dataset_c{batch["xc_class"]}.png', stype='img', sparams={'chw2hwc': True, 'nrow': 2})


        # NOTE: Mask design
        # # M_union_L_xs_xc = ((xslmask + xclmask) - (xslmask * xclmask)).detach()
        # # M_L_xs_mines_xc = (xslmask - (xslmask * xclmask)).detach() # TODO Interpolation
        # # M_C_Union = ((1 - M_union_L_xs_xc) * xsf).detach() #shape:torch.Size([1, 1, 256, 256]) # reconstruct xs
        # # M_xrec_xcl = (xclmask * xsf).detach() # reconstruct xc
        # xcm_gray = ((xclmask * xc * xsf).mean(dim=1, keepdim=True)).detach() # torch.Size([1, 1, 256, 256])
        # C_xcmask = (1-xclmask).detach()

        return batch

    def __init__(self,
        ddconfig,
        lossconfig,
        n_embed,
        embed_dim,
        ckpt=None,
        ignore_keys=[],
        image_key="image", # this `key` is defined in //apps/VQGAN/data/base.py > __getitem__ function
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
        Rfn='', # replace functions
        **kwargs
        ):
        super().__init__()


        self.counter_control = 0
        self.ddconfig = ddconfig
        self.Rfn = Rfn
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig, returnSkipPath=True)
        self.decoder = Decoder(**ddconfig)
        self.Loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_e=n_embed, e_dim=embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        self.vqgan_fn_phi_denormalize = lambda G: ((((G.clamp(-1., 1.))+1)/2)*255)

        print('BEFORE CKPT', self.decoder.up[4].attn[1].k.weight[0][0,0])
        if bool(ckpt):
            self.init_from_ckpt(ckpt, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        
        print('AFTER CKPT', self.decoder.up[4].attn[1].k.weight[0][0,0])
        
        # Notic: [empty string -> Nothing happend] becuse it casted as `False`
        if bool(self.Rfn):
            rfn_list = [elementName for elementName in dir(self) if elementName.endswith(self.Rfn)]
            RfnLen = -len(self.Rfn) # Negetive number
            for fnName in rfn_list:
                setattr(self, fnName[:RfnLen], getattr(self, fnName))
        
        self.__start()

    def decoder_grad_controller000000(self, flag):
        # print('before self.decoder.up[4]', self.decoder.up[4].attn[1].k.weight.requires_grad)
        for param in self.decoder.up[4].parameters():
            param.requires_grad = flag
        for param in self.decoder.norm_out.parameters():
            param.requires_grad = flag
        for param in self.decoder.conv_out.parameters():
            param.requires_grad = flag
        # for param in self.decoder.spade_ilevel1.parameters():
        #     param.requires_grad = flag
        # for param in self.decoder.spade_endDownSampling.parameters():
        #     param.requires_grad = flag
        # print('after self.decoder.up[4]', self.decoder.up[4].attn[1].k.weight.requires_grad)
    
    def __start(self):
        self.counter = [0,0,0]
        self.HGrad = {
            # 'train_opt0_A_IF': 1e7,
            # 'train_opt0_A_ELSE': 1e25,
            # 'train_opt0_B_ELSE': 1e25,
        }
        self.HGrad2 = {
            # 'train_opt0_A_IF': 1,
            # 'train_opt0_A_ELSE': 1e15,
            # 'train_opt0_B_ELSE': 1e15,
        }
        self.IDX = 2
        self.OPT = [
            [0, {'condstep': True}],
            [0, {'VgradViewrFlag': True, 'flag_logdata': True}], 
            [1, {}]
        ]
        self.Z = torch.zeros((1,3,256,256), device='cuda')
        
        self.imglogger = [None, None, None]
        self.acc = {
            'train_': {'d1': 0, 'd2': 0, 'O': 0},
            'val_': {'d1': 0, 'd2': 0, 'O': 0}
        }
        # self.regexp_d1_acc = '^.*_opt1_.*\/d1ACC$'

        self.freeze()

        print(self.encoder.kernel_regressor)
        assert False

        for param in self.encoder.kernel_regressor.parameters():
            param.requires_grad = True
        
        for param in self.quantize.parameters():
            param.requires_grad = True
        
        for param in self.Loss.discriminator.parameters():
            param.requires_grad = True

    def continueStart000(self):

        # print('encoder', self.encoder)
        # print('decoder', self.decoder)
        # print('disc', self.loss.discriminator)

        for param in self.quant_conv.parameters():
            param.requires_grad = False
        for param in self.post_quant_conv.parameters():
            param.requires_grad = False
        for param in self.quantize.parameters():
            param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.loss.discriminator.parameters():
            param.requires_grad = False

        
        if True: 
            # print('before self.encoder.down[4]', self.encoder.down[4].block[1].conv1.weight.requires_grad)
            for param in self.encoder.Qsurface2Qdiagonal.parameters():
                param.requires_grad = True
            for param in self.encoder.netb_diagonal.parameters():
                param.requires_grad = True
            for param in self.encoder.QGfazzy.parameters():
                param.requires_grad = True
            
        if False:
            for param in self.encoder.down[4].parameters():
                param.requires_grad = True
            for param in self.encoder.mid.parameters():
                param.requires_grad = True
            for param in self.encoder.norm_out.parameters():
                param.requires_grad = True
            for param in self.encoder.conv_out.parameters():
                param.requires_grad = True
            # print('after self.encoder.down[4]', self.encoder.down[4].block[1].conv1.weight.requires_grad)
            self.decoder_grad_controller(True)


        # print('before self.loss.discriminator.main[8]', self.loss.discriminator.main[8].weight.requires_grad)
        for pidx in [5, 6, 8, 9, 11]:
            for param in self.loss.discriminator.main[pidx].parameters():
                param.requires_grad = True
        # print('after self.loss.discriminator.main[8]', self.loss.discriminator.main[8].weight.requires_grad)
        
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, tag='train', show_dataset=True)
    
    def validation_step(self, batch, batch_idx):
        return
        return self.step(batch, batch_idx, tag='val')
    
    def bb(self, img, color='red'):
        return torchvision.utils.draw_bounding_boxes(((img.squeeze()+1)*127.5).to(torch.uint8), torch.tensor([0,0, 255,255], dtype=torch.int).unsqueeze(0), colors=color).to(self.device).unsqueeze(0) /127.5 -1
    
    def on_train_epoch_end(self):
        self.save('train')
    
    def on_validation_epoch_end(self):
        return
        self.save('val')
    
    def configure_optimizers(self):
        # lr = self.learning_rate
        lr = 0.02
        opt_ae = torch.optim.Adam(
                                    list(self.encoder.kernel_regressor.parameters())+
                                    # list(self.decoder.parameters())+
                                    list(self.quantize.parameters()),
                                    # list(self.quant_conv.parameters())+
                                    # list(self.post_quant_conv.parameters()),
                                lr=lr, 
                                # betas=(0.5, 0.9)
                            )
        opt_disc = torch.optim.Adam([
                                        {'params': self.Loss.discriminator.graph.parameters()},
                                    ],
                                lr=lr, 
                                # betas=(0.5, 0.9)
                            )

        return [opt_ae, opt_disc], []
        return [opt_ae], []
