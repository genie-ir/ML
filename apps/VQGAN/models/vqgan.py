import os
import torch
import numpy as np
from loguru import logger
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf
from libs.coding import random_string
from libs.basicIO import signal_save

from apps.VQGAN.modules.configuration import Config
from libs.basicIO import compressor
instantiate_from_config = Config.instantiate_from_config

from apps.VQGAN.modules.diffusionmodules.model import Encoder, Decoder
from apps.VQGAN.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
# from apps.VQGAN.modules.vqvae.quantize import GumbelQuantize
# from apps.VQGAN.modules.vqvae.quantize import EMAVectorQuantizer

class VQModel(pl.LightningModule):
    def __init__(self,
        ddconfig,
        lossconfig,
        n_embed,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image", # this `key` is defined in //apps/VQGAN/data/base.py > __getitem__ function
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
        Rfn='' # replace functions
        ):
        super().__init__()
        self.counter_control = 0
        self.ddconfig = ddconfig
        self.Rfn = Rfn
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        if bool(ckpt_path):
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        
        # Notic: [empty string -> Nothing happend] becuse it casted as `False`
        if bool(self.Rfn):
            rfn_list = [elementName for elementName in dir(self) if elementName.endswith(self.Rfn)]
            RfnLen = -len(self.Rfn) # Negetive number
            for fnName in rfn_list:
                setattr(self, fnName[:RfnLen], getattr(self, fnName))
        
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

    def encode(self, x, vetoFlag=False):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return self.quantize(h, vetoFlag=vetoFlag)
        # quant, emb_loss, info = self.quantize(h, vetoFlag=vetoFlag)
        # return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def get_input(self, batch, k):
        if self.Rfn == 'syn':
            return {'x': batch[k], 'y': batch['y']}
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        # x.shape must be: (B, CH, H, W)
        return x.float()
    
    def on_validation_epoch_end(self) -> None:
        if self.Rfn == 'syn':
            dpath = os.path.join(os.getenv('GENIE_ML_CACHEDIR'), 'syn')
            compressor(dpath, '/content/syn.zip', mode='zip')
            assert False, 'END'

    def forward(self, input):
        zshape = [-1,16,16,256]
        if self.Rfn == 'syn':
            input, y = input['x'], input['y']
            # I = R[2].view(zshape[:-1]) # comes from CGAN
            I = input.squeeze().long()
            I2 = I.flatten() # Good
            _quant, _diff, _R = self.quantize(None, I2=I2)
            _dec = self.decode(_quant)
            afn = lambda G: ((((G.clamp(-1., 1.))+1)/2)*255).transpose(0,1).transpose(1,2)
            signal_save(_dec, os.path.join(os.getenv('GENIE_ML_CACHEDIR'), 'syn', str(y[0].item()), f'{random_string()}.png'), stype='img', sparams={'fn': afn})
            return None, None
        else:
            quant, diff, R = self.encode(input)
            dec = self.decode(quant)
            return dec, diff

        
        
    # def on_validation_epoch_end_D2E(self) -> None:
    #     print('****************************')
    #     if self.counter_control == 1:
    #         compressor(os.path.join(os.getenv('GENIE_ML_CACHEDIR'), '1DsignalOfEyepacs'), '/content/pretrained_0_1DsignalOfEyepacs.zip', mode='zip')
    #         assert False, 'END'
    #     self.counter_control += 1
    
    # def forward_D2E(self, input):
    #     min_encoding_indices = self.encode(input, vetoFlag=True)
        
        
        
        
    #     X = min_encoding_indices.unsqueeze(1)
    #     bypass_zshape = [4, 16, 16, 256]
    #     quant = self.quantize(X, vetoFlag=False, bypassFlag=True, bypass_zshape=bypass_zshape)
    #     dec = self.decode(quant)
    #     for i in range(dec.shape[0]):
    #         deci = dec[i]
    #         deci_z = torch.zeros((256,256,3))
    #         deci_z[:,:,0] = deci[0]
    #         deci_z[:,:,1] = deci[1]
    #         deci_z[:,:,2] = deci[2]
    #         deci = deci_z
    #         gi = deci #((deci+1.0)/2.0) * 255
    #         print(gi.min(), gi.max())
    #         # print(deci.shape)
    #         # assert False, 'oooooooooooooooooooo'
            


    #         # gi = (
    #         #         ((deci+1.0)/2.0) * 255
    #         #     ).transpose(0,1).transpose(1,2).squeeze(-1)
            
    #         gi = gi.cpu().detach().numpy().astype(np.uint8)
    #         print('------------->', gi.shape)
    #         signal_save(gi, '/content/{}.jpeg'.format(i))

    #     print(X, X.shape, dec.shape, dec.dtype)
    #     assert False, 'SSSSSSSSSSSSSSSSSSSSS'


        
        
        
    #     return min_encoding_indices
    
    # def training_step_D2E(self, batch, batch_idx, optimizer_idx):
    #     images = self.get_input(batch, 'image')
    #     iclusters = self(images)
    #     for i, x in enumerate(batch['x']):
    #         fname = os.path.split(x)[1].replace('.jpeg', '.npy')
    #         y = batch['y'][i].item()
    #         dpath = os.path.join(os.getenv('GENIE_ML_CACHEDIR'), '1DsignalOfEyepacs', 'train', 'Grade_' + str(y))
    #         os.makedirs(dpath, exist_ok=True)
    #         ici = iclusters[i].cpu().detach().numpy()
    #         np.save(os.path.join(dpath, fname), ici)
    
    # def validation_step_D2E(self, batch, batch_idx):
    #     images = self.get_input(batch, 'image')
    #     iclusters = self(images)
    #     for i, x in enumerate(batch['x']):
    #         fname = os.path.split(x)[1].replace('.jpeg', '.npy')
    #         y = batch['y'][i].item()
    #         dpath = os.path.join(os.getenv('GENIE_ML_CACHEDIR'), '1DsignalOfEyepacs', 'val', 'Grade_' + str(y))
    #         os.makedirs(dpath, exist_ok=True)
    #         ici = iclusters[i].cpu().detach().numpy()
    #         np.save(os.path.join(dpath, fname), ici)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        if self.Rfn == 'syn':
            return
        
        vasl = self.get_input(batch, 'vasl')
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train", cond=vasl)

            # self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            # self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            
            
            self.log("train/aeloss", aeloss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train", cond=vasl)
            # self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            # self.log_dict(log_dict_disc, prog_bar=True, logger=True, on_step=True, on_epoch=True)


            self.log("train/discloss", discloss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss
        
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        
        if self.Rfn == 'syn':
            return
        
        vasl = self.get_input(batch, 'vasl')
        
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="val", cond=vasl)
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="val", cond=vasl)
        # rec_loss = log_dict_ae["val/rec_loss"]

        # self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict({'val/aeloss':aeloss, 'val/discloss':discloss}, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        # return self.log_dict

    # def training_step_Syn(self, batch, batch_idx, optimizer_idx):
    #     return
    
    # def validation_step_Syn(self, batch, batch_idx):
    #     x = batch['x']
    #     X = batch['latentcode']
    #     y = batch['y']
    #     X = X.unsqueeze(1)
    #     bypass_zshape = [X.shape[0], X.shape[2], X.shape[3], self.ddconfig.z_channels]
    #     generated_images = self(X, bypass_zshape=bypass_zshape)
    #     for i, xi in enumerate(x):
    #         gi = (
    #             ((generated_images[i]+1.0)/2.0) * 255
    #         ).transpose(0,1).transpose(1,2).squeeze(-1).cpu().detach().numpy().astype(np.uint8)
            
    #         yi = y[i].item()
    #         xi0, xi1 = os.path.split(xi)
    #         fpath = os.path.join(xi0, xi1.replace('.npy', '.jpeg'))
    #         dpath = os.path.join(os.getenv('GENIE_ML_CACHEDIR'), 'Syn', fpath)
    #         signal_save(gi, dpath)
    
    # def forward_Syn(self, X, bypass_zshape=None):
    #     quant = self.quantize(X, vetoFlag=False, bypassFlag=True, bypass_zshape=bypass_zshape)
    #     dec = self.decode(quant)
    #     return dec
    
    # def on_validation_epoch_end_Syn(self) -> None:
    #     dpath = os.path.join(os.getenv('GENIE_ML_CACHEDIR'), 'Syn')
    #     compressor(dpath, '/content/Syn.zip', mode='zip')
    #     assert False, 'END'
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        """this function is must be exist for ptCallback.ImageLoggerBase"""
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
