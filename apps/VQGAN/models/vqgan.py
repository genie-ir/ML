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
from apps.VQGAN.modules.vqvae.quantize import VectorQuantizer #2 as VectorQuantizer
# from apps.VQGAN.modules.vqvae.quantize import GumbelQuantize
# from apps.VQGAN.modules.vqvae.quantize import EMAVectorQuantizer
from einops import rearrange
from PIL import Image
import albumentations as A 
from albumentations.pytorch import ToTensorV2


from utils.pt.tricks.gradfns import dzq_dz_eq1




from .ocv import ROT


dr_transformer0 = A.Compose([
    ToTensorV2()
])
class VQModel(pl.LightningModule):
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

        # from dependency.BCDU_Net.Retina_Blood_Vessel_Segmentation.pretrain import pretrain as makevaslsegmentation
        # self.vseg = makevaslsegmentation('/content/drive/MyDrive/storage/dr_classifire/unet-segmentation/weight_retina.hdf5')
        # self.vqgan_fn_phi_denormalize = lambda G: ((((G.clamp(-1., 1.))+1)/2)*255)#.transpose(0,1).transpose(1,2)






        # self.dr_classifire_normalize_std = torch.tensor([0.1252, 0.0857, 0.0814]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to('cuda')
        # self.dr_classifire_normalize_mean = torch.tensor([0.3771, 0.2320, 0.1395]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to('cuda')


        self.counter_control = 0
        self.ddconfig = ddconfig
        self.Rfn = Rfn
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig, returnSkipPath=True)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_e=n_embed, e_dim=embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        self.vqgan_fn_phi_denormalize = lambda G: ((((G.clamp(-1., 1.))+1)/2)*255)

        if bool(ckpt):
            self.init_from_ckpt(ckpt, ignore_keys=ignore_keys)
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
        
        self.start()

    def false_all_params(self, m):
        for param in m.parameters():
            param.requires_grad = False
    def true_all_params(self, m):
        for param in m.parameters():
            param.requires_grad = True
    
    
    def get_theta_tx_ty(self, xs_256ch, xcl_256ch):
        print('before xs_256ch', xs_256ch.shape)
        print('before xcl_256ch', xcl_256ch.shape)

        xs_256ch = self.cnn_xscl_256x32_256x16(xs_256ch)
        xs_256ch = xs_256ch + torch.relu(self.cnn_xscl_256x16_256x16(xs_256ch))
        xs_256ch = self.cnn_xscl_bn256(xs_256ch)
        
        xcl_256ch = self.cnn_xscl_256x32_256x16(xcl_256ch)
        xcl_256ch = xcl_256ch + torch.relu(self.cnn_xscl_256x16_256x16(xcl_256ch))
        xcl_256ch = self.cnn_xscl_bn256(xcl_256ch)

        print('after xs_256ch', xs_256ch.shape)
        print('after xcl_256ch', xcl_256ch.shape)

        x = torch.cat([xs_256ch, xcl_256ch], dim=1) # Bx512x16x16
        print('cat', x.shape)

        x = self.cnn_xscl_512x16_512x8(x)
        x = x + torch.relu(self.cnn_xscl_512x8_512x8(x))
        x = self.cnn_xscl_bn512(x)

        x = self.cnn_xscl_512x8_1024x4(x)
        x = x + torch.relu(self.cnn_xscl_1024x4_1024x4(x))
        x = self.cnn_xscl_bn1024(x)

        x = self.cnn_xscl_1024x4_1024x1(x).flatten(1) # Bx1024
        print('end_cnn x', x.shape)

        x = self.fc_xscl(x)
        print('end_cnn x', x.shape)
        theta = self.fc_xscl_theta(x)
        tx = self.fc_xscl_tx(x)
        ty = self.fc_xscl_ty(x)
        return theta, tx, ty
        
    def start(self): # TODO
        self.cnn_xscl_256x32_256x16 = torch.nn.Conv2d(256, 256, 4,2,1)
        self.cnn_xscl_256x16_256x16 = torch.nn.Conv2d(256, 256, 3,1,1)
        self.cnn_xscl_bn256 = torch.nn.BatchNorm2d(256)
        
        self.cnn_xscl_512x16_512x8 = torch.nn.Conv2d(512, 512, 4,2,1)
        self.cnn_xscl_512x8_512x8 = torch.nn.Conv2d(512, 512, 3,1,1)
        self.cnn_xscl_bn512 = torch.nn.BatchNorm2d(512)
        
        self.cnn_xscl_512x8_1024x4 = torch.nn.Conv2d(512, 1024, 4,2,1)
        self.cnn_xscl_1024x4_1024x4 = torch.nn.Conv2d(1024, 1024, 3,1,1)
        self.cnn_xscl_bn1024 = torch.nn.BatchNorm2d(1024)
        
        self.cnn_xscl_1024x4_1024x1 = torch.nn.Conv2d(1024, 1024, 4,2,0)

        self.fc_xscl = torch.nn.Sequential(
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )
        self.fc_xscl_theta = torch.nn.Sequential(
            torch.nn.Linear(32, 1),
            torch.nn.Tanh()
        )
        self.fc_xscl_tx = torch.nn.Sequential(
            torch.nn.Linear(32, 1),
            torch.nn.Tanh()
        )
        self.fc_xscl_ty = torch.nn.Sequential(
            torch.nn.Linear(32, 1),
            torch.nn.Tanh()
        )

        # delete
        # self.false_all_params(self.encoder)
        # self.false_all_params(self.decoder)
        # self.false_all_params(self.quant_conv)
        # self.false_all_params(self.loss)
        # self.true_all_params(self.loss.discriminator)



        # Note
        # self.false_all_params(self.post_quant_conv)
        # self.false_all_params(self.quantize)
        pass

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
        if vetoFlag:
            return self.quantize.fwd_getIndices(h)
        else:
            return self.quantize(h)

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def get_input(self, batch, k):
        return batch['xs'].float()
        if self.Rfn == 'syn':
            return {'x': batch[k], 'y': batch['y']}
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        # x.shape must be: (B, CH, H, W)
        return x.float()
    
    def on_validation_epoch_end_syn(self) -> None:
        print('on_validation_epoch_end_syn')
        dpath = os.path.join(os.getenv('GENIE_ML_CACHEDIR'), 'syn')
        compressor(dpath, '/content/syn.zip', mode='zip')
        assert False, 'END'

    
    def phi2lat(self, x):
        return self.encode(x, vetoFlag=True)
    def lat2phi(self, x):
        _quant = self.quantize.fwd_bpi(x)
        _dec = self.decode(_quant)
        return _dec
    def lat2qua(self, x):
        return self.quantize.fwd_bpi(x)
    def qua2phi(self, x):
        return self.decode(x)

    def save_phi(self, _dec, pathdir=None, fname=None, sreturn=False, afn=None, nrow=0):
        fname = fname if fname else f'{random_string()}.png'
        pathdir = os.getenv('GENIE_ML_CACHEDIR') if pathdir is None else pathdir
        if afn is None:
            afn = lambda G: ((((G.clamp(-1., 1.))+1)/2)*255).transpose(0,1).transpose(1,2)
        return signal_save(_dec, os.path.join(pathdir, 'syn', fname), stype='img', sparams={'fn': afn, 'nrow': nrow or int(_dec.shape[0] ** .5), 'sreturn': bool(sreturn)})
    
    # def forward_syn(self, input):
    #     print('forward_syn')
    #     return
    #     zshape = [-1,16,16,256]
    #     input, y = input['x'], input['y']
    #     # I = R[2].view(zshape[:-1]) # comes from CGAN
    #     I = input.squeeze().long()
    #     I2 = I.flatten() # Good
    #     _quant, _diff, _R = self.fwd_bpi(I2)
    #     _dec = self.decode(_quant)
    #     afn = lambda G: ((((G.clamp(-1., 1.))+1)/2)*255).transpose(0,1).transpose(1,2)
    #     signal_save(_dec, os.path.join(os.getenv('GENIE_ML_CACHEDIR'), 'syn', str(y[0].item()), f'{random_string()}.png'), stype='img', sparams={'fn': afn})
    #     return None, None
    
    
    
    def forward00(self, input):
        quant, diff = self.encode(input)
        dec = self.decode(quant)
        return dec, diff
    
    
    
    
    
    
    
    
    
    def forward(self, xs, xc_lesion):
        _, _, _, h_ilevel4_xcl = self.encoder(xc_lesion)
        h, h_ilevel1, h_endDownSampling, h_ilevel4_xs = self.encoder(xs)
        h = self.quant_conv(h)
        quant, diff = self.quantize(h)
        Q = self.post_quant_conv(quant)
        dec = self.decoder(Q + h, xc_lesion, h_ilevel1, h_endDownSampling) # Note: add skip connection

        theta, tx, ty = self.get_theta_tx_ty(h_ilevel4_xs, h_ilevel4_xcl)
        
        return dec, diff, theta, tx, ty

    
    def get_V(self, Forg, Frec):
        F_rec = self.vqgan_fn_phi_denormalize(Frec).detach()
        V_rec = self.vseg(F_rec.cpu()).detach()
        F_org = ((Forg +1)*127.5).detach()
        V_org = self.vseg(F_org.cpu()).detach()
        V_org = torch.cat([V_org,V_org,V_org], dim=1).detach()
        

        V_org = ((V_org / 255)).detach()
        V_rec = ((V_rec / 255)).detach()
        # V_org = ((V_org / 127.5) - 1).detach()
        # V_rec = ((V_rec / 127.5) - 1).detach()
        



        # V_org = torch.cat([V_org,V_org,V_org], dim=1)
        # V_rec = torch.cat([V_rec,V_rec,V_rec], dim=1)
        
        
        # print('-----------F_rec---------------->', F_rec.shape)
        # print('-----------F_org---------------->', F_org.shape)
        # print('-----------V_rec---------------->', V_rec.shape)
        # print('-----------V_org---------------->', V_org.shape)
        # signal_save(F_org, f'/content/F_org.png', stype='img', sparams={'chw2hwc': True})
        # signal_save(V_org, f'/content/V_org.png', stype='img', sparams={'chw2hwc': True})
        # signal_save(F_rec, f'/content/F_rec.png', stype='img', sparams={'chw2hwc': True})
        # signal_save(V_rec, f'/content/V_rec.png', stype='img', sparams={'chw2hwc': True})
        # assert False
        return V_org, V_rec
    def get_V2(self, Forg, Frec):
        # F_rec = self.vqgan_fn_phi_denormalize(Frec).detach()
        V_rec = self.vseg(Frec.cpu().detach()).detach()
        V_org = self.vseg((((Forg.cpu() +1)*127.5).detach())).detach()
        
        
        # V_org = torch.cat([V_org,V_org,V_org], dim=1)
        # V_rec = torch.cat([V_rec,V_rec,V_rec], dim=1)
         
        
        # print('-----------F_rec---------------->', F_rec.shape)
        # print('-----------F_org---------------->', F_org.shape)
        # print('-----------V_rec---------------->', V_rec.shape)
        # print('-----------V_org---------------->', V_org.shape)
        # signal_save(F_org, f'/content/F_org.png', stype='img', sparams={'chw2hwc': True})
        # signal_save(V_org, f'/content/V_org.png', stype='img', sparams={'chw2hwc': True})
        # signal_save(F_rec, f'/content/F_rec.png', stype='img', sparams={'chw2hwc': True})
        # signal_save(V_rec, f'/content/V_rec.png', stype='img', sparams={'chw2hwc': True})
        # assert False
        return V_org, V_rec
    
    def training_step_syn(self, batch, batch_idx, optimizer_idx):
        print(batch['x'], batch['x'].shape, batch['x'].dtype)
        print(batch['y'], batch['y'].shape, batch['y'].dtype)

        assert False
        return
    
    
    
    
    def dice_lossfn(self, inputs, target):
        num = target.shape[0]
        inputs = inputs.reshape(num, -1)
        target = target.reshape(num, -1).detach()

        intersection = (inputs * target).sum(1)
        union = (inputs.sum(1) + target.sum(1)).detach()
        dice = (2. * intersection) / (union + 1e-8)
        print('intersection, union', intersection.shape, union.shape)
        print('dicr', dice.shape)
        return -dice.log()
        # dice = dice.sum()/num
        # return dice
    
    # NOTE: Syn Idea
    def training_step(self, batch, batch_idx, optimizer_idx):
        h = torch.tensor(0.01).to(self.device)
        # if batch_idx % 500 == 0:
        #     self.log_images(batch, ignore=False)
        # x = self.get_input(batch, self.image_key)

        # print(batch.keys())
        # print(batch['y'])
        # for i in ['xs', 'xs_lesion', 'xs_cunvexhull', 'xs_fundusmask', 'xc', 'xc_lesion', 'xc_cunvexhull', 'xc_fundusmask']:
        #     if isinstance(batch[i], list):
        #         for j in batch[i]:
        #             print(i, '->', j.shape)
        #     else:
        #         print(i, batch[i].shape)


        cidx = 1 # 0 1 2
        xs = batch['xs']
        xs_fundusmask = batch['xs_fundusmask']
        xc_lesion = batch['xc_lesion'][cidx]
        xc_lesion_np = batch['xc_lesion_np'][cidx][0].cpu().numpy()
        xc_cunvexhull = batch['xc_cunvexhull'][cidx][0].cpu().numpy()
        xrec, qloss, theta, tx, ty = self(xs, xc_lesion)
        theta.register_hook(lambda grad: print('theta', grad))
        tx.register_hook(lambda grad: print('tx', grad))
        ty.register_hook(lambda grad: print('ty', grad))

        print('xc_lesion_np', xc_lesion_np.dtype, xc_lesion_np.shape)
        print('xc_cunvexhull', xc_cunvexhull.dtype, xc_cunvexhull.shape)
        m = dr_transformer0(image=ROT(xc_lesion_np, theta=theta, tx=tx, ty=ty))['image'].unsqueeze(0) # is a lead node, considere as a groundtrouth.
        mue = dr_transformer0(image=ROT(xc_cunvexhull, theta=theta, tx=tx, ty=ty))['image'].unsqueeze(0).to(self.device) # this shoulde be define as intermediate node
        mue_plus_h = dr_transformer0(image=ROT(xc_cunvexhull, theta=theta + h, tx=tx + h, ty=ty + h))['image'].unsqueeze(0).to(self.device) # this shoulde be define as intermediate node
        print('m, mue', m.shape, mue.shape, m.dtype, mue.dtype)
        print('xrec', xrec.shape)
        print('qloss', qloss.shape)

        
        iou = self.dice_lossfn(mue, xs_fundusmask).detach()
        iou_plus_h = self.dice_lossfn(mue_plus_h, xs_fundusmask).detach()
        D_tx = ((iou_plus_h - iou) / h).detach()
        D_ty = ((iou_plus_h - iou) / h).detach()
        D_theta = ((iou_plus_h - iou) / h).detach()
        iou = dzq_dz_eq1(iou, D_theta * theta + D_tx * tx + D_ty * ty)
        print('iou', iou.shape, iou.mean().item())
        
        return iou
        assert False

        Vorg, Vrec = self.get_V(x, xrec)
        Vrec = dzq_dz_eq1(Vrec, xrec)
        # print(optimizer_idx, x.shape, xrec.shape, Vorg.shape, Vrec.shape, Vorg.min(), Vorg.max(), Vrec.min(), Vrec.max())

        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="train")
            
            # DELETE
            # VLOSS = 0.5 * torch.mean(torch.abs(Vorg - Vrec) + 0.1 * self.loss.perceptual_loss(Vorg, Vrec)).log()
            vintersection = (Vorg * Vrec)
            VLOSS =  1 - (vintersection / (Vorg + Vrec - vintersection).clamp(1e-8, 1).detach()).mean()


            log_dict_ae['train/VLOSS'] = VLOSS.detach()
            self.log("train/aeloss", aeloss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            
            
            return VLOSS + aeloss
        
        if optimizer_idx == 1:
            discloss, log_dict_disc = self.loss(None, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="train")
            discloss_v, log_dict_disc_v = self.loss(None, Vorg, Vrec, 1, self.global_step, last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            # print('11', discloss_v, discloss)
            return discloss + discloss_v
    
































    # NOTE: real VQGAN training process
    # def training_step(self, batch, batch_idx, optimizer_idx):
    #     # logged = self.log_images(batch, fName='badRec/' + random_string())
    #     # return
    #     if batch_idx % 500 == 0:
    #         self.log_images(batch, ignore=False)
    #     # assert False
    #     # print('training_step')
    #     x = self.get_input(batch, self.image_key)
    #     xrec, qloss = self(x)
    #     Vorg, Vrec = self.get_V(x, xrec)
    #     Vrec = dzq_dz_eq1(Vrec, xrec)
    #     # vasl = None # self.get_input(batch, 'vasl')
    #     # vasl = self.get_input(batch, 'vasl')
    #     if optimizer_idx == 0:
    #         # autoencode
    #         aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train"
    #             # , cond=vasl
    #         )
    #         VLOSS = 0.5 * torch.mean(torch.abs(Vorg - Vrec) + 0.1 * self.loss.perceptual_loss(Vorg, Vrec)).log()
    #         log_dict_ae['train/VLOSS'] = VLOSS.detach()
    #         # self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #         # self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #         self.log("train/aeloss", aeloss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
    #         self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            
    #         return VLOSS + aeloss
    #     if optimizer_idx == 1:
    #         # discriminator
    #         discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train"
    #             # , cond=vasl 
    #         )
    #         # self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #         # self.log_dict(log_dict_disc, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #         self.log("train/discloss", discloss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
    #         self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
    #         print('!!!!!!!!!!!!!', discloss, discloss.shape)
    #         assert False
    #         return discloss



    def drcQ(self, x, w):
        quant, qloss = self.encode(x)
        quant_c = w(quant)
        xrec = self.decode(quant + quant_c)
        return xrec, qloss
    
    def training_step_for_drc(self, x, w, clabel, S, CE, DRC, bi, cv):
        # Vorg = self.vseg((((x.cpu() +1)*127.5).detach())).detach()
        xrec, qloss = self.drcQ(x, w)

        xr = self.vqgan_fn_phi_denormalize(xrec).detach()
        if bi % 400 == 0 or (bi-1) % 400 == 0:
            signal_save(torch.cat([
                (x+1) * 127.5,
                xr,
            ], dim=0), f'/content/syn_0to{cv}.png', stype='img', sparams={'chw2hwc': True, 'nrow': xr.shape[0]})

        # Vrec = self.vseg(xr.cpu()).detach()
        # Vrec = dzq_dz_eq1(Vrec, xrec)
        xr = dzq_dz_eq1(xr, xrec)
        xr = (xr - (self.dr_classifire_normalize_mean * 255)) / (self.dr_classifire_normalize_std * 255)
        drloss = CE(S(DRC(xr)[0]), clabel)


        # Vorg, Vrec = self.get_V2(x, xr)
        

        
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="train"
            # , cond=vasl
        )
        
        # aeloss = torch.mean(torch.abs(x - xrec) + 0.1 * self.loss.perceptual_loss(x, xrec))
        # VLOSS = 0.5 * torch.mean(torch.abs(Vorg - Vrec)).log() #+ 0.1 * self.loss.perceptual_loss(Vorg, Vrec)).log()
        # print(VLOSS, aeloss, drloss)
        return aeloss + drloss
        
        
        # if optimizer_idx == 1:
        #     # discriminator
        #     discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train"
        #         # , cond=vasl 
        #     )
        #     # self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        #     # self.log_dict(log_dict_disc, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        #     self.log("train/discloss", discloss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        #     self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        #     print('!!!!!!!!!!!!!', discloss, discloss.shape)
        #     assert False
        #     return discloss
        
    
    
    def validation_step_syn(self, batch, batch_idx):
        print('validation_step_syn')
        return
    def validation_step(self, batch, batch_idx):
        # print('validation_step')
        # logged = self.log_images(batch, fName='badRec/' + random_string())
        return
        # return
        # T = A.Compose([
        #     A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
        #     ToTensorV2()
        # ])
        # R = []
        # rr = self.vqgan_fn_phi_denormalize(logged['inputs'])
        # for i in range(logged['inputs'].shape[0]):
        #     r = rr[i]
        #     R.append(T(image=rearrange(r, 'c h w -> h w c').cpu().detach().numpy().astype(np.uint8))['image'].unsqueeze(0).to('cuda'))
        # signal_save(torch.cat([
        #     rr,
        #     torch.cat(R, dim=0)
            
        # ], dim=0), '/content/D1.png', stype='img', sparams={'chw2hwc': True, 'nrow': 4})
        
        
        
        
        # Tf0 = A.Compose([
        #     A.Resize(256, 256),
        #     ToTensorV2()
        # ])
        # Tf = A.Compose([
        #     A.Resize(256, 256),
        #     A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
        #     A.ToGray(),
        #     ToTensorV2()
        # ])
        # Tm = A.Compose([
        #     A.Resize(256, 256),
        #     ToTensorV2()
        # ])
        # fundus_drive = np.array(Image.open('/content/dataset_drive/DRIVE/training/images/24_training.tif'))
        # fd = Tf0(image=fundus_drive)['image'].unsqueeze(0)
        # # fundus_drive = (fundus_drive[:,:,0] + fundus_drive[:,:,1] + fundus_drive[:,:,2]) / 3
        # fundus_drive = fundus_drive.astype(np.uint8)
        # fundus_mask = np.array(Image.open('/content/dataset_drive/DRIVE/training/1st_manual/24_manual1.gif'))
        
        # fundus_drive = Tf(image=fundus_drive)['image'].unsqueeze(0)
        # fundus_mask = Tm(image=fundus_mask)['image'].unsqueeze(0)
        # fundus_mask = torch.cat([fundus_mask,fundus_mask,fundus_mask], dim=1)
        # print(fundus_drive.shape, fundus_mask.shape)
        
        # patches_f0 = fd.unfold(2, 64, 32).unfold(3, 64, 32)
        # patches_f = fundus_drive.unfold(2, 64, 32).unfold(3, 64, 32)
        # patches_m = fundus_mask.unfold(2, 64, 32).unfold(3, 64, 32)




        # signal_save(torch.cat([#fundus_drive, fundus_mask, 
        #     torch.cat([
        #         patches_f0[0:1, :, 3,3,:,:], patches_f[0:1, :, 3,3,:,:], patches_m[0:1, :, 3,3,:,:],
        #         patches_f0[0:1, :, 2,5,:,:], patches_f[0:1, :, 2,5,:,:], patches_m[0:1, :, 2,5,:,:]
        # ], dim=0)
        # ], dim=0), '/content/dri2.png',stype='img', sparams={'chw2hwc': True, 'nrow': 6})
        
        # self.save_phi(torch.cat([logged['inputs'], logged['reconstructions']], dim=0), '/content/inp.png', nrow=4)
        
        
        
        
        
        
        # assert False
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        Vorg, Vrec = self.get_V(x, xrec)
        # vasl = None # self.get_input(batch, 'vasl')
        # vasl = self.get_input(batch, 'vasl')
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="val"
            # , cond=vasl
        )
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="val"
            # , cond=vasl
        )
        # rec_loss = log_dict_ae["val/rec_loss"]
        # self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict({'val/aeloss':aeloss, 'val/discloss':discloss}, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        # return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
                                  list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
        return [opt_ae], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        # if kwargs.get('ignore', True):
        #     return
        """this function is must be exist for ptCallback.ImageLoggerBase"""
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        
        # T = A.Compose([
        #     A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
        #     ToTensorV2()
        # ])
        # R = []
        # rr = self.vqgan_fn_phi_denormalize(x)
        # for i in range(x.shape[0]):
        #     r = rr[i]
        #     R.append(T(image=rearrange(r, 'c h w -> h w c').cpu().detach().numpy().astype(np.uint8))['image'].unsqueeze(0).to('cuda'))
        # x = torch.cat(R, dim=0)
        # print('--------------------->', x.shape)

        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        pathimg = f'/content/{kwargs.get("fName", "rec")}.png'
        signal_save(torch.cat([
            (x + 1 ) * 127.5,
            self.vqgan_fn_phi_denormalize(xrec)
        ], dim=0), pathimg, stype='img', sparams={'chw2hwc': True, 'nrow': 4})
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
