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

import signal as sig
from signal import signal
dr_transformer0 = A.Compose([
    ToTensorV2()
])


def fold3d(x, gp=None):
    """
        x is x3d
        gp is grid patch size
    """
    B, ch, h, w = x.shape
    gp = gp if gp else int(ch ** .5)
    return x.view(B, gp, gp, 1, h, w).permute(0, 3, 1, 4, 2, 5).contiguous().view(B, 1, gp*h, gp*w) 

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
    
    
    def get_theta_tx_ty(self, xs_256ch, xcl_256ch): # DELETE 40 M parameter!!!!!
        # print('before xs_256ch', xs_256ch.shape)
        # print('before xcl_256ch', xcl_256ch.shape)

        xs_256ch = self.cnn_xscl_256x32_256x16(xs_256ch)
        xs_256ch = xs_256ch + torch.relu(self.cnn_xscl_256x16_256x16(xs_256ch))
        xs_256ch = self.cnn_xscl_bn256(xs_256ch)
        
        xcl_256ch = self.cnn_xscl_256x32_256x16(xcl_256ch)
        xcl_256ch = xcl_256ch + torch.relu(self.cnn_xscl_256x16_256x16(xcl_256ch))
        xcl_256ch = self.cnn_xscl_bn256(xcl_256ch)

        # print('after xs_256ch', xs_256ch.shape)
        # print('after xcl_256ch', xcl_256ch.shape)

        x = torch.cat([xs_256ch, xcl_256ch], dim=1) # Bx512x16x16
        # print('cat', x.shape)

        x = self.cnn_xscl_512x16_512x8(x)
        x = x + torch.relu(self.cnn_xscl_512x8_512x8(x))
        x = self.cnn_xscl_bn512(x)

        x = self.cnn_xscl_512x8_1024x4(x)
        x = x + torch.relu(self.cnn_xscl_1024x4_1024x4(x))
        x = self.cnn_xscl_bn1024(x)

        x = self.cnn_xscl_1024x4_1024x1(x).flatten(1) # Bx1024
        # print('end_cnn x', x.shape)

        x = self.fc_xscl(x)
        # print('end_cnn x', x.shape, x.sum())
        theta = self.fc_xscl_theta(x)
        tx = self.fc_xscl_tx(x)
        ty = self.fc_xscl_ty(x)
        # print('theta, tx, ty', theta, tx, ty)
        return theta, tx, ty
        
    def start(self): # TODO
        # self.theta = 0.0
        # self.tx = 0.0
        # self.ty = 0.0

        self.conv_catskip_0 = torch.nn.Conv2d(512, 256, kernel_size=1)
        self.conv_crosover_adjustion_in_ch = torch.nn.Conv2d(512, 256, kernel_size=1)

        # self.cnn_xscl_256x32_256x16 = torch.nn.Conv2d(256, 256, 4,2,1) # 1 M
        # self.cnn_xscl_256x16_256x16 = torch.nn.Conv2d(256, 256, 3,1,1)
        # self.cnn_xscl_bn256 = torch.nn.BatchNorm2d(256)
        
        # self.cnn_xscl_512x16_512x8 = torch.nn.Conv2d(512, 512, 4,2,1) # 4.2 M
        # self.cnn_xscl_512x8_512x8 = torch.nn.Conv2d(512, 512, 3,1,1) # 2.4 M
        # self.cnn_xscl_bn512 = torch.nn.BatchNorm2d(512)
        
        # self.cnn_xscl_512x8_1024x4 = torch.nn.Conv2d(512, 1024, 4,2,1) # 8.4 M
        # self.cnn_xscl_1024x4_1024x4 = torch.nn.Conv2d(1024, 1024, 3,1,1) # 9.4 M
        # self.cnn_xscl_bn1024 = torch.nn.BatchNorm2d(1024)
        
        # self.cnn_xscl_1024x4_1024x1 = torch.nn.Conv2d(1024, 1024, 4,2,0) # 16.8 M

        # self.fc_xscl = torch.nn.Sequential(
        #     torch.nn.Linear(1024, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.2),
        #     torch.nn.Linear(256, 32),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.2),
        # )
        # self.fc_xscl_theta = torch.nn.Sequential(
        #     torch.nn.Linear(32, 1),
        #     torch.nn.Tanh()
        # )
        # self.fc_xscl_tx = torch.nn.Sequential(
        #     torch.nn.Linear(32, 1),
        #     torch.nn.Tanh()
        # )
        # self.fc_xscl_ty = torch.nn.Sequential(
        #     torch.nn.Linear(32, 1),
        #     torch.nn.Tanh()
        # )

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
    
    def unfold(self, x, Ps, Nk):
        return x.unfold(2, Ps, Ps).unfold(3, Ps, Ps).contiguous().view(-1, int(Nk*Nk), Ps, Ps).permute(1,0,2,3).contiguous()

    # def fold(self, x, batchsize=4):
    #     print('@@@@@@@@@@@@', x.shape)
    #     num_patches, B_CH, jigsaw_h, jigsaw_w = x.shape
    #     c = B_CH // batchsize
    #     batch_size = batchsize
    #     # x = x.unsqueeze(0)
    #     grid_size = int(num_patches ** .5)
    #     grid_size = (grid_size, grid_size)
    #     # x shape is batch_size x num_patches x c x jigsaw_h x jigsaw_w
    #     # batch_size, num_patches, c, jigsaw_h, jigsaw_w = x.size()
    #     # print('****************', batch_size, num_patches, c, jigsaw_h, jigsaw_w)
    #     x_image = x.view(batch_size, grid_size[0], grid_size[1], c, jigsaw_h, jigsaw_w)
    #     output_h = grid_size[0] * jigsaw_h
    #     output_w = grid_size[1] * jigsaw_w
    #     x_image = x_image.permute(0, 3, 1, 4, 2, 5).contiguous()
    #     x_image = x_image.view(batch_size, c, output_h, output_w)
    #     return x_image.view(batchsize, -1, output_h, output_w)
    
    def fold(self, x, grid_size):
        x = x.unsqueeze(0)
        grid_size = (grid_size,grid_size)
        # x shape is batch_size x num_patches x c x jigsaw_h x jigsaw_w
        batch_size, num_patches, c, jigsaw_h, jigsaw_w = x.size()
        assert num_patches == grid_size[0] * grid_size[1]
        x_image = x.view(batch_size, grid_size[0], grid_size[1], c, jigsaw_h, jigsaw_w)
        output_h = grid_size[0] * jigsaw_h
        output_w = grid_size[1] * jigsaw_w
        x_image = x_image.permute(0, 3, 1, 4, 2, 5).contiguous()
        x_image = x_image.view(batch_size, c, output_h, output_w)
        return x_image


    def forward(self, xs, Xc, xcl_pure):
        """
            xs: source color fundus
            Xc: conditional color fundus | ROT version
            # xcl_pure: none ROT version of Xcl (attendend)
            # UPDATE: xcl_pure: is xcl === ROT (attendend)
        """
        Sk = 64 # patch size
        Nk = 4  # num patches in each row and column
        q_eye16 = torch.eye(16, dtype=torch.float32, device=self.device).detach()
        
        # signal_save(torch.cat([
        #     (xs+1)* 127.5, 
        #     (Xc+1)* 127.5, #ROT
        #     (xcl_pure+1)* 127.5, # none ROT 
        # ], dim=0), f'/content/export/fip.png', stype='img', sparams={'chw2hwc': True, 'nrow': 4})
        
        xs0 = xs
        Xc0 = Xc
        xs = self.unfold(xs, Sk, Nk) # PATCH version | self.ssf1(xs0, self.fold(xs, Nk), xs)
        Xc = self.unfold(Xc, Sk, Nk) # PATCH version | self.ssf1(xc0, self.fold(xc, Nk), xc)

        # print('############ 1', xs0.shape, Xc0.shape)
        # print('############ 2', xs.shape, Xc.shape)
        # signal_save(torch.cat([
        #     (xs+1)* 127.5, 
        #     (Xc+1)* 127.5, 
        # ], dim=0), f'/content/export/fp.png', stype='img', sparams={'chw2hwc': True, 'nrow': 4})
        
        # signal_save(torch.cat([
        #     (xs0+1)* 127.5, 
        #     (Xc0+1)* 127.5, 
        # ], dim=0), f'/content/export/fnp.png', stype='img', sparams={'chw2hwc': True, 'nrow': 4})
        

        # signal_save(torch.cat([
        #     (self.fold(xs, Nk)+1)* 127.5, # unpatch
        #     (self.fold(Xc, Nk)+1)* 127.5, # unpatch
        # ], dim=0), f'/content/export/fnp2.png', stype='img', sparams={'chw2hwc': True, 'nrow': 4})

        ###############################################################assert False

        hc, h_ilevel1_xcl, h_endDownSampling_xcl = self.encoder(Xc) # INFO patch version
        # print('before', hc.shape, h_ilevel1_xcl.shape, h_endDownSampling_xcl.shape)
        hc = self.fold(hc, Nk) # before: torch.Size([16, 256, 4, 4])
        h_ilevel1_xcl = self.fold(h_ilevel1_xcl, Nk) # before: torch.Size([16, 128, 64, 64])
        h_endDownSampling_xcl = self.fold(h_endDownSampling_xcl, Nk) # before: torch.Size([16, 512, 4, 4])
        # print('after', hc.shape, h_ilevel1_xcl.shape, h_endDownSampling_xcl.shape)

        hc = self.quant_conv(hc)
        # print('!!! hc', hc.shape)
        quanth, diff_xc = self.quantize(hc)
        # print('!!! quanth', quanth.shape)
        hc_new = self.post_quant_conv(quanth)
        # print('!!! hc_new', hc_new.shape)
        _Qh = self.conv_catskip_0(torch.cat([hc_new, hc], dim=1))
        # print('!!! _Qh', _Qh.shape)
        Qh = q_eye16 * _Qh
        # print('!!! Qh', Qh.shape)
        Qj = (1-q_eye16) * _Qh
        # print('!!! Qj', Qj.shape)

        h, h_ilevel1, h_endDownSampling = self.encoder(xs) # INFO patch version
        # print('@@ before', h.shape, h_ilevel1.shape, h_endDownSampling.shape)
        h = self.fold(h, Nk)
        h_ilevel1 = self.fold(h_ilevel1, Nk)
        h_endDownSampling = self.fold(h_endDownSampling, Nk)
        # print('@@ after', h.shape, h_ilevel1.shape, h_endDownSampling.shape)


        h = self.quant_conv(h)
        # print('$$$ h', h.shape)
        quant, diff = self.quantize(h)
        # print('$$$ quant', quant.shape)
        h_new = self.post_quant_conv(quant)
        # print('$$$ h_new', h_new.shape)
        Qorg = self.conv_catskip_0(torch.cat([h_new, h], dim=1))
        # print('$$$ Qorg', Qorg.shape)
        Qcrossover = (1-q_eye16) * Qorg + Qh # crossover/exchange of latent codes.
        # print('$$$ Qcrossover', Qcrossover.shape)
        Q = self.conv_crosover_adjustion_in_ch(torch.cat([Qcrossover, Qorg], dim=1))
        # print('$$$ Q', Q.shape)
        Q0 = self.conv_crosover_adjustion_in_ch(torch.cat([Qorg, Qorg], dim=1))
        # print('$$$ Q0', Q0.shape)

        dec_Xc = self.decoder( # Xc -> Xcl (attendend version) ; givven only digonal of Qh.
            Qh, # PATCH version
            None,
            h_ilevel1_xcl,
            h_endDownSampling_xcl,
            flag=False, # spade off
            flag2=False
        ) # Note: add skip connection
        # print('dec_Xc ))) before', dec_Xc.shape) # dec_Xc ))) before torch.Size([1, 1, 256, 256])
        dec_Xc = Xc0 - 0.8 * Xc0 * (1 - torch.sigmoid(dec_Xc))
        # print('dec_Xc ))) after', dec_Xc.shape) # dec_Xc ))) after torch.Size([1, 3, 256, 256])
        # dec_Xc is ROT

        dec_xscl = self.decoder( # xs, xcl -> xscl ; givven digonal of Qh and others of Q.
            Q, #Q, # PATCH version 
            xcl_pure, # SPADE # `xcl ROT version`
            h_ilevel1, 
            h_endDownSampling
        ) # Note: add skip connection
        # print('dec_xscl', dec_xscl.shape) # dec_xscl torch.Size([1, 1, 256, 256])

        dec_Xs = self.decoder( # xs, xcl -> xscl ; givven digonal of Qh and others of Q.
            Q0, 
            None,
            h_ilevel1, 
            h_endDownSampling,
            flag=False # spade off
        ) # Note: add skip connection
        # print('dec_Xs', dec_Xs.shape) # dec_Xs torch.Size([1, 1, 256, 256])
        
        return xs0 + dec_Xs, xs0 + dec_xscl, diff, dec_Xc, diff_xc

    
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
    
    
    def dice_static_metric(self, inputs, target):
        num = target.shape[0]
        inputs = inputs.reshape(num, -1)
        target = target.reshape(num, -1).detach()

        intersection = (inputs * target).sum(1)
        union = (inputs.sum(1) + target.sum(1)).detach()
        dice = (2. * intersection) / (union + 1e-8)
        # print()
        # print('intersection, union', intersection.shape, union.shape, intersection.sum(), union.sum())
        # print('dice', dice.shape, dice.dtype, dice.sum(), -dice.log())
        return -dice.log()
        # dice = dice.sum()/num
        # return dice
    
    # NOTE: Syn Idea
    def training_step(self, batch, batch_idx, optimizer_idx):
        cidx = 1
        return self.training_step_slave(batch, batch_idx, optimizer_idx, cidx=cidx, split='train_')
    
    def training_step_slave(self, batch, batch_idx, optimizer_idx, cidx, split):
        xs = batch['xs']
        xsl = batch['xsl']
        xsc = batch['xsc']
        xsf = batch['xsf']
        xslmask = batch['xslmask']

        xc = batch['xc'][cidx] # ROT
        xcl = batch['xcl'][cidx] # ROT
        xcc = batch['xcc'][cidx] # ROT
        xcf = batch['xcf'][cidx] # ROT
        xclmask = batch['xclmask'][cidx] # ROT
        
        ynl = batch['ynl'][cidx][0] # I dont know why is a tuple!!
        y_edit = batch['y_edit'].item()
        print('@@@@@@@@@@@@', y_edit)
        assert False


        # DELETE
        # mRGB = xs.detach().mean(dim=[2,3]).clone().detach()
        # m_rgb = (torch.zeros((1,3,256,256), dtype=self.dtype) + torch.tensor(mRGB, device=self.device).unsqueeze(-1).unsqueeze(-1)).clone().detach()


        # print('@@@@@@@@@@@', batch['yl'], batch['y_edit'])
        # print(xs.shape, xs.dtype, xs.min().item(), xs.max().item())
        # print(xsl.shape, xsl.dtype, xsl.min().item(), xsl.max().item())
        # print(xsc.shape, xsc.dtype, xsc.min().item(), xsc.max().item())
        # print(xsf.shape, xsf.dtype, xsf.min().item(), xsf.max().item())
        # print(xslmask.shape, xslmask.dtype, xslmask.min().item(), xslmask.max().item())
        
        # print('!!!!!!!!!!!', ynl) # !!!!!!!!!!! [01] # NOTE: in this case in another case it can be '2' or '[34]'
        # print(xc.shape, xc.dtype, xc.min().item(), xc.max().item())
        # print(xcl.shape, xcl.dtype, xcl.min().item(), xcl.max().item())
        # print(xcc.shape, xcc.dtype, xcc.min().item(), xcc.max().item())
        # print(xcf.shape, xcf.dtype, xcf.min().item(), xcf.max().item())
        # print(xclmask.shape, xclmask.dtype, xclmask.min().item(), xclmask.max().item())
        # print('-'*30)



        # NOTE: Mask design
        M_union_L_xs_xc = ((xslmask + xclmask) - (xslmask * xclmask)).detach()
        M_L_xs_mines_xc = (xslmask - (xslmask * xclmask)).detach() # TODO Interpolation
        M_C_Union = ((1 - M_union_L_xs_xc) * xsf).detach() #shape:torch.Size([1, 1, 256, 256]) # reconstruct xs
        M_xrec_xcl = (xclmask * xsf).detach() # reconstruct xc
        # signal_save(torch.cat([
        #     (M_union_L_xs_xc) * 255, 
        #     (M_L_xs_mines_xc) * 255, 
        #     (M_xrec_xs) * 255, 
        #     (M_xrec_xcl) * 255, 
        # ], dim=0), f'/content/export/masks.png', stype='img', sparams={'chw2hwc': True, 'nrow': 4})

        xh = (M_C_Union * xs + xclmask * xcl).mean(dim=1, keepdim=True)
        print('@@@@@@@@@@@@@@', xh.shape)
        
        rec_xs, rec_xscl, qloss, rec_xcl, qcloss = self(xs, xc, xcl) # xcl is ROT.

        # signal_save(torch.cat([
        #     (rec_xs+1) * 127.5, 
        #     (rec_xscl+1) * 127.5, 
        #     (rec_xcl+1) * 127.5, 
        # ], dim=0), f'/content/export/rec.png', stype='img', sparams={'chw2hwc': True, 'nrow': 3})


        xscl_final = self.synf(xh, xs, xcl, rec_xscl, M_C_Union, xclmask)
        # print(rec_xs.shape, rec_xscl.shape, qloss.shape, rec_xcl.shape, qcloss.shape) # torch.Size([1, 3, 256, 256]) torch.Size([1, 3, 256, 256]) torch.Size([]) torch.Size([1, 3, 256, 256]) torch.Size([])


        if optimizer_idx == 0: # reconstruction/generator process
            # INFO *** reconstruction xcl diesis === diagonal learning *** => only Geometry loss calcualated!
            rec_xcl = rec_xcl * xcf # learning diagonal of quantizer such that save disease features -> reconstruction Xcl
            Xcl__groundtrouth = xcl * xcf
            Xcl_aeloss, Xcl_log_dict_ae = self.loss(qcloss, Xcl__groundtrouth, rec_xcl, 0, self.global_step, 
                                            last_layer=self.get_last_layer(flag2=False), dw=0, split=split + 'xcl')

            # INFO: reconstruction xs and xs~ => only Geometry loss calcualated!
            rec__xs0 = xsf * rec_xs
            rec__xs_graoundtrouth0 = (xsf * xs).detach()
            xss_aeloss, xss_log_dict_ae = self.loss(qloss, rec__xs_graoundtrouth0, rec__xs0, 0, self.global_step, 
                                            last_layer=self.get_last_layer(flag2=True), dw=0, split=split + 'xss')
            
            # INFO: reconstruction xs surface === complement of union => only Geometry loss calculated!
            rec__xs = M_C_Union * rec_xscl # outside of both diesis features -> reconstruction xs
            rec__xs_graoundtrouth = (M_C_Union * xs).detach()
            xs_aeloss, xs_log_dict_ae = self.loss(qloss, rec__xs_graoundtrouth, rec__xs, 0, self.global_step, 
                                            last_layer=self.get_last_layer(flag2=True), dw=0, split=split + 'xs')
            
            # INFO: reconstruction xc diesis === xclmask a part of union === step1: import leasions from xc => only Geometry loss calculated!
            # rec__Xc = M_xrec_xcl * rec_xscl # on xc diesis features -> reconstruction  xc
            # rec__Xc_graoundtrouth = (M_xrec_xcl * xc).detach()
            # Xc_aeloss, Xc_log_dict_ae = self.loss(qloss, rec__Xc_graoundtrouth, rec__Xc, 0, self.global_step, 
            #                                 last_layer=self.get_last_layer(flag2=True), dw=0, split=split + 'xc')
            
            # INFO: synthesis process on uinon area === xscl_final === step2: 
            


            Xc_aeloss = 0 # DELETE
            total_loss = xss_aeloss + xs_aeloss + Xc_aeloss + Xcl_aeloss

            # print('###############', total_loss.item(), xss_aeloss.item(), xs_aeloss.item(), Xc_aeloss.item(), Xcl_aeloss.item()) # ############### 1.8206264972686768 0.3734074831008911 0.4133034348487854 0.3042736053466797 0.7296419143676758
            
            return total_loss

        if optimizer_idx == 1: # discriminator
            assert False

    def synf(self, xh, xs, xcl, xscl, m_c_union, xclmask):
        """xscl0 has information on `1 - m_union` and we want here, add information in `m_union area` to xscl0"""
        
        # signal_save(torch.cat([
        #     (m_rgb +1) * 127.5, 
        # ], dim=0), f'/content/export/m_rgb.png', stype='img', sparams={'chw2hwc': True, 'nrow': 3})
        
        print('22 @@@@@@@@@@@@@@@@@@@@@@@@@',xclmask.shape)
        signal_save(torch.cat([
            (torch.cat([xh, xh, xh], dim=1)+1) * 127.5, 
            (xs+1) * 127.5, 
            (xcl+1) * 127.5, 
            (xscl+1) * 127.5, 
            torch.cat([m_c_union, m_c_union, m_c_union], dim=1) * 255,
            torch.cat([xclmask, xclmask, xclmask], dim=1) * 255,
        ], dim=0), f'/content/export/syn.png', stype='img', sparams={'chw2hwc': True, 'nrow': 3})


        assert False
        self.encoder.fwd_syn_step()
        return xscl0




    def test00000(self):

        h = torch.tensor(0.01).to(self.device)
        # if batch_idx % 500 == 0:
        #     self.log_images(batch, ignore=False)
        # x = self.get_input(batch, self.image_key)

        cidx = 2 # 0:G(0,1) 1:G(2) 2:G(3,4)
        
        
        
        xs = batch['xs'] # fundus source. bipolar
        xc = batch['xc'][cidx] # fundus condition. bipolar. shape:(Bxchxhxw)
        xc_np = batch['xc_np'][cidx].cpu().numpy()[0] # fundus condition. bipolar. shape:(Bxhxwxch)
        xs_lesion = batch['xs_lesion']
        xc_lesion = batch['xc_lesion'][cidx] # fundus condition attendend version. bipolar. shape:(Bxchxhxw)
        xc_lesion_np = batch['xc_lesion_np'][cidx].cpu().numpy()[0] # remove batch dimention. # RGB fundus condition. bipolar. shape:(Bxchxhxw)
        xs_fundusmask = batch['xs_fundusmask'][0] # remove batch dimention # binary
        xc_fundusmask = batch['xc_fundusmask'][cidx][0] # remove batch dimention # binary
        xc_fundusmask_np = batch['xc_fundusmask'][cidx][0].cpu().numpy() # remove batch dimention # binary
        xs_cunvexhull = batch['xs_cunvexhull'][0] # remove batch dimention # Bxhxwxch=1
        xc_cunvexhull = batch['xc_cunvexhull'][cidx][0] # Bxhxwxch=1
        xc_cunvexhull_np = batch['xc_cunvexhull'][cidx].cpu().numpy()[0] # Bxhxwxch=1
        Lmask_xs = batch['Lmask_xs'][0] # remove batch dimention # binary of diesis features
        Lmask_xc = batch['Lmask_xc'][cidx][0] # remove batch dimention # binary of diesis features
        Lmask_xc_np = batch['Lmask_xc'][cidx].cpu().numpy()[0] # remove batch dimention # binary of diesis features
        
        # INFO: print variables!
        # print('xs', xs.shape, xs.dtype, xs.min().item(), xs.max().item())
        # print('xc', xc.shape, xc.dtype, xc.min().item(), xc.max().item())
        # print('xc_np', xc_np.shape, xc_np.dtype, xc_np.min().item(), xc_np.max().item())
        # print('xs_lesion', xs_lesion.shape, xs_lesion.dtype, xs_lesion.min().item(), xs_lesion.max().item())
        # print('xc_lesion', xc_lesion.shape, xc_lesion.dtype, xc_lesion.min().item(), xc_lesion.max().item())
        # print('xc_lesion_np', xc_lesion_np.shape, xc_lesion_np.dtype, xc_lesion_np.min().item(), xc_lesion_np.max().item())
        # print('xs_fundusmask', xs_fundusmask.shape, xs_fundusmask.dtype, xs_fundusmask.min().item(), xs_fundusmask.max().item())
        # print('xc_fundusmask', xc_fundusmask.shape, xc_fundusmask.dtype, xc_fundusmask.min().item(), xc_fundusmask.max().item())
        # print('xs_cunvexhull', xs_cunvexhull.shape, xs_cunvexhull.dtype, xs_cunvexhull.min().item(), xs_cunvexhull.max().item())
        # print('xc_cunvexhull', xc_cunvexhull.shape, xc_cunvexhull.dtype, xc_cunvexhull.min().item(), xc_cunvexhull.max().item())
        # print('xc_cunvexhull_np', xc_cunvexhull_np.shape, xc_cunvexhull_np.dtype, xc_cunvexhull_np.min().item(), xc_cunvexhull_np.max().item())
        # print('Lmask_xs', Lmask_xs.shape, Lmask_xs.dtype, Lmask_xs.min().item(), Lmask_xs.max().item())
        # print('Lmask_xc', Lmask_xc.shape, Lmask_xc.dtype, Lmask_xc.min().item(), Lmask_xc.max().item())
        # print('Lmask_xc_np', Lmask_xc_np.shape, Lmask_xc_np.dtype, Lmask_xc_np.min().item(), Lmask_xc_np.max().item())
        
        # INFO: affine parammetters
        theta = torch.tensor(self.theta, dtype=torch.float32, device=self.device) + .3
        tx = torch.tensor(self.tx, dtype=torch.float32, device=self.device) - .1
        ty = torch.tensor(self.ty, dtype=torch.float32, device=self.device) + .4

        # INFO: ROT
        Xc = dr_transformer0(image=ROT(xc_np, theta=theta, tx=tx, ty=ty))['image'].unsqueeze(0).to(self.device)
        Xcl = dr_transformer0(image=ROT(xc_lesion_np, theta=theta, tx=tx, ty=ty))['image'].unsqueeze(0).to(self.device)
        Xcf = dr_transformer0(image=ROT(xc_fundusmask_np, theta=theta, tx=tx, ty=ty))['image'].squeeze().to(self.device)
        Xcm = dr_transformer0(image=ROT(Lmask_xc_np, theta=theta, tx=tx, ty=ty))['image'].squeeze().to(self.device)
        mue = dr_transformer0(image=ROT(xc_cunvexhull_np, theta=theta, tx=tx, ty=ty))['image'].squeeze().to(self.device)
        mue_plus_h_tx = dr_transformer0(image=ROT(xc_cunvexhull_np, theta=theta, tx=tx + h, ty=ty))['image'].squeeze().to(self.device)
        mue_plus_h_ty = dr_transformer0(image=ROT(xc_cunvexhull_np, theta=theta, tx=tx, ty=ty + h))['image'].squeeze().to(self.device)
        mue_plus_h_theta = dr_transformer0(image=ROT(xc_cunvexhull_np, theta=theta + h, tx=tx, ty=ty))['image'].squeeze().to(self.device)

        # INFO: print ROT
        # Xc torch.Size([1, 3, 256, 256]) torch.float32 -1.0 1.0
        # Xcl torch.Size([1, 3, 256, 256]) torch.float32 -1.0 0.2851102352142334
        # Xcm torch.Size([256, 256]) torch.float32 0.0 1.0
        # mue torch.Size([256, 256]) torch.float32 0.0 1.0
        # mue_plus_h_tx torch.Size([256, 256]) torch.float32 0.0 1.0
        # mue_plus_h_ty torch.Size([256, 256]) torch.float32 0.0 1.0
        # mue_plus_h_theta torch.Size([256, 256]) torch.float32 0.0 1.0
        # ###############################################################################
        # print('Xc', Xc.shape, Xc.dtype, Xc.min().item(), Xc.max().item()) # 1x3x256x256
        # print('Xcl', Xcl.shape, Xcl.dtype, Xcl.min().item(), Xcl.max().item()) # 1x3x256x256
        # print('Xcm', Xcm.shape, Xcm.dtype, Xcm.min().item(), Xcm.max().item()) # 256x256
        # print('mue', mue.shape, mue.dtype, mue.min().item(), mue.max().item()) # 256x256
        # print('mue_plus_h_tx', mue_plus_h_tx.shape, mue_plus_h_tx.dtype, mue_plus_h_tx.min().item(), mue_plus_h_tx.max().item()) # 256x256
        # print('mue_plus_h_ty', mue_plus_h_ty.shape, mue_plus_h_ty.dtype, mue_plus_h_ty.min().item(), mue_plus_h_ty.max().item()) # 256x256
        # print('mue_plus_h_theta', mue_plus_h_theta.shape, mue_plus_h_theta.dtype, mue_plus_h_theta.min().item(), mue_plus_h_theta.max().item()) # 256x256


        rec_xscl, qloss, rec_Xcl, qcloss = self(xs, Xc, xc_lesion) # xc_lesion is none rot version of Xcl.

        # theta, tx, ty = self.get_theta_tx_ty(h_ilevel4_xs.detach(), h_ilevel4_xcl)
        # theta.register_hook(lambda grad: print('theta', grad))
        # tx.register_hook(lambda grad: print('tx', grad))
        # ty.register_hook(lambda grad: print('ty', grad))

        # INFO: signal save
        # signal_save(torch.cat([
        #     (xs+1) * 127.5,
        #     (xc+1) * 127.5, # same as xc_np
        #     (Xc+1) * 127.5, # ROT version of xc
        #     (xc_lesion+1) * 127.5, # is pure
        #     self.ssf0(xs_fundusmask * 255),
        #     self.ssf0(xc_fundusmask * 255),
        #     self.ssf0(Xcf * 255),
        #     (Xcl+1) * 127.5, # ROT version of xc_lesion
        #     self.ssf0(Lmask_xs * 255),
        #     self.ssf0(Lmask_xc * 255), # pure
        #     self.ssf0(Xcm * 255), # rot version of Lmask_xc
        #     self.ssf0(xc_cunvexhull * 255), # pure
        #     self.ssf0(mue * 255), # rot version of xc_cunvexhull
        #     self.ssf0(mue_plus_h_tx * 255),
        #     self.ssf0(mue_plus_h_ty * 255),
        #     self.ssf0(mue_plus_h_theta * 255),
        # ], dim=0), f'/content/export/1.png', stype='img', sparams={'chw2hwc': True, 'nrow': 4})

        


        # INFO: loss function
        iou = self.dice_static_metric(mue, xs_fundusmask).detach()
        iou_plus_h_tx = self.dice_static_metric(mue_plus_h_tx, xs_fundusmask).detach()
        iou_plus_h_ty = self.dice_static_metric(mue_plus_h_ty, xs_fundusmask).detach()
        iou_plus_h_theta = self.dice_static_metric(mue_plus_h_theta, xs_fundusmask).detach()

        print('11111111', iou, iou_plus_h_tx, iou_plus_h_ty, iou_plus_h_theta)

        D_tx = ((iou_plus_h_tx - iou) / h).flatten().detach()
        D_ty = ((iou_plus_h_ty - iou) / h).flatten().detach()
        D_theta = ((iou_plus_h_theta - iou) / h).flatten().detach()
        print('22222222', D_tx, D_ty, D_theta)

        # iou = dzq_dz_eq1(iou, D_theta * theta + D_tx * tx + D_ty * ty)

        print('333333333333', iou, iou.shape, iou.dtype)


        if optimizer_idx == 0: # reconstruction
            M_union_L_xs_xc = ((Lmask_xs + Xcm) - (Lmask_xs * Xcm)).detach()
            M_L_xs_mines_xc = (Lmask_xs - (Lmask_xs * Xcm)).detach() # Interpolation
            M_xrec_xs = ((1 - M_union_L_xs_xc) * xs_fundusmask).detach() # reconstruct xs
            M_xrec_xcl = (Xcm * xs_fundusmask).detach() # reconstruct xc
            
            
            print('!!!!!!!!!', M_xrec_xs.min().item(), M_xrec_xs.max().item(), M_xrec_xs.sum().item(), M_xrec_xs.shape)

            rec__xs = M_xrec_xs * rec_xscl # outside of both diesis features -> reconstruction xs
            rec__xs_graoundtrouth = (M_xrec_xs * xs).detach()
            xs_aeloss, xs_log_dict_ae = self.loss(qloss, rec__xs_graoundtrouth, rec__xs, 0, self.global_step, 
                                            last_layer=self.get_last_layer(flag=True), split="train")
            
            rec__Xc = M_xrec_xcl * rec_xscl # on Xc diesis features -> reconstruction  Xc #TODO ???
            rec__Xc_graoundtrouth = (M_xrec_xcl * Xc).detach()
            Xc_aeloss, Xc_log_dict_ae = self.loss(qloss, rec__Xc_graoundtrouth, rec__Xc, 0, self.global_step, 
                                            last_layer=self.get_last_layer(flag=True), split="train")
            
            rec_Xcl = rec_Xcl * Xcf # learning diagonal of quantizer such that save disease features -> reconstruction Xcl
            Xcl__groundtrouth = Xcl * Xcf
            Xcl_aeloss, Xcl_log_dict_ae = self.loss(qcloss, Xcl__groundtrouth, rec_Xcl, 0, self.global_step, 
                                            last_layer=self.get_last_layer(flag=False), split="train")

            
            # on xc diesease features -> interpolation!! -> adversialloss
            
            assert False

        if optimizer_idx == 1: # discriminator
            pass

        return iou
    
    def test(self):

        
        
        
        xs_groundtrouth = (xs * M_xrec_xs).detach() # groundtrouth
        lesion_ROT_groundtrouth = (lesion_ROT * M_xrec_xcl).detach() # groundtrouth
        # recloss(xrec * M_xrec_xs, xs_groundtrouth)
        # recloss(xrec * M_xrec_xcl, lesion_ROT_groundtrouth)



        assert False
        return iou

        # Vorg, Vrec = self.get_V(x, xrec)
        # Vrec = dzq_dz_eq1(Vrec, xrec)
        # print(optimizer_idx, x.shape, xrec.shape, Vorg.shape, Vrec.shape, Vorg.min(), Vorg.max(), Vrec.min(), Vrec.max())

        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="train")
            
            # DELETE
            # VLOSS = 0.5 * torch.mean(torch.abs(Vorg - Vrec) + 0.1 * self.loss.perceptual_loss(Vorg, Vrec)).log()
            # vintersection = (Vorg * Vrec)
            # VLOSS =  1 - (vintersection / (Vorg + Vrec - vintersection).clamp(1e-8, 1).detach()).mean()
            # log_dict_ae['train/VLOSS'] = VLOSS.detach()


            self.log("train/aeloss", aeloss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            
            return aeloss
            # return VLOSS + aeloss
        
        if optimizer_idx == 1:
            discloss, log_dict_disc = self.loss(None, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="train")
            discloss_v, log_dict_disc_v = self.loss(None, Vorg, Vrec, 1, self.global_step, last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            # print('11', discloss_v, discloss)
            return discloss + discloss_v
    

    def ssf0(self, t):
        t = t.unsqueeze(0).unsqueeze(0)
        return torch.cat([t,t,t], dim=1)

    # def ssf1(self, x, xrecombine, patches):
    #     signal_save(torch.cat([
    #         (x+1) * 127.5, # same as xc_np
    #         (xrecombine+1) * 127.5,
    #     ], dim=0), f'/content/export/patches/r256.png', stype='img', sparams={'chw2hwc': True, 'nrow': 2})
    #     signal_save((patches+1) * 127.5, f'/content/export/patches/r64.png', stype='img', sparams={'chw2hwc': True, 'nrow': 4})


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
        return
        # print('validation_step')
        # logged = self.log_images(batch, fName='badRec/' + random_string())
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

    def get_last_layer(self, flag2=True):
        if flag2:
            return self.decoder.conv_out_1ch_main.weight
        else:
            return self.decoder.conv_out_1ch.weight

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
