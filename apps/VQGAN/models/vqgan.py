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

import time
from apps.VQGAN.models.metrics import SQLiteLogger
from utils.pl.plLogger import GenieLoggerBase

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
        self.automatic_optimization = False


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
        
        self.start()

    def start(self):
        self.select_query_d1_acc = '/d1ACC$'

        self.expected_acc_val = dict()
        self.expected_acc_train = dict()

        # self.gl = GenieLoggerBase()
        self.metrics = SQLiteLogger(db='/content/metrics.db')

        # print('encoder', self.encoder)
        # print('decoder', self.decoder)
        # print('disc', self.loss.discriminator)

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.loss.discriminator.parameters():
            param.requires_grad = False

        
        
        # print('before self.encoder.down[4]', self.encoder.down[4].block[1].conv1.weight.requires_grad)
        for param in self.encoder.Qsurface2Qdiagonal.parameters():
            param.requires_grad = True
        for param in self.encoder.netb_diagonal.parameters():
            param.requires_grad = True
        
        for param in self.encoder.down[4].parameters():
            param.requires_grad = True
        for param in self.encoder.mid.parameters():
            param.requires_grad = True
        for param in self.encoder.norm_out.parameters():
            param.requires_grad = True
        for param in self.encoder.conv_out.parameters():
            param.requires_grad = True
        # print('after self.encoder.down[4]', self.encoder.down[4].block[1].conv1.weight.requires_grad)


        # print('before self.decoder.up[4]', self.decoder.up[4].attn[1].k.weight.requires_grad)
        for param in self.decoder.up[4].parameters():
            param.requires_grad = True
        for param in self.decoder.norm_out.parameters():
            param.requires_grad = True
        for param in self.decoder.conv_out.parameters():
            param.requires_grad = True
        # for param in self.decoder.spade_ilevel1.parameters():
        #     param.requires_grad = True
        # for param in self.decoder.spade_endDownSampling.parameters():
        #     param.requires_grad = True
        # print('after self.decoder.up[4]', self.decoder.up[4].attn[1].k.weight.requires_grad)

        # print('before self.loss.discriminator.main[8]', self.loss.discriminator.main[8].weight.requires_grad)
        for pidx in [5, 6, 8, 9, 11]:
            for param in self.loss.discriminator.main[pidx].parameters():
                param.requires_grad = True
        # print('after self.loss.discriminator.main[8]', self.loss.discriminator.main[8].weight.requires_grad)
        

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
    
    def netConditins(self, simg):
        h_ilevel1, h_endDownSampling, q_eye16, Qsurface, Qorg, Qdiagonal = self.net(simg)
        y = self.decoder(
            Qsurface,
            None, 
            h_ilevel1, 
            h_endDownSampling,
            flag=False
        ) # Note: add skip connection
        
        # signal_save(torch.cat([
        #     (simg+1) * 127.5,
        #     (y+1) * 127.5,
        # ], dim=0), f'/content/export/netConditins.png', stype='img', sparams={'chw2hwc': True, 'nrow': 3})
        # assert False

        return y


    def net(self, x):
        Sk = 64 # patch size
        Nk = 4  # num patches in each row and column
        q_eye16 = torch.eye(16, dtype=torch.float32, device=self.device).detach()
        # patching!!
        # x = self.unfold(x0, Sk, Nk)
        h, h_ilevel1, h_endDownSampling = self.encoder(x) 
        # unpatching!!
        # h = self.fold(h, Nk) 
        # h_ilevel1 = self.fold(h_ilevel1, Nk) 
        # h_endDownSampling = self.fold(h_endDownSampling, Nk)

        h = self.quant_conv(h)
        quant, diff = self.quantize(h)
        h_new = self.post_quant_conv(quant)
        # Qorg = self.encoder.catconv_hnew_h(torch.cat([h_new, h], dim=1))
        Qorg = h_new
        
        Qsurface = (1-q_eye16) * Qorg
        Qdiagonal = self.encoder.Qsurface2Qdiagonal(Qsurface.detach())
        
        return h_ilevel1, h_endDownSampling, q_eye16, Qsurface, Qorg, Qdiagonal
        
    def netA(self, simg, smask):
        h_ilevel1, h_endDownSampling, q_eye16, Qsurface, Qorg, Qdiagonal = self.net(simg)
        Qsurface = Qsurface.detach()
        Qcrossover = Qsurface + q_eye16 * Qdiagonal
        y = self.decoder(
            Qcrossover,
            None, 
            h_ilevel1, 
            h_endDownSampling,
            flag=False
        ) # Note: add skip connection
        
        # signal_save(torch.cat([
        #     (simg+1) * 127.5,
        #     (y+1) * 127.5,
        #     torch.cat([smask, smask, smask], dim=1) * 255,
        # ], dim=0), f'/content/export/netA.png', stype='img', sparams={'chw2hwc': True, 'nrow': 3})
        # assert False
        
        return y
    
    def netB(self, simg, smask, sinfgray):
        n = 16
        ch = 256
        sinfgray_diesis = self.loss.Ro(torch.cat([sinfgray,sinfgray,sinfgray], dim=1)).detach()
        v = self.encoder.netb_diagonal(sinfgray_diesis).view(ch, n, 1, 1)
        z = torch.zeros(ch, n, n, dtype=torch.float32, device=self.device).detach()
        V = (v + z.unsqueeze(-1)).view((1, ch, n, n))
        # `sinfgray_diesis` -> torch.Size([1, 256]) tensor(-0.5701) tensor(0.5360)
        # `v` -> torch.Size([256, 16, 1, 1]) tensor(-0.2625, grad_fn=<MinBackward1>) tensor(0.2622, grad_fn=<MaxBackward1>)
        # `V` -> torch.Size([1, 256, 16, 16]) tensor(-0.2625, grad_fn=<MinBackward1>) tensor(0.2622, grad_fn=<MaxBackward1>)


        h_ilevel1, h_endDownSampling, q_eye16, Qsurface, Qorg, Qdiagonal = self.net(simg)
        Qsurface = Qsurface.detach()
        Qdiagonal = Qdiagonal.detach()
        Qdb = Qdiagonal + V
        Qcrossover = Qsurface + q_eye16 * Qdb
        y = self.decoder(
            Qcrossover,
            None,
            h_ilevel1, 
            h_endDownSampling,
            flag=False
        ) # Note: add skip connection
        
        # signal_save(torch.cat([
        #     (simg+1) * 127.5,
        #     (y+1) * 127.5,
        #     torch.cat([smask, smask, smask], dim=1) * 255,
        #     (torch.cat([sinfgray, sinfgray, sinfgray], dim=1)+1) * 127.5,
        # ], dim=0), f'/content/export/netB.png', stype='img', sparams={'chw2hwc': True, 'nrow': 2})
        # assert False
        
        return y

    
    def pipline(self, xs, Xc, 
                split,
                optidx,
                y_edit, y_edit_xc, xsmask, xcmask, C_xsmask, C_xcmask, xcm_gray
    ):
        """
            By: ***alihejrati***
            xs: source color fundus
            Xc: conditional color fundus | ROT version
            # xcl_pure: none ROT version of Xcl (attendend)
            # UPDATE: xcl_pure: is xcl === ROT (attendend)
        """
        # Conditins)
        # Qsurface should be contain all information for reconstructiong none gray area part of xs. # NOTE: Geometry loss
        Cond_loss_logdict = {}
        if optidx == 0:
            xs_noneGrayAreaPart_gtru  = xs * C_xsmask
            xs_noneGrayAreaPart_pred = self.netConditins(xs_noneGrayAreaPart_gtru)
            Cond_loss, Cond_loss_logdict = self.loss.geometry(xs_noneGrayAreaPart_gtru, xs_noneGrayAreaPart_pred, split=split + 'Cond_Geo')
            # print('Conditins) OPTIDX0)', Cond_loss, Cond_loss.shape)

        # A)
        # punching xs only in xsmask Not in Union of lesions and getting it as xss.
        # Interpolating xss and getting 𝝍s_tm. (reggression baft!!)
        if y_edit == 0: # 𝝍s_tm, xs ===> # NOTE: geometry loss
            # here xcmask was used as random mask.
            xss = xs * C_xcmask
            𝝍s_tm = xss + xcmask * self.netA(xss, xcmask)
            𝝍s_tm_final = xs
            if optidx == 0:
                A_loss, A_loss_logdict = self.loss.geometry(xs, 𝝍s_tm, split=split + 'A_Geo')
                # print('A) IF) OPTIDX0)', A_loss, A_loss.shape)
            else:
                A_loss0, A_d0 = self.loss.omega_of_phi(xs, flag=True, split=split + 'A_if1_OFxs') # OK!
                A_loss5, A_d5 = self.loss.omega_of_phi(Xc, split=split + 'A_if1_ORxc') # OK!
                A_loss1, A_d1 = self.loss.D12(xs, l1=1, l2=1, split=split + 'A_if1_Rxs')
                A_loss2, A_d2 = self.loss.D12(Xc, l1=1, l2=1, split=split + 'A_if1_Rxc')
                A_loss3, A_d3 = self.loss.D12(𝝍s_tm, l1=1, l2=1, flag=True, split=split + 'A_if1_Fpsistm')
                A_loss4, A_d4 = self.loss.D12(xss, l1=1, l2=1, flag=True, split=split + 'A_if1_Fxss')
                A_loss = A_loss0 + A_loss1 + A_loss2 + A_loss3 + A_loss4 + A_loss5
                A_loss_logdict = {
                    **A_d0,
                    **A_d5,
                    **A_d1,
                    **A_d2,
                    **A_d3,
                    **A_d4,
                }
                # print('A) IF) OPTIDX1)', A_loss0, A_loss1, A_loss2, A_loss3, A_loss4, A_loss5, A_loss, A_loss.shape)
        else: # 𝝍s_tm ===> #NOTE: adversial loss
            xss = xs * C_xsmask
            𝝍s_tm = xss + xsmask * self.netA(xss, xsmask)
            𝝍s_tm_final = 𝝍s_tm
            if optidx == 0:
                A_loss0, A_d0 = self.loss.omega_of_phi(𝝍s_tm, flag=True, split=split + 'A_el0_OFpsistm') # OK!
                A_loss1, A_d1 = self.loss.D12(𝝍s_tm, l1=1, l2=1, split=split + 'A_el0_Rpsistm')
                A_loss = A_loss0 + A_loss1
                A_loss_logdict = {
                    **A_d0,
                    **A_d1
                }
                # print('A) ELSE) OPTIDX0)', A_loss0, A_loss1, A_loss, A_loss.shape)
            else:
                A_loss0, A_d0 = self.loss.omega_of_phi(xs, split=split + 'A_el1_ORxs') # OK!
                A_loss1, A_d1 = self.loss.D12(xs, l1=1, l2=1, split=split + 'A_el1_Rxs')
                A_loss2, A_d2 = self.loss.D12(Xc, l1=1, l2=1, split=split + 'A_el1_Rxc')
                A_loss3, A_d3 = self.loss.D12(𝝍s_tm, l1=1, l2=1, flag=True, split=split + 'A_el1_Fpsistm')
                A_loss4, A_d4 = self.loss.D12(xss, l1=1, l2=1, flag=True, split=split + 'A_el1_Fxss')
                A_loss = A_loss0 + A_loss1 + A_loss2 + A_loss3 + A_loss4
                A_loss_logdict = {
                    **A_d0,
                    **A_d1,
                    **A_d2,
                    **A_d3,
                    **A_d4,
                }
                # print('A) ELSE) OPTIDX1)', A_loss0, A_loss1, A_loss2, A_loss3, A_loss4, A_loss, A_loss.shape)

        # B)
        # using 𝝍s_tm_final (xs with absolutly no diesis) and punching it only in `xcmask` and considder gray information of `xc lessions` as xcm_gray.
        # Interpolationg 𝝍s_tm_final_s and getting 𝝍s_tp. (reggression bimari!!)
        if y_edit_xc == '[01]': # 𝝍s_tp, xs ===> # Note: geometry loss
            xsss = xs * C_xsmask
            xsm_gray = (xs * xsmask).mean(dim=1, keepdim=True).detach()
            𝝍s_tp = xsss + xsmask * self.netB(xsss, xsmask, xsm_gray)
            𝝍s_tp_final = 𝝍s_tm_final
            if optidx == 0:
                B_loss, B_loss_logdict = self.loss.geometry(xs, 𝝍s_tp, split=split + 'B_Geo')
                # print('B) IF) OPTIDX0)', B_loss, B_loss.shape)
            else:
                B_loss0, B_d0 = self.loss.omega_of_phi(Xc, flag=True, split=split + 'B_if1_OFxc') # OK!
                B_loss1, B_d1 = self.loss.omega_of_phi(xs, split=split + 'B_if1_ORxs') # OK!
                B_loss2, B_d2 = self.loss.D12(xs, l1=1, l2=1, split=split + 'B_if1_Rxs')
                B_loss3, B_d3 = self.loss.D12(Xc, l1=1, l2=1, split=split + 'B_if1_Rxc')
                B_loss4, B_d4 = self.loss.D12(xsss, l1=1, l2=1, flag=True, split=split + 'B_if1_Fxsss')
                B_loss5, B_d5 = self.loss.D12(𝝍s_tp, l1=1, l2=1, flag=True, split=split + 'B_if1_Fpsistp')
                B_loss = B_loss0 + B_loss1 + B_loss2 + B_loss3 + B_loss4 + B_loss5
                B_loss_logdict = {
                    **B_d0,
                    **B_d1,
                    **B_d2,
                    **B_d3,
                    **B_d4,
                    **B_d5,
                }
                # print('B) IF) OPTIDX1)', B_loss0, B_loss1, B_loss2, B_loss3, B_loss4, B_loss5, B_loss, B_loss.shape)
        else: # 𝝍s_tp ===> # Note: adversial loss
            𝝍s_tm_final_s = (𝝍s_tm_final * C_xcmask).detach()
            𝝍s_tp = 𝝍s_tm_final_s + xcmask * self.netB(𝝍s_tm_final_s, xcmask, xcm_gray)
            𝝍s_tp_final = 𝝍s_tp
            if optidx == 0:
                R_𝝍s_tp = self.loss.Ro(𝝍s_tp)
                B_loss0, B_d0 = self.loss.omega_of_phi_givvenRo(R_𝝍s_tp, split=split + 'B_el0_ORropsistp') # OK!
                B_loss1, B_d1 = self.loss.D12(𝝍s_tp, l1=1, l2=1, split=split + 'B_el0_Rpsistp')
                B_loss2, B_d2 = self.loss.geometry(self.loss.Ro(Xc), R_𝝍s_tp, pw=0, recln1p=True, split=split + 'B_Geo_Ro', landa1=0.01)
                B_loss = B_loss0 + B_loss1 + B_loss2
                B_loss_logdict = {
                    **B_d0,
                    **B_d1,
                    **B_d2,
                }
                # print('B) ELSE) OPTIDX0)', B_loss0, B_loss1, B_loss2, B_loss, B_loss.shape)
            else:
                B_loss0, B_d0 = self.loss.omega_of_phi(Xc, split=split + 'B_el1_ORxc') # OK!
                B_loss1, B_d1 = self.loss.D12(xs, l1=1, l2=1, split=split + 'B_el1_Rxs')
                B_loss2, B_d2 = self.loss.D12(Xc, l1=1, l2=1, split=split + 'B_el1_Rxc')
                B_loss3, B_d3 = self.loss.D12(𝝍s_tp, l1=1, l2=1, flag=True, split=split + 'B_el1_Fpsistp')
                B_loss4, B_d4 = self.loss.D12(𝝍s_tm_final_s, l1=1, l2=1, flag=True, split=split + 'B_el1_Fpsistmfs')
                B_loss = B_loss0 + B_loss1 + B_loss2 + B_loss3 + B_loss4
                B_loss_logdict = {
                    **B_d0,
                    **B_d1,
                    **B_d2,
                    **B_d3,
                    **B_d4,
                }
                # print('B) ELSE) OPTIDX1)', B_loss0, B_loss1, B_loss2, B_loss3, B_loss4, B_loss, B_loss.shape)
        
        if optidx == 0:
            loss = Cond_loss + A_loss + B_loss
            # print(optidx, 'Condloss, Aloss, Bloss, loss', Cond_loss, A_loss, B_loss, loss)
        else:
            loss = A_loss + B_loss
            # print(optidx, 'Aloss, Bloss, loss', A_loss, B_loss, loss)
        # print('-'*30)
        
        return loss, {
            **Cond_loss_logdict,
            **A_loss_logdict,
            **B_loss_logdict,
            "{}/total_loss".format(split): loss.clone().detach().mean().item(),
            "{}/total_A_loss".format(split): A_loss.clone().detach().mean().item(),
            "{}/total_B_loss".format(split): B_loss.clone().detach().mean().item(),
        }

    # NOTE: Syn Idea
    def training_step(self, batch, batch_idx):
        if batch_idx >= 5:
            return
        # start_time = time.time()
        opt_ae, opt_disc = self.optimizers()
        for cidx in range(2):
            for optimizer_idx in range(2):
                # print(f'batch_idx={batch_idx} | optimizer_idx={optimizer_idx} | cidx={cidx}')
                opt_ae.zero_grad()
                opt_disc.zero_grad()
                loss, logdict = self.training_step_slave(batch, batch_idx, optimizer_idx, cidx=cidx, split='train_')
                
                
                self.manual_backward(loss)
                if optimizer_idx == 0:
                    opt_ae.step()
                else:
                    opt_disc.step()
                
                
                self.metrics.log('train', logdict)
        # execution_time_in_sec = (time.time() - start_time)
        # print(f'batch_idx={batch_idx} | execution_time_in_sec={execution_time_in_sec}')
        # assert False
    
    def validation_step(self, batch, batch_idx):
        if batch_idx >= 10:
            return
        for cidx in range(2):
            for optimizer_idx in range(2):
                # print(f'batch_idx={batch_idx} | optimizer_idx={optimizer_idx} | cidx={cidx}')
                loss, logdict = self.training_step_slave(batch, batch_idx, optimizer_idx, cidx=cidx, split='val_')
                self.metrics.log('val', logdict)
        # assert False

    def on_train_epoch_end(self):
        R = self.metrics.save('train')
        self.metrics.inference(self.select_query_d1_acc, R)
        # for i in ['']:
        #     pass
        # self.expected_acc_train['']
    
    def on_validation_epoch_end(self):
        R = self.metrics.save('val')
        self.metrics.inference(self.select_query_d1_acc, R)
    
    def training_step_slave(self, batch, batch_idx, optimizer_idx, cidx, split='train_'):
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

        # NOTE: Mask design
        M_union_L_xs_xc = ((xslmask + xclmask) - (xslmask * xclmask)).detach()
        # M_L_xs_mines_xc = (xslmask - (xslmask * xclmask)).detach() # TODO Interpolation
        M_C_Union = ((1 - M_union_L_xs_xc) * xsf).detach() #shape:torch.Size([1, 1, 256, 256]) # reconstruct xs
        # M_xrec_xcl = (xclmask * xsf).detach() # reconstruct xc
        xcm_gray = (xclmask * xc).mean(dim=1, keepdim=True).detach() # torch.Size([1, 1, 256, 256])
        
        C_xsmask = (1-xslmask).detach()
        C_xcmask = (1-xclmask).detach()
        
        y_edit = batch['y_edit'].item()
        y_edit_xc = batch['ynl'][cidx][0]

        # print(y_edit, type(y_edit), y_edit_xc, type(y_edit_xc))
        # 0 <class 'int'> 2 <class 'str'>

        # print('xs, ...')
        # print(xs.shape, xs.dtype, xs.min().item(), xs.max().item())
        # print(xsl.shape, xsl.dtype, xsl.min().item(), xsl.max().item())
        # print(xsc.shape, xsc.dtype, xsc.min().item(), xsc.max().item())
        # print(xsf.shape, xsf.dtype, xsf.min().item(), xsf.max().item())
        # print(xslmask.shape, xslmask.dtype, xslmask.min().item(), xslmask.max().item())
        
        # print('xc, ...')
        # print(xc.shape, xc.dtype, xc.min().item(), xc.max().item())
        # print(xcl.shape, xcl.dtype, xcl.min().item(), xcl.max().item())
        # print(xcc.shape, xcc.dtype, xcc.min().item(), xcc.max().item())
        # print(xcf.shape, xcf.dtype, xcf.min().item(), xcf.max().item())
        # print(xclmask.shape, xclmask.dtype, xclmask.min().item(), xclmask.max().item())
        # print('-'*30)

        # print('mask...')
        # print(xcm_gray.shape, xcm_gray.dtype, xcm_gray.min().item(), xcm_gray.max().item())
        # print(C_xsmask.shape, C_xsmask.dtype, C_xsmask.min().item(), C_xsmask.max().item())
        # print(C_xcmask.shape, C_xcmask.dtype, C_xcmask.min().item(), C_xcmask.max().item())
        # print(M_union_L_xs_xc.shape, M_union_L_xs_xc.dtype, M_union_L_xs_xc.min().item(), M_union_L_xs_xc.max().item())
        # print(M_C_Union.shape, M_C_Union.dtype, M_C_Union.min().item(), M_C_Union.max().item())
        # print('-'*30)

        # xs, ...
        # torch.Size([1, 3, 256, 256]) torch.float32 -1.0 1.0
        # torch.Size([1, 3, 256, 256]) torch.float32 -1.0 0.6392157077789307
        # torch.Size([1, 1, 256, 256]) torch.float32 0.0 1.0
        # torch.Size([1, 1, 256, 256]) torch.float32 0.0 1.0
        # torch.Size([1, 1, 256, 256]) torch.float32 0.0 1.0
        # xc, ...
        # torch.Size([1, 3, 256, 256]) torch.float32 -1.0 1.0
        # torch.Size([1, 3, 256, 256]) torch.float32 -1.0 0.20784318447113037
        # torch.Size([1, 1, 256, 256]) torch.float32 0.0 1.0
        # torch.Size([1, 1, 256, 256]) torch.float32 0.0 1.0
        # torch.Size([1, 1, 256, 256]) torch.float32 0.0 1.0
        # ------------------------------
        # mask...
        # torch.Size([1, 1, 256, 256]) torch.float32 -0.6287581324577332 0.0
        # torch.Size([1, 1, 256, 256]) torch.float32 0.0 1.0
        # torch.Size([1, 1, 256, 256]) torch.float32 0.0 1.0
        # torch.Size([1, 1, 256, 256]) torch.float32 0.0 1.0
        # torch.Size([1, 1, 256, 256]) torch.float32 0.0 1.0

        # signal_save(torch.cat([
        #     (xs+1) * 127.5,
        #     (xsl+1) * 127.5,
        #     torch.cat([xsc, xsc, xsc], dim=1) * 255,
        #     torch.cat([xsf, xsf, xsf], dim=1) * 255,
        #     torch.cat([xslmask, xslmask, xslmask], dim=1) * 255,
            
        #     (xc+1) * 127.5,
        #     (xcl+1) * 127.5,
        #     torch.cat([xcc, xcc, xcc], dim=1) * 255,
        #     torch.cat([xcf, xcf, xcf], dim=1) * 255,
        #     torch.cat([xclmask, xclmask, xclmask], dim=1) * 255,
            
        #     (torch.cat([xcm_gray, xcm_gray, xcm_gray], dim=1) +1) * 127.5,
        #     torch.cat([C_xsmask, C_xsmask, C_xsmask], dim=1) * 255,
        #     torch.cat([C_xcmask, C_xcmask, C_xcmask], dim=1) * 255,
        #     torch.cat([M_C_Union, M_C_Union, M_C_Union], dim=1) * 255,
        #     torch.cat([M_union_L_xs_xc, M_union_L_xs_xc, M_union_L_xs_xc], dim=1) * 255,

        # ], dim=0), f'/content/export/data.png', stype='img', sparams={'chw2hwc': True, 'nrow': 5})

        loss, losslogdict = self.pipline(xs, xc, 
            split=split,
            optidx=optimizer_idx,
            y_edit=y_edit, y_edit_xc=y_edit_xc, xsmask=xslmask, xcmask=xclmask, C_xsmask=C_xsmask, C_xcmask=C_xcmask, xcm_gray=xcm_gray
        )
        losslogdict['epoch'] = self.current_epoch
        return loss, losslogdict

    def validation_step_syn(self, batch, batch_idx):
        print('validation_step_syn')
        return
    def validation_step0000000(self, batch, batch_idx):
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
        # lr = self.learning_rate
        lr = 0.02
        opt_ae = torch.optim.Adam(
                                    list(self.encoder.parameters())+
                                    list(self.decoder.parameters())+
                                    list(self.quantize.parameters())+
                                    list(self.quant_conv.parameters())+
                                    list(self.post_quant_conv.parameters()),
                                lr=lr, 
                                # betas=(0.5, 0.9)
                            )
        opt_disc = torch.optim.Adam(
                                    list(self.loss.discriminator.parameters())+
                                    list(self.loss.discriminator_large.parameters())+
                                    list(self.loss.vgg16.parameters())+
                                    list(self.loss.vgg16_head.parameters()),
                                lr=lr, 
                                # betas=(0.5, 0.9)
                            )
        # opt_ae = torch.optim.Adam(
        #                         list(self.encoder.parameters())+
        #                         list(self.decoder.parameters())+
        #                         list(self.quantize.parameters())+
        #                         list(self.quant_conv.parameters())+
        #                         list(self.post_quant_conv.parameters()),
        #                         lr=lr, 
        #                         # betas=(0.5, 0.9)
        #                     )
        # opt_disc = torch.optim.Adam(
        #                         list(self.loss.discriminator.parameters())+
        #                         list(self.loss.discriminator_large.parameters())+
        #                         list(self.loss.vgg16.parameters())+
        #                         list(self.loss.vgg16_head.parameters()),
        #                         lr=lr, 
        #                         # betas=(0.5, 0.9)
        #                     )
        


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
