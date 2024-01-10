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

import torchvision
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
        
        self.__start()

    
    def decoder_grad_controller(self, flag):
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
        setattr(self, 'forward', self.pipline)
        self.imglogger = [None, None, None]
        self.QclassDict = {
            '[01]': 0,
            '2': 1,
            '[34]': 2
        }
        
        self.acc = {
            'train_': {'d1': 0, 'd2': 0, 'O': 0},
            'val_': {'d1': 0, 'd2': 0, 'O': 0}
        }

        self.regexp_d1_acc = '^.*1_.*\/d1ACC$'
        self.regexp_d2_acc = '^.*1_.*\/d2ACC$'
        self.regexp_OP_acc = '^.*1_.*\/ACC$'

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


        self.decoder_grad_controller(True)


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
        self.decoder_grad_controller(True)
        h_ilevel1, h_endDownSampling, q_eye16, Qsurface, Qorg, Qdiagonal = self.net(simg)
        y = self.decoder(
            Qorg,
            # (q_eye16) * Qorg,
            # Qsurface,
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

    def net(self, x, Qd_wg=1):
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
        Qdiagonal = self.encoder.Qsurface2Qdiagonal(Qsurface.detach(), self.VgradViewrFlag, Qd_wg)
        
        return h_ilevel1, h_endDownSampling, q_eye16, Qsurface, Qorg, Qdiagonal
        
    def netA(self, simg, smask, wg=1):
        h_ilevel1, h_endDownSampling, q_eye16, Qsurface, Qorg, Qdiagonal = self.net(simg, Qd_wg=wg)
        Qsurface = Qsurface.detach()
        Qcrossover = Qsurface + q_eye16 * Qdiagonal
        y = self.decoder(
            q_eye16 * Qdiagonal, #Qcrossover,
            None, 
            h_ilevel1, 
            h_endDownSampling,
            flag=False
        )
        
        # signal_save(torch.cat([
        #     (simg+1) * 127.5,
        #     (y+1) * 127.5,
        #     torch.cat([smask, smask, smask], dim=1) * 255,
        # ], dim=0), f'/content/export/netA.png', stype='img', sparams={'chw2hwc': True, 'nrow': 3})
        # assert False
        
        return y
    
    def netB(self, simg, smask, sinfgray, Qclass):
        n = 16
        ch = 256
        sinfgray_diesis = self.loss.Ro(torch.cat([sinfgray,sinfgray,sinfgray], dim=1)).view(-1, 1, 16, 16).detach()
        v = self.encoder.netb_diagonal(sinfgray_diesis, self.QclassDict[Qclass], sinfgray).view(ch, n, 1, 1)
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
            q_eye16 * Qdb, #Qcrossover,
            None,
            h_ilevel1, 
            h_endDownSampling,
            flag=False
        )
        
        # signal_save(torch.cat([
        #     (simg+1) * 127.5,
        #     (y+1) * 127.5,
        #     torch.cat([smask, smask, smask], dim=1) * 255,
        #     (torch.cat([sinfgray, sinfgray, sinfgray], dim=1)+1) * 127.5,
        # ], dim=0), f'/content/export/netB.png', stype='img', sparams={'chw2hwc': True, 'nrow': 2})
        # assert False
        
        return y

    def report(self, name, N, listimgs):
        signal_save((torch.cat(listimgs, dim=0)+1)*127.5, f'/content/export/{name}.png', stype='img', sparams={'chw2hwc': True, 'nrow': N})

    def pipline(self, xs, Xc, xsf,
                split,
                optidx,
                y_edit, y_edit_xc, xsmask, xcmask, C_xsmask, C_xcmask, xcm_gray, condstep=False,
                **kwargs
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
        if optidx == 0 and condstep == True:
            Cond_loss_logdict = {}
            xs_noneGrayAreaPart_gtru  = xs * C_xsmask
            xs_noneGrayAreaPart_pred = xsf * self.netConditins(xs_noneGrayAreaPart_gtru)
            Cond_loss, Cond_loss_logdict = self.loss.geometry(xs_noneGrayAreaPart_gtru, xs_noneGrayAreaPart_pred, split=split + 'Cond_Geo', losscontroller='CondGeo')
            # print('Conditins) OPTIDX0)', Cond_loss, Cond_loss.shape)
            self.report('Cond', 2, [xs_noneGrayAreaPart_gtru, xs_noneGrayAreaPart_pred])
            # assert False
            return Cond_loss, Cond_loss_logdict, None

        # A)
        # punching xs only in xsmask Not in Union of lesions and getting it as xss.
        # Interpolating xss and getting 𝝍s_tm. (reggression baft!!)
        if y_edit == 0: # 𝝍s_tm, xs ===> # NOTE: geometry loss
            # here xcmask was used as random mask.
            xss = xs * C_xcmask
            
            if self.VgradViewrFlag:
                print('IF ---------')
            
            𝝍s_tm = xsf * (xss + xcmask * self.netA(xss, xcmask, 1e5))
            𝝍s_tm_final = xs
            if optidx == 0:
                A_loss, A_loss_logdict = self.loss.geometry(xs, 𝝍s_tm, split=split + 'A_Geo', hoo=True)
                self.report('netA_Geo', 2, [xs, 𝝍s_tm])
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
            
            if self.VgradViewrFlag:
                print('ELSE -------------------------')
            
            
            𝝍s_tm = xsf * (xss + xsmask * self.netA(xss, xsmask, 1e9))
            𝝍s_tm_final = 𝝍s_tm
            if optidx == 0:
                A_loss0, A_d0 = self.loss.omega_of_phi(𝝍s_tm, flag=True, split=split + 'A_el0_OFpsistm', l=self.acc[split]['O']) # OK!
                A_loss1, A_d1 = self.loss.D12(𝝍s_tm, l1=self.acc[split]['d1'], l2=self.acc[split]['d2'], split=split + 'A_el0_Rpsistm')
                A_loss = A_loss0 + A_loss1
                A_loss_logdict = {
                    **A_d0,
                    **A_d1
                }
                # print('A) ELSE) OPTIDX0)', A_loss0, A_loss1, A_loss, A_loss.shape)
                self.report('netA_adv', 2, [xss, 𝝍s_tm])
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
            𝝍s_tp = xsf * (xsss + xsmask * self.netB(xsss, xsmask, xsm_gray, y_edit_xc))
            𝝍s_tp_final = 𝝍s_tm_final
            if optidx == 0:
                B_loss, B_loss_logdict = self.loss.geometry(xs, 𝝍s_tp, split=split + 'B_Geo')
                self.report('netB_Geo', 4, [xs, 𝝍s_tp, xsss, 𝝍s_tp_final])
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
            𝝍s_tp = xsf * (𝝍s_tm_final_s + xcmask * self.netB(𝝍s_tm_final_s, xcmask, xcm_gray, y_edit_xc))
            𝝍s_tp_final = 𝝍s_tp
            if optidx == 0:
                R_𝝍s_tp = self.loss.Ro(𝝍s_tp)
                B_loss0, B_d0 = self.loss.omega_of_phi_givvenRo(R_𝝍s_tp, split=split + 'B_el0_ORropsistp', l=self.acc[split]['O']) # OK!
                B_loss1, B_d1 = self.loss.D12(𝝍s_tp, l1=self.acc[split]['d1'], l2=self.acc[split]['d2'], split=split + 'B_el0_Rpsistp')
                B_loss2, B_d2 = self.loss.geometry(self.loss.Ro(Xc), R_𝝍s_tp, pw=0, recln1p=True, split=split + 'B_Geo_Ro', landa1=0.001)
                B_loss = B_loss0 + B_loss1 + B_loss2
                B_loss_logdict = {
                    **B_d0,
                    **B_d1,
                    **B_d2,
                }
                # print('B) ELSE) OPTIDX0)', B_loss0, B_loss1, B_loss2, B_loss, B_loss.shape)
                self.report('netB_adv', 2, [𝝍s_tm_final_s, 𝝍s_tp])
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
        
        logdata = {
            '𝝍s_tm_final': 𝝍s_tm_final.detach(),
            '𝝍s_tp_final': 𝝍s_tp_final.detach()
        }
        
        loss = A_loss + B_loss
        
        return loss, {
            **A_loss_logdict,
            **B_loss_logdict,
            "{}/total_loss".format(split): loss.clone().detach().mean().item(),
            "{}/total_A_loss".format(split): A_loss.clone().detach().mean().item(),
            "{}/total_B_loss".format(split): B_loss.clone().detach().mean().item(),
        }, logdata

    # NOTE: Syn Idea
    # def batch(self, batch): # TODO
    #     return batch
    def step(self, batch, batch_idx, **kwargs):
        # if batch_idx >= 20:
        #     return
        
        tag = kwargs['tag']
        optFlag = tag == 'train' or kwargs.get('force_train', False)
        
        # start_time = time.time()
        if optFlag:
            opt_ae, opt_disc = self.optimizers()
        
        # xsl = batch['xsl']
        # xsc = batch['xsc']
        xsf = batch['xsf']
        xs = batch['xs'] * xsf
        xslmask = batch['xslmask']
        C_xsmask = (1-xslmask).detach()

        y_edit = batch['y_edit'].item()
        
        flag_logdata = False
        pack_logdata = dict()
        if self.imglogger[y_edit] == None and tag == 'train':
            flag_logdata = True
            pack_logdata['xs'] = xs
            pack_logdata['y_edit'] = y_edit
        else:
            return

        
        for cidx in range(2):
            self.decoder_grad_controller(False)
            # xcl = batch['xcl'][cidx] # ROT
            # xcc = batch['xcc'][cidx] # ROT
            xcf = batch['xcf'][cidx] # ROT
            xc = batch['xc'][cidx] * xcf # ROT
            xclmask = batch['xclmask'][cidx] # ROT

            # NOTE: Mask design
            # M_union_L_xs_xc = ((xslmask + xclmask) - (xslmask * xclmask)).detach()
            # M_L_xs_mines_xc = (xslmask - (xslmask * xclmask)).detach() # TODO Interpolation
            # M_C_Union = ((1 - M_union_L_xs_xc) * xsf).detach() #shape:torch.Size([1, 1, 256, 256]) # reconstruct xs
            # M_xrec_xcl = (xclmask * xsf).detach() # reconstruct xc
            xcm_gray = ((xclmask * xc * xsf).mean(dim=1, keepdim=True)).detach() # torch.Size([1, 1, 256, 256])
            C_xcmask = (1-xclmask).detach()

            # if y_edit == 1 and cidx == 1:
            #     signal_save(torch.cat([
            #         (torch.cat([xcm_gray,xcm_gray,xcm_gray], dim=1)+1) * 127.5
            #     ], dim=0), f'/content/export/xcm_gray.png', stype='img', sparams={'chw2hwc': True, 'nrow': 4})
            #     assert False
            # continue


            # if kwargs.get('show_dataset', False):
            #     if y_edit == 1 and cidx == 1:
            #         signal_save(torch.cat([
            #             (xs+1)*127.5, (xsl+1)*127.5, self.cat3d(xsf)*255, self.cat3d(xslmask)*255, 
            #             (xc+1)*127.5, (xcl+1)*127.5, self.cat3d(xcf)*255, self.cat3d(xclmask)*255, 
            #             ((xs*C_xsmask)+1)*127.5, (self.cat3d(xcm_gray)+1)*127.5, xsf * ((xs*M_C_Union + xc*xclmask)+1)*127.5, xsf * self.cat3d(M_union_L_xs_xc)*255, 
            #         ], dim=0), f'/content/export/dataset/B{batch_idx}.png', stype='img', sparams={'chw2hwc': True, 'nrow': 4})


            y_edit_xc = batch['ynl'][cidx][0]

            if flag_logdata:
                pack_logdata[f'xc{cidx}'] = xc

            for optimizer_idx, optimizer_params in [[0, {'VgradViewrFlag': True}], [0, {'condstep': True}], [1, {}]]:
                self.VgradViewrFlag = optimizer_params.get('VgradViewrFlag', False)
                # print('Qs) start', optimizer_idx, optimizer_params)#, self.encoder.Qsurface2Qdiagonal.z0.convt.weight[0,0])
                # print(optimizer_idx, optimizer_params, self.encoder.netb_diagonal.c0.convt.weight[0][0,0])
                # print(f'before optidx={optimizer_idx}',optimizer_params, self.decoder.up[4].attn[1].k.weight.requires_grad, self.decoder.up[4].attn[1].k.weight.sum().item())
                # print(f'batch_idx={batch_idx} | optimizer_idx={optimizer_idx} | cidx={cidx}')
                if optFlag:
                    opt_ae.zero_grad()
                    opt_disc.zero_grad()
                
                loss, logdict, logdata = self.pipline(xs, xc, xsf,
                    split=f'{tag}_',
                    optidx=optimizer_idx,
                    y_edit=y_edit, y_edit_xc=y_edit_xc, xsmask=xslmask, xcmask=xclmask, C_xsmask=C_xsmask, C_xcmask=C_xcmask, xcm_gray=xcm_gray,
                    **optimizer_params
                )
                logdict['epoch'] = self.current_epoch
                
                if optFlag:
                    self.manual_backward(loss)
                    if optimizer_idx == 0:
                        opt_ae.step()
                    else:
                        opt_disc.step()
                
                self.metrics.log(tag, logdict)
                
                if flag_logdata and logdata != None:
                    pack_logdata[f'c{cidx}_optidx{optimizer_idx}_pipline'] = logdata
                
                # print(f'after optidx={optimizer_idx}',optimizer_params, self.decoder.up[4].attn[1].k.weight.requires_grad, self.decoder.up[4].attn[1].k.weight.sum().item())

        if flag_logdata:
            self.imglogger[y_edit] = pack_logdata

        # execution_time_in_sec = (time.time() - start_time)
        # print(f'batch_idx={batch_idx} | execution_time_in_sec={execution_time_in_sec}')
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, tag='train', show_dataset=True)
    
    def validation_step(self, batch, batch_idx):
        return
        p = torch.rand(1)
        force_train = False # (p>.5).item()
        return self.step(batch, batch_idx, tag='val', force_train=force_train)
    
    
    
    def cat3d(self, x):
        return torch.cat([x,x,x], dim=1) 
    def bb(self, img):
        return torchvision.utils.draw_bounding_boxes(((img.squeeze()+1)*127.5).to(torch.uint8), torch.tensor([0,0, 255,255], dtype=torch.int).unsqueeze(0), colors='red').to(self.device).unsqueeze(0) /127.5 -1
    
    def on_train_epoch_end(self):
        # self.imglogger --> save!!
        signal_save((torch.cat([
            self.imglogger[0]['xc0'], self.imglogger[0]['xc1'], self.bb(self.imglogger[0]['xs']), self.imglogger[0]['c0_optidx0_pipline']['𝝍s_tp_final'], self.imglogger[0]['c1_optidx0_pipline']['𝝍s_tp_final'],
            self.imglogger[1]['xc0'], self.imglogger[1]['xc1'], self.imglogger[1]['c0_optidx0_pipline']['𝝍s_tp_final'], self.bb(self.imglogger[1]['xs']), self.imglogger[1]['c1_optidx0_pipline']['𝝍s_tp_final'],
            self.imglogger[2]['xc0'], self.imglogger[2]['xc1'], self.imglogger[2]['c0_optidx0_pipline']['𝝍s_tp_final'], self.imglogger[2]['c1_optidx0_pipline']['𝝍s_tp_final'], self.bb(self.imglogger[2]['xs'])
        ], dim=0)+1)*127.5, f'/content/export/E{self.current_epoch}.png', stype='img', sparams={'chw2hwc': True, 'nrow': 5})

        self.metrics.save('train')
        last_d1_acc = self.metrics.inference('train', self.regexp_d1_acc)
        # last_d2_acc = self.metrics.inference('train', self.regexp_d2_acc)
        last_op_acc = self.metrics.inference('train', self.regexp_OP_acc)
        self.acc['train_'] = {'d1': last_d1_acc, 
                              'd2': 0, 
                              'O': last_op_acc}
        print('train_', self.acc['train_'])
        self.imglogger = [None, None, None]
    
    def on_validation_epoch_end(self):
        return
        self.metrics.save('val')
        last_d1_acc = self.metrics.inference('val', self.regexp_d1_acc)
        # last_d2_acc = self.metrics.inference('val', self.regexp_d2_acc)
        last_op_acc = self.metrics.inference('val', self.regexp_OP_acc)
        self.acc['val_'] = {'d1': last_d1_acc, 
                            'd2': 0, 
                            'O': last_op_acc}
        print('val_', self.acc['val_'])
    
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
        opt_disc = torch.optim.Adam([
                                        {'params': self.loss.discriminator.parameters()},
                                        # {'params': self.loss.discriminator_large.parameters()},
                                        {'params': self.loss.vgg16.parameters(), 'lr': .02},
                                        {'params': self.loss.vgg16_head.parameters(), 'lr': .02}
                                    ],
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
