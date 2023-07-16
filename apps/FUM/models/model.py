import torch
from torch import nn
import torch.nn.functional as F
from utils.pt.nnModuleBase import nnModuleBase
from utils.pl.plModuleBase import plModuleBase
from libs.basicIO import signal_save, compressor
from utils.pt.BB.Scratch.Transformer.transformer import Transformer
from utils.pt.BB.Quantizer.VectorQuantizer import VectorQuantizer2 as VectorQuantizer

class FUM(plModuleBase):
    def validation_step(self, batch, batch_idx, split='val'):
        pass
    
    def __start(self):
        self.seqlen = 3 # 256
        self.seqdim = 2 # 1
        self.vocab_size = 451
        self.transformer = Transformer(
            heads=getattr(self, 'heads', 1),
            maxlen=getattr(self, 'maxlen', 1e3),
            dropout=getattr(self, 'dropout', 0),
            fwd_expan=getattr(self, 'fwd_expan', 4),
            num_layers=getattr(self, 'num_layers', 8),
            trg_mask=getattr(self, 'trg_mask', True),
            src_mask=getattr(self, 'src_mask', False),
            embed_size=getattr(self, 'embed_size', 256),
            trg_vocab_size=getattr(self, 'trg_vocab_size', 1e3),
            src_vocab_size=getattr(self, 'src_vocab_size', 1e3)
        )

    def generator_step_3(self, batch):
        print('++>', batch[self.signal_key])
        B = 2
        src = torch.randint(0, self.vocab_size, (B, 3), device=self.device)
        trg = torch.randint(0, self.vocab_size, (B, 8), device=self.device)
        out = self.transformer(src, trg)
        print('++++++++>', out, out.shape)
        assert False

    def generator_step(self, batch):
        latent = batch[self.signal_key].float()
        old_rec_metric = -1
        shape = (batch['batch_size'], self.phi_ch, self.phi_wh, self.phi_wh)
        s1 = torch.zeros(shape, device=self.device)
        # s2 = torch.zeros(shape, device=self.device)
        for N in range(1, self.phi_it + 1):
            phi, q = self.vqgan.rec_phi(x=latent, flag=True)

            s1 = s1 + phi
            # s2 = s2 + phi ** 2
            latent_rec = self.vqgan.rec_lat(phi).float()
            rec_metric = (latent-latent_rec).abs().sum()
            # print('--lm-->', rec_metric)
            # print('--lm-->', rec_metric, rec_metric.shape, rec_metric.requires_grad, rec_metric.dtype)
            # print('--phi-->', phi.shape, phi.requires_grad, phi.dtype)
            
            # phir, qr = self.vqgan.rec_phi(x=latent_rec, flag=True)
            # print('---qm--->', (q-qr).abs().sum())
            # print('---phim--->', (phir-phi).abs().sum())
            
            latent = latent_rec
            self.vqgan.save_phi(phi, pathdir=self.pathdir, fname=f'phi-{str(N)}.png')
            if rec_metric < 1e-6 or old_rec_metric == rec_metric:
                break
            old_rec_metric = rec_metric
        # compressor(self.pathdir, self.pathdir + '/phi.zip')
        mue = s1 / N # R
        mue_l = self.vqgan.rec_lat(mue).float() # r
        print('!!!!!!!!!!!!!! mue_l', mue_l, mue_l.shape, mue_l.dtype, mue_l.requires_grad)
        assert False
        mue_lq = self.scodebook(mue_l)
        # mue_rec, mue_q = self.vqgan.rec_phi(x=mue_l, flag=True)
        
        self.vqgan.save_phi(mue, pathdir=self.pathdir, fname=f'mue-{str(N)}.png')
        # std = ((s2 + ((mue ** 2) * N) + (-2 * mue * s1)) / (N)).clamp(0).sqrt()
        # sample = (std) * torch.randn(shape, device=self.device) + mue
        # self.vqgan.save_phi(mue_rec, pathdir=self.pathdir, fname=f'mue_rec-{str(N)}.png')
        # self.vqgan.save_phi(sample, pathdir=self.pathdir, fname=f'sample-{str(N)}.png')

        # mue_loss = -torch.mean(self.vqgan.loss.discriminator(mue.contiguous()))
        mue_loss = -torch.mean(self.vqgan.loss.discriminator(mue))
        
        
        
        # print('g_loss', g_loss.shape, g_loss, g_loss.requires_grad)
        assert False
        return g_loss, {'loss': g_loss.item()}
    
    def start(self):
        print('!!!!!!!', self.ncluster, self.embed_size)
        self.scodebook = VectorQuantizer(n_e=self.ncluster, e_dim=self.embed_size, beta=0.25)
        self.ccodebook = VectorQuantizer(n_e=(self.ncrosses * self.ncluster), e_dim=self.embed_size, beta=0.25)
    
    def generator_step00(self, batch):
        x = self.codebook(batch[self.signal_key])
        phi = self.vqgan.rec_phi({'x': x, 'y': batch['y']})
        self.vqgan.save_phi(phi, pathdir='/content')

        g_loss = -torch.mean(self.vqgan.loss.discriminator(phi.contiguous()))
        print('g_loss', g_loss.shape, g_loss, g_loss.requires_grad)
        assert False
        return g_loss, {'loss': g_loss.item()}
