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
            embed_size=getattr(self, 'latent_dim', 256),
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
        z = torch.randint(0, self.latent_range, (batch['batch_size'], self.latent_dim, 1, 1), device=self.device)
        latent = self.ccodebook(z)[0].view(-1, self.qwh, self.qwh)
        print('!!!!!!!!!!!!!!!!!!!!!!', latent.shape, latent.dtype, latent.requires_grad)
        assert False
        # latent = batch[self.signal_key].float()
        old_rec_metric = -1
        phi_shape = (batch['batch_size'], self.phi_ch, self.phi_wh, self.phi_wh)
        s1 = torch.zeros(phi_shape, device=self.device)
        # s2 = torch.zeros(phi_shape, device=self.device)
        for N in range(1, self.phi_steps + 1):
            phi = self.vqgan.lat2phi(latent)
            s1 = s1 + phi
            # s2 = s2 + phi ** 2
            latent_rec = self.vqgan.phi2lat(phi).float()
            rec_metric = (latent-latent_rec).abs().sum()
            # print('--lm-->', rec_metric)
            latent = latent_rec
            self.vqgan.save_phi(phi, pathdir=self.pathdir, fname=f'phi-{str(N)}.png')
            if rec_metric < 1e-6 or old_rec_metric == rec_metric:
                break
            old_rec_metric = rec_metric
        # compressor(self.pathdir, self.pathdir + '/phi.zip')
        mue = s1 / N
        # print('!!!!!!!!!!!!! mue', mue.shape, mue.dtype, mue.requires_grad)
        m = self.vqgan.phi2lat(mue).float().flatten(1).unsqueeze(-1).unsqueeze(-1)
        # print('!!!!!!!!!!!!!! m', m.shape, m.dtype, m.requires_grad)
        s, sloss = self.scodebook(m)
        # print('########### s', s.shape, s.dtype, s.requires_grad)
        sq = self.qw * self.vqgan.lat2qua(s) + self.qb # sq = w x sq + b
        print('!!!!!!!!!! sq', sq.shape, sq.dtype, sq.requires_grad)
        sphi = self.vqgan.qua2phi(sq)
        
        self.vqgan.save_phi(mue, pathdir=self.pathdir, fname=f'mue-{str(N)}.png')
        self.vqgan.save_phi(sphi, pathdir=self.pathdir, fname=f'sphi-{str(N)}.png')
        
        
        
        
        
        
        
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
        self.qshape = (self.qch, self.qwh, self.qwh)
        self.qw = nn.Parameter(torch.randn(self.qshape))
        self.qb = nn.Parameter(torch.randn(self.qshape))
        self.scodebook = VectorQuantizer(n_e=self.ncluster, e_dim=self.latent_dim, beta=0.25, zwh=1)
        self.ccodebook = VectorQuantizer(n_e=(self.ncrosses * self.ncluster), e_dim=self.latent_dim, beta=0.25, zwh=1)
    
    # def generator_step00(self, batch):
    #     x = self.codebook(batch[self.signal_key])
    #     phi = self.vqgan.rec_phi({'x': x, 'y': batch['y']})
    #     self.vqgan.save_phi(phi, pathdir='/content')

    #     g_loss = -torch.mean(self.vqgan.loss.discriminator(phi.contiguous()))
    #     print('g_loss', g_loss.shape, g_loss, g_loss.requires_grad)
    #     assert False
    #     return g_loss, {'loss': g_loss.item()}
