import torch
from torch import nn
import torch.nn.functional as F
from utils.pt.nnModuleBase import nnModuleBase
from utils.pl.plModuleBase import plModuleBase
from libs.basicIO import signal_save, compressor
from utils.pt.BB.Scratch.Transformer.transformer import Transformer

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
        # xf = error_grade(batch[self.signal_key], 3)
        # xf = self.generator(x=xf)
        # xt = denormalizing(xf)
        latent = batch[self.signal_key].float()
        pathdir=f'/content/phi'
        old_rec_metric = -1
        shape = (batch['batch_size'], self.phi_ch, self.phi_wh, self.phi_wh)
        s1 = torch.zeros(shape, device=self.device)
        s2 = torch.zeros(shape, device=self.device)
        old_phi = None
        for N in range(1, self.phi_it + 1):
            phi, q = self.vqgan.rec_phi(x=latent, flag=True)
            if old_phi is not None:
                print('!!!!!!!!', (phi-old_phi).abs().sum().item())
            old_phi = phi

            s1 = s1 + phi
            s2 = s2 + phi ** 2
            latent_rec = self.vqgan.rec_lat(phi).float()
            rec_metric = (latent-latent_rec).abs().sum()
            # print('--lm-->', rec_metric)
            # print('--lm-->', rec_metric, rec_metric.shape, rec_metric.requires_grad, rec_metric.dtype)
            # print('--phi-->', phi.shape, phi.requires_grad, phi.dtype)
            
            
            # phir, qr = self.vqgan.rec_phi(x=latent_rec, flag=True)
            # print('---qm--->', (q-qr).abs().sum())
            # print('---phim--->', (phir-phi).abs().sum())
            
            latent = latent_rec
            self.vqgan.save_phi(phi, pathdir=pathdir, fname=f'phi-{str(N)}.png')
            if rec_metric < 1e-6 or old_rec_metric == rec_metric:
                break
            old_rec_metric = rec_metric
        # compressor(pathdir, pathdir + '/phi.zip')
        mue = s1 / N # R
        mue_latent_rec = self.vqgan.rec_lat(mue).float() # r
        # mue_rec, mue_q = self.vqgan.rec_phi(x=mue_latent_rec, flag=True)
        
        std = ((s2 + ((mue ** 2) * N) + (-2 * mue * s1)) / (N)) ** .5
        
        sample = (std) * torch.randn(shape, device=self.device) + mue
        sample2 = (std) * torch.randn(shape, device=self.device) + mue
        sample3 = (std) * torch.randn(shape, device=self.device) + mue
        self.vqgan.save_phi(mue, pathdir=pathdir, fname=f'mue-{str(N)}.png')
        # self.vqgan.save_phi(mue_rec, pathdir=pathdir, fname=f'mue_rec-{str(N)}.png')
        print('########', mue.shape, mue.dtype, sample.shape, sample.dtype)
        print('########', mue.min().item(), mue.max().item(), std.min().item(), std.max().item())
        print('######ms', (mue-sample).abs().sum().item())
        print('######ms2', (mue-sample2).abs().sum().item())
        print('######ms3', (mue-sample3).abs().sum().item())
        self.vqgan.save_phi(sample, pathdir=pathdir, fname=f'sample-{str(N)}.png')
        self.vqgan.save_phi(sample2, pathdir=pathdir, fname=f'sample2-{str(N)}.png')
        self.vqgan.save_phi(sample3, pathdir=pathdir, fname=f'sample3-{str(N)}.png')

        g_loss = -torch.mean(self.vqgan.loss.discriminator(phi.contiguous()))
        print('g_loss', g_loss.shape, g_loss, g_loss.requires_grad)
        assert False
        return g_loss, {'loss': g_loss.item()}
    
    def start(self):
        print('!!!!!!!', self.ncluster, self.embed_size)
        self.codebook = nn.Embedding(self.ncluster, self.embed_size)
    
    def generator_step00(self, batch):
        x = self.codebook(batch[self.signal_key])
        phi = self.vqgan.rec_phi({'x': x, 'y': batch['y']})
        self.vqgan.save_phi(phi, pathdir='/content')

        g_loss = -torch.mean(self.vqgan.loss.discriminator(phi.contiguous()))
        print('g_loss', g_loss.shape, g_loss, g_loss.requires_grad)
        assert False
        return g_loss, {'loss': g_loss.item()}
