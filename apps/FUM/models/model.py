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
        for i in range(50):
            phi = self.vqgan.rec_phi(x=latent)
            latent_rec = self.vqgan.rec_lat(phi).float()
            rec_metric = (latent-latent_rec).abs().sum()
            print('------>', rec_metric)
            latent = latent_rec
            self.vqgan.save_phi(phi, pathdir=pathdir, fname=f'{str(i)}.png')
            if rec_metric < 1e-6 or old_rec_metric == rec_metric:
                break
            old_rec_metric = rec_metric
        compressor(pathdir, pathdir + '/phi.zip')

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
