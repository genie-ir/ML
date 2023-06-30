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
    
    def start(self):
        self.seqnum = 5
        self.seqlen = 3 # 256
        self.seqdim = 2 # 1
        self.vocab_size = 451
        self.transformer = Transformer(
            heads=1,
            maxlen=self.seqlen,
            dropout=0,
            fwd_expan=4,
            num_layers=8,
            embed_size=self.seqdim,
            src_mask=False,
            trg_mask=True,
            src_vocab_size=self.vocab_size,
            trg_vocab_size=self.vocab_size
        )

    def generator_step(self, batch):
        print('++>', batch[self.signal_key])

        src = torch.randint(0, self.vocab_size, (self.seqnum, self.seqlen), device=self.device)
        trg = torch.randint(0, self.vocab_size, (self.seqnum, self.seqlen), device=self.device)
        print('$$$$$$$$$$$$$$', src.device, trg.device)
        print('@@@@@@@@@@@@@@', trg.shape, trg[:, :-1].shape)
        print('++++++++>', self.transformer(src, trg[:, :-1]))
        assert False

    def generator_step2(self, batch):
        y = batch['y']
        # xf = error_grade(batch[self.signal_key], 3)
        # xf = self.generator(x=xf)
        # xt = denormalizing(xf)
        xt = batch[self.signal_key].float()
        phi = self.vqgan.rec_phi({'x': xt, 'y': y})
        self.vqgan.save_phi(phi, pathdir='/content')

        g_loss = -torch.mean(self.vqgan.loss.discriminator(phi.contiguous()))
        print('g_loss', g_loss.shape, g_loss, g_loss.requires_grad)

        return g_loss, {'loss': g_loss.item()}
