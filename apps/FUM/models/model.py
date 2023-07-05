import torch
from torch import nn
import torch.nn.functional as F
from utils.pt.nnModuleBase import nnModuleBase
from utils.pl.plModuleBase import plModuleBase
from libs.basicIO import signal_save, compressor
try:
    from utils.pt.BB.Scratch.Transformer.transformer import Transformer
    from utils.preprocessing.text.tokenizer import Tokenizer
except Exception as e:
    print(e)
    assert False

class FUM(plModuleBase):
    def validation_step(self, batch, batch_idx, split='val'):
        pass
    
    def start(self):
        T = Tokenizer('en', 'de')
        for B in T.dataloaders['train']:
            print(B[0], B[0].shape, B[1].shape)
            assert False

    def start2(self):
        self.seqnum = 5
        self.seqlen = 3 # 256
        self.seqdim = 2 # 1
        self.vocab_size = 451
        self.transformer = Transformer(
            heads=1,
            maxlen=10,#self.seqlen,
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

        src = torch.randint(0, self.vocab_size, (self.seqnum, 3), device=self.device)
        trg = torch.randint(0, self.vocab_size, (self.seqnum, 8), device=self.device)
        out = self.transformer(src, trg)
        print('++++++++>', out, out.shape)
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
