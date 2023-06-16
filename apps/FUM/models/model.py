import torch
from torch import nn
import torch.nn.functional as F
from utils.pt.nnModuleBase import nnModuleBase
from utils.pl.plModuleBase import plModuleBase
from libs.basicIO import signal_save, compressor

class FUM(plModuleBase):
    def generator_step(self, batch):
        phi = self.vqgan.rec_phi({
            'x': batch[self.signal_key],
            'y': batch['y']
        })
        # self.vqgan.save_phi(phi, pathdir='/content')

        # print(phi.shape)
        # print(self.vqgan.loss.discriminator(phi).shape)

        for i in range(3):
            logits_fake = self.vqgan.loss.discriminator(phi.contiguous())
            g_loss = -torch.mean(logits_fake)
            # print('logits_fake', logits_fake.shape, logits_fake)
            print('g_loss', g_loss.shape, g_loss)
            phi = torch.randn(phi.shape, device=self.device)
        assert False
        return None, {'loss': -1}