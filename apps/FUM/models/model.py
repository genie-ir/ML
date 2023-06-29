import torch
from torch import nn
import torch.nn.functional as F
from utils.pt.nnModuleBase import nnModuleBase
from utils.pl.plModuleBase import plModuleBase
from libs.basicIO import signal_save, compressor

class FUM(plModuleBase):
    def validation_step(self, batch, batch_idx, split='val'):
        pass

    def generator_step(self, batch):
        print('------>', batch)
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
