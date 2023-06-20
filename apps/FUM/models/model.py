import torch
from torch import nn
import torch.nn.functional as F
from utils.pt.nnModuleBase import nnModuleBase
from utils.pl.plModuleBase import plModuleBase
from libs.basicIO import signal_save, compressor
from apps.FUM.data.eyepacs import denormalizing
from utils.pt.tricks.error_grade import error_grade

class FUM(plModuleBase):
    def start(self):
        self.counter = -1

    def validation_step(self, batch, batch_idx, split='val'):
        pass

    def generator_step(self, batch):
        self.counter += 1
        if self.counter == 4:
            assert False
        else:
            print(self.generator.net_seq0[0].weight[0, :10])

        y = batch['y']
        xf = error_grade(batch[self.signal_key], 3)
        xf = self.generator(x=xf)

        xt = denormalizing(xf)
        phi = self.vqgan.rec_phi({'x': xt, 'y': y})
        self.vqgan.save_phi(phi, pathdir='/content')

        g_loss = -torch.mean(self.vqgan.loss.discriminator(phi.contiguous()))
        print('g_loss', g_loss.shape, g_loss, g_loss.requires_grad)

        return g_loss, {'loss': g_loss.item()}