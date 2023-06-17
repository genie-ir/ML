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

    def generator_step(self, batch):
        y = batch['y']
        # print(batch[self.signal_key].min().item(), batch[self.signal_key].max().item())
        xf = error_grade(batch[self.signal_key], 3)
        
        print(self.generator)
        self.counter += 1
        if self.counter == 2:
            assert False
        
        xt = torch.tensor(denormalizing(xf.detach().cpu().numpy()), device=self.device, dtype=torch.float)
        phi = self.vqgan.rec_phi({
            'x': xt,
            'y': y
        })
        # self.vqgan.save_phi(phi, pathdir='/content')

        g_loss = -torch.mean(self.vqgan.loss.discriminator(phi.contiguous()))
        print('g_loss', g_loss.shape, g_loss, g_loss.requires_grad)

        assert False
        return None, {'loss': -1}