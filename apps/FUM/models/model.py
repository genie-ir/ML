import torch
from torch import nn
import torch.nn.functional as F
from utils.pt.nnModuleBase import nnModuleBase
from utils.pl.plModuleBase import plModuleBase
from libs.basicIO import signal_save, compressor
from apps.FUM.data.eyepacs import denormalizing

class FUM(plModuleBase):
    def generator_step(self, batch):
        y = batch['y']
        xf0 = batch[self.signal_key]
        xfN = 5.5 * torch.ones_like(xf0, device=self.device) / 1e4
        print(xf0.min().item(), xf0.max().item())
        print(xfN.min().item(), xfN.max().item())
        xf = xf0 + xfN
        xt = torch.tensor(denormalizing(xf.detach().cpu().numpy()), device=self.device, dtype=torch.float)
        phi = self.vqgan.rec_phi({
            'x': xt,
            'y': y
        })
        self.vqgan.save_phi(phi, pathdir='/content')

        g_loss = -torch.mean(self.vqgan.loss.discriminator(phi.contiguous()))
        print('g_loss', g_loss.shape, g_loss, g_loss.requires_grad)

        print(xf.shape, xt.shape, phi.shape)

        assert False
        return None, {'loss': -1}