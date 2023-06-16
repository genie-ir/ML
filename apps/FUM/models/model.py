import torch
from torch import nn
import torch.nn.functional as F
from utils.pt.nnModuleBase import nnModuleBase
from utils.pl.plModuleBase import plModuleBase
from libs.basicIO import signal_save, compressor

class FUM(plModuleBase):
    def generator_step(self, batch):
        print('generator step')

        print(batch.keys())
        phi = self.vqgan.rec_phi({
            'x': batch[self.signal_key],
            'y': batch['y']
        })
        print(phi.shape)

        assert False
        return None, {'loss': -1}