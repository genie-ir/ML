import torch
from torch import nn
import torch.nn.functional as F
from utils.pt.nnModuleBase import nnModuleBase
from utils.pl.plModuleBase import plModuleBase
from libs.basicIO import signal_save, compressor

class FUM(plModuleBase):
    
    
    def generator_step(self, batch):
        print('generator step')
        return None, {'loss': -1}
    
    def discriminator_step(self, batch):
        print('discriminator step')
        return None, {'dloss': -1}

class D(nnModuleBase):
    def forward(self, **kargs):
        kargs['x'] = (kargs['x'] - .5) / .5
        return super().forward(**kargs)