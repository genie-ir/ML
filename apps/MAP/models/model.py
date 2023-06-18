import torch
from torch import nn
import torch.nn.functional as F
from utils.pt.nnModuleBase import nnModuleBase
from utils.pl.plModuleBase import plModuleBase
from libs.basicIO import signal_save, compressor

class MAP(plModuleBase):
    def OH_step(self, batch):
        # print('Xi', batch['Xi'].shape)
        # print('Yi', batch['Yi'].shape)
        # print(batch['Xi'][0,0,0], batch['Yi'][0])
        
        yi = self.OH(x=batch['Xi'])
        yip1 = self.OH(x=batch['Xip1'])

        print(yi.shape, yip1.shape)

        assert False