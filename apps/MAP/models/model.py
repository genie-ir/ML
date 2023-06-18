import torch
from torch import nn
import torch.nn.functional as F
from utils.pt.nnModuleBase import nnModuleBase
from utils.pl.plModuleBase import plModuleBase
from libs.basicIO import signal_save, compressor

class MAP(plModuleBase):
    def validation_step(self, batch, batch_idx, split='val'):
        return
    def OH_step(self, batch):
        print(
            '--->', self.OH.net_seq0[0].weight[0,0]
        )
        return self.OHLoss(
            batch['Yi'].long(), 
            batch['Yip1'].long(),
            self.OH(x=batch['Xi']),
            self.OH(x=batch['Xip1']),
        )