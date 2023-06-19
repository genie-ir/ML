import torch
from torch import nn
import torch.nn.functional as F
from utils.pt.nnModuleBase import nnModuleBase
from utils.pl.plModuleBase import plModuleBase
from libs.basicIO import signal_save, compressor

class MAP(plModuleBase):
    def validation_step(self, batch, batch_idx, split='val'):
        print(
            'val --->', self.OH.net_seq0[0].weight[0,0]
        )
        return super().validation_step(batch, batch_idx, split)
    
    def training_step(self, batch, batch_idx, split='train'):
        print(
            'train --->', self.OH.net_seq0[0].weight[0,0]
        )
        return super().training_step(batch, batch_idx, split)
    
    def OH_step(self, batch):
        return self.OHLoss(
            batch['Yi'].long(), 
            batch['Yip1'].long(),
            self.OH(x=batch['Xi']),
            self.OH(x=batch['Xip1']),
        )