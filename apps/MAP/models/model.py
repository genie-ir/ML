import torch
from torch import nn
import torch.nn.functional as F
from utils.pt.nnModuleBase import nnModuleBase
from utils.pl.plModuleBase import plModuleBase
from libs.basicIO import signal_save, compressor

class MAP(plModuleBase):
    def OH_step(self, batch):
        print(batch.keys())

        assert False