import torch
from torch import nn
import torch.nn.functional as F
from utils.pt.nnModuleBase import nnModuleBase
from utils.pl.plModuleBase import plModuleBase
from libs.basicIO import signal_save, compressor

class FUM(plModuleBase):
    def start(self):
        print(self.VQGAN_PATH)
        assert False
        # self.vqgan = self.get_pretrained_model()
    
    def generator_step(self, batch):
        print('generator step')
        return None, {'loss': -1}