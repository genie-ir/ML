import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.pt.building_block import BB

class SPADE(BB): 
    def start(self):
        self.fwd = str(self.kwargs.get('fwd', ''))
        if self.fwd == 'ilevel1': # torch.Size([1, 128, 256, 256])
            pass
        if self.fwd == 'endDownSampling': # torch.Size([1, 512, 16, 16])
            pass
        # self.alphaconv = nn.Sequential(
        #     torch.nn.Conv2d(self.xch, self.alphach, int(self.alphaconv_ksp[0]), stride=int(self.alphaconv_ksp[1]), padding=int(self.alphaconv_ksp[2])),
        #     torch.nn.Conv2d(self.alphach, 2*self.alphach, int(self.alphaconv_ksp[0]), stride=int(self.alphaconv_ksp[1]), padding=int(self.alphaconv_ksp[2])),
        #     torch.nn.BatchNorm2d(2*self.alphach)
        # )
    
    def forward(self, x, featuremap):
        """
            x.shape is torch.Size([1, 3, 256, 256])
        """
        print('SPADE', x.shape, featuremap.shape)
        alpha = self.alphaconv(x)
        beta = self.betaconv(alpha)
        gamma = self.gammaconv(alpha)
        return self.fconv(self.bn(featuremap)) * gamma + beta