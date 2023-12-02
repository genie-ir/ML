import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.pt.building_block import BB

def fold3d(x, gp=None):
    """
        x is x3d
        gp is grid patch size
    """
    B, ch, h, w = x.shape
    gp = gp if gp else int(ch ** .5)
    return x.view(B, gp, gp, 1, h, w).permute(0, 3, 1, 4, 2, 5).contiguous().view(B, 1, gp*h, gp*w) 


class SPADE(BB): 
    def start(self):
        self.fwd = str(self.kwargs.get('fwd', ''))
        setattr(self, 'forward', getattr(self, f'forward_{self.fwd}'))
        if self.fwd == 'ilevel1': # torch.Size([1, 128/256, 256, 256])
            # self.fconv = # TODO
            pass
        if self.fwd == 'endDownSampling': # torch.Size([1, 512/1024, 16, 16])
            pass
        # self.alphaconv = nn.Sequential(
        #     torch.nn.Conv2d(self.xch, self.alphach, int(self.alphaconv_ksp[0]), stride=int(self.alphaconv_ksp[1]), padding=int(self.alphaconv_ksp[2])),
        #     torch.nn.Conv2d(self.alphach, 2*self.alphach, int(self.alphaconv_ksp[0]), stride=int(self.alphaconv_ksp[1]), padding=int(self.alphaconv_ksp[2])),
        #     torch.nn.BatchNorm2d(2*self.alphach)
        # )
    
    # def forward0(self, x, featuremap):
    #     """
    #         x.shape is torch.Size([1, 3, 256, 256])
    #     """
    #     alpha = self.alphaconv(x)
    #     beta = self.betaconv(alpha)
    #     gamma = self.gammaconv(alpha)
    #     return self.fconv(self.bn(featuremap)) * gamma + beta
    
    def forward_ilevel1(self, x, featuremap):
        """1x3x256x256  1x256x256x256"""
        featuremap = fold3d(featuremap)
        print('forward_ilevel1', featuremap.shape)



        alpha = self.alphaconv(x)
        beta = self.betaconv(alpha)
        gamma = self.gammaconv(alpha)
        return self.fconv(self.bn(featuremap)) * gamma + beta