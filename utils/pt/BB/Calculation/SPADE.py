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
        
        if self.fwd == 'ilevel1': # torch.Size([1, 128/256, 256, 256])
            self.fconvbn = nn.Sequential(
                torch.nn.Conv2d(1, 32, 5, 4, 1),
                torch.nn.BatchNorm2d(32),
                nn.ReLU(),
                torch.nn.Conv2d(32, 64, 3, 2, 1),
                torch.nn.BatchNorm2d(64),
                nn.ReLU(),
                torch.nn.Conv2d(64, 128, 3, 2, 1),
            )
            self.alphaconv = nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, 1, 1), # 1x1x1024x1024
            )
            self.betaconv = nn.Sequential(
                torch.nn.Conv2d(1, 64, 3, 2, 1),
                torch.nn.BatchNorm2d(64),
                nn.ReLU(),
                torch.nn.Conv2d(64, 128, 3, 2, 1),
            )
            self.gammaconv = nn.Sequential(
                torch.nn.Conv2d(1, 64, 3, 2, 1),
                torch.nn.BatchNorm2d(64),
                nn.ReLU(),
                torch.nn.Conv2d(64, 128, 3, 2, 1),
            )
        
        if self.fwd == 'endDownSampling': # torch.Size([1, 512/1024, 16, 16])
            self.fconvbn = nn.Sequential(
                torch.nn.Conv2d(1, 128, 5, 4, 1),
                torch.nn.BatchNorm2d(128),
                nn.ReLU(),
                torch.nn.Conv2d(128, 256, 5, 4, 1),
                torch.nn.BatchNorm2d(256),
                nn.ReLU(),
                torch.nn.Conv2d(256, 512, 3, 2, 1),
            )
            self.alphaconv = nn.Sequential(
                torch.nn.Conv2d(3, 4, 3, 2, 1), #128**2
            )
            self.betaconv = nn.Sequential(
                torch.nn.Conv2d(4, 256, 5, 4, 1),
                torch.nn.BatchNorm2d(256),
                nn.ReLU(),
                torch.nn.Conv2d(256, 512, 5, 4, 1),
            )
            self.gammaconv = nn.Sequential(
                torch.nn.Conv2d(4, 256, 5, 4, 1),
                torch.nn.BatchNorm2d(256),
                nn.ReLU(),
                torch.nn.Conv2d(256, 512, 5, 4, 1),
            )
    




    def forward(self, xclpure, fmap, flag):
        print('1 ->', self.fwd, fmap.shape)
        femap = fold3d(fmap)
        print('2 ->', self.fwd, femap.shape)
        featuremap = self.fconvbn(femap)
        print('3 ->', self.fwd, featuremap.shape)
        if not flag:
            return featuremap

        alpha = fold3d(self.alphaconv(xclpure))
        beta = self.betaconv(alpha)
        gamma = self.gammaconv(alpha)
        r =  featuremap * gamma + beta
        print('4 ->', self.fwd, r.shape)

        return r