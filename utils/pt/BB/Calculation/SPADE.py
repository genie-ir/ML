import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.pt.building_block import BB

class SPADE(BB): 
    def start(self):
        self.fch = int(self.kwargs.get('fch', -1))
        self.xch = int(self.kwargs.get('xch', -1))
        
        
        self.alphach = int(self.kwargs.get('alphach', -1))
        self.betach = int(self.kwargs.get('betach', self.alphach))
        self.gammach = int(self.kwargs.get('gammach', self.alphach))
        assert self.betach == self.gammach
        
        
        self.ksp = str(self.kwargs.get('ksp', '311'))
        self.fconv_ksp = str(self.kwargs.get('fconv_ksp', self.ksp))
        self.alphaconv_ksp = str(self.kwargs.get('alphaconv_ksp', self.ksp))
        self.betaconv_ksp = str(self.kwargs.get('betaconv_ksp', self.ksp))
        self.gammaconv_ksp = str(self.kwargs.get('gammaconv_ksp', self.ksp))
        
        
        self.bn = torch.nn.BatchNorm2d(self.fch)
        self.fconv = torch.nn.Conv2d(self.fch, self.gammach, int(self.fconv_ksp[0]), stride=int(self.fconv_ksp[1]), padding=int(self.fconv_ksp[2]))
        
        
        self.alphaconv = torch.nn.Conv2d(self.xch, self.alphach, int(self.alphaconv_ksp[0]), stride=int(self.alphaconv_ksp[1]), padding=int(self.alphaconv_ksp[2]))
        self.betaconv = torch.nn.Conv2d(self.alphach, self.betach, int(self.betaconv_ksp[0]), stride=int(self.betaconv_ksp[1]), padding=int(self.betaconv_ksp[2]))
        self.gammaconv = torch.nn.Conv2d(self.alphach, self.gammach, int(self.gammaconv_ksp[0]), stride=int(self.gammaconv_ksp[1]), padding=int(self.gammaconv_ksp[2]))

    
    def forward(self, x, featuremap):
        alpha = self.alphaconv(x)
        beta = self.betaconv(alpha)
        gamma = self.gammaconv(alpha)
        return self.fconv(self.bn(featuremap)) * gamma + beta