import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.pt.building_block import BB

class CC(BB): 
    def start(self):
        self.C = self.kwargs.get('C', dict())
        self.C0 = self.kwargs.get('C0', {**self.C})
        self.C1 = self.kwargs.get('C1', {**self.C})
        self.C1['in_channels'] = self.C0['out_channels']

        self.double_conv = nn.Sequential(
            nn.Conv2d(**self.C0),
            nn.BatchNorm2d(self.C0['out_channels']),
            nn.ReLU(inplace=True),
            nn.Conv2d(**self.C1),
            nn.BatchNorm2d(self.C1['out_channels']),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)