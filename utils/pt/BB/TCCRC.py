import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.pt.building_block import BB

class TCCRC(BB):
    def start(self):
        self.T = self.kwargs.get('T', dict())
        self.C = self.kwargs.get('C', dict())

        self.stage0 = nn.Sequential(*[
            nn.ConvTranspose2d(**self.T),
            nn.BatchNorm2d(num_features=self.T['out_channels']),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.T['out_channels'], out_channels=self.T['out_channels']*2, **self.C),
            nn.BatchNorm2d(num_features=self.T['out_channels']*2),
            nn.ReLU(),
        ])

        self.stage1 = nn.Sequential(*[
            nn.Conv2d(in_channels=self.T['out_channels']*2, out_channels=self.T['out_channels']*2, **self.C),
            nn.BatchNorm2d(num_features=self.T['out_channels']*2),
            nn.ReLU(),
        ])

        self.stage2 = nn.Sequential(*[
            nn.Conv2d(in_channels=self.T['out_channels']*2, out_channels=self.T['out_channels']*4, **self.C),
            nn.BatchNorm2d(num_features=self.T['out_channels']*4),
            nn.ReLU(),
        ])
    
    def forward(self, x):
        x = self.stage0(x)
        r = x
        x = self.stage1(x)
        x = self.stage2(x + r)
        return x

        