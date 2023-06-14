import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.pt.building_block import BB

class CCR(BB):
    def start(self):
        self.archcode = int(self.kwargs.get('archcode', 0))
        self.C0 = self.kwargs.get('C0', dict())
        self.C1 = self.kwargs.get('C1', dict())

        if self.archcode == 0:
            # spec: skip connection apply on C0.out and C1.out
            self.ac0_stage0 = nn.Sequential(*[
                nn.BatchNorm2d(num_features=self.C0['in_channels']),
                nn.ReLU(),
                nn.Conv2d(**self.C0),
            ])
            self.ac0_stage1 = nn.Sequential(*[
                nn.BatchNorm2d(num_features=self.C1['in_channels']),
                nn.ReLU(),
                nn.Conv2d(**self.C1)
            ])
        elif self.archcode == 1:
            # spec: C0.in_ch == C0.out_ch == C1.in_ch == C1.out_ch | start with conv | skip connection apply on end to end
            self.ac1_stage = nn.Sequential(*[
                nn.Conv2d(**self.C0),
                nn.BatchNorm2d(num_features=self.C1['in_channels']),
                nn.ReLU(),
                nn.Conv2d(**self.C1)
            ])
        else:
            assert False, '`self.archcode={}` | Does not supported, please do code for this'.format(self.archcode)

        setattr(self, 'forward', getattr(self, 'forward{}'.format(self.archcode)))
        

    def forward0(self, x):
        x = self.ac0_stage0(x)
        return x + self.ac0_stage1(x)
    
    def forward1(self, x):
        return x + self.ac1_stage(x)

        