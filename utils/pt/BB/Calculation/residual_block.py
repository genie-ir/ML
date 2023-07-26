import torch
import functools
from torch import nn
from utils.pt.building_block import BB
from utils.pt.nonlinearity import swish

class MAC(BB):
    """MAC unit performs `multiplication` and `accumulation` process"""
    def start(self):
        self.fwd = str(self.kwargs.get('fwd', 'f1'))
        self.units = int(self.kwargs.get('units', 1))
        self.shape = list(self.kwargs.get('shape', []))
        self.w = nn.ParameterList([
            nn.Parameter(torch.randn(self.shape))
            for unit in range(self.units)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.randn(self.shape))
            for unit in range(self.units)
        ])
        setattr(self, 'forward', getattr(self, self.fwd))

    def f1(self, x):
        x0 = x
        for i, p in enumerate(self.w):
            x = swish(x0 + (self.w[i] * x + self.b[i]))
        return x
    
    def f2(self, x):
        for i, p in enumerate(self.w):
            x = swish(x + (self.w[i] * x + self.b[i]))
        return x