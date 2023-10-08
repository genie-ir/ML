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
        if self.fwd == 'fConv2d':
            ch = int(self.kwargs.get('ch', 3))
            self.w = [
                torch.nn.Conv2d(ch,    30*ch, 3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device='cuda', dtype=None),
                torch.nn.Conv2d(30*ch, 20*ch, 3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device='cuda', dtype=None),
                torch.nn.Conv2d(20*ch, 10*ch, 3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device='cuda', dtype=None),
                torch.nn.Conv2d(10*ch, ch,    1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device='cuda', dtype=None)
                #   for unit in range(self.units)
            ]
        else:
            self.w = nn.ParameterList([self.nnParameter(shape=self.shape) for unit in range(self.units)])
            # self.b = nn.ParameterList([self.nnParameter(shape=self.shape) for unit in range(self.units)])
        setattr(self, 'forward', getattr(self, self.fwd))

    def fConv2d(self, x):
        for i, p in enumerate(self.w):
            x = swish(self.w[i](x))
        return x
    
    def f1(self, x):
        x0 = x
        for i, p in enumerate(self.w):
            x = swish(x0 + (self.w[i] * x))
        return x
    
    def f2(self, x):
        for i, p in enumerate(self.w):
            x = swish(x + (self.w[i] * x))
        return x
    # def f1(self, x):
    #     x0 = x
    #     for i, p in enumerate(self.w):
    #         x = swish(x0 + (self.w[i] * x + self.b[i]))
    #     return x
    
    # def f2(self, x):
    #     for i, p in enumerate(self.w):
    #         x = swish(x + (self.w[i] * x + self.b[i]))
    #     return x