import os
import sys
import torch
import cowsay
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class BB(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        kwargs_new = dict()
        for k, v in kwargs.items():
            if isinstance(k, str) and ':' in k:
                k_split = k.split(':')
                assert len(k_split) == 2, '`len(k_split)={}` | It must be `2` | looking this maybe useful: `k={}`'.format(len(k_split), k)
                kwargs_new[k_split[0]] = kwargs_new.get(k_split[0], dict())
                kwargs_new[k_split[0]][k_split[1]] = v
            else:
                kwargs_new[k] = v

        self.kwargs = kwargs_new
        self.start()
    
    def start(self):
        cowsay.cow('NotImplementedError:\nplease define `{}` function for `{}` building block.'.format('start', self.__class__.__name__))
        sys.exit()
    
    def visualize(self, **kwargs):
        # print('----------->', id(self), kwargs)
        # kwargs['create_node'](id(self))
        # kwargs['create_node'](self.__class__.__name__)
        pass
        