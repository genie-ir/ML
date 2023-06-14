import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.pt.building_block import BB

class Print(BB): 
    def start(self):
        self.singleFlag = bool(self.kwargs.get('single', True))
        self.msg = str(self.kwargs.get('msg', ''))
        self.msg_shape = bool(self.kwargs.get('shape', True))
        self.msg_input = bool(self.kwargs.get('input', False))

        if self.singleFlag:
            setattr(self, 'forward', getattr(self, 'forward_single'))
    
    def forward_single(self, x):
        print('{} {} {}'.format(
            self.msg,
            'shape: {}'.format(getattr(x, 'shape', '?')) if self.msg_shape else '',
            x if self.msg_input else ''
        ))
        return x