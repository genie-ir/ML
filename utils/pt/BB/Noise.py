import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.pt.building_block import BB

class Noise(BB): 
    def start(self):
        self.low = int(self.kwargs.get('low', -1))
        self.high = int(self.kwargs.get('high', -1))
        self.NoiseType = str(self.kwargs.get('type', 'randn'))
        self.NoiseShape = list(self.kwargs.get('shape', []))
        _NoiseShape = []
        for e in self.NoiseShape:
            if isinstance(e, str):
                if e.startswith('$'):
                    eName = e[1:]
                else:
                    eName = e
                _NoiseShape.append(int(self.kwargs[eName]))
            else:
                _NoiseShape.append(int(e))
        self.NoiseShape = _NoiseShape
        setattr(self, 'forward', getattr(self, f'forward_{self.NoiseType}'))
    
    def forward_randn(self, x):
        return torch.randn([x.shape[0]] + self.NoiseShape, device=x.device)

    def forward_randint(self, x):
        size = [x.shape[0]] + self.NoiseShape
        return torch.randint(self.low, self.high, size=size, device=x.device)