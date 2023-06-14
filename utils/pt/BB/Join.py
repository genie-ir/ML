import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.pt.building_block import BB

class Join(BB): # skip connection
    def start(self):
        key = 'needed' # Notic: Dont change it becuse it used in other places
        needed_list = []
        self.kwargs[key] = self.kwargs.get(key, dict())
        assert isinstance(self.kwargs[key], dict), '`type(self.kwargs[key])={}` | it must be dict | `key={}` | looking this maybe useful: `self.kwargs[key]={}`'.format(type(self.kwargs[key]), key, self.kwargs[key])
        for k, v in self.kwargs[key].items():
            needed_list.append({k: [int(vi) for vi in str(v).split('_')]})
        self.kwargs[key] = needed_list
        setattr(self, key, self.kwargs[key])
        del self.kwargs[key]

        self.kwargs['op'] = str(self.kwargs.get('op', 'cat'))
        self.op_fn = self.kwargs['op']
        del self.kwargs['op']
        setattr(self, 'forward', getattr(self, f'forward_{self.op_fn}'))
    
    def forward_cat(self, *pargs_x):
        return torch.cat(pargs_x, **self.kwargs)
    
    def forward_sum(self, *pargs_x):
        x = pargs_x[0]
        for i in range(1, len(pargs_x)):
            x = x + pargs_x[i]
        return x
    
    def forward_mul(self, *pargs_x):
        x = pargs_x[0]
        for i in range(1, len(pargs_x)):
            x = x * pargs_x[i]
        return x
    