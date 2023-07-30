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
        self.DEBUG = bool(self.kwargs.get('DEBUG', False)) or (os.getenv('DEBUG') == 'True')
        self.__map_id2name = dict()
        self.start()
        if bool(self.kwargs.get('apply_weights_init', True)): # TODO
            self.apply(getattr(self, self.kwargs.get('weights_init_fn_name', 'weights_init')))
    
    def start(self):
        cowsay.cow('NotImplementedError:\nplease define `{}` function for `{}` building block.'.format('start', self.__class__.__name__))
        sys.exit()
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and hasattr(m, 'weight'):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1 and hasattr(m, 'weight') and hasattr(m, 'bias'):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def visualize(self, **kwargs):
        # print('----------->', id(self), kwargs)
        # kwargs['create_node'](id(self))
        # kwargs['create_node'](self.__class__.__name__)
        pass

    def nnParameter(self, **pkwargs):
        t = torch.randn(pkwargs.get('shape', []))
        requires_grad = bool(pkwargs.get('requires_grad', True))
        p = nn.Parameter(t, requires_grad=requires_grad)
        hooks = pkwargs.get('hooks', [])
        if requires_grad:
            self.sethooks(p, hooks=hooks, leaf=True)
        pname = str(pkwargs.get('pkwargs', ''))
        if pname:
            setattr(pkwargs.get('Self', self), pname, p)
        return p
    
    def sethooks(self, p, hooks=[], leaf=False):
        if not isinstance(hooks, (list, tuple)):
            hooks = [hooks]
        self.__map_id2name[id(p)] = id(p)
        if self.DEBUG:
            p.register_hook(lambda grad: print(self.__map_id2name[id(p)], grad))
        for hook in hooks:
            p.register_hook(hook)
        # if self.DEBUG and (not leaf): # even though retain_grad is defined but inaccessable from outside the forward method and also inaccessble inside the forward method becuse at that time backward doesnt happend!
        #     p.retain_grad()


class EBB(BB):
    """Empty BB"""
    
    def start(self):
        pass
    
    def forward(self):
        return