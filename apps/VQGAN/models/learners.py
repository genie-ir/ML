# NOTE: use it in repo asli!!!!!!!!!!
import torch
from torch import nn
from apps.VQGAN.models.kernel_py_classes.basic import PYBASE

class Grad(PYBASE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__start()

    def __start(self):
        pass
    
    def sethook(tensor, callback):
        tensor.register_hook(lambda grad: callback(grad))
    
    def dzq_dz_eq1(self, zq, z, w=1):
        """
            # NOTE: if zq has gradient and z hasnt requires_grad then gradient of zq is fucked:)
            transfer gradients from `zq` to `z`  | (zq -> z)
            `zq` and `z` must be the same shape
            (Notic): zq not change in terms of numerically but here we define a drevative path from zq to z such that (dzq/dz = 1)
            Example: 
                zq = dzq_dz_eq1(zq, z)
        """
        return (w * z) + (zq - (w * z)).detach()

    def safe(self, x, cb, w=1, **kwargs):
        """kar dare hanoz!!"""
        xzq = cb(x.detach(), **kwargs)
        zxq = self.dzq_dz_eq1(xzq, x, w=w)
        return zxq

class Lerner(PYBASE, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__start()

    def __start(self):
        self.Grad = Grad()
        # self.m
        # self.pl.device self.device = 'cuda' # dynamic it...
        pass

    def List(self, list1D_components):
        return nn.ModuleList(list1D_components)
    
    def component(self, T, **kwargs):
        if isinstance(T, str):
            return getattr(self, T, getattr(nn, T))(**kwargs)
        return T(**kwargs)

class Activation(Lerner):
    def __init__(self, T='tanh', **kwargs):
        super().__init__(**kwargs)
        
        if T.lower() == 'tanh': # sign of derivative is positive
            T = 'Tanh'
        
        if T.lower() == 'sig': # sign of derivative is positive
            T = 'Sigmoid'
        
        self.T = T
        self.__start()

    def __start(self):
        self.act = self.component(self.T, **self.kwargs)
    
    def forward(self, x):
        y = self.act(x)
        y_requires_grad = y.requires_grad
        y = y.detach()
        y.requires_grad = y_requires_grad
        y = self.Grad.dzq_dz_eq1(y, x)

        return y

class T(Lerner):
    def __init__(self, T='conv', us=False, ds=True, b=1, n=100, act='tanh', actparams=None, **kwargs):
        super().__init__(**kwargs)

        self.us = bool(us) # up sampling
        self.ds = bool(ds) # down sampling
        assert (self.ds == True and self.us == False) or (self.ds == False and self.us == True)

        self.b = int(b) # branching factor
        self.n = int(n) # sinusoidal factor

        if 'conv' in T.lower():
            k = None
            s = None
            p = None

            if T.lower() == 'c2d' or T.lower() == 'conv':
                if self.ds:
                    T = 'Conv2d'
                    k = self.kwargs.get('k', 3)
                    s = self.kwargs.get('s', 2)
                    p = self.kwargs.get('p', 1)
                else:
                    T = 'ConvTranspose2d'
                    k = self.kwargs.get('k', 4)
                    s = self.kwargs.get('s', 2)
                    p = self.kwargs.get('p', 1)
            
            inch = self.kwargs.get('inch', None)
            outch = self.kwargs.get('outch', None)

            if inch != None:
                self.kwargs['in_channels'] = int(inch)
            
            if outch != None:
                self.kwargs['out_channels'] = int(outch)
            
            if k != None:
                self.kwargs['kernel_size'] = int(k)
            
            if s != None:
                self.kwargs['stride'] = int(s)
            
            if p != None:
                self.kwargs['padding'] = int(p)

        self.T = T

        self.act = act
        self.actparams = actparams if actparams != None else dict()

        self.__start()

    def __start(self):
        fwd = '_fwd'
        if self.act != None:
            fwd = '_fwd_act'
            self.act = Activation(self.act, **self.actparams)
        
        setattr(self, 'fwd', getattr(self, fwd))
        
        self.l = self.List([self.component(self.T, **self.kwargs) for b in range(self.b)])

        bfwd = 'fwd'
        if self.b > 1:
            bfwd = '_bfwd'
            self.merge_params()
        
        setattr(self, 'bfwd', getattr(self, bfwd))


    def merge(self, l):
        """overwitable method"""
        return self._merge(torch.cat(l, dim=1))
    
    def merge_params(self):
        """overwitable method"""
        self._merge = self.component(
            'Conv2d', 
            in_channels = self.b * self.kwargs['out_channels'],
            out_channels = self.kwargs['out_channels'],
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _fwd(self, x, i=0):
        y = self.l[i](x)
        # TODO control derivative
        return y
    
    def _fwd_act(self, x, i=0):
        return self.act(self._fwd(x, i))
    
    def _bfwd(self, x):
        l = []
        for b in range(self.b):
            l.append(self.fwd(x, b))
        return self.merge(l)
    
    def forward(self, x):
        y = self.bfwd(x)
        # sinsoide!!!!!!