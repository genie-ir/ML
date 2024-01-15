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
    
    def normalizer(self, g):
        t = g.clone().detach()
        frexp = torch.tensor(t).frexp()
        m = frexp.mantissa
        e = frexp.exponent

        ln2 = torch.tensor(2.0).log()
        ln10 = torch.tensor(10.0).log()

        n = (e * (ln2 / ln10))
        new_n = n
        # new_n = 0
        new_e = new_n * (ln10 / ln2)
        # new_e = torch.tensor(0)
        x = torch.ldexp(m, e) * (10**(-n))
        (t-x)
    
    def sethook(self, tensor, callback):
        tensor.register_hook(callback)
    
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

    def __safe(self, x, cb, w=1, **kwargs):
        """kar dare hanoz!!"""
        xzq = cb(x.detach(), **kwargs)
        zxq = self.dzq_dz_eq1(xzq, x, w=w)
        return zxq

# Bank of Basic Components
class Activation(nn.Module):
    def __init__(self, T: str = 'tanh', fwd='nsd', r=1, **kwargs):
        """change fwd to None for orginal physical/semantic functionality; #NOTE nsd is default becuse I belive activation function should not been has contribution to changing scale of Derivative, becuse it should not been any samantic purpose just it should be operate as noneLinerity contributer to last function and it shoulde be contribute as extra bounding controller"""
        super().__init__()
        self.r = float(r) # search radius; # NOTE: each activation compute its static parammetters based on r.
        self.Grad = Grad()

        T_lower = T.lower()

        if T_lower == 'tanh': # 0<=D<=1
            T = 'Tanh'
            self.tanh_scale = 1 if self.r == 1 else 'BAYAD MOHASEBE BASHE ALAN BALAD NISTAM :)' # TODO

        if T_lower == 'sig': # 0<=D<=1
            T = 'Sigmoid'
            self.sig_scale = 1 if self.r == 1 else 'BAYAD MOHASEBE BASHE ALAN BALAD NISTAM :)' # TODO

        # if T_lower == '?': # NOTE: define a new activation function!!
        #     T = '?'

        fwd = fwd or 'fwd'
        self.activation = getattr(self, T, getattr(nn, T, None))(**kwargs)
        setattr(self, 'forward', getattr(self, fwd))
        setattr(self, 'variant', getattr(self, f'variant_{T}'))

    def fwd(self, x):
        return self.activation(x)
    
    def nsd(self, x):
        """nsd: None Scale Derivative"""
        self.Grad.sethook(x, lambda grad: print('------------------->', grad.mean().item()))
        y = self.activation(x)
        y_requires_grad = y.requires_grad
        y = self.variant(y).detach()
        y.requires_grad = y_requires_grad
        y = self.Grad.dzq_dz_eq1(y, x)
        return y

    def variant_Tanh(self, y):
        return self.tanh_scale * y
    
    def variant_Sigmoid(self, y):
        return self.sig_scale * y



class Lerner(PYBASE, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__start()

    def __start(self):
        self.Grad = Grad()
        self.Activation = Activation
        # self.m
        # self.pl.device self.device = 'cuda' # dynamic it...
        self.device = 'cuda'

    def List(self, list1D_components):
        return nn.ModuleList(list1D_components)
    
    def component(self, __T, __name: str = None, **kwargs) -> None:
        T = __T
        name = __name
        if isinstance(T, str):
            _component = getattr(self, T, getattr(nn, T, None))(**kwargs)
        else:
            _component = T(**kwargs)
        
        if name != None:
            setattr(self, name, _component)
        
        return _component
    
    # Linear Algebra Component
    def LA(self, **kwargs): # TODO
        """instantiation"""
        return self.__la
    
    def __la(self):
        """forward"""
        return

class T(Lerner):
    def __init__(self, T='conv', us=False, ds=True, b=1, n=100, act='tanh', actparams=None, **kwargs):
        super().__init__(**kwargs)

        self.us = bool(us) # up sampling
        self.ds = bool(ds) # down sampling
        assert (self.ds == True and self.us == False) or (self.ds == False and self.us == True)

        self.b = int(b) # branching factor
        self.n = int(n) # sinusoidal factor

        if T.lower() == 'la':
            T = 'LA'
            pass # TODO
        
        if 'conv' in T.lower():
            inch = self.kwargs.get('inch', None)
            outch = self.kwargs.get('outch', None)

            if self.ds:
                k = self.kwargs.get('k', 3)
                s = self.kwargs.get('s', 2)
                p = self.kwargs.get('p', 1)
            else:
                k = self.kwargs.get('k', 4)
                s = self.kwargs.get('s', 2)
                p = self.kwargs.get('p', 1)

            if T.lower() == 'conv2d' or T.lower() == 'conv': # 2D
                if self.ds:
                    T = 'Conv2d'
                else:
                    T = 'ConvTranspose2d'
            
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
            # self.component(self.Activation, 'afn1', T=self.act, **self.actparams)
            self.component(self.Activation, 'afn1', T='tanh', **self.actparams)
            self.component(self.Activation, 'afn2', T='sig', **self.actparams)
        
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
        return self.activation(self._fwd(x, i))
    
    def _bfwd(self, x):
        l = []
        for b in range(self.b):
            l.append(self.fwd(x, b))
        return self.merge(l)
    
    def forward(self, x):
        y = self.bfwd(x)
        # sinsoide!!!!!!