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
    def __init__(self, r=1.0, e=0.15, λ=0.5, α=10, **kwargs):
        super().__init__()
        self.α = float(α) # scaler for loss function GSL
        self.λ = float(λ) # scaler for loss function GSL
        self.e = float(e) # fault tolerance
        self.r = float(r) # search radius; # NOTE: each activation compute its static parammetters based on r.
        self.s = float(1.0)
        self.Grad = Grad()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x_cd = x.clone().detach() # non_preemptive; hasnt share memory and hasnt derevative path
        y = self.tanh(x)
        y = (self.s * y).detach()
        y = self.Grad.dzq_dz_eq1(y, x)
        if y.requires_grad:
            self.Grad.sethook(y, lambda grad: self.GSL(grad.clone().detach(), x_cd))
        return y

    def S(self, x_np):
        """Tanh satisfaction loss function"""
        x_np2 = 2 * x_np.abs()
        return torch.min(torch.max((-x_np2+1), ((x_np2/5) - (1/5))), x_np**0)
    
    def GSL(self, g, x_np):
        """gradient scaler loss function by Tanh properties"""
        g_sign = g.sign().clone().detach()
        
        μ, β = g.frexp()
        γ = β - 1
        γ_new = γ / (γ.abs().max() + 1) # γ_new is in (-1, 1)
        g_new = self.α * torch.ldexp(μ, 1+γ_new)

        return (g_new.abs() * (1 + self.λ * self.S(x_np))) * g_sign
    
    def binary_decision(self, logit):
        """logit is came from Tanh"""
        logit_cd = logit.clone().detach()
        decision = 0.5 * torch.ones_like(logit_cd, requires_grad=False) # every where has init value 0.5 === No Idea
        decision.masked_fill_((logit_cd - 0.5).abs() <= self.e, 1.0) # +1/2 -> True  === 1.0
        decision.masked_fill_((logit_cd + 0.5).abs() <= self.e, 0.0) # -1/2 -> False === 0.0
        decision = decision.clone().detach()
        # decision.requires_grad = True # DELETE this line shoude be deleted I comment it out, becuse you know that, I understand what i done! :)
        decision = self.Grad.dzq_dz_eq1(decision, logit)
        return decision

    def decimal_decision(self, *logits):
        """logit(s) is came from Tanh(s)"""
        pass
    
    def binary_loss(self, logit, groundtruth):
        """logit is came from Tanh"""
        prediction = self.binary_decision(logit)







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