# NOTE: use it in repo asli!!!!!!!!!!
import torch
from torch import nn
from apps.VQGAN.models.kernel_py_classes.basic import PYBASE


r = 1.0     # search radius
e = 0.15    # fault tolerance
β = 4       # bits for regression; 4 supports precision of 0.93 for regression between zero and one
ζ = 3       # depth of BST
λgs = 1     # GSL scaler
λsl = 0.5   # GSL scaler
λmc = 2.0   # misclassification loss coefficient
λlc = 4.0   # lazy classification loss coefficient


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

class BaseLerner(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.__start()

    def __start(self):
        self.Grad = Grad()
        self.device = 'cuda'
        
        self.r = float(self.kwargs.get('r', r))
        self.e = float(self.kwargs.get('e', e))
        self.β = int(self.kwargs.get('β', β))
        self.ζ = int(self.kwargs.get('ζ', ζ))
        self.λgs = float(self.kwargs.get('λgs', λgs))
        self.λsl = float(self.kwargs.get('λsl', λsl))
        self.λmc = float(self.kwargs.get('λmc', λmc))
        self.λlc = float(self.kwargs.get('λlc', λlc))

    def binary_decision(self, logit):
        """logit is came from Tanh and output will be a binary decision === 0,1,0.5"""
        logit_cd = logit.clone().detach()
        decision = 0.5 * torch.ones_like(logit_cd, requires_grad=False) # every where has init value 0.5 === No Idea
        decision.masked_fill_((logit_cd - 0.5).abs() <= self.e, 1.0) # +1/2 -> True  === 1.0
        decision.masked_fill_((logit_cd + 0.5).abs() <= self.e, 0.0) # -1/2 -> False === 0.0
        decision = decision.clone().detach()
        # decision.requires_grad = True # DELETE this line shoude be deleted I comment it out, becuse you know that, I understand what i done! :)
        decision = self.Grad.dzq_dz_eq1(decision, logit)
        return decision

class Loss(BaseLerner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__start()

    def __start(self):
        pass

    def binary(self, logit, groundtruth: bool = True, make_decision=True, tag: str = ''):
        """
            logit is came from Tanh
            groundtruth is single bool:       True===1===positive       False===0===negative
        """
        if make_decision:
            prediction = self.binary_decision(logit)
        else:
            prediction = logit
        pred = prediction.clone().detach()
        loss = self.λlc * torch.ones_like(pred)
        
        TP, TN, FP, FN = 0, 0, 0, 0
        
        if groundtruth == True: # groundtruth is positive
            TP_Mask = pred == 1.0
            FN_Mask = pred == 0.0
            loss.masked_fill_(TP_Mask, 0.0)
            loss.masked_fill_(FN_Mask, self.λmc)
            TP = TP + TP_Mask.sum()
            FN = FN + FN_Mask.sum()
        else: # groundtruth is negative
            TN_Mask = pred == 0.0
            FP_Mask = pred == 1.0
            loss.masked_fill_(TN_Mask, 0.0)
            loss.masked_fill_(FP_Mask, self.λmc)
            TN = TN + TN_Mask.sum()
            FP = FP + FP_Mask.sum()
        loss = loss.clone().detach()
        loss = self.Grad.dzq_dz_eq1(loss, prediction, w=loss.detach())
        
        tag = tag.upper()
        log = {
            "{}/TP:reduction_ignore".format(tag): TP,
            "{}/TN:reduction_ignore".format(tag): TN,
            "{}/FP:reduction_ignore".format(tag): FP,
            "{}/FN:reduction_ignore".format(tag): FN,
            "{}/ACC:reduction_accuracy".format(tag): None,
            "{}/LOSS".format(tag): loss.clone().detach().mean().item(),
        }
        
        return loss, log

class Lerner(BaseLerner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__start()

    def __start(self):
        self.Loss = Loss()

    def List(self, layers):
        return nn.ModuleList(layers) # layers is pythonic list
    
    def Seq(self, n, layers):
        seq = []
        for i in range(n):
            for layer in layers:
                seq.append(layer)
        return nn.Sequential(*seq)

class System(Lerner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__start()

    def __start(self):
        self.F_INIT()
        setattr(self, 'forward', self.F)

    def F_INIT(self):
        # parameterized by θs
        self.f = nn.Conv2d(self.kwargs['outch'], self.kwargs['outch'], 3, 1, 1)
    
    def F(self, bipolar):
        """
            changes inside a coordinate system
            Bipolar current enters and the output is none-band
        """
        return self.f(bipolar)


class Activation(Lerner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.s = float(1.0) # TODO compute it based on r
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        """None-band current enters and the output is bipolar"""
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
        g_new = self.λgs * torch.ldexp(μ, 1+γ_new)

        return (g_new.abs() * (1 + self.λsl * self.S(x_np))) * g_sign
    
class BST(Lerner):
    """classifire: # NOTE   0===False     1===True     0.5===Unknown"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__start()

    def __start(self):
        self.bst = self.Seq(self.ζ, [
            System(**self.kwargs),
            Activation(**self.kwargs)
        ])
    
    def forward(self, bipolar):
        """bipolar current enters and the output is bipolar"""
        return self.bst(bipolar)

class BST_Regressor(Lerner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__start()

    def __start(self):
        self.BST = self.List([BST(**self.kwargs) for b in range(self.β)])

    def forward(self, bipolar):
        """single bipolar current signal enters and the output is single bipolar signal"""
        μ = torch.zeros_like(bipolar, requires_grad=False, dtype=torch.float32)
        for b in range(self.β):
            bst_b = self.binary_decision(self.BST[b](bipolar))
            bst_B = (bst_b.detach() * (2 ** (-(b+1)))).detach() # 0:ignores the bit position # 1:Keeps the bit position # 0.5: keeps the half bit posotion === here this is a good feature for regression
            bst_B = self.Grad.dzq_dz_eq1(bst_B, bst_b)
            μ = μ + bst_B
        μ_bipolar = μ.detach()
        μ_bipolar = (μ_bipolar * 2 - 1).detach()
        μ_bipolar = self.Grad.dzq_dz_eq1(μ_bipolar, μ)
        
        return μ_bipolar

class Node(Lerner): # TODO Node -> dim -> (chxhxw)  s.t. h=w
    def __init__(self, regressor=True, graph=None, **kwargs):
        super().__init__(**kwargs)
        self.graph = graph
        self.regressor = bool(regressor)
        self.__start()

    def __start(self):
        if self.regressor:
            self.bst = BST_Regressor(**self.kwargs)
        else:
            self.bst = BST(**self.kwargs)
        self.G_INIT()

    def G_INIT(self): # TODO
        self.g = nn.Conv2d(self.kwargs['inch'], self.kwargs['outch'], self.kwargs['k'], self.kwargs['s'], self.kwargs['p'])
    
    def G(self, bipolar):
        """changes between coordinate devices"""
        return self.g(bipolar)
    
    def forward(self, bipolar):
        return self.bst(self.G(bipolar))

class Graph(Lerner): # TODO
    def __init__(self, nodes=1, leafs=0, Node=Node, **kwargs):
        super().__init__(**kwargs)
        self._Node = Node
        self._nodes = int(nodes)
        self._leafs = int(leafs)
        self.__start()

    def __start(self):
        self.nodes = self.List(
            [self._Node(graph=self) for n in range(self._nodes)] +
            [self._Node(graph=self, regressor=False) for n in range(self._leafs)]
        )
        self.edges = None # TODO PGM
    
    def forward(self, *bipolars): # TODO
        pass

class FUM_Disc_Graph(Lerner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nodes = nn.Sequential(*[
            Node(regressor=False, inch=3,  outch=8,   k=3, s=2, p=1), # 8x128
            Node(regressor=False, inch=8,  outch=16,  k=3, s=2, p=1), # 16x64
            Node(regressor=False, inch=16, outch=32,  k=3, s=2, p=1), # 32x32
            Node(regressor=False, inch=32, outch=64,  k=3, s=2, p=1), # 64x16
            Node(regressor=False, inch=64, outch=128, k=3, s=2, p=1), # 128x8
        ])
    
    def forward(self, x, groundtruth, tag):
        y = self.nodes(x)
        return self.Loss.binary(y, groundtruth=groundtruth, tag=tag)

class FUM_H_Graph(Lerner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = Node(inch=256, outch=256, k=3, s=1, p=1)
        self.b = Node(inch=256, outch=256, k=3, s=1, p=1) 
        self.c = Node(inch=512, outch=256, k=3, s=1, p=1, ζ=5)
    
    def forward(self, x1, x2):
        y1 = self.a(x1)
        y2 = self.b(x2)
        y = self.c(torch.cat([y1, y2], dim=1))
        return y
    



    

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