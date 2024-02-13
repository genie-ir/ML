# NOTE: use it in repo asli!!!!!!!!!!
import torch
from torch import nn
import torch.nn.functional as F
from apps.VQGAN.models.kernel_py_classes.basic import PYBASE


ρ = 1.0             # search radius
α = 1e2             # cosine coefficient scaler
β = 2               # branching factor of decision tree (for BST is 2) 
ζ = 10              # depth of decision tree
κ = 3               # kernel size (the width of cosines)
λgs = 3             # gradient scaler loss coefficient
λmc = 2.0           # misclassification loss coefficient


def NG(t):
    """gamma scaler | `ts` is the scale of tensor `t`, that keeps the order"""
    m, e = t.frexp()
    ga = e - 1
    gb = 1 + (ga / (1 + ga.abs().max())) # ∈ (0, 2)
    ts = torch.ldexp(torch.sign(m), gb) # {-1, 0, +1} * (1, 4) = (-4, -1) union {0} union (1, 4)
    return m, ts

class Grad(PYBASE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__start()

    def __start(self):
        pass
    
    def sethook(self, tensor, callback=None):
        if tensor.requires_grad:
            if callback == None:
                tensor.register_hook(self.GS)
            else:
                tensor.register_hook(callback)
    
    def GS(self, g):
        """gradient scaler | keeps the order"""
        gm, gs = NG(g)
        return λgs * gs # (-4λgs, -λgs) union {0} union (λgs, 4λgs)

    def dzq_dz_eq1(self, zq, z, w=1, **kwargs):
        """
            # NOTE: if zq has gradient and z hasnt requires_grad then gradient of zq is fucked:)
            transfer gradients from `zq` to `z`  | (zq -> z)
            `zq` and `z` must be the same shape
            (Notic): zq not change in terms of numerically but here we define a drevative path from zq to z such that (dzq/dz = 1)
            Example: 
                zq = dzq_dz_eq1(zq, z)
        """
        return (w * z) + (zq - (w * z)).detach()

class Lerner(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.__start()

    def __start(self):
        self.Grad = Grad()
        self.device = 'cuda'
        
        self.ρ = float(self.kwargs.get('ρ', ρ))
        self.α = float(self.kwargs.get('α', α))
        self.β = int(self.kwargs.get('β', β))
        self.ζ = int(self.kwargs.get('ζ', ζ))
        self.κ = int(self.kwargs.get('κ', κ))
        self.λgs = float(self.kwargs.get('λgs', λgs))
        self.λmc = float(self.kwargs.get('λmc', λmc))

        assert self.β >= 2
        assert self.ζ >= 1
        assert (self.κ >= 1) and (self.κ % 2 != 0)

        # changes inside a coordinate system # NOTE: static params
        self.κs = int(1)
        self.κp = int(self.κ // 2)

    def List(self, layers):
        return nn.ModuleList(layers) # layers is pythonic list

class BB(Lerner):
    def __init__(self, T: str = 'inc', **kwargs):
        super().__init__(**kwargs)
        self.T = str(T).upper()
        self.__start()

    def __start(self):
        getattr(self, f'{self.T}_INIT')()
        setattr(self, 'forward', getattr(self, f'{self.T}_FORWARD'))

    def CONV_INIT(self):
        """BxCHxHxW"""
        self.inch = int(self.kwargs['inch'])
        self.outch = int(self.kwargs['outch'])

        # OPTIONAL
        self.k = int(self.kwargs.get('k', self.κ))
        self.s = int(self.kwargs.get('s', self.κs))
        self.p = int(self.kwargs.get('p', self.κp))
        
        self.conv = nn.Conv2d(self.inch, self.outch, self.k, self.s, self.p, **self.kwargs.get('conv', dict(bias=False)))
        self.bn = nn.BatchNorm2d(self.outch, **self.kwargs.get('bn', dict(eps=1e-3)))
        self.lrelu = nn.LeakyReLU(**self.kwargs.get('lrelu', dict(negative_slope=1e-1)))

    def CONV_FORWARD(self, x):
        y = self.conv(x)
        self.Grad.sethook(y)
        yn = self.bn(y)
        self.Grad.sethook(yn)
        return self.lrelu(yn)
    
    def INC_INIT(self):
        """BxCHxHxW"""
        self.inch = int(self.kwargs['inch'])
        self.outch = int(self.kwargs['outch'])

        # OPTIONAL
        self.k = int(self.kwargs.get('k', self.κ))
        self.s = int(self.kwargs.get('s', self.κs))
        self.p = int(self.kwargs.get('p', self.κp))
        
        self.kout = int(self.kwargs.get('kout', 1))
        self.sout = int(self.kwargs.get('sout', 1))
        self.pout = int(self.kwargs.get('pout', 0))

        self.outch_branch3x3 = int(self.kwargs.get('branch3x3', 64))
        self.outch_branch3x3dbl_1 = int(self.kwargs.get('branch3x3dbl_1', 32))
        self.outch_branch3x3dbl_2 = int(self.kwargs.get('branch3x3dbl_2', 64))

        self.branch3x3 = BB(T='conv', outch=self.outch_branch3x3, inch=self.inch, k=self.k, s=self.s, p=self.p)
        
        self.branch3x3dbl_1 = BB(T='conv', outch=self.outch_branch3x3dbl_1, inch=self.inch, k=1, s=1, p=0)
        self.branch3x3dbl_2 = BB(T='conv', outch=self.outch_branch3x3dbl_2, inch=self.outch_branch3x3dbl_1, k=self.k, s=self.s, p=self.p)
        self.branch3x3dbl_3 = BB(T='conv', outch=self.outch_branch3x3dbl_2, inch=self.outch_branch3x3dbl_2, k=self.k, s=self.s, p=self.p)
        
        self.MaxPool2d = nn.MaxPool2d(self.k, self.s, self.p)

        self.inc = BB(T='conv',
            inch=(self.outch_branch3x3 + self.outch_branch3x3dbl_2 + self.inch), 
            outch=self.outch,  
            k=self.kout, s=self.sout, p=self.pout)

    def INC_FORWARD(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.MaxPool2d(x)

        z = torch.cat([branch3x3, branch3x3dbl, branch_pool], dim=1)
        y = self.inc(z)
        return y

class Loss(Lerner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__start()

    def __start(self):
        pass

    def bit(self, logit, groundtruth: bool = True, λacc=1, tag: str = ''):
        """
            ***This is bit loss***
            logit is came from Tanh
            0<=λacc<=1 ; it gonna be a controller and provide by master!
            groundtruth is single bool:       True===1===positive       False===0===negative
        """
        prediction = logit # NOTE: binary_decision at end of the BSTC has been done.
        pred = prediction.clone().detach()

        # if prediction.requires_grad:
        #     print('prediction', prediction)
        #     self.Grad.sethook(prediction, lambda grad: print('prediction.grad', grad))
        N = pred.numel()
        unk = (pred == 0.5).sum() / N
        neg = (pred == 0.0).sum() / N
        pos = (pred == 1.0).sum() / N
        print('prediction', unk, neg, pos)

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
        loss = (λacc * loss).clone().detach()
        loss = self.Grad.dzq_dz_eq1(loss, prediction, w=loss.detach())
        self.Grad.sethook(loss, lambda grad: torch.ones_like(grad))
        
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
 
class System(Lerner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__start()

    def __start(self):
        self.F_INIT()

    def F_INIT(self):
        """changes inside a coordinate system ; parameterized by θs that cames from Bayesian framework""" # TODO
        self.d0 = self.kwargs['d0']
        self.Nshape = self.kwargs['Nshape'] # (√N, √N)

        self.ew = self.List([BB(inch=self.d0, outch=1, **self.kwargs) for bfidx in range(self.kwargs['β'])])
        self.eb = self.List([BB(inch=self.d0, outch=1, **self.kwargs) for bfidx in range(self.kwargs['β'])])
        self.classifier = BB(inch=self.d0, outch=self.β, kout=self.κ, sout=self.κs, pout=0, **self.kwargs)

        self.Unfold = nn.Unfold(kernel_size=self.κ, dilation=1, padding=self.κp, stride=self.κs)
        self.Fold = nn.Fold(output_size=self.Nshape, kernel_size=self.κ, dilation=1, padding=self.κp, stride=self.κs)
        
    def gradadder(self, _b, _bidx, _csel, _cobj, g):
        h = torch.zeros((_b, self.β), device=self.device)
        h[_bidx][_csel] = g.abs().sum().detach()
        _cobj.add_(h)
        return g
    
    def forward(self, x):
        b = x.shape[0]

        fnτ = self.Unfold(x)
        fτ = fnτ.permute(0, 2, 1).sum(dim=1).view(b, self.d0, self.κ, self.κ)

        c = self.classifier(fτ).view(b, self.β)
        self.Grad.sethook(c, lambda grad: _cgrad)
        _cgrad = torch.zeros((b, self.β), device=self.device)
        cselect = c.argmax(-1) # (b,)

        planes_ew = []
        planes_eb = []
        x_detach = x.detach()

        csum = c.sum() # virtual derivative path to `c` tensor

        for bidx in range(b):
            csel = cselect[bidx].item()
            pew = self.ew[csel](x_detach[bidx:bidx+1]) + 0 * csum
            peb = self.eb[csel](x_detach[bidx:bidx+1]) + 0 * csum

            self.Grad.sethook(pew, lambda grad, _b=b, _bidx=bidx, _csel=csel, _cobj=_cgrad: self.gradadder(_b, _bidx, _csel, _cobj, grad))
            self.Grad.sethook(peb, lambda grad, _b=b, _bidx=bidx, _csel=csel, _cobj=_cgrad: self.gradadder(_b, _bidx, _csel, _cobj, grad))

            planes_ew.append(pew)
            planes_eb.append(peb)

        planes_ew = torch.cat(planes_ew, dim=0)
        planes_eb = torch.cat(planes_eb, dim=0)

        









        


class DT(Lerner):
    """Decision Tree"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__start()

    def __start(self):
        self.dt = self.List([System(**self.kwargs) for z in range(self.ζ)])

    # def binary_decision(self, logit):
    #     """logit is came from Tanh and output will be a binary decision === 0,1,0.5"""
    #     logit_cd = logit.clone().detach()
    #     decision = 0.5 * torch.ones_like(logit_cd, requires_grad=False) # every where has init value 0.5 === No Idea
    #     decision.masked_fill_((logit_cd - 0.5).abs() <= self.e, 1.0) # +1/2 -> True  === 1.0
    #     decision.masked_fill_((logit_cd + 0.5).abs() <= self.e, 0.0) # -1/2 -> False === 0.0
    #     decision = decision.clone().detach()
    #     # decision.requires_grad = True # DELETE this line shoude be deleted I comment it out, becuse you know that, I understand what i done! :)
    #     decision = self.Grad.dzq_dz_eq1(decision, logit)
    #     return decision
    
    def forward(self, f0):
        fip1 = f0
        for z in range(self.ζ):
            fi, fip1, ci = self.dt[z](fip1)
        return fi, fip1, ci

class Node(Lerner): # TODO Node -> dim -> (chxhxw)  s.t. h=w
    def __init__(self, ch: int, hw: int, rc: bool =True, graph=None, **kwargs):
        super().__init__(**kwargs)
        self.ch = int(ch)
        self.hw = int(hw)
        self.rc = bool(rc)
        self.graph = graph
        self.__start()

    def __start(self):
        self.N = int(self.hw * self.hw)
        self.d0 = int(self.ch) # effective dimension
        self.d = int(self.d0 * self.κ * self.κ) # original dimension
        self.dshape = (self.d0, self.κ, self.κ)
        self.Nshape = (self.hw, self.hw)
        self.decision_tree = DT(
            N=self.N,
            d0=self.d0,
            d=self.d,
            dshape=self.dshape,
            Nshape=self.Nshape,
            **self.kwargs
        )
        self.fwd = str('regressor' if self.rc else 'classifier')
        
        setattr(self, 'forward', getattr(self, self.fwd))
        self.G_INIT()

    def G_INIT(self): # TODO
        """changes between coordinate devices"""
        pass
        # self.G = nn.Conv2d(self.kwargs['inch'], self.kwargs['outch'], self.kwargs['k'], self.kwargs['s'], self.kwargs['p'])
    
    def regressor(self, x):
        fi, fip1, ci = self.decision_tree(self.G(x))
        return fi
    
    def classifier(self, x):
        fi, fip1, ci = self.decision_tree(self.G(x))
        return ci























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
        # kwargs['ζ'] = 10
        self.nodes = nn.Sequential(*[
            # Node(inch=3, fwd='test',  outch=8,   k=3, s=2, p=1, **kwargs), # 8x128**2
            # Node(inch=8,  outch=16,  k=3, s=2, p=1, **kwargs), # 16x64**2
            # Node(inch=16, outch=32,  k=3, s=2, p=1, **kwargs), # 32x32**2
            # Node(inch=32, outch=64,  k=3, s=2, p=1, **kwargs), # 64x16**2
            # Node(inch=64, outch=64,  k=3, s=2, p=1, **kwargs), # 64x8**2
            # Node(regressor=False, inch=64, outch=4, k=3, s=2, p=1, **kwargs), # 4x4**2 # BSTC => one bit estimator
        ])
    
    def forward(self, x, groundtruth, λacc, tag):
        return self.Loss.bit(self.nodes(x), groundtruth=groundtruth, λacc=λacc, tag=tag)

class FUM_H_Graph(Lerner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # kwargs['ζ'] = 10
        # self.a = Node(inch=256, outch=256, k=3, s=1, p=1, **kwargs)
        # self.b = Node(inch=256, outch=256, k=3, s=1, p=1, **kwargs) 
        # self.c = Node(inch=256, outch=256, k=3, s=1, p=1, **kwargs)
    
    def forward(self, x1, x2):
        y1 = self.a(x1)
        y2 = self.b(x2)
        ymean = (y1 + y2) / 2
        if ymean.requires_grad:
            self.Grad.sethook(ymean, lambda grad: print('ymean.grad', grad.mean().item()))
        y = self.c(ymean)
        return y
    

