import torch
from einops import rearrange

def L2S(a, b):
    """
        (Square of L2 Norm)
        a.shape: Nxd
        b.shape: Mxd
    """
    return torch.sum(a**2, dim=1, keepdim=True) \
            + torch.sum(b**2, dim=1) \
            -2 * torch.matmul(a, b.t())

def L2S_VQ(a, b, argminFlag=False):
    d = torch.sum(a ** 2, dim=1, keepdim=True) + \
        torch.sum(b ** 2, dim=1) - 2 * \
        torch.einsum('bd,dn->bn', a, rearrange(b, 'n d -> d n'))
    
    if argminFlag:
        return torch.argmin(d, dim=1)
    return d