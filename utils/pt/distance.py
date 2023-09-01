import torch
from einops import rearrange

# def L2S(a, b):
#     """
#         (Square of L2 Norm)
#         a.shape: Nxd
#         b.shape: Mxd
#     """
#     return torch.sum(a**2, dim=1, keepdim=True) \
#             + torch.sum(b**2, dim=1) \
#             -2 * torch.matmul(a, b.t())

def L2S(a, b, argmin=False, topk=False):
    """Square of L2 Norm"""
    a = a.round()
    b = b.round()
    d = torch.sum(a ** 2, dim=1, keepdim=True) + \
        torch.sum(b ** 2, dim=1) - 2 * \
        torch.einsum('bd,dn->bn', a, rearrange(b, 'n d -> d n'))
    
    if argmin:
        return torch.argmin(d, dim=1)
    
    if topk:
        return torch.topk(d, topk, largest=False, dim=1)
    
    return d