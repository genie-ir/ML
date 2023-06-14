import torch

def L2S(a, b):
    """
        (Square of L2 Norm)
        a.shape: Nxd
        b.shape: Mxd
    """
    return torch.sum(a**2, dim=1, keepdim=True) \
            + torch.sum(b**2, dim=1) \
            -2 * torch.matmul(a, b.t())