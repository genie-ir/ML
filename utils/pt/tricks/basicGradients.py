import torch

def preserve_gradients(z, z_q):
    """
        transfer gradients from `z_q` to `z`
        `z_q` and `z` must be the same shape
    """
    z_q = z + (z_q - z).detach()
    return z_q