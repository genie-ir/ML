import torch
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift

def x2xe(x: np.ndarray):
    """
        x is a `time space` `real` signal.
        [function output] is a `time space` `real` `even` signal. (make up on `x`)
    """
    assert x.ndim == 2, '`x.ndim={}` | It must be `2` if you want support more dims please do code for it'.format(x.ndim)
    B, N = x.shape[0], x.shape[1]
    return np.concatenate([x[:, ::-1], np.zeros((B, 1)), x[:, :-1]], 1)

def x2fr(x: np.ndarray):
    """
        x  is a `time space` `real` signal.
        xe is a `time space` `real` `even` signal. (make up on `x`)
        fr is a 
    """
    B, N = x.shape[0], x.shape[1]
    xe = x2xe(x)
    return fftshift(fft(ifftshift(xe))).real[:, :N+1] # imag part is zero becuse `xe` is `real` and `even` signal

def fr2x(fr: np.ndarray):
    """
        fr is a `effective values` of `fft` taken from `even version` of `real time space signal x`
        [function output] is a `time space` `real` signal.
    """
    assert fr.ndim == 2, '`fr.ndim={}` | It must be `2` if you want support more dims please do code for it'.format(fr.ndim)
    N = fr.shape[1] - 1
    z = np.concatenate([fr, fr[:, ::-1][:, 1:-1]], 1)
    xt = ifftshift(ifft(fftshift(z)))
    return xt[:, :N][:, ::-1].real

def fr2x_torch(fr: torch.tensor):
    """
        fr is a `effective values` of `fft` taken from `even version` of `real time space signal x`
        [function output] is a `time space` `real` signal.
    """
    N = fr.shape[1] - 1
    z = torch.cat([fr, fr.fliplr()[:, 1:-1]], 1)
    xt = torch.fft.ifftshift(torch.fft.ifft(torch.fft.fftshift(z)))
    return xt[:, :N][:, ::-1].real

# Example:
# FR = x2fr(x)
# xp = fr2x(FR)
# print(xp, xp.shape, FR, FR.shape)


def test():
    pass