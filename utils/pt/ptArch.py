import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from fightingcv_attention.attention.CBAM import SpatialAttention
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Normalizer
class Simple_LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(Simple_LayerNorm, self).__init__()
        self._gama = nn.Parameter(torch.ones(features))
        self._beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # mean = x.mean(-1, keepdim=True)
        # std = x.std(-1, keepdim=True)
        # return self._gama * (x - mean) / (std + self.eps) + self._beta
        return x / 255.0    


# Residual "Connection"
class SimpleResidualConnection(nn.Module):
    """
        A residual connection followed by a layer norm.
        Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout, LayerNorm=nn.LayerNorm):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(size) # size is must be embedding_dim for TS data.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class ResidualConnection(nn.Module):
    """
        A residual connection followed by a layer norm.
        Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout=0, LayerNorm=Simple_LayerNorm):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(sublayer(x)))


# Masking
class Mask():
    def lookahead_mask(size, rank=2, diagonal=1):
        map_shape = tuple((1 for r in range(2, rank))) + (size, size)
        return torch.triu(torch.ones(map_shape), diagonal=diagonal).bool()

    def generate_square_subsequent_mask(sz: int):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


# Functional
def get_activation_fn(activation: str):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu

    raise RuntimeError('activation {} is undefined.'.format(activation))


def test():
    x = torch.randn(2, 3, 128,128)
    satn = SpatialAttention(kernel_size=5)
    y = satn(x)
    print(y.shape)
test()