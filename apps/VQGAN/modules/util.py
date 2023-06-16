import torch
from torch import nn
from utils.pt.BB.Norm.ActNorm import ActNorm


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class Labelator(AbstractEncoder):
    """Net2Net Interface for Class-Conditional Model"""
    def __init__(self, n_classes, quantize_interface=True):
        super().__init__()
        self.n_classes = n_classes
        self.quantize_interface = quantize_interface

    def encode(self, c):
        # print('EEEEEEEEEEEEEEE', c, c.shape, c.dtype) # EEEEEEEEEEEEEEE tensor([1, 0], device='cuda:0') torch.Size([2]) torch.int64
        c = c[:,None]
        # print('EEEEEEEEEEEEEEE 2', c, c.shape, c.dtype) #EEEEEEEEEEEEEEE 2 tensor([[1],[0]], device='cuda:0') torch.Size([2, 1]) torch.int64
        if self.quantize_interface:
            return c, None, [None, None, c.long()]
        return c


class SOSProvider(AbstractEncoder):
    # for unconditional training
    def __init__(self, sos_token, quantize_interface=True):
        super().__init__()
        self.sos_token = sos_token
        self.quantize_interface = quantize_interface

    def encode(self, x):
        # get batch size from data and replicate sos_token
        c = torch.ones(x.shape[0], 1)*self.sos_token
        c = c.long().to(x.device)
        if self.quantize_interface:
            return c, None, [None, None, c]
        return c
