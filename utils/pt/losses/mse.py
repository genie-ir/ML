import torch
from torch import nn
from utils.pt.loss import Loss

class MSE_Loss(Loss):
    def __init__(self, **kwargs):
        kwargs['prefix'] = kwargs.get('prefix', 'mse')
        super().__init__(**kwargs)

    def start(self):
        self.mse = nn.MSELoss()

    def mse_loss(self, y, t):
        loss = self.mse(y, t)
        log = {
            'loss': loss.clone().detach().mean(),
        }
        return loss, log