import torch
from torch import nn
from utils.pt.loss import Loss

class CGAN_Loss(Loss):
    def __init__(self, **kwargs):
        kwargs['prefix'] = kwargs.get('prefix', 'mse')
        super().__init__(**kwargs)

    def start(self):
        self.mse = nn.MSELoss()
    
    def mse_generator_loss(self, d_fake, Real):
        loss = self.mse(d_fake, Real * torch.ones_like(d_fake, device=d_fake.device))
        log = {
            'gloss': loss.clone().detach().mean(),
        }
        return loss, log
        
    def mse_discriminator_loss(self, d_real, d_fake, Real, Fake):
        loss_real = self.mse(d_real, Real * torch.ones_like(d_real, device=d_real.device))
        loss_fake = self.mse(d_fake, Fake * torch.ones_like(d_fake, device=d_fake.device))
        loss = loss_real + loss_fake
        log = {
            'dloss': loss.clone().detach().mean(),
        }
        return loss, log