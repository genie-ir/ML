import torch
from torch import nn
from utils.pt.loss import Loss
import torch.nn.functional as F

class VQVAE_Loss(Loss):
    def vqvae_loss(self, x, quantized_x, lambda_e_latent=1):
        e_latent_loss = F.mse_loss(quantized_x.detach(), x)
        q_latent_loss = F.mse_loss(quantized_x, x.detach())
        loss = q_latent_loss + lambda_e_latent * e_latent_loss
        log = {
            'loss': loss.clone().detach().mean(),
        }
        return loss, log