import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.pt.losses.mse import MSE_Loss
from utils.pt.losses.cgan import CGAN_Loss

class Loss(MSE_Loss):
    pass

class CGANLoss(CGAN_Loss):
    pass