import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.pt.building_block import BB

class Pass(BB): 
    def start(self):
        pass
    
    def forward(self, x):
        return x