import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.pt.building_block import BB

class DilatedCNN(BB): 
    def start(self):
        self.convlayers = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 9, stride = 1, padding = 0, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size = 3, stride = 1, padding= 0, dilation = 2),
            nn.ReLU(),
        )    
        self.fclayers = nn.Sequential(
            nn.Linear(2304,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )  
    
    def forward(self,x):
        x = self.convlayers(x)
        x = x.view(-1,2304)
        x = self.fclayers(x)
        return x