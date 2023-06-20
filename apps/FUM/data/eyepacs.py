import yaml
import torch
import numpy as np
from os.path import join, exists
from utils.pt.datasets.D import D_Base
from os import system, getenv, makedirs
from utils.analysis.fourier.basic import x2fr, fr2x_torch
from data.config.eyepacs.D import eyepacsTrain as eyepacsTrainBase, eyepacsValidation as eyepacsValidationBase

# N0, N1, B0 = 1, 1, 0
# N0, N1, B0 = 600, 512, 50
# M0, M1 = 100, 50

# def normalizing(signal: np.ndarray):
#     signal = x2fr(np.reshape(signal, (1, -1))).squeeze() / N1
#     return ((((signal + B0) / N0) * M0) - M1).astype(np.float32)

# def denormalizing(signal: torch.tensor):
#     return fr2x_torch((((signal + M1) / M0) * N0 - B0) * N1) 

class D(D_Base):
    def fetch(self, signal_path):
        return {
            'latentcode': np.load(signal_path)
        }

class eyepacsTrain(eyepacsTrainBase): 
    def preparation(self, **kwargs):
        self.D = D
    
class eyepacsValidation(eyepacsValidationBase):
    def preparation(self, **kwargs):
        self.D = D