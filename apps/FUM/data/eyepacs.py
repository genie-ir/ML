import yaml
import numpy as np
from os.path import join, exists
from utils.pt.datasets.D import D_Base
from os import system, getenv, makedirs
from utils.analysis.fourier.basic import x2fr, fr2x
from data.config.eyepacs.D import eyepacsTrain as eyepacsTrainBase, eyepacsValidation as eyepacsValidationBase

def normalizing(signal):
    signal = x2fr(np.reshape(signal, (1, -1))).squeeze() / 512
    return (signal +50) / 600

def denormalizing(signal):
    return fr2x((signal * 600 - 50) * 512).round().astype(np.int32)

class D(D_Base):
    def fetch(self, signal_path):
        return {
            'latentcode': normalizing(np.load(signal_path))
        }

class eyepacsTrain(eyepacsTrainBase): 
    def preparation(self, **kwargs):
        self.D = D
    
class eyepacsValidation(eyepacsValidationBase):
    def preparation(self, **kwargs):
        self.D = D