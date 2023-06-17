import yaml
import numpy as np
from os.path import join, exists
from utils.pt.datasets.D import D_Base
from os import system, getenv, makedirs
from utils.analysis.fourier.basic import x2fr
from data.config.eyepacs.D import eyepacsTrain as eyepacsTrainBase, eyepacsValidation as eyepacsValidationBase

class D(D_Base):
    def fetch(self, signal_path):
        signal = np.reshape(np.load(signal_path), (1, -1)).astype(np.float32)
        print(signal.shape)
        assert False
        return {
            'image': signal
        }

class eyepacsTrain(eyepacsTrainBase): 
    def preparation(self, **kwargs):
        self.D = D
    
class eyepacsValidation(eyepacsValidationBase):
    def preparation(self, **kwargs):
        self.D = D