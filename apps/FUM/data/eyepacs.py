import yaml
import torch
import numpy as np
from libs.basicIO import dfread
from os.path import join, exists
from utils.pt.datasets.D import D_Base
from os import system, getenv, makedirs
from data.config.eyepacs.D import eyepacsTrain as eyepacsTrainBase, eyepacsValidation as eyepacsValidationBase

class D(D_Base):
    def start(self):
        self._length = 0
        # dfread()
        print('@@@@@@@@@@@@', self.kwargs['df_candidate_path'])
        assert False

    def fetch(self, signal_path):
        return {
            # 'latentcode': np.load(signal_path)
            # 'noisecode': 
        }

class eyepacsTrain(eyepacsTrainBase): 
    def preparation(self, **kwargs):
        self.D = D
    
class eyepacsValidation(eyepacsValidationBase):
    def preparation(self, **kwargs):
        self.D = D