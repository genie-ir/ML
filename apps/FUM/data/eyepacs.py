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
        print('#############', self.kwargs['UPPER_PATH'])
        # df_candidate = dfread(self.kwargs['DF_CANDIDATE_PATH'])
        # self.init_clusters = np.array([np.load(join(, dfc_row_imgid)) for dfc_row_imgid in df_candidate.image_id])
        # print(self.init_clusters)

    def fetch(self, signal_path):
        assert False
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