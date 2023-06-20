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
        STATIC_PATH = join('1DsignalOfEyepacs', self.kwargs['DATASET_CATEGORY'], 'Grade_')
        print('------>', STATIC_PATH)
        self._length = 0
        df_candidate = dfread(self.kwargs['DF_CANDIDATE_PATH'])
        self.init_clusters = np.array([np.load(join(
            self.kwargs['UPPER_PATH'],
            STATIC_PATH + df_candidate.iloc[dfc_idx].dr,
            df_candidate.iloc[dfc_idx].image_id)) for dfc_idx in range(len(df_candidate))])
        print(self.init_clusters, self.init_clusters.shape)
        assert False

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