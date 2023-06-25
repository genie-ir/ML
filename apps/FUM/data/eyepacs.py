import yaml
import torch
import numpy as np
from libs.basicIO import dfread
from os.path import join, exists
from utils.pt.datasets.D import D_Base
from os import system, getenv, makedirs
from data.config.eyepacs.D import eyepacsTrain as eyepacsTrainBase, eyepacsValidation as eyepacsValidationBase

from utils.pt.distance import L2S_VQ
from utils.np.statfns import correlation
from utils.pt.BB.KDE.kde import plot_kde

class D(D_Base):
    def start(self):
        STATIC_PATH = join('1DsignalOfEyepacs', self.kwargs['DATASET_CATEGORY'], 'Grade_')
        print('------>', STATIC_PATH)
        self._length = 0
        df_candidate = dfread(self.kwargs['DF_CANDIDATE_PATH'])
        self.init_clusters = dict()
        for dfc_dr in df_candidate.dr.unique():
            df_candidate_dr = df_candidate[df_candidate.dr == dfc_dr]
            self.init_clusters['class_' + str(dfc_dr)] = np.array([np.load(join(
                self.kwargs['UPPER_PATH'],
                STATIC_PATH + str(dfc_dr),
                df_candidate_dr.iloc[dfc_idx].image_id)).flatten().astype(np.float32) for dfc_idx in range(len(df_candidate_dr))])
        self.all_unique_init_clusters = np.unique(np.array([self.init_clusters[k] for k in self.init_clusters])) 
        
        for k in self.init_clusters:
            n = torch.tensor(self.init_clusters[k]) / (30*1024)
            plot_kde(D=n, h=1e2, r=.001, s=.01, path='/content/KDE/{}.png'.format(k))
            # assert False

            # m = n.unique()
            # print(k, m.shape)
            # m = n.corrcoef()
            # m = np.abs(correlation(n.detach().numpy(), kill_itself=True))
            m = torch.topk(L2S_VQ(n, n), 2, largest=False).values[:, 1]
            # M = m[m<1e-2]
            # M = M[M>0]
            print(k, m.min(), m.max(), m.dtype, m.shape)
            
            # for kj in self.init_clusters:
            #     if k == kj:
            #         continue
            #     _n = torch.tensor(self.init_clusters[kj])
            #     m = np.abs(correlation(n.detach().numpy(), _n.detach().numpy()))
            #     print('class {} -> {}'.format(k, kj), m.min(), m.max(), m.dtype, m.shape)
            # print()

        # print('all unique', self.all_unique_init_clusters, len(self.all_unique_init_clusters))
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