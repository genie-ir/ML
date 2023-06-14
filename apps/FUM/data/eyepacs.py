import yaml
import numpy as np
from os.path import join, exists
from utils.pt.datasets.D import D_Base
from os import system, getenv, makedirs
from utils.pt.datasets.imageNet import ImageNetTrain, ImageNetValidation

class D(D_Base):
    def fetch(self, signal_path):
        return {
            'image': np.load(signal_path).astype(np.float32)
        }

class eyepacsTrain(ImageNetTrain):
    def download_dataset(self, **kwargs):
        makedirs(kwargs['real_fdir'], exist_ok=True)
        makedirs(getenv('GENIE_ML_STORAGE0'), exist_ok=True)
        src = join(getenv('GENIE_ML_STORAGE0'), '..', '..', self.config.get('SRC'))
        if not exists(src):
            src = join('/content', self.config.get('SRC'))
        system('cp -R {} {}/datasetfile.zip'.format(
            src,
            kwargs['real_fdir']
        ))
    
    def preparation(self, **kwargs):
        self.D = D # real dataset wrapper
    
class eyepacsValidation(ImageNetValidation):
    def download_dataset(self, **kwargs):
        makedirs(kwargs['real_fdir'], exist_ok=True)
        makedirs(getenv('GENIE_ML_STORAGE0'), exist_ok=True)
        src = join(getenv('GENIE_ML_STORAGE0'), '..', '..', self.config.get('SRC'))
        if not exists(src):
            src = join('/content', self.config.get('SRC'))
        system('cp -R {} {}/datasetfile.zip'.format(
            src,
            kwargs['real_fdir']
        ))
    
    def preparation(self, **kwargs):
        self.D = D