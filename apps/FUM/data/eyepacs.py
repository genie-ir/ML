# NOTE: YOU SHOUDE DELETE THIS FILE LATER

import yaml
import torch
import numpy as np
from PIL import Image
from libs.basicIO import dfread
from os.path import join, exists
from utils.pt.datasets.D import D_Base
from data.config.eyepacs.D import eyepacsTrain as eyepacsTrainBase, eyepacsValidation as eyepacsValidationBase
from albumentations.pytorch import ToTensorV2
import albumentations as A

class D(D_Base):
    def fetch(self, signal_path):
        return {
            'X': np.load(signal_path)
        }


dr_transformer = A.Compose([
    ToTensorV2()
])
class D_DR(D_Base):
    def fetch(self, signal_path):
        return {
            'X': dr_transformer(np.array(Image.open(signal_path)))
        }

class eyepacsTrain(eyepacsTrainBase): 
    def preparation(self, **kwargs):
        self.D = D
    
class eyepacsValidation(eyepacsValidationBase):
    def preparation(self, **kwargs):
        self.D = D


from os import getenv, system
from utils.pt.datasets.imageNet import ImageNetTrain, ImageNetValidation

class DTrain(ImageNetTrain):
    def download_dataset(self, **kwargs):
        src = join(getenv('GENIE_ML_STORAGE0'), '..', '..', self.config.get('SRC'))
        if not exists(src):
            src = join('/content', self.config.get('SRC'))
        system('cp -R {} {}/datasetfile.zip'.format(
            src,
            kwargs['real_fdir']
        ))
    
    def preparation(self, **kwargs):
        self.D = D_DR
    
class DVal(ImageNetValidation):
    def download_dataset(self, **kwargs):
        src = join(getenv('GENIE_ML_STORAGE0'), '..', '..', self.config.get('SRC'))
        if not exists(src):
            src = join('/content', self.config.get('SRC'))
        system('cp -R {} {}/datasetfile.zip'.format(
            src,
            kwargs['real_fdir']
        ))
    
    def preparation(self, **kwargs):
        self.D = D_DR