# NOTE: YOU SHOUDE DELETE THIS FILE LATER
import os
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
    def fetch(self, signal_path, **kwargs):
        y = kwargs['y']
        # print('--------------------------------->', y)
        if y == 0 or y == 1:
            y_edit = 0
        elif y == 2 or y == 3:
            y_edit = 1
        elif y == 4:
            y_edit = 2
        else:
            assert False
        return {
            'X0': np.load(signal_path),
            'y_edit': y_edit
        }


dr_transformer = A.Compose([
    # A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
    ToTensorV2()
])
class D_DR(D_Base):
    def fetch(self, signal_path, **kwargs):
        print(signal_path)
        assert False

        return {
            'xs': dr_transformer(image=np.array(Image.open(signal_path)))['image'],
            'xc1': dr_transformer(image=np.array(Image.open(signal_path)))['image'],
            'xc2': dr_transformer(image=np.array(Image.open(signal_path)))['image'],
            # 'y_edit': kwargs['y'] # DELETE: any other case of DR it must be comment out.
        }


class DDR_TRAIN(D_DR):
    def start(self):
        super().start()
        self.path_grade2 = '/content/root/ML_Framework/FUM/cache/autoencoders/data/fum_dataset/data/dataset/train/Grade_2'
        self.grade2 = os.listdir(self.path_grade2)
        print('------------------------------------>', self.grade2, len(self.grade2))

class DDR_VAL(D_DR):
    def start(self):
        super().start()
        self.path_grade2 = '/content/root/ML_Framework/FUM/cache/autoencoders/data/fum_dataset/data/dataset/val/Grade_2'
        self.grade2 = os.listdir(self.path_grade2)
        print('------------------------------------>', self.grade2, len(self.grade2))

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
        self.D = DDR_TRAIN
    
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
        self.D = DDR_VAL