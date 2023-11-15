
APP_NAME = 'VQGAN' # FUM/VQGAN

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
from libs.basicIO import signal_save
import glob
# from apps.FUM.data.extract_ma import findMA
try:
    from dependency.Local_Convergence_Index_Features.B_GadientWeighting import main as ma_ditector_fn
except Exception as e:
    print(e)
    assert False


DATASET_PATH = '/content/root/ML_Framework/VQGAN/cache/autoencoders/data/eyepacs_all_for_cgan/data/fumdata'


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




LANDA = (256*256)

dr_transformer0 = A.Compose([
    ToTensorV2()
])
# dr_transformer = A.Compose([
#     A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=0.5),
#     ToTensorV2()
# ])
class D_DR(D_Base):
    def fetch(self, signal_path, **kwargs):
        y = kwargs['y']
        # print('--------------------------------->', y)
        # if y == 0:
        #     y_edit = 0
        # elif y == 1 or y == 2 or y == 3:
        #     y_edit = 1
        # elif y == 4:
        #     y_edit = 2
        # else:
        #     assert False


        
        # [[575 105   0]
        #  [ 57 392   0]
        #  [ 11  61   1]]
        # if y == 0 or y == 1:
        #     y_edit = 0
        # elif y == 2 or y == 3:
        #     y_edit = 1
        # elif y == 4:
        #     y_edit = 2
        # else:
        #     assert False
        
        
        
        
        
        
        if y == 0 or y == 1:
            y_edit = 0
        elif y == 2:
            y_edit = 1
        elif y == 3 or y == 4:
            y_edit = 2
        else:
            assert False


        
        
        print(signal_path)

        
        xs_fundus = dr_transformer0(image=np.array(Image.open(signal_path)))['image']
        xs_lesion = dr_transformer0(image=np.array(Image.open(signal_path.replace('/fundus/', 'lesion'))))['image']
        xs_cunvechull = dr_transformer0(image=np.array(Image.open(signal_path.replace('/fundus/', 'cunvexhull'))))['image']
        xs_fundusmask = dr_transformer0(image=np.array(Image.open(signal_path.replace('/fundus/', 'fundus-mask'))))['image']


        # xc1 = (dr_transformer(image=np.array(Image.open(
        #         os.path.join(self.path_grade2, self.grade2[kwargs['i'] % self.grade2_len])
        #     )))['image'] / 127.5) - 1
        # xc2 = (dr_transformer(image=np.array(Image.open(
        #         os.path.join(self.path_grade4, self.grade4[kwargs['i'] % self.grade4_len])
        #     )))['image'] / 127.5) - 1

        return {
            'xs_fundus': (xs_fundus / 127.5) - 1,
            'xs_lesion': (xs_lesion / 127.5) - 1,
            'xs_cunvechull': (xs_cunvechull / 127.5) - 1,
            'xs_fundusmask': (xs_fundusmask / 127.5) - 1,
            # 'xs_ma': (xs_ma / 127.5) -1,
            # 'xc': [
            #     xc1, xc2
            # ],
            'y_edit': y_edit # DELETE: any other case of DR it must be comment out.
        }


class DDR_TRAIN(D_DR):
    def start(self):
        super().start()
        # print('TRAIN', self.__len__())
        
        
        category = 'train'
        self.path_grade2 = f'/content/root/ML_Framework/VQGAN/cache/autoencoders/data/eyepacs_all_for_cgan/data/fumdata/{category}/fundus/2/*.jpg'
        self.grade2 = glob.glob(self.path_grade2)
        self.grade2_len = len(self.grade2)
        
        
        self.path_grade01 = f'/content/root/ML_Framework/VQGAN/cache/autoencoders/data/eyepacs_all_for_cgan/data/fumdata/{category}/fundus/[01]/*.jpg'
        self.grade01 = glob.glob(self.path_grade01)
        self.grade01_len = len(self.grade01)
        
        
        self.path_grade34 = f'/content/root/ML_Framework/VQGAN/cache/autoencoders/data/eyepacs_all_for_cgan/data/fumdata/{category}/fundus/[34]/*.jpg'
        self.grade34 = glob.glob(self.path_grade34)
        self.grade34_len = len(self.grade34)
        
        
        
        # print('@@@@@@@@@@@@2', self.grade2_len, self.grade2)
        # print('@@@@@@@@@@@@34', self.grade34_len, self.grade34)
        # print('@@@@@@@@@@@@01', self.grade01_len, self.grade01)

class DDR_VAL(D_DR):
    def start(self):
        super().start()
        # print('VAL', self.__len__())


        category = 'gg'
        self.path_grade2 = f'/content/root/ML_Framework/VQGAN/cache/autoencoders/data/eyepacs_all_for_cgan/data/fumdata/{category}/fundus/2/*.jpg'
        self.grade2 = glob.glob(self.path_grade2)
        self.grade2_len = len(self.grade2)
        
        
        self.path_grade01 = f'/content/root/ML_Framework/VQGAN/cache/autoencoders/data/eyepacs_all_for_cgan/data/fumdata/{category}/fundus/[01]/*.jpg'
        self.grade01 = glob.glob(self.path_grade01)
        self.grade01_len = len(self.grade01)
        
        
        self.path_grade34 = f'/content/root/ML_Framework/VQGAN/cache/autoencoders/data/eyepacs_all_for_cgan/data/fumdata/{category}/fundus/[34]/*.jpg'
        self.grade34 = glob.glob(self.path_grade34)
        self.grade34_len = len(self.grade34)

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
        # kwargs['real_fdir'] = '/content/root/ML_Framework/VQGAN/cache/autoencoders/data/eyepacs_all_for_cgan'
        # src = join(getenv('GENIE_ML_STORAGE0'), '..', '..', self.config.get('SRC'))
        src = self.config.get('SRC')
        print(src)
        # if not exists(src):
        #     src = join('/content', self.config.get('SRC'))
        system('cp -R {} {}/fumdata.zip'.format(
            src,
            kwargs['real_fdir']
        ))
    
    def preparation(self, **kwargs):
        self.D = DDR_TRAIN
    
class DVal(ImageNetValidation):
    def download_dataset(self, **kwargs):
        # kwargs['real_fdir'] = '/content/root/ML_Framework/VQGAN/cache/autoencoders/data/eyepacs_all_for_cgan'
        # src = join(getenv('GENIE_ML_STORAGE0'), '..', '..', self.config.get('SRC'))
        src = self.config.get('SRC')
        print(src)
        # if not exists(src):
        #     src = join('/content', self.config.get('SRC'))
        system('cp -R {} {}/fumdata.zip'.format(
            src,
            kwargs['real_fdir']
        ))

    def preparation(self, **kwargs):
        self.D = DDR_VAL