
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

def imgNormalizer(img):
    return (img / 127.5) - 1 

# dr_transformer = A.Compose([
#     A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=0.5),
#     ToTensorV2()
# ])
class D_DR(D_Base):
    def fetch(self, signal_path, **kwargs):
        print('!!!!!!!!!!!!!!!!!!!!', self.__len__())
        y = kwargs['y']
        
        if y == 0 or y == 1:
            y_edit = 0
        elif y == 2:
            y_edit = 1
        elif y == 3 or y == 4:
            y_edit = 2
        else:
            assert False

        
        xs = dr_transformer0(image=np.array(Image.open(signal_path)))['image']
        xs_lesion = dr_transformer0(image=np.array(Image.open(signal_path.replace('/fundus/', '/lesion/'))))['image']
        xs_cunvexhull = dr_transformer0(image=np.array(Image.open(signal_path.replace('/fundus/', '/cunvexhull/'))))['image']
        xs_fundusmask = dr_transformer0(image=np.array(Image.open(signal_path.replace('/fundus/', '/fundus-mask/'))))['image']

        xc = [None for n in range(3)]
        xc_lesion = [None for n in range(3)]
        xc_cunvexhull = [None for n in range(3)]
        xc_fundusmask = [None for n in range(3)]
        for cidx, cval in enumerate(['[01]', '2', '[34]']):
            xc_idx = kwargs['i'] % self.grade_len[cval]
            cpath = self.grade[cval][xc_idx]
            
            xc[cidx] = imgNormalizer(dr_transformer0(image=np.array(Image.open(cpath)))['image'])
            xc_lesion[cidx] = imgNormalizer(dr_transformer0(image=np.array(Image.open(cpath.replace('/fundus/', '/lesion/'))))['image'])
            xc_cunvexhull[cidx] = imgNormalizer(dr_transformer0(image=np.array(Image.open(cpath.replace('/fundus/', '/cunvexhull/'))))['image'])
            xc_fundusmask[cidx] = imgNormalizer(dr_transformer0(image=np.array(Image.open(cpath.replace('/fundus/', '/fundus-mask/'))))['image'])

        return {
            'xs': imgNormalizer(xs),
            'xs_lesion': imgNormalizer(xs_lesion),
            'xs_cunvexhull': imgNormalizer(xs_cunvexhull),
            'xs_fundusmask': imgNormalizer(xs_fundusmask),
            'xc': xc,
            'xc_lesion': xc_lesion,
            'xc_cunvexhull': xc_cunvexhull,
            'xc_fundusmask': xc_fundusmask,
            'y_edit': y_edit
        }


class DDR_TRAIN(D_DR):
    def start(self):
        super().start()
        # print('TRAIN', self.__len__())
        
        
        category = 'train'
        self.grade = dict()
        self.grade_len = dict()
        for cidx, cval in enumerate(['[01]', '2', '[34]']):
            path_grade = f'/content/root/ML_Framework/VQGAN/cache/autoencoders/data/eyepacs_all_for_cgan/data/fumdata/{category}/fundus/{cval}/*.jpg'
            self.grade[cval] = glob.glob(path_grade)
            self.grade_len[cval] = len(self.grade[cval])
        
        
class DDR_VAL(D_DR):
    def start(self):
        super().start()
        # print('VAL', self.__len__())


        category = 'val'
        self.grade = dict()
        self.grade_len = dict()
        for cidx, cval in enumerate(['[01]', '2', '[34]']):
            path_grade = f'/content/root/ML_Framework/VQGAN/cache/autoencoders/data/eyepacs_all_for_cgan/data/fumdata/{category}/fundus/{cval}/*.jpg'
            self.grade[cval] = glob.glob(path_grade)
            self.grade_len[cval] = len(self.grade[cval])

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