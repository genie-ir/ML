
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
import signal as sig
from signal import signal
# from apps.FUM.data.extract_ma import findMA
try:
    from dependency.Local_Convergence_Index_Features.B_GadientWeighting import main as ma_ditector_fn
except Exception as e:
    print(e)
    assert False
from os.path import join as ospjoin

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
    ToTensorV2() # hxwxch -> chxhxw
])
dr_transformer_e = A.Compose([ # doesnt affect order of channells
])

def imgNormalizer(img): # bipolar normalizer
    return (img / 127.5) - 1 

# dr_transformer = A.Compose([
#     A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=0.5),
#     ToTensorV2()
# ])
class D_DR(D_Base):
    def fetch(self, signal_path, **kwargs):
        y = kwargs['y']
        
        if y == 0 or y == 1:
            y_edit = 0
            yl = '[01]'
            ynl = ['2', '[34]']
        elif y == 2:
            y_edit = 1
            yl = '2'
            ynl = ['[01]', '[34]']
        elif y == 3 or y == 4:
            y_edit = 2
            yl = '[34]'
            ynl = ['[01]', '2']
        else:
            assert False

        fname = signal_path.split('/')[-1].replace('.jpg', '').replace('_clahe', '')

        xs = dr_transformer0(image=np.array(Image.open(signal_path)).astype(np.float32))['image']
        xsl = dr_transformer0(image=np.array(Image.open(signal_path.replace('/fundus/', '/lesion/'))).astype(np.float32))['image']
        xsc = np.array(Image.open(signal_path.replace('/fundus/', '/cunvexhull/'))).astype(np.float32)[:,:,0]
        xsf = np.array(Image.open(signal_path.replace('/fundus/', '/fundus-mask/'))).astype(np.float32)
        xslmask = np.array(Image.open(signal_path.replace('/fundus/', '/lmask/'))).astype(np.float32)[:,:,0]

        xsc = dr_transformer0(image=xsc)['image']
        xsf = dr_transformer0(image=xsf)['image']
        xslmask = dr_transformer0(image=xslmask)['image']


        xc = [None for n in range(2)]
        xcl = [None for n in range(2)]
        # xclNrot = [None for n in range(2)]
        xcc = [None for n in range(2)]
        xcf = [None for n in range(2)]
        xclmask = [None for n in range(2)]
        
        # for cidx, cval in enumerate(['[01]', '2', '[34]']):
        for cidx, cval in enumerate(ynl):
            # xc_idx = kwargs['i'] % self.grade_len[cval]
            # cpath = self.grade[cval][xc_idx]
            cpath = ospjoin('/content/RetinaLessions', fname, cval)

            xc[cidx] = imgNormalizer(dr_transformer0(image=np.array(Image.open(ospjoin(cpath, 'fundus.jpg'))).astype(np.float32))['image'])
            xcl[cidx] = imgNormalizer(dr_transformer0(image=np.array(Image.open(ospjoin(cpath, 'lesion.jpg'))).astype(np.float32))['image'])
            # xclNrot[cidx] = imgNormalizer(dr_transformer0(image=np.array(Image.open(ospjoin(cpath, '??lesion.jpg'))).astype(np.float32))['image'])
            
            xcc[cidx] = np.array(Image.open(ospjoin(cpath, 'cvh.jpg'))).astype(np.float32) / 255.0 # binary
            xcf[cidx] = np.array(Image.open(ospjoin(cpath, 'fmask.jpg'))).astype(np.float32) / 255.0 # single channell binary
            xclmask[cidx] = np.array(Image.open(ospjoin(cpath, 'lmask.jpg'))).astype(np.float32)[:,:,0] / 255.0 # binary
            
            xcc[cidx] = dr_transformer0(image=xcc[cidx])['image']
            xcf[cidx] = dr_transformer0(image=xcf[cidx])['image']
            xclmask[cidx] = dr_transformer0(image=xclmask[cidx])['image']
            
            # print(xc[cidx].shape)
            # print(xcl[cidx].shape)
            # print(xcc[cidx].shape)
            # print(xcf[cidx].shape)
            # print(xclmask[cidx].shape)
            # print('-'*30)

        
        
        # print(xs.shape)
        # print(xsl.shape)
        # print(xsf.shape)
        # print(xsc.shape)
        # print(xslmask.shape)
        # print('+'*30)
        
        return {
            'xs': imgNormalizer(xs),
            'xsl': imgNormalizer(xsl),
            'xsc': xsc / 255.0, # binary
            'xsf': xsf / 255.0, # binary
            'xslmask': xslmask / 255.0, # binary
            'xc': xc,
            'xcl': xcl,
            # 'xclNrot': xclNrot,
            'xcc': xcc,
            'xcf': xcf,
            'xclmask': xclmask,
            'ynl': ynl,
            'yl': yl,
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
        # print(src)
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
        # print(src)
        # if not exists(src):
        #     src = join('/content', self.config.get('SRC'))
        system('cp -R {} {}/fumdata.zip'.format(
            src,
            kwargs['real_fdir']
        ))

    def preparation(self, **kwargs):
        self.D = DDR_VAL