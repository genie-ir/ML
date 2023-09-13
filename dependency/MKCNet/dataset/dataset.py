import torch
import os.path as osp
from torch.utils.data import Dataset
from PIL import Image


from einops import rearrange
import torchvision, numpy as np
from libs.basicIO import signal_save
class basic_dataset(Dataset):
    def __init__(self, root, split='empty', transform=None, **kwargs):
        self.kwargs = kwargs
        self.root = root['ROOT']
        self.datadir = root['DATADIR']
        self.mapsplit = root['MAPSPLIT']
        self.transform = transform
        # self.data = []
        # self.label_T = []
        # self.label_IQ = []
        # self.label_M = []
        self.split = split



    def read_data(self, split, dataset_name, num_T):
        txt_path = osp.join(self.root, dataset_name, split + '.txt')
        with open(txt_path, 'r') as f:
            next(f)
            for line in f:
                line = self._modifyline_(line, dataset_name) # modify the label of DEEPDR and EYEQ
                fs, sc = line[0].split('/')
                scn = sc.split('_')[0]
                # print(self.transform)
                T = self.transform(image=np.array((self._readimage_(osp.join(self.mapsplit[split], fs, scn, sc), dataset_name))))['image'].float()
                T = T.unsqueeze(0).to('cuda')
                signal_save(T, f'/content/dataset/{scn}.png', stype='img', sparams={'chw2hwc': True})
                print('---------------------->', T.shape, T.dtype)
                softmax = torch.nn.Softmax(dim=1)

                print(softmax(self.kwargs['tasknet'](torch.cat([T, T], dim=0))[0]))

                

                

                # lat = self.vqgan.phi2lat(T)
                # print(T.shape, lat.shape)
                lt = (int(line[1]))
                liq = (int(line[2]))
                lm = (int(line[2]) * num_T + int(line[1]))
                print('lt', lt)
                print('liq', liq)
                print('lm', lm)
                assert False
                
                
                # self.label_T.append(int(line[1]))
                # self.label_IQ.append(int(line[2]))
                # self.label_M.append(int(line[2]) * num_T + int(line[1]))
    
    def _modifyline_(self, line, dataset_name):
        line = line.strip().split(',')
        if dataset_name in ['DEEPDR', 'EYEQ']:
            if line[1] == '4': line[1] = '2'
            elif line[1] in ['2', '3']: line[1] = '1'

        return line
    
    def _readimage_(self, path, dataset_name):
        if dataset_name in ['DEEPDR', 'EYEQ']:
            return Image.open(path).convert('RGB')
        else:
            return Image.open(path).convert('L')

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label_T[index], self.label_IQ[index], self.label_M[index]

    def __len__(self):
        return len(self.data)
    
class DEEPDR(basic_dataset):
    def __init__(self, root, split, transform=None, **kwargs):
        super(DEEPDR, self).__init__(root, split, transform, **kwargs)
        self.read_data(split, 'DEEPDR', 3)

class EYEQ(basic_dataset):
    def __init__(self, root, split, transform=None):
        super(EYEQ, self).__init__(root, split, transform)
        self.read_data(split, 'EYEQ', 3)

class DRAC(basic_dataset):
    def __init__(self, root, split, transform=None):
        super(DRAC, self).__init__(root, split, transform)
        self.read_data(split, 'DRAC', 3)

class IQAD_CXR(basic_dataset):
    def __init__(self, root, split, transform=None):
        super(IQAD_CXR, self).__init__(root, split, transform)
        self.read_data(split, 'IQAD_CXR', 2)

class IQAD_CT(basic_dataset):
    def __init__(self, root, split, transform=None):
        super(IQAD_CT, self).__init__(root, split, transform)
        self.read_data(split, 'IQAD_CT', 2)