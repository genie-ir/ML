import torch
import cv2
import os.path as osp
from torch.utils.data import Dataset
from PIL import Image
from utils.preprocessing.image.example.fundus.eyepacs_vasl_extraction import vaslExtractor
from albumentations.pytorch import ToTensorV2
import albumentations as A
from einops import rearrange
import torchvision, numpy as np
from libs.basicIO import signal_save



def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    return labeled_img



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
        softmax = torch.nn.Softmax(dim=1)
        NSTD  =  torch.tensor([0.1252, 0.0857, 0.0814]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to('cuda')
        NMEAN =  torch.tensor([0.3771, 0.2320, 0.1395]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to('cuda')
        txt_path = osp.join(self.root, dataset_name, split + '.txt')
        with open(txt_path, 'r') as f:
            next(f)
            for idx, line in enumerate(f):
                
                line = self._modifyline_(line, dataset_name) # modify the label of DEEPDR and EYEQ
                fs, sc = line[0].split('/')
                scn = sc.split('_')[0]
                
                img = np.array((self._readimage_(osp.join(self.mapsplit[split], fs, scn, sc), dataset_name)))
                img_clahe = A.Compose([
                    A.Resize(256, 256),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
                    ToTensorV2()
                ])(image=img)['image']

                T = self.transform(image=img)['image'].float()
                T = T.unsqueeze(0).to('cuda')
                
                
                
                pred = softmax(self.kwargs['tasknet'](torch.cat([T, T], dim=0))[0])
                DR_label = (int(line[1]))
                target = 0
                if DR_label == 1 or DR_label == 2:
                    target = 1
                elif DR_label == 3 or DR_label == 4:
                    target = 2
                
                
                # quality = (int(line[2]))
                # print('pred', pred)
                # print('DR_label', DR_label)
                # print('liq', quality)

                # signal_save(T * (255 * NSTD) + (255 * NMEAN), f'/content/dataset/fundus/{target}/{scn}.png', stype='img', sparams={'chw2hwc': True})
                # signal_save(img_clahe, f'/content/dataset/fundus-clahe/{target}/{scn}.png', stype='img', sparams={'chw2hwc': True})
                r = vaslExtractor(rearrange(img_clahe, 'c h w -> h w c').contiguous().numpy())
                # print('@@@@@@@@@', img_clahe.shape, r.shape)
                # r = cv2.GaussianBlur(r,(13,13),0)
                # r = cv2.threshold(r, 100, 255, cv2.THRESH_BINARY)[1]


                r = cv2.threshold(r, 127, 255, cv2.THRESH_BINARY)#[1]  # ensure binary
                print('@@@@@@@@@@@@@', r.shape)
                num_labels, labels_im = cv2.connectedComponents(r)
                print('!!!!!!!!!!', labels_im)
                r = imshow_components(labels_im)
                print('#########', r)

                # print('!!!!!!!!!', thresh.shape)
                # assert False
                r = torch.tensor(r)
                r = rearrange(r, 'h w c -> c h w').contiguous()
                # print(r.shape, r.dtype)
                signal_save(r, f'/content/dataset/fundus-vasl/{target}/{scn}.png', stype='img', sparams={'chw2hwc': True})
                assert False
                
                # lat = self.vqgan.phi2lat(T)
                # print(T.shape, lat.shape)
                
                
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