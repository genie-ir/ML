import bisect
import numpy as np
import albumentations
from PIL import Image
from os.path import join
from loguru import logger
from utils.preprocessing.image.example.fundus.eyepacs_vasl_extraction import vaslExtractor
from torch.utils.data import Dataset, ConcatDataset


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx

class D(Dataset):
    """custom Dataset"""
    def __init__(self, size=None, random_crop=False, labels=None, **kw):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self._length = len(self.labels['x'])

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose([
                self.rescaler, 
                self.cropper
            ])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = np.array(image).astype(np.uint8)

        # logger.info('image!! {}'.format(image))
        return image

    def __getitem__(self, i):
        x = self.labels['x'][i]
        y = self.labels['y'][i]
        fpath = join(self.labels['#upper_path'], x)
        
        example = dict()
        vasl = vaslExtractor(fpath)
        image = self.preprocess_image(fpath)
        # logger.critical('{} | {} | {}'.format(np.unique(vasl), vasl.shape, image.shape))
        T = self.preprocessor(image=image, mask=vasl)
        
        example['vasl'] = (T['mask']/127.5 - 1.0).astype(np.float32)
        example['image'] = (T['image']/127.5 - 1.0).astype(np.float32)

        # logger.warning('{} | {} | {}'.format(np.unique(example['vasl']), example['vasl'].shape, example['image'].shape))

        for k in self.labels: # (#:ignore) (@:function(y)) ($:function(x))
            if k[0] == '#':
                pass
            elif k[0] == '@':
                example[k[1:]] = self.labels[k](y)
            elif k[0] == '$':
                example[k[1:]] = self.labels[k](x)
            else:
                example[k] = self.labels[k][i]
        
        return example






class D_latent(Dataset):
    """custom Dataset"""
    def __init__(self, size=None, random_crop=None, labels=None):
        self.labels = dict() if labels is None else labels
        self._length = len(self.labels['x'])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        # print(i)
        # return i
        x = self.labels['x'][i]
        y = self.labels['y'][i]
        fpath = join(self.labels['#upper_path'], x)
        
        example = dict()
        example['latentcode'] = (np.load(fpath)).astype(np.float32)

        for k in self.labels: # (#:ignore) (@:function(y)) ($:function(x))
            if k[0] == '#':
                pass
            elif k[0] == '@':
                example[k[1:]] = self.labels[k](y)
            elif k[0] == '$':
                example[k[1:]] = self.labels[k](x)
            else:
                example[k] = self.labels[k][i]
        
        return example


class NumpyPaths(D):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
