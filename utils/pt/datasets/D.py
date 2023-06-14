import cowsay, sys
import numpy as np
from PIL import Image
from os.path import join, exists
from torch.utils.data import Dataset
from os import system, getenv, makedirs

class D_Base(Dataset):
    """custom Dataset"""
    def __init__(self, labels=None, **kwargs):
        self.kwargs = kwargs
        self.labels = dict() if labels is None else labels
        self._length = len(self.labels['x'])

    def __len__(self):
        return self._length

    def fetch(self, signal_path):
        """It must be overwrite in child class"""
        cowsay.cow('NotImplementedError:\nplease define `{}:fetch` function.'.format(self.__class__.__name__))
        sys.exit()

    def __getitem__(self, i):
        x = self.labels['x'][i]
        y = self.labels['y'][i]
        fpath = join(self.labels['#upper_path'], x)
        
        example = self.fetch(fpath) # `example` must be `dict`

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