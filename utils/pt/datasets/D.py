import cowsay, sys
import numpy as np
from PIL import Image
from os.path import join, exists
from os import system, getenv, makedirs
from torch.utils.data import Dataset, TensorDataset

# Example: -> TensorDataset <-
# x1 = torch.randint(0,1023, (10, 4))
# x2 = torch.randint(0,1023, (10, 3, 4))
# x3 = torch.randint(0,1023, (10, 4))
# x4 = torch.randint(0,1023, (10, 4))
# for single_row_of_data in TensorDataset(x1,x2,x3,x4):
#     print(single_row_of_data)

class D_Base(Dataset):
    """custom Dataset"""
    def __init__(self, labels=None, **kwargs):
        self.kwargs = kwargs
        self.config = self.kwargs.get('config', dict())
        self.labels = dict() if labels is None else labels
        self.start()
    
    def start(self):
        self.set_length()
    
    def set_length(self, length: int =None):
        if length is None:
            self.__length = len(self.labels['x'])
        else:
            self.__length = int(length)

    @property
    def get_length(self):
        return self.__length
    
    def __len__(self):
        return self.__length

    def fetch(self, signal_path, **kwargs):
        """It must be overwrite in child class"""
        cowsay.cow('NotImplementedError:\nplease define `{}:fetch` function.'.format(self.__class__.__name__))
        sys.exit()

    def __getitem__(self, i):
        x = self.labels['x'][i]
        y = self.labels['y'][i]
        fpath = join(self.labels['#upper_path'], x)
        
        example = self.fetch(fpath, **{'y': y, 'i': i}) # `example` must be `dict`

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