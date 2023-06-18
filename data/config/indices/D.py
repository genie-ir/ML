import numpy as np
from utils.pt.datasets.D import D_Base
from utils.pt.datasets.imageNet import ImageNetTrain, ImageNetValidation

class D(D_Base):
    def start(self):
        self._length = int(self.kwargs.get('N', -1))

    def __getitem__(self, i):
        print('$$$$$$$$$$$$$$$$$$$', i)
        assert False

    def fetch(self, signal_path):
        return {
            'X': np.load(signal_path).astype(np.float32)
        }