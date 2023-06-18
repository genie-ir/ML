import numpy as np
from utils.pt.datasets.D import D_Base
from utils.pt.datasets.imageNet import ImageNetTrain, ImageNetValidation

class D(D_Base):
    def start(self):
        print('###############', self.kwargs)
        self._length = 1024
        assert False

    def fetch(self, signal_path):
        return {
            'X': np.load(signal_path).astype(np.float32)
        }