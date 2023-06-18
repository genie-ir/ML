import numpy as np
from utils.pt.datasets.D import D_Base

class D(D_Base):
    def start(self):
        self.M = int(self.kwargs.get('M', 1))
        self._length = int(self.kwargs.get('N', -1))

    def __getitem__(self, i):
        X = np.zeros((self.M, 1)) + i

        print(X)
        assert False

        return {
            'X': i,
            'Y': 0
        }