import numpy as np
from utils.pt.datasets.D import D_Base

class D(D_Base):
    def start(self):
        self.M = int(self.kwargs.get('M', 1))
        self.N = int(self.kwargs.get('N', -1))
        self.oh = np.eye(self.N).astype(np.float32)
        self._length = self.N - 1

    def __getitem__(self, i):

        return {
            'Xi': np.random.uniform(0, .5, (self.M, 1)).astype(np.float32) + i,
            'Xip1': np.random.uniform(.5, 1, (self.M, 1)).astype(np.float32) + i,
            'Yi': self.oh[i],
            'Yip1': self.oh[i+1]
        }