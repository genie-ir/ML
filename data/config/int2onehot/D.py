import numpy as np
from utils.pt.datasets.D import D_Base

class D(D_Base):
    def start(self):
        self.M = int(self.kwargs.get('M', 1))
        self.N = int(self.kwargs.get('N', -1))
        self.oh = np.eye(self.N)
        self._length = self.N - 1

    def __getitem__(self, i):
        Xi = np.random.uniform(0, .5, (self.M, 1)) + i
        Xip1 = np.random.uniform(.5, 1, (self.M, 1)) + i + 1

        print(Xi, self.oh[i])
        print('-'*30)
        print(Xip1, self.oh[i+1])
        assert False

        return {
            'Xi': Xi,
            'Xip1': Xip1,
            'Yi': self.oh[i],
            'Yi+1': self.oh[i+1]
        }