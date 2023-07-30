import numpy as np
from utils.pt.datasets.D import D_Base

class D(D_Base):
    def start(self):
        self.shape = list(self.config.get('shape', []))
        self.length = int(self.config.get('length', 1e3))
        self.key = str(self.config.get('key', 'indices'))
        self.getitemfn = str(self.config.get('getitemfn', 'identity'))
        setattr(self, '__getitem__', getattr(self, self.getitemfn))
        self.set_length(self.length)

        if self.getitemfn == 'randint':
            self.low = int(self.config.get('low', 0))
            self.high = int(self.config.get('high', None))
    
    def identity(self, i):
        return {
            self.key: i
        }
    
    def randint(self, i):
        return {
            self.key: np.random.randint(self.low, self.high, self.shape)
        }
    
    def randn(self, i):
        return {
            self.key: np.random.normal(size=self.shape)
        }
