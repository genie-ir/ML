import numpy as np
from utils.pt.datasets.D import D_Base

class D(D_Base):
    def start(self):
        self.shape = list(self.config.get('shape', []))
        self.range = int(self.config.get('range', 1e3))
        self.length = int(self.config.get('length', 1e3))
        self.set_length(self.length)
    
    def __getitem__(self, i):
        return {
            'indices': np.random.randint(0, self.range, self.shape)
        }
