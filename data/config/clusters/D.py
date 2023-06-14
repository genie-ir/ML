
import numpy as np
from PIL import Image
from utils.pt.datasets.D import D_Base
from utils.pt.datasets.imageNet import ImageNetTrain, ImageNetValidation

class D(D_Base):
    def fetch(self, signal_path):
        dspec = self.kwargs.get('dspec', [])
        signal = np.random.randint(dspec[2], dspec[3]+1, (dspec[0],dspec[1])).astype(np.float32)
        
        return {
            'X': signal
        }

class Train(ImageNetTrain):
    def preparation(self, **kwargs):
        self.D = D
    
class Validation(ImageNetValidation):
    def preparation(self, **kwargs):
        self.D = D