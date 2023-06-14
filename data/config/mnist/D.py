
import numpy as np
from PIL import Image
from utils.pt.datasets.D import D_Base
from utils.pt.datasets.imageNet import ImageNetTrain, ImageNetValidation

class D(D_Base):
    def fetch(self, signal_path):
        r = self.kwargs.get('repeat', 1)
        rc = int(self.kwargs.get('random_crop', 0))
        signal = (np.array(Image.open(signal_path)) / 255.0).astype(np.float32)
        signal = np.repeat(signal, r, axis=0)
        signal = np.repeat(signal, r, axis=1)
        
        if rc:
            signal = signal[0:rc, 0:rc]
        
        return {
            'X': signal
        }

class MNIST_Train(ImageNetTrain):
    def preparation(self, **kwargs):
        self.D = D
    
class MNIST_Validation(ImageNetValidation):
    def preparation(self, **kwargs):
        self.D = D