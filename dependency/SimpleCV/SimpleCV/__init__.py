__version__ = '1.3.0'

from .base import *
from .Camera import *
from .Color import *
from .Display import *
from .Features import *
from .ImageClass import *
from .Stream import *
from .Font import *
from .ColorModel import *
from .DrawingLayer import *
from .Segmentation import *
from .MachineLearning import *
from .LineScan import *
from .DFT import DFT

if (__name__ == '__main__'):
    from SimpleCV.Shell import *
    main(sys.argv)
