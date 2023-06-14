from loguru import logger
from pytorch_lightning.callbacks import Callback
from utils.pl.plCallback import ModelCheckpointBase, SetupCallbackBase, CustomProgressBarBase, SignalLoggerBase, CBBase

class ModelCheckpoint(ModelCheckpointBase):
    pass

class SetupCallback(SetupCallbackBase):
    pass

class CustomProgressBar(CustomProgressBarBase):
    pass

class SignalLogger(SignalLoggerBase):
    pass

class CB(CBBase):
    pass