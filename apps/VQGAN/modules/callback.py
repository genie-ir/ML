from loguru import logger
from utils.pl.plCallback import ModelCheckpointBase, SetupCallbackBase, CustomProgressBarBase, SignalLoggerBase, CBBase
# from pytorch_lightning.callbacks import ModelCheckpoint as ModelCheckpointBase, Callback, LearningRateMonitor


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