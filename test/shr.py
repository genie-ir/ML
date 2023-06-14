from libs.basicHR import SHR

class SignalHandler:
    def __init__(self, trainer):
        SignalHandler.trainer = trainer
        SHR(SIGUSR1=SignalHandler.melk)

    @staticmethod
    def melk(*args, **kwargs): # allow checkpointing via USR1
        print('**[melk]**', SignalHandler.trainer)