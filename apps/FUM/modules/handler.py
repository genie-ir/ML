from os.path import join
from libs.basicHR import SHR

class SignalHandler:
    def __init__(self, trainer, ckptdir):
        SignalHandler.trainer = trainer
        SignalHandler.ckptdir = ckptdir
        SHR(SIGUSR1=SignalHandler.melk, SIGUSR2=SignalHandler.divein)

    @staticmethod
    def melk(*args, **kwargs): # allow checkpointing via USR1
        # run all checkpoint hooks
        if SignalHandler.trainer.global_rank == 0:
            print('Summoning checkpoint.')
            ckpt_path = join(SignalHandler.ckptdir, 'last.ckpt')
            SignalHandler.trainer.save_checkpoint(ckpt_path)

    @staticmethod
    def divein(*args, **kwargs):
        if SignalHandler.trainer.global_rank == 0:
            import pudb; pudb.set_trace()


