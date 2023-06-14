import os
from loguru import logger
from libs.basicHR import EHR 
from utils.pl.plAppBase import AppBase
from apps.MVLAB.modules.args import Parser
from apps.MVLAB.modules.configuration import Config
from apps.MVLAB.modules.handler import SignalHandler

class App(AppBase):
    def __new__(cls, **kwargs):
        cls.Parser = Parser
        return super().__new__(cls, **kwargs)

    @classmethod
    def main(cls):
        try:
            cls.model, cls.trainer, cls.data = Config(
                ckptdir=cls.ckptdir, cfgdir=cls.cfgdir, logdir=cls.logdir,
                opt=cls.opt, unknown=cls.unknown, nowname=cls.nowname, now=cls.now,
                nondefault_trainer_args=Parser.nondefault_trainer_args
            )

            SignalHandler(cls.trainer, cls.ckptdir)

            # run
            cls.validate()
            try:
                cls.fit()
            except Exception:
                SignalHandler.melk()
                raise
            cls.validate()
            cls.test()
        except Exception as e:
            if cls.opt.debug and cls.trainer.global_rank==0:
                EHR(e)
            raise
        finally:
            # move newly created debug project to debug_runs
            if cls.opt.debug and not cls.opt.resume and cls.trainer.global_rank==0:
                dst, name = os.path.split(cls.logdir)
                dst = os.path.join(dst, 'debug_runs', name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                os.rename(cls.logdir, dst)