from os import getenv
from loguru import logger
from libs.basicTime import getTimeHR_V0

class AppBase:
    def __new__(cls, **kwargs):
        super().__new__(cls)
        cls.now = getattr(cls, 'now_function', getTimeHR_V0)()
        cls.opt, cls.unknown, (cls.ckptdir, cls.cfgdir, cls.logdir, cls.nowname) = cls.Parser(now=cls.now)
        return getattr(cls, getenv('GENIE_ML_APP_FN'))(**kwargs)

    @classmethod
    def main(cls):
        raise NotImplementedError()
    
    @classmethod
    def validate(cls):
        if not cls.opt.no_validate and not cls.trainer.interrupted:
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            cls.trainer.validate(cls.model, cls.data)

    @classmethod
    def fit(cls):
        if cls.opt.train:
            for tl in cls.trainer.loggers:
                unlockFlag = getattr(tl, 'unlockFlag', lambda *args, **kwargs: None)
                unlockFlag()
            if cls.data is None:
                print('00000000000000000000000000000000000000000000000')
                cls.trainer.fit(cls.model)
            else:
                cls.trainer.fit(cls.model, cls.data)
            for tl in cls.trainer.loggers:
                lockFlag = getattr(tl, 'lockFlag', lambda *args, **kwargs: None)
                lockFlag()

    @classmethod
    def test(cls):
        if not cls.opt.no_test and not cls.trainer.interrupted:
            cls.trainer.test(cls.model, cls.data)
        
    @classmethod
    def plot(cls, **kwargs):
        col_names = kwargs.get('col_names', 'val__discloss_epoch, step, epoch')
        index = kwargs.get('index', 0)
        y_name = col_names.split(',')[index].strip()

        from utils.plots.plot1d import Plot1D
        neon = Plot1D(xlabel=kwargs.get('xlabel', 'epoch'), ylabel=kwargs.get('ylabel', y_name))
        neon.plot_metrics(
            tbl = kwargs.get('tbl', '2022_12_14t19_46_35_eyepacs_vqgan'),
            hash = kwargs.get('hash', '7cea1c511ce7e9bab00be269201cc16effa8ad12'),
            col_names = col_names,
            # col_names = 'val__discloss_epoch, step, epoch',
            db = kwargs.get('db', '/media/alihejrati/3E3009073008C83B/Code/Genie-ML/logs/vqgan/13/metrics.db'),
            smoothing=kwargs.get('smoothing', True),
            # smooth_both=kwargs.get('smooth_both', True),
            # label=smooth_both.get('label', 'loss'),
            index=index
        )