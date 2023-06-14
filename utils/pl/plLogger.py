from os import getenv, makedirs
import inflect
from os.path import join
from loguru import logger
from libs.coding import sha1
import pytorch_lightning as pl
from utils.metrics import Metrics
from libs.dbms.sqlite_dbms import SqliteDBMS
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

class GenieLoggerBase(Logger):
    def __init__(
        self, 
        save_dir: str = None,
        name: Optional[str] = 'GeineLogger',
        fn_name: Optional[str] = '_genie',
        select_storage: Optional[str] = 'GENIE_ML_STORAGE0',
        version: str = '0.1',
        hash_ignore = [],
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.hash_ignore = hash_ignore
        self._save_dir = save_dir
        self._name = name.lower().strip().replace(' ', '_').replace('-', '_')
        self._fn_name = fn_name
        self._version = version
        self.select_storage = select_storage
        self.all_metrics_tbls = dict()

        self.db_fname = 'metrics'
        self.db_path_dir = getenv(self.select_storage)
        makedirs(self.db_path_dir, exist_ok=True)
        self.sqlite_dbms = SqliteDBMS(join(self.db_path_dir, self.db_fname))
        table_names = self.sqlite_dbms.get_tables()
        self.table_numbers = sum([1 for t in table_names if self._name in t])
        self.table_information = [t for t in table_names if self._name in t]
        
        self.inflect_engine = inflect.engine()
        self.flag_lock = True
        
        for tn_inf in self.table_information:
            cols = self.sqlite_dbms.get_colnames(tn_inf)
            for di in ['step', 'timestamp']:
                if di in cols:
                    cols.remove(di)
            cols = [c.replace('__', '/') for c in cols]
            reconstructrd_hash = sha1(' | '.join(sorted(list(cols))))
            if not (reconstructrd_hash in self.hash_ignore):
                self.all_metrics_tbls[reconstructrd_hash] = self.create_metrics_table(list(cols), bypass_tblname=tn_inf)
    
    @property
    def name(self):
        return self._name
    
    @property
    def fn_name(self):
        return self._fn_name

    @property
    def version(self):
        # Return the experiment version, int or str.
        return self._version

    @property
    def save_dir(self) -> Optional[str]:
        """Return the root directory where experiment logs get saved, or `None` if the logger does not save data locally."""
        return self._save_dir
    
    def create_metrics_table(self, metrics_items, bypass_tblname=None):
        if bypass_tblname is None:
            self.table_numbers = self.table_numbers + 1
            ord_number = self.inflect_engine.ordinal(self.table_numbers)
            new_table_name = 'tbl_' + self._name + f'_{ord_number}' # this is `nowname`. (one dir after `logs` in `logdir`) Notic: `nowname` is constant when resuming.
        else:
            new_table_name = bypass_tblname
        
        # logger.info('metric table `{}` was created.'.format(new_table_name))
        return Metrics(
            self.db_path_dir,
            self.db_fname,
            new_table_name,
            [str(m).replace('/', '__') for m in metrics_items]
        )

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        logger.critical('log_hyperparams | params={}'.format(params))

    @rank_zero_only
    def unlockFlag(self):
        self.flag_lock = False
    
    @rank_zero_only
    def lockFlag(self):
        self.flag_lock = True
    
    @rank_zero_only
    def setter_handiCall(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    @rank_zero_only
    def log_metrics_handiCall(self, **kwargs):
        return self.log_metrics(**kwargs)
    
    @rank_zero_only
    def log_metrics(self, metrics, step):
        col0 = list(metrics.keys())[0]
        # print(col0, col0.lower().startswith('val') and col0.lower().endswith('epoch'))
        if self.flag_lock:
            return
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        if metrics.get('epoch', None) is None:
            metrics['epoch'] = self._trainer_obj.current_epoch
        
        if col0.lower().startswith('val') and col0.lower().endswith('epoch'):
            step = int(metrics['epoch'])
        
        if col0.lower().startswith('val') and col0.lower().endswith('step'):
            return
        
        hash_metrics_keys = sha1(' | '.join(sorted(list(metrics.keys()))))
        tbl = self.all_metrics_tbls.get(hash_metrics_keys, None)
        if tbl is None:
            self.all_metrics_tbls[hash_metrics_keys] = self.create_metrics_table(list(metrics.keys()))
            tbl = self.all_metrics_tbls[hash_metrics_keys]
        M = {str(m).replace('/', '__'): v for m, v in metrics.items()}
        tbl.add({**M, 'step': step})

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # super().save()
        # logger.critical('save')
        # try:
        #     print(self.hparams)
        # except Exception as e:
        #     print(e)
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        # logger.critical('finalize | status={}'.format(status))
        # print(self.get('hparams', None))
        # if self._experiment is not None:
        #     self.experiment.flush()
        #     self.experiment.close()
        # self.save()
        pass