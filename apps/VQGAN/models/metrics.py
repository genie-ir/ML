# DELETE not use it in repo asli:)


from apps.VQGAN.models.kernel_py_classes.basic import PYBASE
import re


# https://github.com/coleifer/peewee

import datetime
from peewee import *

class SQLite(PYBASE):
    def __init__(self, db: str = None, **kwargs):
        super().__init__(**kwargs)
        self.db_path = db or '@HOME/database/SQLite/test.db'
        if not self.db_path.endswith('.db'):
            self.db_path = self.db_path + '.db'
        self.__start()

    def __start(self):
        self.db = SqliteDatabase(self.db_path)
        self.db.connect()
        self.orm = self.ORM()
        self.tables = dict()

    def create_tables(self, tables):
        for tn, tc in tables.items(): # tn: table name, tc: table columns informations
            columns = dict() # table columns instancess
            for cn, _cv in tc.items(): # cn: column name, cv: column value
                if isinstance(_cv, str):
                    cv = dict(type=_cv, params=dict())
                else:
                    cv = _cv
                columns[cn] = eval(cv['type'])(**cv.get('params', dict()))
            
            self.tables[tn] = type(
                tn,
                (self.orm,),
                {
                    **columns,
                    'timestamp': DateTimeField(default=datetime.datetime.now)
                },
            )
        self.db.create_tables(list(self.tables.values()))
    
    def ORM(self):
        db = self.db
        class BaseModel(Model):
            class Meta:
                database = db
        return BaseModel





class Metrics(PYBASE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__start()

    def __start(self):
        self.R = dict()
        self.metrics = dict()
        self.reductions = dict()

    def logger(self, tag: str, **kwargs):
        """this shoud be overwrite in each Logger class"""
        print(self.reductions[tag])

    def log(self, tag: str, logdict, **kwargs):
        """assuming that each `mv` is a scaller value!! if you dont this mind you can freely change this function such that compatability also exist:)"""
        for mk, mv in logdict.items():
            try:
                self.metrics[tag][mk].append(mv)
            except Exception as e:
                self.metrics[tag] = self.metrics.get(tag, dict())
                self.metrics[tag][mk] = self.metrics[tag].get(mk, [])
                self.metrics[tag][mk].append(mv)

    def save(self, tag: str, **kwargs):
        for MK, mv in self.metrics.get(tag, dict()).items():
            mk, mr = (MK + ':reduction_mean').split(':')[:2]
            mrv = getattr(self, mr)(tag, mk, mv)
            if mrv is None:
                continue
            try:
                self.reductions[tag][mk] = mrv
            except Exception as e:
                self.reductions[tag] = self.reductions.get(tag, dict())
                self.reductions[tag][mk] = mrv
        
        self.logger(tag, **kwargs)

        self.R[tag] = self.reductions[tag]
        self.metrics[tag] = dict()
        self.reductions[tag] = dict()

        return self.R[tag]

    def inference(self, tag: str, regexp: str, **kwargs):
        R = kwargs.get('R', self.R[tag]) # OPTIONAL
        reduction = kwargs.get('reduction', 'reduction_mean') # OPTIONAL
        RV = []
        pattern = re.compile(regexp)
        for rk, rv in R.items():
            if pattern.match(rk):
                RV.append(rv)
        assert len(RV) > 0, f'`regexp={regexp}` Not Found in: `{list(R.keys())}`'
        return getattr(self, reduction)(tag, None, RV)

    def reduction_sum(self, tag: str, mk: str, mv):
        return sum(mv)
    
    def reduction_mean(self, tag: str, mk: str, mv):
        return sum(mv) / len(mv)
    
    def reduction_ignore(self, tag: str, mk: str, mv):
        return None
    
    def reduction_accuracy(self, tag: str, mk: str, mv):
        globalname, localname = mk.split('/')
        subname = '{}/{}'.format(globalname, localname.replace('ACC', ''))
        
        TP = sum(self.metrics[tag][f'{subname}TP:reduction_ignore'])
        TN = sum(self.metrics[tag][f'{subname}TN:reduction_ignore'])
        FP = sum(self.metrics[tag][f'{subname}FP:reduction_ignore'])
        FN = sum(self.metrics[tag][f'{subname}FN:reduction_ignore'])
        
        A = TP + TN
        B = TP + TN + FP + FN
        
        ACC = 0
        if B > 0:
            ACC = A / B
        
        return ACC


class SQLiteLogger(Metrics):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__start()

    def __start(self):
        self.sqlite = SQLite(db=self.kwargs['db'])

    def logger(self, tag: str, **kwargs):
        # print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')
        # print(self.reductions[tag])
        # print(self.reductions[tag]['TRAIN_OPT1_A_IF1_FPSISTM/ACC'])
        # print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')
        try:
            self.sqlite.tables[tag].create(**self.reductions[tag])
        except Exception as e:
            if self.sqlite.tables.get(tag, None) is None:
                self.sqlite.create_tables({
                    tag: dict((mk, dict(type='FloatField', params=dict())) for mk, mv in self.reductions[tag].items())
                    # TODO currently only we assume each field is FloatField and have no params for FloatField class, you can extend types later...
                })
                self.sqlite.tables[tag].create(**self.reductions[tag])
            else:
                raise e
            
