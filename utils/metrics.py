import time
import sqlite3
import orm_sqlite
import pandas as pd
from os.path import join
from loguru import logger

class Metrics():
    def __init__(self, db_path_dir: str, fname: str, tableName: str =None, items: list = None):
        try:
            from libs.basicIO import pathBIO
        except Exception as e:
            pathBIO = lambda x: x
        
        self._real_items_NOT_USED = items
        self._processed_items = [str(item).lower().replace(' ', '_').replace('-', '_') for item in items] 
        self.tableName = tableName.lower().strip().replace(' ', '_').replace('-', '_')
        self.db_path = join(pathBIO(db_path_dir), f'{fname}.db')
        self.DB = orm_sqlite.Database(self.db_path)
        
        Metric = type(
            self.tableName,
            (orm_sqlite.Model,),
            {
                'step': orm_sqlite.IntegerField(primary_key=True),
                **{p_item: orm_sqlite.FloatField() for p_item in self._processed_items},
                'timestamp': orm_sqlite.StringField()
            },
        )

        self.orderedFields = [
            'step',
            *[p_item for p_item in self._processed_items],
            # 'timestamp'
        ]

        Metric.objects.backend = self.DB
        self.Model = Metric

    def add(self, _spec):
        _spec_keys_map = {str(sk).lower().replace(' ', '_').replace('-', '_'): sk for sk in list(_spec.keys())}
        spec = {}
        for of in self.orderedFields:
            spec[of] = _spec[_spec_keys_map[of]]

        assert 'step' in spec, 'metric record does not have "step" key.'
        spec['timestamp'] = str(time.time())
        statuscode = self.Model(spec).save()
        if statuscode == -1:
            pk = spec.get('step', None)
            obj = self.Model.objects.get(pk=pk)
            assert obj is not None, f'metric record with step={pk} does not exist.'
            for key in spec.keys():
                obj[key] = spec.get(key, None)
            statuscode = self.Model.objects.update(obj)
        return statuscode
    
    def delete(self, where='true'):
        where = f'step >= {where}' if isinstance(where, int) else where
        return self.DB.execute(f'delete from {self.tableName} where {where}', *[], autocommit=True)
    
    def select(self, sql: str):
        "Returns query results, a list of sqlite3.Row objects."
        return self.DB.select(sql, *[], size=None)

    def sql(self, sql: str):
        "Executes an SQL statement and returns rows affected."
        return self.DB.execute(sql, *[], autocommit=True)
    
    def to_csv(self, sql=None, dist=None):
        sql = sql if sql else f'SELECT * FROM {self.tableName}'
        dist = self.db_path.replace('.db', '.csv') if dist is None else dist
        conn = sqlite3.connect(self.db_path, isolation_level=None, detect_types=sqlite3.PARSE_COLNAMES)
        db_df = pd.read_sql_query(sql, conn)
        db_df.to_csv(dist, index=False)
        conn.close()

"""
Example:
#database handler
# metrics = Metrics(
#     f'//apps/{opt.app}',
#     'metrics',
#     opt.metrics_tbl if opt.metrics_tbl else f'{CONFIG.logs.model.name}__{CONFIG.logs.data.name}__{getTimeHR(split="", dateFormat="%YY%mM%dD", timeFormat="%HH%MM%SS")}',
#     CONFIG.logs.metrics
# )
Example:
    import database as my_db
    metrics = Metrics(my_db, 'my-db', ['loss', 'val_loss'])
    for s, m in zip([-3, 2, 21, 4, 232, 44, 23], [(6.12,3.21), (0.01,1.78), (6.3,1.5), (2.1,5.2), (12.36, 41.25), (54.58, 66.225), (22.456, 6.3)]):
        metrics.add({
            'step': s,
            'loss': m[0],
            'val_loss': m[1]
        })
    metrics.add({
        'FID': 66, # 'FiD': 66 # 'fid': 66 # all of this is ok.
        'step': 6,
        'loss': 36,
        'val_loss': 37
    })
    print(metrics.select('select * from metric'))
    print(metrics.delete())
    print(metrics.db_path)
    metrics.to_csv()
    metrics.to_csv(dist='/home/test.csv', sql='select step, loss from metric where step > 0')
"""

if __name__ == '__main__':
    m = Metrics(
        '/media/alihejrati/3E3009073008C83B/Code/Genie-ML/logs/vqgan/firstStage-IdrId/garbage',
        'metrics',
        'tbl_test2',
        ['loss', 'val_loss', 'x'] # step, timestamp are automatically considired and timestamp is automatically filled.
    )
    m.add({
        'val_loss': 2,
        'step': 6, 
        'x': -23.5,
        'x2': -13.5,
        'loss': 14.261
    })
    m.add({
        'x': 11,
        'step': 24, 
        'loss': -14.372,
        'val_loss': 6,
    })
    m.add({
        'x': -4.5,
        'step': 2, 
        'val_loss': 4.261,
        'loss': 2.2361,
    })
    m.add({
        'x': 13.5,
        'val_loss': 12,
        'loss': 114.261,
        'step': -24,
    })