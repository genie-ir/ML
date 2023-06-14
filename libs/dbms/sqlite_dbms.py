import sqlite3
from loguru import logger
from libs.basicIO import pathBIO

class SqliteDBMS:
    def __init__(self, fullpath: str):
        self.fullpath = pathBIO(fullpath)
        if not self.fullpath.endswith('.db'):
            self.fullpath = self.fullpath + '.db' 
        self.con = sqlite3.connect(self.fullpath)
        self.cursor = self.con.cursor()

    def select(self, Q: str):
        self.cursor.execute(Q)
        res = self.cursor.fetchall()
        return res
    
    def get_colnames(self, table_name):
        res = self.con.execute('select * from {}'.format(table_name))
        res = res.description
        res = [r[0] for r in res]
        return res


    def get_tables(self):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        res = self.cursor.fetchall()
        return [r[0] for r in res]

    # def to_csv(self, sql=None, dist=None):
    #     sql = sql if sql else f'SELECT * FROM {self.tableName}'
    #     dist = self.db_path.replace('.db', '.csv') if dist is None else dist
    #     conn = sqlite3.connect(self.db_path, isolation_level=None, detect_types=sqlite3.PARSE_COLNAMES)
    #     db_df = pd.read_sql_query(sql, conn)
    #     db_df.to_csv(dist, index=False)
    #     conn.close()