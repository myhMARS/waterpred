import pandas as pd

from SQL.SqlManager import MysqlConnecter

sqlobject = MysqlConnecter()
df = pd.read_csv("fallraw_63000200.csv")

df = df.dropna(how='all', axis=1)
df = df.dropna()

columns = list(df.columns)
columns_str = ", ".join(columns)
placeholders = ", ".join(["%s"] * len(columns))
sql = f"INSERT INTO api_waterinfo ({columns_str}) VALUES ({placeholders})"

data_list = df.values.tolist()
sqlobject.cursor.executemany(sql, data_list)
sqlobject.conn.commit()
