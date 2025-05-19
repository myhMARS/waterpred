import pandas as pd
import numpy as np
import pymysql
from pymysql.cursors import DictCursor
from pymysql import Error
from dotenv import load_dotenv
import os

load_dotenv()
df = pd.read_csv("data.csv")
areaweather = df[["times", "temperature", "humidity", "winddirection", "windpower"]].copy()
areaweather["city"] = "杭州"
areaweather["county"] = "临安"

stationinfo = pd.DataFrame({
    "name": ['里畈西坑溪', '里畈水库', '桥东村'],
    "id": ["63000110", "63000100", "63000200"],
    "city": ["杭州"] * 3,
    "county": ["临安"] * 3,
    "flood_limit": [None, 234.73, None],
    "guaranteed": [None, None, 85.66],
    "warning": [None, None, 84.66],
}).replace({np.nan: None})

station63000110 = df[["times", "waterlevels63000120"]].copy()
station63000110.rename(columns={"waterlevels63000120": "waterlevels"}, inplace=True)
station63000110["rains"] = None
station63000110["station_id"] = "63000110"

station63000100 = df[["times", "rains63000100", "waterlevels63000100"]].copy()
station63000100.rename(columns={"rains63000100": "rains", "waterlevels63000100": "waterlevels"}, inplace=True)
station63000100["station_id"] = "63000100"

station63000200 = df[["times", "rains", "waterlevels"]].copy()
station63000200["station_id"] = "63000200"

dependence = pd.DataFrame({
    "station_id": ["63000200", '63000100'],
})

print(areaweather)
print(stationinfo)
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PWD"),
    'database': os.getenv("DB_NAME"),
    'charset': 'utf8mb4',
    'cursorclass': DictCursor
}


def insert_dataframe_to_mysql(df, table_name, batch_size=1000):
    """
    将DataFrame插入MySQL数据库
    参数:
        df: 要插入的DataFrame
        table_name: 目标表名
        batch_size: 分批插入的每批行数
    """
    connection = None
    try:
        # 1. 建立数据库连接
        connection = pymysql.connect(**db_config)

        with connection.cursor() as cursor:
            # 2. 准备SQL语句
            columns = ', '.join([f'`{col}`' for col in df.columns])
            placeholders = ', '.join(['%s'] * len(df.columns))
            sql = f"INSERT INTO `{table_name}` ({columns}) VALUES ({placeholders})"

            # 3. 分批插入数据
            total_rows = len(df)
            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i + batch_size]
                cursor.executemany(sql, batch.values.tolist())
                connection.commit()
                print(f"已插入 {min(i + batch_size, total_rows)}/{total_rows} 行")

        print("数据插入完成！")

    except Error as e:
        print(f"数据库错误: {e}")
        connection.rollback()
    finally:
        if connection:
            connection.close()


# 执行插入
insert_dataframe_to_mysql(areaweather, "api_areaweatherinfo")
insert_dataframe_to_mysql(stationinfo, "api_stationinfo")
insert_dataframe_to_mysql(station63000110, "api_waterinfo")
insert_dataframe_to_mysql(station63000200, "api_waterinfo")
insert_dataframe_to_mysql(station63000100, "api_waterinfo")
insert_dataframe_to_mysql(dependence, "lstm_predictstations")
