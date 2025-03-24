import os
import pymysql
import yaml
from dotenv import load_dotenv

load_dotenv()

class MysqlConnecter:
    def __init__(self) -> None:
        self.config = dict()
        self.conn = pymysql.Connection()

        self.load_config()
        self.get_db_conn()

        self.cursor = self.conn.cursor()

        self.init_db()

    def load_config(self):
        with open("./SQL/sqlconfig.yaml", "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        for key, value in config["database"].items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                config["database"][key] = os.getenv(env_var,"")
        self.config = config


    def get_db_conn(self):
        conn = pymysql.connect(
            host=self.config["database"]["host"],
            port=int(self.config["database"]["port"]),
            user=self.config["database"]["user"],
            password=self.config["database"]["pwd"],
            db='waterpred',
            charset='utf8mb4'
        )
        self.conn = conn


    def init_db(self):
        self.create_tables()

    def create_tables(self):
        for table_name, table_info in self.config["tables"].items():
            columns = ",".join([f"{col} {data_type}" for col, data_type in table_info["columns"].items()])
            print(columns)
            sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
            self.cursor.execute(sql)
        self.conn.commit()

