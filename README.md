# LSTM 深度学习洪水预测系统

### 1. 项目运行（开发服务器）

配置本地.env文件
后端前端部分，分别参照各个.env.example文件

```bash
git clone https://github.com/myhMARS/waterpred.git
```
安装项目环境
```text
python>=3.12, nodejs>=18, mysql>=8.0, redis
```

python 依赖包
```bash
pip install -r requirements.txt
```
nodejs包
```bash
cd frontend
npm install
```

参照.env.example配置.env文件
初始化数据库
```bash
cd backend
python manage mikemirgations
python manage migrate
```
启动数据源数据库
```bash
cd waterinfo_output
python output_api.py
```
启动后端服务
```bash
cd backend
python manage runserver
celery -A backend worker -l INFO -P threads  # celery异步工作池 windows版不支持多进程
celery -A backend beat -l INFO  # 定时器
```
启动前端服务
```bash
cd frontend
npm run dev
```
访问127.0.0.1:5173即可访问

### 2. 项目部署

#### 1. 使用BT面板初始化 *gunicorn* 配置
添加python网站项目填写相关端口及参数,配置mysql及redis
#### 2. build vue项目并将编译完成后的dist文件夹上传至服务器
```bash
npm run build
```
#### 3. 在backend settings中添加配置
```text
DEBUG = False //修改该项
STATICFILES_DIRS = ["...在这里填写项目路径/waterpred/dist/"]
STATIC_ROOT = os.path.join(BASE_DIR,'collected_static')
```
收集静态文件
```bash
cd backend
python manage.py collectstatic
```
初始化服务器(确认.env文件已完成相关配置)
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser # 创建管理员账号
```
初始模型训练(将后端.env文件复制至waterinfo_output文件夹下)
```bash
cd waterinfo_output
python init_db.py
cd ../backend
python manage.py runserver
```
```bash
celery -A backend worker -l info -P threads
```
手动触发lstm/train接口开始训练
训练完成后删除部分数据表中的数据并在django后台启用训练完成的模型
```sql
delete from api_areaweatherinfo;
delete from api_waterinfo;
delete from api_waterpred;
delete from api_warningnotice;
delete from api_statistics;
```
停止django开发服务器和celery
#### 4. 配置 nginx 及其他相关服务
配置nginx，配置文件示例如下
```
server
{
    listen 80;
    server_name xxx; //此处填写服务器ip或者域名
    index index.html;
    root /www/wwwroot/project/waterpred/backend;
    
    //相关链接配置
    location /media/TrainResult {
        alias /www/wwwroot/project/waterpred/backend/media/TrainResult;
    }
    
    location /static {
        alias /www/wwwroot/project/waterpred/backend/collected_static;
    }
    
    location /assets{
        alias /www/wwwroot/project/waterpred/backend/collected_static/assets;
    }
    
    location /images {
        alias /www/wwwroot/project/waterpred/backend/collected_static/images;
    }
    location /geo {
        alias /www/wwwroot/project/waterpred/backend/collected_static/geo;
    }
    
    location ~ (^/api|^/lstm|^/auth|^/accounts|^/admin) {
        proxy_set_header Host $host;
        proxy_pass http://127.0.0.1:9000;
    }
    
    location / {
        try_files /collected_static/index.html =404;
    }
}
```
生成supervisor配置
```bash
echo_supervisord_conf > supervisord.conf
```
supervisor配置最后添加示例(gunicorn的相关配置文件已由bt面板自动生成)
```text
[program:gunicorn]
command=/www/wwwroot/project/waterpred/.venv/bin/gunicorn backend.wsgi:application -c gunicorn_conf.py
directory=/www/wwwroot/project/waterpred/backend
autostart=true
autorestart=true
stderr_logfile=/www/wwwroot/project/waterpred/log/gunicorn.err.log
stdout_logfile=/www/wwwroot/project/waterpred/log/gunicorn.out.log

[program:celery]
command=/www/wwwroot/project/waterpred/.venv/bin/celery -A backend worker -l info -P threads
directory=/www/wwwroot/project/waterpred/backend
autostart=true
autorestart=true
stderr_logfile=/www/wwwroot/project/waterpred/log/celery.err.log
stdout_logfile=/www/wwwroot/project/waterpred/log/celery.out.log

[program:beat]
command=/www/wwwroot/project/waterpred/.venv/bin/celery -A backend beat -l info
directory=/www/wwwroot/project/waterpred/backend
autostart=true
autorestart=true
stderr_logfile=/www/wwwroot/project/waterpred/log/beat.err.log
stdout_logfile=/www/wwwroot/project/waterpred/log/beat.out.log

[program:datasource]
command=/www/wwwroot/project/waterpred/.venv/bin/python output_api.py
directory=/www/wwwroot/project/waterpred/waterinfo_output
autostart=true
autorestart=true
stderr_logfile=/www/wwwroot/project/waterpred/log/flask.err.log
stdout_logfile=/www/wwwroot/project/waterpred/log/flask.out.log 
```
启动后端服务
```bash
supervisord -c supervisord.conf
```


