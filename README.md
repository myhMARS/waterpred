# LSTM 深度学习洪水预测系统

## 1. 目录结构

```bash
├── README.md                               # 项目说明文件
├── backend                                 # 后端 Django 项目
│   ├── accounts                            # 用户账户管理模块
│   │   ├── __init__.py                     # 包初始化文件
│   │   ├── admin.py                        # 注册模型到 Django 管理后台
│   │   ├── apps.py                         # 应用配置
│   │   ├── migrations/                     # 数据库迁移文件
│   │   ├── models.py                       # 定义与用户相关的数据库模型
│   │   ├── permissions.py                  # 自定义权限逻辑
│   │   ├── serializers.py                  # 数据序列化规则
│   │   ├── tests.py                        # 与账户相关的单元测试
│   │   ├── urls.py                         # 账户相关的 URL 路由
│   │   └── views.py                        # 账户相关的视图函数
│   ├── api                                 # API 模块
│   │   ├── __init__.py                     # 包初始化文件
│   │   ├── admin.py                        # 注册 API 模型到 Django 后台
│   │   ├── apps.py                         # 应用配置
│   │   ├── middleware.py                   # 自定义中间件
│   │   ├── migrations/                     # 数据库迁移文件
│   │   ├── models.py                       # API 相关的数据库模型
│   │   ├── serializers.py                  # 数据序列化规则
│   │   ├── tasks.py                        # 异步任务
│   │   ├── tests.py                        # 与 API 相关的单元测试
│   │   ├── urls.py                         # API 路由
│   │   ├── utils.py                        # 工具函数
│   │   └── views.py                        # API 视图函数
│   ├── lstm                                # LSTM 模型模块
│   │   ├── __init__.py                     # 包初始化文件
│   │   ├── admin.py                        # 注册 LSTM 模型到 Django 后台
│   │   ├── apps.py                         # 应用配置
│   │   ├── migrations/                     # 数据库迁移文件
│   │   ├── models.py                       # LSTM 相关数据库模型
│   │   ├── serializers.py                  # LSTM 模型的序列化器
│   │   ├── tasks.py                        # LSTM 相关异步任务
│   │   ├── train_src/                      # 模型训练源码
│   │   │   ├── model_net/                  # 神经网络架构
│   │   │   │   ├── data_loader.py          # 数据加载器
│   │   │   │   └── net.py                  # 网络模型定义
│   │   │   └── train.py                    # 模型训练脚本
│   │   ├── urls.py                         # LSTM 模型相关路由
│   │   ├── utils.py                        # LSTM 工具函数
│   │   └── views.py                        # LSTM 视图函数
│   ├── backend                             # Django 后端配置
│   │   ├── __init__.py                     # 包初始化文件
│   │   ├── asgi.py                         # 配置 ASGI 服务器
│   │   ├── celery.py                       # Celery 配置文件
│   │   ├── logging_config.py               # 日志配置
│   │   ├── settings.py                     # 项目核心配置
│   │   ├── urls.py                         # 全局 URL 配置
│   │   └── wsgi.py                         # 配置 WSGI 服务器
│   ├── locale                              # 本地化文件
│   │   └── zh_Hans/LC_MESSAGES             # 中文简体翻译文件
│   │       ├── django.mo                   # 编译后的翻译文件
│   │       └── django.po                   # 翻译源文件
├── frontend                                # 前端 Vue.js 项目
│   ├── README.md                           # 前端项目说明文件
│   ├── index.html                          # HTML 模板文件
│   ├── jsconfig.json                       # JavaScript 项目的配置文件
│   ├── package-lock.json                   # 前端依赖锁定文件
│   ├── package.json                        # 前端依赖配置文件
│   ├── public/                             # 公共文件
│   │   ├── favicon.ico                     # 网站图标
│   │   └── images/                         # 存放图片
│   │       └── logo/                       # Logo 图标
│   │           └── auth-logo.svg           # 认证页面 Logo
│   ├── src/                                # 源代码
│   │   ├── App.vue                         # Vue 应用根组件
│   │   ├── assets/                         # 静态资源
│   │   │   ├── css/                        # CSS 文件
│   │   │   │   └── main.css                # 主要样式文件
│   │   │   ├── img/                        # 图片资源
│   │   │   │   ├── Aurora.png              # 图片示例
│   │   │   │   └── Eternity.png            # 图片示例
│   │   │   └── logo.svg                    # Logo 图标
│   │   ├── components/                     # Vue 组件
│   │   │   ├── Header.vue                  # 页头组件
│   │   │   ├── common/                     # 公共组件
│   │   │   │   ├── CommonGridShape.vue     # 网格形状组件
│   │   │   │   └── ComponentCard.vue       # 组件卡片
│   │   │   ├── layout/                     # 布局组件
│   │   │   │   └── AdminLayout.vue         # 管理页面布局
│   │   │   ├── profile/                    # 用户个人资料组件
│   │   │   │   ├── AddressCard.vue         # 地址卡片组件
│   │   │   │   └── ProfileCard.vue         # 个人资料卡片组件
│   │   │   ├── ui/                         # UI 组件
│   │   │   │   └── notification/           # 通知组件
│   │   │   │       └── Notification.vue    # 通知显示组件
│   │   ├── router/                         # Vue 路由配置
│   │   │   └── index.js                    # 路由定义
│   │   ├── stores/                         # Vuex 存储
│   │   │   ├── authStatus.js               # 用户认证状态
│   │   │   └── counter.js                  # 计数器状态
│   │   ├── utils/                          # 工具函数
│   │   │   └── vuemessage.js               # Vue 消息显示工具
│   ├── vite.config.js                      # Vite 配置文件
├── model_src                               # 模型训练相关源码
│   ├── SQL/                                # SQL 数据管理
│   │   ├── SqlManager.py                   # SQL 管理脚本
│   │   └── sqlconfig.yaml                  # SQL 配置文件
│   ├── api.py                              # API 交互脚本
│   ├── data.csv                            # 输入数据文件
│   ├── data.ipynb                          # 数据分析 Jupyter 笔记本
│   ├── evaluate.py                         # 模型评估脚本
│   ├── fallraw_63000200.csv                # 特定数据集 CSV 文件
│   ├── model/                              # 模型相关代码
│   │   ├── __init__.py                     # 包初始化文件
│   │   ├── data_loader.py                  # 数据加载器
│   │   └── net.py                          # 神经网络模型
│   ├── predict.py                          # 预测脚本
│   ├── scaler.pkl                          # 归一化器文件
│   ├── train.py                            # 模型训练脚本
│   └── utils.py                            # 工具函数
├── requirements.txt                        # 项目依赖文件
└── waterinfo_output                        # 输出目录
    ├── data.csv                            # 处理后的数据文件
    ├── model                               # 模型输出目录
    └── train.py                            # 训练脚本
```

### 2. 项目运行（开发服务器）
```bash
git clone https://github.com/myhMARS/waterpred.git
```
安装项目环境,python>=3.12, nodejs>=18, mysql>=8.0, redis

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

### 3. 项目部署
 -[ ] TODO


