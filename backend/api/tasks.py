from .models import WaterInfo

import requests
from celery import shared_task
from datetime import datetime


@shared_task
def get_water_info(url):
    """水文数据获取任务
    这里需要根据实际 api 需求进行构造
    任务计划运行在 Django 后台进行配置
    """
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        if not data:
            return 0
        else:
            times = datetime.strptime(data[0], "%Y-%m-%d %H:%M:%S")
            temperature = float(data[1])
            humidity = float(data[2])
            winddirection = data[3]
            windpower = float(data[4])
            rains = float(data[5])
            waterlevels63000120 = float(data[6])
            rains63000100 = float(data[7])
            waterlevels63000100 = float(data[8])
            waterlevels = float(data[9])

            WaterInfo.objects.create(
                times=times,
                temperature=temperature,
                humidity=humidity,
                winddirection=winddirection,
                windpower=windpower,
                rains=rains,
                waterlevels63000120=waterlevels63000120,
                rains63000100=rains63000100,
                waterlevels63000100=waterlevels63000100,
                waterlevels=waterlevels,
            )
            return 1
    except:
        return 0
