from datetime import datetime

import numpy as np
import pandas as pd
import requests
from celery import shared_task
from django.utils import timezone

from .models import WaterInfo, WaterPred, StationInfo
from .serializers import WaterInfoDataSerializer
from .utils import predict


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
            raise Exception('api未获取到数据')
        else:
            times = timezone.make_aware(datetime.strptime(data[0], "%Y-%m-%d %H:%M:%S"))
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

            waterinfo = WaterInfo.objects.order_by("-times")[:18][::-1]
            waterinfo_data = WaterInfoDataSerializer(waterinfo, many=True)
            if len(waterinfo_data.data) >= 12:
                data = np.array(pd.DataFrame(waterinfo_data.data).tail(12))
                output = predict(data).tolist()[0]

                fields = {f"waterlevel{i + 1}": level for i, level in enumerate(output)}
                WaterPred.objects.create(
                    times=times,
                    station=StationInfo.objects.filter(id="63000200").first(),
                    **fields,
                )
            return 1

    except Exception as e:
        print(e)
        return 0


@shared_task
def send_warning_email():
    pass
