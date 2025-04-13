import asyncio
import time
from datetime import datetime

import httpx
from celery import shared_task
from django.utils import timezone

from backend.logging_config import get_task_logger
from lstm.models import LSTMModels, PredictDependence
from lstm.serializers import PredictDependenceSerializer
from .models import WaterInfo, WaterPred, StationInfo, AreaWeatherInfo
from .utils import predict, WaterInfoDependenceParser

logger = get_task_logger()


async def fetch_data(keys_dict_list, max_concurrency=50):
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded_fetch(client, keys_dic):
        async with semaphore:
            try:
                response = await client.get(**keys_dic)
                response.raise_for_status()
                return response
            except Exception as e:
                logger.error(f"请求失败: {keys_dic.get('url')} - {str(e)}")
                return None

    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [bounded_fetch(client, keys_dic) for keys_dic in keys_dict_list]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for response in responses:
            # 处理异常情况
            if isinstance(response, Exception):
                logger.error(f"请求失败: {type(response).__name__}: {response}")
                continue

            # 确保是 httpx.Response 对象
            if not isinstance(response, httpx.Response):
                logger.error(f"无效的响应类型: {type(response)}")
                continue

            try:
                response.raise_for_status()
                data = response.json()
                data['type'] = 'station' if data.get('station_id') else 'area'
                results.append(data)
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP 错误 [{e.response.status_code}]: {e.request.url}")
            except ValueError as e:
                logger.error(f"JSON 解析失败: {e}")
            except Exception as e:
                logger.error(f"未知错误: {e}")
        return results


def insert_water_data(data):
    times = timezone.make_aware(datetime.strptime(data['times'], "%Y-%m-%d %H:%M:%S"))
    station_id = data['station_id']
    rains = float(data['rains']) if data['rains'] is not None else None
    waterlevels = float(data['waterlevels']) if data['waterlevels'] is not None else None

    WaterInfo.objects.update_or_create(
        times=times,
        station_id=station_id,
        rains=rains,
        waterlevels=waterlevels
    )
    logger.debug(f"插入waterinfo {station_id}-{times}")


def insert_weather_data(data):
    times = timezone.make_aware(datetime.strptime(data['times'], "%Y-%m-%d %H:%M:%S"))
    city = data['city']
    county = data['county']
    humidity = float(data['humidity']) if data['humidity'] is not None else None
    temperature = float(data['temperature']) if data['temperature'] is not None else None
    winddirection = data['winddirection'] if data['winddirection'] is not None else None
    windpower = float(data['windpower']) if data['windpower'] is not None else None

    AreaWeatherInfo.objects.update_or_create(
        times=times,
        city=city,
        county=county,
        humidity=humidity,
        temperature=temperature,
        winddirection=winddirection,
        windpower=windpower,
    )

    logger.debug(f"插入weather {(city, county)}-{times}")


def update_predict(station_id):
    queryset = PredictDependence.objects.get(station_id=station_id)
    dependence = PredictDependenceSerializer(queryset)
    dependence_parser = WaterInfoDependenceParser(dependence.data, 12)
    predict_availbel, data = dependence_parser.get_dataset()
    times = dependence_parser.times
    if predict_availbel:
        output = predict(station_id, data).tolist()[0]

        fields = {f"waterlevel{i + 1}": level for i, level in enumerate(output)}
        WaterPred.objects.update_or_create(
            times=times,
            station=StationInfo.objects.filter(id=station_id).first(),
            **fields,
        )
        logger.debug(f'插入预测数据{station_id}-{times}')
        return 1
    return 0


@shared_task
def update(weather_url, water_url):
    """水文数据获取任务
    这里需要根据实际 api 需求进行构造
    任务计划运行在 Django 后台进行配置
    """
    queryset = StationInfo.objects.all()
    request_list = []
    area_set = set()
    for obj in queryset:
        request_list.append({
            "url": water_url,
            "params": {
                "station_id": obj.id,
            }
        })
        area_set.add((obj.city, obj.county))

    for area in area_set:
        city, county = area
        request_list.append({
            "url": weather_url,
            "params": {
                "city": city,
                "county": county
            }
        })

    strat_time = time.time()

    result = asyncio.run(fetch_data(request_list))
    logger.INFO('-' * 256)
    net_time = time.time()
    for data in result:
        if data['type'] == 'station':
            insert_water_data(data)
        elif data['type'] == 'area':
            insert_weather_data(data)
        else:
            logger.error(f'Unknown type data: {data}')
    sql_time = time.time()

    running_model_list = LSTMModels.objects.filter(is_activate=True).values_list('station_id', flat=True)
    for station in running_model_list:
        update_predict(station)
    lstm_time = time.time()
    logger.INFO(
        f'Request CostTime: {net_time - strat_time} '
        f'/ SQL CostTime: {sql_time - net_time} '
        f'/ LSTM CostTime: {lstm_time - sql_time}'
    )
    logger.INFO('-' * 256)

    return 1
