import logging
from datetime import datetime

import requests
from celery import shared_task
from django.utils import timezone

from lstm.models import LSTMModels, PredictDependence
from lstm.serializers import PredictDependenceSerializer
from .models import WaterInfo, WaterPred, StationInfo, AreaWeatherInfo
from .utils import predict, WaterInfoDependenceParser

from backend.logging_config import get_task_logger

logger = get_task_logger()


def update_waterinfo(url, station):
    params = {
        'station_id': station
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data:
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
            logger.info(f"插入waterinfo {station_id}-{times}")
        return 1
    except requests.exceptions.HTTPError as e:
        logger.error(e)


def update_weather(url, area):
    params = {
        'city': area[0],
        'county': area[1]
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data:
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

            logger.info(f"插入weather {(city, county)}-{times}")
        return 1
    except requests.exceptions.HTTPError as e:
        logger.error(e)


@shared_task
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
        return 1
    return 0


@shared_task
def update(weather_url, water_url):
    """水文数据获取任务
    这里需要根据实际 api 需求进行构造
    任务计划运行在 Django 后台进行配置
    """
    queryset = StationInfo.objects.all()
    station_set = []
    area_set = set()
    for obj in queryset:
        station_set.append(obj.id)
        area_set.add((obj.city, obj.county))

    for station in station_set:
        update_waterinfo(water_url, station)

    for area in area_set:
        update_weather(weather_url, area)

    running_model_list = LSTMModels.objects.filter(is_activate=True).values_list('station_id', flat=True)
    for station in running_model_list:
        update_predict(station)

    return 1
