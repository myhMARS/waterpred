import asyncio
import time
from datetime import datetime, timedelta

import httpx
from celery import shared_task
from django.utils import timezone
from django.db.models import Max

from backend.logging_config import get_task_logger
from lstm.models import LSTMModels
from .models import WaterInfo, WaterPred, StationInfo, AreaWeatherInfo, WarningNotice, Statistics
from .serializers import WaterInfoDataSerializer
from .utils import predict, sendWarning

logger = get_task_logger()


def insert_waring_statistics(times: datetime, station_id: str):
    Statistics.objects.update_or_create(
        year=times.year,
        month=times.month,
        day=times.day,
        station_id=station_id,
    )


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
    queryset = LSTMModels.objects.filter(station_id=station_id, is_activate=True)
    if queryset.exists():
        update_predict(station_id)
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
    queryset = WaterInfo.objects.filter(station_id=station_id).order_by('-times').all()[:12][::-1]
    waterinfos = WaterInfoDataSerializer(queryset, many=True)
    times = [item['times'] for item in waterinfos.data]
    current_time = waterinfos.data[-1]['times']
    current_time = datetime.fromisoformat(current_time)
    predict_flag = True
    current_waterlevel = [waterinfos.data[-1]['waterlevels']]
    output = []
    times_check = [datetime.fromisoformat(time_str) for time_str in times]
    for i in range(1, len(times)):
        time_diff = times_check[i] - times_check[i - 1]
        if time_diff != timedelta(hours=1):
            predict_flag = False
            break
    if predict_flag and len(queryset) >= 12:
        data = [float(item.waterlevels) for item in queryset]
        output = predict(station_id, data).tolist()[0]

        fields = {f"waterlevel{i + 1}": level for i, level in enumerate(output)}
        WaterPred.objects.update_or_create(
            times=current_time,
            station=StationInfo.objects.filter(id=station_id).first(),
            **fields,
        )
        logger.debug(f'插入预测数据{station_id}-{times}')
    output = current_waterlevel + output
    check_warning(timezone.make_naive(current_time), station_id, output)
    return 0


def check_warning(times, station_id, waterlevels):
    queryset = StationInfo.objects.get(id=station_id)
    station_name = queryset.name
    limits = [queryset.flood_limit, queryset.warning, queryset.guaranteed]
    types = ['汛限水位', '警戒水位', '保证水位']
    res = [0] * len(limits)
    for waterlevel in waterlevels:
        for i in range(len(limits)):
            if limits[i] and waterlevel >= limits[i]:
                res[i] += 1
    if any(res):
        index = 0
        insert_waring_statistics(times, station_id)
        for i in range(len(res) - 1, -1, -1):
            if res[i] != 0:
                index = i
                break
        logger.debug('进入了一次校验')
        try:
            queryset = WarningNotice.objects.filter(station_id=station_id, isCanceled=False)
            if queryset.exists():
                current_max = max(waterlevels)
                existing_max = queryset.aggregate(Max('max_level'))['max_level__max']
                if current_max > existing_max:
                    queryset.update(max_level=max(waterlevels))
                    queryset.update(isSuccess=True)
            else:
                obj = WarningNotice.objects.create(
                    station_id=station_id,
                    noticetime=timezone.make_aware(times),
                    noticetype=types[index],
                    max_level=max(waterlevels),
                )
                sendWarning(times, station_name, waterlevels, types[index], limits[index])
                obj.isSuccess = True
                obj.save()
                logger.info('成功发送了一次邮件')
        except Exception as e:
            logger.error(f'error info: {e}, station_id: {station_id}, times: {times} , noticetype: {types[index]}')


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

    net_time = time.time()
    for data in result:
        if data['type'] == 'station':
            insert_water_data(data)
        elif data['type'] == 'area':
            insert_weather_data(data)
        else:
            logger.error(f'Unknown type data: {data}')
    sql_time = time.time()
    logger.debug(
        f'Request CostTime: {net_time - strat_time} '
        f'/ SQL CostTime: {sql_time - net_time} '
    )

    return 1


@shared_task
def WarngingNoticeManage():
    print('checkchek')
