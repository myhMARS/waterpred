import hashlib
import os
import shutil
from datetime import datetime

from celery import shared_task
from django.core.files import File

from .models import LSTMModels, ScalerPT, PredictStations, TrainResult
from api.models import StationInfo, WaterInfo
from .serializers import PredictStationsSerializer
from .train_src.train import train_lstm_model


def get_file_md5(filepath):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@shared_task
def start_train():
    queryset = PredictStations.objects.all()
    stations = PredictStationsSerializer(queryset, many=True)
    for station in stations.data:
        station_id = station["station"]
        dataset = WaterInfo.objects.filter(station_id=station_id).order_by("times").values_list("waterlevels", flat=True)
        train_available = False
        if dataset.exists() and dataset.count() > 5000:
            train_available = True
        if train_available:
            time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_size = 1
            hidden_size = 128
            output_size = 1
            lstm_filename = f'waterlevel_model_{station_id}_{input_size}_{hidden_size}_{output_size}_{time_stamp}.pt'
            rmse = train_lstm_model(dataset, input_size, hidden_size, output_size, time_stamp, lstm_filename)
            if rmse:

                md5 = get_file_md5(f'temp/{lstm_filename}')
                with open(f'temp/{lstm_filename}', 'rb') as f:
                    lstm_instance = LSTMModels.objects.create(
                        name=lstm_filename,
                        file=File(f, name=lstm_filename),
                        input_size=input_size,
                        hidden_size=hidden_size,
                        output_size=output_size,
                        rmse=rmse,
                        md5=md5,
                        station=StationInfo.objects.get(id=station_id),
                    )
                with open(f'temp/scaler_{time_stamp}.pkl', 'rb') as f:
                    ScalerPT.objects.create(
                        name=f'scaler_{time_stamp}',
                        file=File(f, name=f'scaler_{time_stamp}.pkl'),
                        lstm_model=lstm_instance
                    )
                with open(f'temp/train_res_{time_stamp}.svg', 'rb') as f:
                    TrainResult.objects.create(
                        image=File(f, name=f'train_res_{time_stamp}.svg'),
                        lstm_model=lstm_instance
                    )
                temp_dir = "temp/"
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    os.makedirs(temp_dir)
    return 1
