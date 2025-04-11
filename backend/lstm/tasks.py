import hashlib
import logging
import os
import shutil
from datetime import datetime

from celery import shared_task
from django.core.files import File

from .models import LSTMModels, ScalerPT, PredictDependence, TrainResult
from api.models import StationInfo
from .serializers import PredictDependenceSerializer
from .train_src.train import train_lstm_model
from .utils import DependenceParser


def get_file_md5(filepath):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@shared_task
def start_train():
    queryset = PredictDependence.objects.all()
    dependence = PredictDependenceSerializer(queryset, many=True)
    for dependence in dependence.data:
        station = dependence['station']
        dependence_parser = DependenceParser(dependence)
        if dependence_parser.train_availble:
            time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_size = dependence_parser.input_size
            hidden_size = 128
            output_size = 6
            X_list = dependence_parser.X
            y_list = dependence_parser.y
            rmse = train_lstm_model(X_list, y_list, input_size, hidden_size, output_size, time_stamp)
            if rmse:
                lstm_filename = f'waterlevel_model_{input_size}_{hidden_size}_{output_size}_{time_stamp}.pt'
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
                        station=StationInfo.objects.get(id=station),
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
    return 0
