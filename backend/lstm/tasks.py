import os
import hashlib
import logging
import shutil
from datetime import datetime
from celery import shared_task
from django.core.files import File
from .train_src.train import train_lstm_model
from api.models import WaterInfo
from .serializers import WaterInfoSerializer
from .models import LSTMModels, ScalerPT

def get_file_md5(filepath):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@shared_task
def start_train():
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_size = 8
    hidden_size = 64
    output_size = 6

    queryset = WaterInfo.objects.order_by("-times")
    data = WaterInfoSerializer(queryset, many=True).data
    if len(data) < 6000:
        logging.info("data not enough")
        return 0
    rmse = train_lstm_model(data, input_size, hidden_size, output_size, time_stamp)
    if rmse:
        lstm_filename = f'waterlevel_model_{input_size}_{hidden_size}_{output_size}_{time_stamp}.pt'
        md5 = get_file_md5(f'temp/{lstm_filename}')
        with open(f'temp/{lstm_filename}', 'rb') as f:
            lstm_instance = LSTMModels.objects.create(
                name=lstm_filename,
                file=File(f, name=lstm_filename),
                rmse=rmse,
                md5=md5
            )
        with open(f'temp/scaler_{time_stamp}.pkl', 'rb') as f:
            ScalerPT.objects.create(
                name=f'scaler_{time_stamp}',
                file=File(f, name=f'scaler_{time_stamp}.pkl'),
                lstm_model=lstm_instance
            )
        temp_dir = "temp/"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
        return 1
    return 0
