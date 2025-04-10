import hashlib

from django.core.files import File
from django.db import migrations


def get_file_md5(filepath):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def add_initial_data(apps, schema_editor):
    LSTM_Model = apps.get_model('lstm', 'LSTMModels')
    Scaler_Model = apps.get_model('lstm', 'ScalerPT')

    model_path = 'media/init/waterlevel_model_8_64_6_init.pt'
    md5 = get_file_md5(model_path)
    scaler_path = 'media/init/scaler_init.pkl'
    with open(model_path, 'rb') as f:
        lstm_instance = LSTM_Model.objects.create(
            name='waterlevel_model_8_64_6_init',
            file=File(f, name='waterlevel_model_8_64_6_init.pt'),
            rmse=0.015890525424569143,
            md5=md5,
            is_activate=True,
        )

    with open(scaler_path, 'rb') as f:
        Scaler_Model.objects.create(
            name='scaler_init',
            file=File(f, name='scaler_init.pkl'),
            lstm_model=lstm_instance
        )


class Migration(migrations.Migration):
    dependencies = [
        ('lstm', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(add_initial_data),
    ]
