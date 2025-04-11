from django.db import models
# from api.models import StationInfo  # feature


class LSTMModels(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    date = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='LSTM_Models/')
    rmse = models.FloatField()
    md5 = models.CharField(max_length=32, unique=True)  # 确保唯一性
    is_activate = models.BooleanField(default=False)


class ScalerPT(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    date = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='ScalerPT_Models/')
    lstm_model = models.ForeignKey(
        LSTMModels,
        on_delete=models.CASCADE,
        related_name='scalers',
        db_column='lstm_md5',
        to_field='md5'
    )


class TrainResult(models.Model):
    id = models.AutoField(primary_key=True)
    image = models.FileField(upload_to='TrainResult/')
    lstm_model = models.ForeignKey(
        LSTMModels,
        on_delete=models.CASCADE,
        related_name='train_res',
        db_column='lstm_md5',
        to_field='md5'
    )


""" Feature: 预测依赖链
class PredictDependnece(models.Model):
    station = models.ForeignKey(
        StationInfo,
        on_delete=models.CASCADE,
        related_name='predict_depend',
        db_column='station_id',
        to_field='id',
        primary_key=True
    )
    '''JSON字段构成如下
    {
        data:
            AreaWeatherInfo: [fields],
            Station_id: [fields],
            ...
        target:
            Station_id: [field] 
    }
    '''
    dependence = models.JSONField(verbose_name="训练数据构成")
"""
