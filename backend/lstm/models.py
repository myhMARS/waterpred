from django.db import models


# Create your models here.
class LSTMModels(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    date = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='LSTM_Models/')
    rmse = models.FloatField()
    md5 = models.CharField(max_length=32, unique=True)  # 确保唯一性


class ScalerPT(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    date = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='ScalerPT_Models/')
    lstm_model = models.ForeignKey(  # 直接关联到LSTMModels
        LSTMModels,
        on_delete=models.CASCADE,  # 级联删除（可选：PROTECT或SET_NULL）
        related_name='scalers',  # 反向查询名称
        db_column='lstm_md5',  # 数据库中仍保留lstm_md5字段名
        to_field='md5'  # 关联到LSTMModels的md5字段
    )


class TrainResult(models.Model):
    id = models.AutoField(primary_key=True)
    image = models.FileField(upload_to='TrainResult/')
    lstm_model = models.ForeignKey(  # 直接关联到LSTMModels
        LSTMModels,
        on_delete=models.CASCADE,  # 级联删除（可选：PROTECT或SET_NULL）
        related_name='train_res',  # 反向查询名称
        db_column='lstm_md5',  # 数据库中仍保留lstm_md5字段名
        to_field='md5'  # 关联到LSTMModels的md5字段
    )
