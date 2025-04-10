from django.db import models
from django.contrib.auth.models import User


class StationInfo(models.Model):
    id = models.CharField(primary_key=True, max_length=10, verbose_name="站点编号")
    name = models.CharField(max_length=20, verbose_name="站名")
    city = models.CharField(max_length=20, verbose_name="城市")
    county = models.CharField(max_length=20, verbose_name="区/县")
    flood_limit = models.FloatField(verbose_name="汛限水位", null=True)
    guaranteed = models.FloatField(verbose_name="保证水位", null=True)
    warning = models.FloatField(verbose_name="警告水位", null=True)


class WaterInfo(models.Model):
    id = models.AutoField(primary_key=True)
    times = models.DateTimeField(verbose_name='上报时间')
    temperature = models.FloatField(verbose_name='气温')
    humidity = models.FloatField(verbose_name='湿度')
    winddirection = models.CharField(verbose_name='风向', max_length=4)
    windpower = models.FloatField(verbose_name='风力')
    rains = models.FloatField(verbose_name='桥东村降水量')
    waterlevels63000120 = models.FloatField(verbose_name='东坑溪水位')
    rains63000100 = models.FloatField(verbose_name="库区降水")
    waterlevels63000100 = models.FloatField(verbose_name="库区水位")
    waterlevels = models.FloatField(verbose_name="桥东村水位")


class WaterPred(models.Model):
    id = models.AutoField(primary_key=True)
    times = models.DateTimeField(verbose_name='预测起始时间')
    station = models.ForeignKey(
        StationInfo,
        on_delete=models.CASCADE,
        related_name='water_pred',
        db_column='station_id',
        to_field='id'
    )
    waterlevel1 = models.FloatField(verbose_name='time+1')
    waterlevel2 = models.FloatField(verbose_name='time+2')
    waterlevel3 = models.FloatField(verbose_name='time+3')
    waterlevel4 = models.FloatField(verbose_name='time+4')
    waterlevel5 = models.FloatField(verbose_name='time+5')
    waterlevel6 = models.FloatField(verbose_name='time+6')


class WarningNotice(models.Model):
    id = models.AutoField(primary_key=True)
    station = models.ForeignKey(
        StationInfo,
        on_delete=models.CASCADE,
        related_name='warning_notice',
        db_column='station_id',
        to_field='id'
    )
    noticetime = models.DateTimeField(auto_now_add=True, verbose_name="通知时间")
    isSuccess = models.BooleanField(default=True, verbose_name="通知发送状态")
    isCanceled = models.BooleanField(default=False, verbose_name="通知确认状态")
    executor = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='executor',
        db_column='executor',
        to_field='username'
    )
    canceltime = models.DateTimeField(verbose_name="确认时间")
