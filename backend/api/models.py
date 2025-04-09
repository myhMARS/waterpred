from django.db import models


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
    times = models.DateTimeField(verbose_name='预测起始时间')
    waterlevel1 = models.FloatField(verbose_name='time+1')
    waterlevel2 = models.FloatField(verbose_name='time+2')
    waterlevel3 = models.FloatField(verbose_name='time+3')
    waterlevel4 = models.FloatField(verbose_name='time+4')
    waterlevel5 = models.FloatField(verbose_name='time+5')
    waterlevel6 = models.FloatField(verbose_name='time+6')

