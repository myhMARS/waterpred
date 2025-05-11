from django.db import models
from django.contrib.auth.models import User
from shortuuidfield import ShortUUIDField


class Profile(models.Model):
    uid = ShortUUIDField(max_length=32, primary_key=True, unique=True, verbose_name="uid")
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=10, verbose_name="姓名")
    phone = models.CharField(max_length=20, verbose_name='手机号', db_index=True)
    email = models.EmailField(max_length=20, verbose_name='邮箱', db_index=True)
    manager = models.CharField(max_length=20, verbose_name='职务', blank=True, null=True, db_index=True)
    location = models.CharField(max_length=20, verbose_name='管理地区', blank=True, null=True, db_index=True)

    class Meta:
        db_table = 'profile'
        verbose_name = '用户信息'
        verbose_name_plural = '用户信息'
