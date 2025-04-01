import re

from rest_framework import serializers

from api.models import WaterInfo
from lstm.models import LSTMModels, ScalerPT


class WaterInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = WaterInfo
        fields = ["temperature", "humidity", "rains", "rains63000100",
                  "windpower", "waterlevels63000100", "waterlevels63000120", "waterlevels"]


class ModelChangeSerializer(serializers.Serializer):
    md5 = serializers.CharField(
        max_length=32,
        min_length=32,
        help_text="MD5哈希值，应为32个字符的十六进制字符串"
    )

    def validate_md5(self, value):
        """验证MD5值的格式是否正确"""
        if not re.match(r'^[a-fA-F0-9]{32}$', value):
            raise serializers.ValidationError("无效的MD5格式，必须是32位十六进制字符")
        return value.lower()


class ModelListSerializer(serializers.ModelSerializer):
    class Meta:
        model = LSTMModels
        fields = ['date', 'name', 'rmse', 'md5']
