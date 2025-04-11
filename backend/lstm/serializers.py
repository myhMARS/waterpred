import re

from rest_framework import serializers

from lstm.models import LSTMModels, PredictDependence


class ModelChangeSerializer(serializers.Serializer):
    station_id = serializers.CharField()
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
        fields = ['date', 'name', 'station_id', 'rmse', 'md5', 'is_activate']


class PredictDependenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictDependence
        fields = ["station", "dependence"]


class ModelInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = LSTMModels
        fields = ['date', 'name', 'rmse', 'md5']
