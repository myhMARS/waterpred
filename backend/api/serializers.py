from rest_framework import serializers

from .models import WaterInfo, WaterPred, StationInfo, Statistics, WarningNotice, AreaWeatherInfo


class WarnCancelDataSerializer(serializers.Serializer):
    station_id = serializers.CharField()
    detail = serializers.CharField(allow_blank=True)


class AreaInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = AreaWeatherInfo
        fields = ['city', 'county', 'times', 'temperature', 'humidity', 'winddirection', 'windpower']


class WaterInfoDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = WaterInfo
        fields = ["station", "times", "rains", "waterlevels"]


class WaterInfoTimeSerializer(serializers.ModelSerializer):
    class Meta:
        model = WaterInfo
        fields = ["times"]


class WaterPredDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = WaterPred
        fields = ["waterlevel1", "waterlevel2", "waterlevel3", "waterlevel4", "waterlevel5", "waterlevel6"]


class StationInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = StationInfo
        fields = ["id", "name", "city", "county", "flood_limit", "guaranteed", "warning"]


class StatisticsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Statistics
        fields = ['year', 'month', 'day', 'station_id']


class WarngingsSerializer(serializers.ModelSerializer):
    max_level = serializers.DecimalField(max_digits=10, decimal_places=2)
    station_name = serializers.CharField(source='station.name', read_only=True)

    class Meta:
        model = WarningNotice
        fields = ['station', 'station_name', 'noticetype', 'max_level', 'isSuccess', 'noticetime', 'isCanceled',
                  'executor', 'canceltime']
