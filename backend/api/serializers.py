from rest_framework import serializers

from .models import WaterInfo, WaterPred, StationInfo, Statistics


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
