from rest_framework import serializers

from .models import WaterInfo, WaterPred, StationInfo


class PredictInputSerializer(serializers.Serializer):
    features = serializers.ListField(
        child=serializers.ListField(
            child=serializers.FloatField(),
            min_length=8,  # 每个样本至少有 8 个特征
            max_length=8,
            error_messages={
                "max_length": "每个时间步至少具有8个特征值",
                "min_length": "每个时间步至少具有8个特征值",
            }
        ),
        min_length=12,  # 至少传入 1 个样本
        max_length=12,
        error_messages={
            "max_length": "需要传入12个时间步的特征",
            "min_length": "需要传入12个时间步的特征",
        }
    )


class PredictOutputSerializer(serializers.Serializer):
    prediction = serializers.ListField(
        child=serializers.ListField(
            child=serializers.FloatField(),
            min_length=6,  # 每个样本至少有 8 个特征
            max_length=6
        ),
        min_length=1  # 至少传入 1 个样本
    )


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
