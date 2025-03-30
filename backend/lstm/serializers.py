from rest_framework import serializers

from api.models import WaterInfo


class WaterInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = WaterInfo
        fields = ["temperature", "humidity", "rains", "rains63000100",
                  "windpower", "waterlevels63000100", "waterlevels63000120", "waterlevels"]
