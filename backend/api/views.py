import numpy as np
from django.http import Http404
from datetime import datetime
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import WaterInfo, WaterPred
from .serializers import WaterInfoDataSerializer, WaterInfoTimeSerializer, WaterPredDataSerializer
from .utils import predict, WaterInfoDependenceParser

from lstm.models import PredictDependence
from lstm.serializers import PredictDependenceSerializer


class TaskTest(APIView):
    def get(self, request):
        from .tasks import update
        update.delay('http://127.0.0.1:5000/api/weather', 'http://127.0.0.1:5000/api/waterinfo')
        return Response({'status': 'ok'})


class Water_Info(APIView):
    def get(self, request):
        station_id = request.query_params.get("station_id")
        time_length = request.query_params.get("length", default=18)
        if not station_id:
            return Response({"detail": "prarms station_id required"}, status=status.HTTP_400_BAD_REQUEST)
        waterinfo = WaterInfo.objects.filter(station=station_id).order_by('-times')[:time_length][::-1]
        if not waterinfo:
            return Response({"detail": "station info notfound"}, status=status.HTTP_400_BAD_REQUEST)
        waterinfo_data = WaterInfoDataSerializer(waterinfo, many=True)

        waterinfo_time = WaterInfoTimeSerializer(waterinfo, many=True)
        times = []
        for time in waterinfo_time.data:
            times.extend(time.values())
        response_data = {
            "times": times,
            "data": waterinfo_data.data,
        }
        waterpred = WaterPred.objects.filter(times=times[-1], station=station_id)
        if waterpred.exists():
            waterpred_data = WaterPredDataSerializer(waterpred, many=True)
            response_data["pred"] = list(waterpred_data.data[0].values())

        return Response(response_data, status=status.HTTP_200_OK)
