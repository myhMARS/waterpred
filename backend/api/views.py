from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import WaterInfo, WaterPred, StationInfo
from .serializers import WaterInfoDataSerializer, WaterInfoTimeSerializer, WaterPredDataSerializer


class TaskTest(APIView):
    def get(self, request):
        return Response(status=status.HTTP_200_OK)


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


class StationCount(APIView):
    def get(self, request):
        stations = StationInfo.objects.all()
        stations_count = len(stations)
        nomal_count = 0
        for station in stations:
            wateinfo_obj = WaterInfo.objects.filter(station=station.id).order_by('-times').first()
            comparedata = [station.warning, station.guaranteed, station.flood_limit]
            comparedata_filter = [item for item in comparedata if item is not None]
            if comparedata_filter:
                if wateinfo_obj.waterlevels < min(comparedata_filter):
                    nomal_count += 1
            else:
                nomal_count += 1
        areacount = stations.values('county').distinct().count()

        data = {
            'stationCount': stations_count,
            'normalCount': nomal_count,
            'areaCount': areacount,
        }
        return Response(data, status=status.HTTP_200_OK)
