from collections import defaultdict
from datetime import datetime

from django.db.models import QuerySet
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from django.utils import timezone

from .models import WaterInfo, WaterPred, StationInfo, Statistics, WarningNotice
from .serializers import WaterInfoDataSerializer, WaterInfoTimeSerializer, WaterPredDataSerializer, \
    WarngingsSerializer
from django.contrib.auth.models import Group
from .tasks import update_predict
from .utils import predict


class TaskTest(APIView):
    def get(self, request):
        print(predict('63000200', [84.3 - i / 10 for i in range(12)]))
        return Response(status=status.HTTP_200_OK)


class Water_Info(APIView):
    def get(self, request):
        station_id: str = request.query_params.get("station_id")
        time_length: str = request.query_params.get("length", default='18')
        if not station_id:
            return Response({"detail": "prarms station_id required"}, status=status.HTTP_400_BAD_REQUEST)
        waterinfo = WaterInfo.objects.filter(station=station_id).order_by('-times')[:int(time_length)][::-1]
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


class StatisticsInfo(APIView):
    def get(self, request):
        time_filter = request.query_params.get("time_filter")
        if not time_filter:
            return Response({"detail": "time_filter field required"}, status=status.HTTP_400_BAD_REQUEST)
        date_queryset = WaterInfo.objects.order_by('-times').first()
        local_time: datetime = timezone.localtime(date_queryset.times)
        year = local_time.year
        month = local_time.month
        quarter = (month - 1) // 3 + 1
        if time_filter == 'month':
            data = Statistics.objects.filter(year=year, month=month)
            res = defaultdict(int)
            for i in data:
                res[f'{i.year}-{i.month}-{i.day}'] += 1

            response = {
                'year': year,
                'month': month,
                'day': local_time.day,
                "data": res
            }
            return Response(response, status=status.HTTP_200_OK)

        elif time_filter == 'quarter':
            start_month = (quarter - 1) * 3 + 1
            end_month = start_month + 2
            data = Statistics.objects.filter(year=year, month__gte=start_month, month__lte=end_month)
            res = defaultdict(int)
            for i in data:
                res[f'{i.year}-{i.month}-{i.day}'] += 1
            response = {
                'year': year,
                'month': month,
                'day': local_time.day,
                'data': res
            }

            return Response(response, status=status.HTTP_200_OK)

        elif time_filter == 'year':
            data = Statistics.objects.filter(year=year)
            res = defaultdict(int)
            for i in data:
                res[f'{i.year}-{i.month}-{i.day}'] += 1
            response = {
                'year': year,
                'month': month,
                'day': local_time.day,
                'data': res
            }
            return Response(response, status=status.HTTP_200_OK)

        return 0


class WarningInfo(APIView):
    def get(self, request):
        isCancel: str = request.query_params.get("isCancel")
        isSuccess: str = request.query_params.get("isSuccess")

        filters = {}
        print(isSuccess, isCancel)
        if isCancel:
            filters["isCanceled"] = int(isCancel)
        if isSuccess:
            filters["isSuccess"] = int(isSuccess)
        queryset: QuerySet = WarningNotice.objects.filter(**filters)

        data = WarngingsSerializer(queryset, many=True).data
        return Response(data, status=status.HTTP_200_OK)
