from collections import defaultdict
from datetime import datetime

import httpx
from django.db.models import QuerySet, Subquery
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from django.utils import timezone

from .models import WaterInfo, WaterPred, StationInfo, Statistics, WarningNotice, WarningCloseDetail, AreaWeatherInfo
from .serializers import WaterInfoDataSerializer, WaterInfoTimeSerializer, WaterPredDataSerializer, \
    WarngingsSerializer, WarnCancelDataSerializer, AreaInfoSerializer
from .utils import predict


class TaskTest(APIView):
    def get(self, request):
        print(predict('63000200', [84.3 - i / 10 for i in range(12)]))
        return Response(status=status.HTTP_200_OK)


class AreaStationCount(APIView):
    def get(self, request):
        filters = {}
        city = request.GET.get('city')
        county: str = request.query_params.get('county')
        if county:
            filters['county'] = county
        if city:
            filters['city'] = city
        stations = StationInfo.objects.filter(**filters)
        warns = 0
        for station in stations:
            warns += WarningNotice.objects.filter(station=station,isCanceled=False).count()
        res = {
            'count': stations.count(),
            'station_status': warns,
        }
        return Response(res, status=status.HTTP_200_OK)


class StationList(APIView):
    def get(self, request):
        stationid: str = request.query_params.get("stationid")
        county: str = request.query_params.get("county")
        filters = {}
        if stationid:
            filters['id'] = stationid
        if county:
            filters['county'] = county

        queryset = StationInfo.objects.filter(**filters)
        response = []
        for obj in queryset:
            stationinfo = dict()
            stationinfo['id'] = obj.id
            stationinfo['name'] = obj.name
            stationinfo['position'] = f'{obj.city}/{obj.county}'
            stationinfo['flood_limit'] = obj.flood_limit
            stationinfo['guaranteed'] = obj.guaranteed
            stationinfo['warning'] = obj.warning
            station_waterlevel = WaterInfo.objects.filter(station=obj.id).order_by('-times').first()
            stationinfo['time'] = station_waterlevel.times
            stationinfo['rain'] = station_waterlevel.rains
            stationinfo['waterlevel'] = station_waterlevel.waterlevels
            stationinfo['status'] = WarningNotice.objects.filter(station=obj.id, isCanceled=False).count()
            response.append(stationinfo)
        return Response(response, status=status.HTTP_200_OK)


class AreaList(APIView):
    def get(self, request):
        unique_counties = AreaWeatherInfo.objects.values_list('county', flat=True).distinct()
        response = []
        for county in unique_counties:
            queryset = AreaWeatherInfo.objects.filter(county=county).order_by('-times').first()
            data = AreaInfoSerializer(queryset).data
            response.append(data)

        return Response(response, status=status.HTTP_200_OK)


class AreaDetail(APIView):
    def get(self, request):
        county: str = request.query_params.get("county")
        queryset = AreaWeatherInfo.objects.filter(county=county).order_by('-times')[:720][::-1]
        data = AreaInfoSerializer(queryset, many=True).data
        return Response(data, status=status.HTTP_200_OK)


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
        station: str = request.query_params.get("station")
        isCancel: str = request.query_params.get("isCancel")
        isSuccess: str = request.query_params.get("isSuccess")

        filters = {}
        if station:
            filters["station"] = station
        if isCancel:
            filters["isCanceled"] = bool(int(isCancel))
        if isSuccess:
            filters["isSuccess"] = bool(int(isSuccess))
        print(filters)
        queryset: QuerySet = WarningNotice.objects.filter(**filters).order_by("-noticetime")
        data = WarngingsSerializer(queryset, many=True).data
        return Response(data, status=status.HTTP_200_OK)


class RecentData(APIView):
    def get(self, request):
        station_id: str = request.query_params.get("stationid")
        if not station_id:
            return Response({"detail": "stationid required"}, status=status.HTTP_400_BAD_REQUEST)
        rainfield: str = request.query_params.get("rain")
        waterlevel: str = request.query_params.get("waterlevel")
        if rainfield:
            res = WaterInfo.objects.filter(station=station_id).order_by('-times').values_list('rains', flat=True)[:12][
                  ::-1]
            return Response(res, status=status.HTTP_200_OK)
        if waterlevel:
            res = WaterInfo.objects.filter(station=station_id).order_by('-times').values_list('waterlevels', flat=True)[
                  :12][::-1]
            return Response(res, status=status.HTTP_200_OK)
        return Response({"detail": "Rain or Waterlevel field flag required"}, status=status.HTTP_400_BAD_REQUEST)


class WarnCancel(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        cancelinfo = WarnCancelDataSerializer(data=request.data)
        if not cancelinfo.is_valid():
            return Response(cancelinfo.errors, status=status.HTTP_400_BAD_REQUEST)
        station_id: str = cancelinfo.validated_data["station_id"]
        detail: str = cancelinfo.validated_data["detail"]

        warning = WarningNotice.objects.filter(station=station_id, isCanceled=False).first()
        if not warning:
            return Response({"detail": '不存在告警'}, status=status.HTTP_404_NOT_FOUND)
        warning.isCanceled = True
        warning.executor = request.user
        warning.canceltime = timezone.now()
        WarningCloseDetail.objects.create(warning=warning, detail=detail)
        warning.save()
        return Response('Success', status=status.HTTP_200_OK)


class GetLocation(APIView):
    def get(self, request):
        station_name: str = request.query_params.get("station_name")
        params = {
            "areaFlag": "1",
            "sss": "全部",
            "zl": "RR,ZZ,ZQ,DD,TT,",
            "sklx": "4,5,3,2,1,9,",
            "sfcj": "0",
            "bxdj": "1,2,3,4,5,",
            "zm": station_name,
            "bx": "0"
        }
        try:
            with httpx.Client(verify=False, timeout=5.0) as client:
                resp = client.get("https://sqfb.slt.zj.gov.cn/rest/newList/getNewDataList", params=params)
            return Response(resp.json())

        except httpx.RequestError as exc:
            return Response({"error": f"请求失败: {str(exc)}"}, status=500)
