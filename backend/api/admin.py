from django.contrib import admin

from .models import WaterInfo, WaterPred, StationInfo, WarningNotice, AreaWeatherInfo, Statistics


@admin.register(WaterInfo)
class WaterInfoAdmin(admin.ModelAdmin):
    list_display = ('times', "stationId", 'station_name', 'rains', 'waterlevels')
    list_filter = ('station__name',)
    search_fields = ('times', 'station__name')

    def station_name(self, obj):
        return obj.station.name

    def stationId(self, obj):
        return obj.station.id

    stationId.short_description = '站点编号'
    station_name.short_description = "站名"


@admin.register(WaterPred)
class WaterPredAdmin(admin.ModelAdmin):
    list_display = (
        "times", "station_name", "waterlevel1", "waterlevel2", "waterlevel3",
        "waterlevel4", "waterlevel5", "waterlevel6"
    )
    search_fields = ('times',)
    list_filter = ('station__name',)

    def station_name(self, obj):
        return obj.station.name

    station_name.short_description = "站名"


@admin.register(StationInfo)
class StationInfoAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "city", "county", "flood_limit", "guaranteed", "warning")
    list_filter = ("city", "county")
    search_fields = ("id", "name", "city", "county")


@admin.register(WarningNotice)
class WarningNoticeAdmin(admin.ModelAdmin):
    list_display = ("station_name", "noticetime", "isCanceled", "canceltime")

    def station_name(self, obj):
        return obj.station.name

    station_name.short_description = "站名"

    list_filter = ("station__name", "isCanceled")
    search_fields = ("station__name", 'station__id')


@admin.register(AreaWeatherInfo)
class AreaWeatherInfoAdmin(admin.ModelAdmin):
    list_display = ('times', 'city', 'county', 'temperature', 'humidity', 'winddirection', 'windpower')
    list_filter = ('city', 'county')
    search_fields = ('times', 'city', 'county')


@admin.register(Statistics)
class StatisticsAdmin(admin.ModelAdmin):
    list_display = ('year', 'month', 'day', 'station_name')

    @staticmethod
    def station_name(obj):
        return obj.station.name
