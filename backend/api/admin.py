from django.contrib import admin

from .models import WaterInfo, WaterPred, StationInfo, WarningNotice, AreaWeatherInfo


@admin.register(WaterInfo)
class WaterInfoAdmin(admin.ModelAdmin):
    list_display = ('times', 'station_id', 'rains', 'waterlevels')
    list_filter = ('station__name',)
    search_fields = ('times', 'station__name')


@admin.register(WaterPred)
class WaterPredAdmin(admin.ModelAdmin):
    list_display = (
        "times", "waterlevel1", "waterlevel2", "waterlevel3",
        "waterlevel4", "waterlevel5", "waterlevel6"
    )
    search_fields = ('times',)


@admin.register(StationInfo)
class StationInfoAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "city", "county", "flood_limit", "guaranteed", "warning")
    list_filter = ("city", "county")
    search_fields = ("id", "name", "city", "county")


@admin.register(WarningNotice)
class WarningNoticeAdmin(admin.ModelAdmin):
    list_display = ("station", "noticetime", "isCanceled", "canceltime")
    list_filter = ("station__name", "isCanceled")
    search_fields = ("station__name", 'station__id')


@admin.register(AreaWeatherInfo)
class AreaWeatherInfoAdmin(admin.ModelAdmin):
    list_display = ('times', 'city', 'county', 'temperature', 'humidity', 'winddirection', 'windpower')
    list_filter = ('city', 'county')
    search_fields = ('times', 'city', 'county')
