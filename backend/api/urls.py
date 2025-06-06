from django.urls import path

from . import views

app_name = "api"
urlpatterns = [
    path("waterinfo/", views.Water_Info.as_view()),
    path("stationCount/", views.StationCount.as_view()),
    path("statistics/", views.StatisticsInfo.as_view()),
    path("warnings/", views.WarningInfo.as_view()),
    path("warncancel/", views.WarnCancel.as_view()),
    path("stationlist/", views.StationList.as_view()),
    path("arealist/", views.AreaList.as_view()),
    path("areadetail/", views.AreaDetail.as_view()),
    path("areastationcount/", views.AreaStationCount.as_view()),
    path("recent/", views.RecentData.as_view()),
    path("getlocation/", views.GetLocation.as_view()),
]
