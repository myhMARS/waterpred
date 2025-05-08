from django.urls import path

from . import views

app_name = "api"
urlpatterns = [
    path("waterinfo/", views.Water_Info.as_view()),
    path("stationCount/", views.StationCount.as_view()),
    path("statistics/", views.StatisticsInfo.as_view()),
    path("warnings/", views.WarningInfo.as_view()),
    path("test/", views.TaskTest.as_view()),
]
