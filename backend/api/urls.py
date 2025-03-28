from django.urls import path
from . import views


app_name = "api"
urlpatterns = [
    path("predict/", views.Water_Predict.as_view()),
    path("waterinfo/", views.Water_Info.as_view()),
]
