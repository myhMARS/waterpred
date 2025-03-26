from django.urls import path
from . import views


app_name = "api"
urlpatterns = [
    path("predict/", views.Water_Info.as_view()),
]
