from django.urls import path

from . import views

app_name = "lstm"
urlpatterns = [
    path("train/", views.TrainAPI.as_view()),
]
