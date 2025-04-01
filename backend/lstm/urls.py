from django.urls import path

from . import views

app_name = "lstm"
urlpatterns = [
    path("train/", views.TrainAPI.as_view()),
    path("change/", views.ChangeModel.as_view()),
    path("info/", views.ModelInfo.as_view()),
    path("list/", views.ModelList.as_view()),
]
