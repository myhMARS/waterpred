from django.urls import path, include
from . import views


app_name = "api"
urlpatterns = {
    path("", views.Water_Info.as_view()),
}
