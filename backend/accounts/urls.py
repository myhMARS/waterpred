from django.urls import path

from . import views

app_name = "account"
urlpatterns = [
    path("userinfo/", views.UserInfo.as_view()),
]
