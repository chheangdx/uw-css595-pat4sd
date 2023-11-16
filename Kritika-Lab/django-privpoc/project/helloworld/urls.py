# pages/urls.py
from django.urls import path
from .views import homePageView, schedulerDemoView

urlpatterns = [
    path("", homePageView, name="home"),
    path("scheduler_demo", schedulerDemoView, name="scheduler_demo")
]
