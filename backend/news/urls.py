from django.conf.urls import url
from django.urls import path, re_path

from . import views

urlpatterns = [
    re_path(r'^(?P<corp>[a-zA-Z0-9]{6,7})/$', views.NewsView.as_view()),
]