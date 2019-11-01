from django.conf.urls import url
from django.urls import path, re_path

from . import views

urlpatterns = [
    re_path(r'^(?P<corp>.*)/$', views.NewsView.as_view()),
]