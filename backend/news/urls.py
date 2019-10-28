from django.conf.urls import url
from django.urls import path, re_path

from . import views

urlpatterns = [
    url(r'^$', views.NewsView.as_view()),
]