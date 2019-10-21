from django.conf.urls import url
from django.urls import path

from . import views

urlpatterns = [
    url(r'^$', views.StockView.as_view()),
    path('<slug:code>', views.StockDetailView.as_view()),
]