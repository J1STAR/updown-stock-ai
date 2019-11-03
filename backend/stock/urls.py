from django.conf.urls import url
from django.urls import path, re_path

from . import views

urlpatterns = [
    url(r'^$', views.StockView.as_view()),
    re_path(r'^page/(?P<page>[0-9]*)/$', views.StockView.as_view()),
    re_path(r'^corp/(?P<code>[a-zA-Z0-9]{6,7})/$', views.StockDetailView.as_view()),
    path('businessTypes/', views.StockBusinessView.as_view()),
    path('businessTypes/<slug:business_code>/', views.StockCorpView.as_view())
]