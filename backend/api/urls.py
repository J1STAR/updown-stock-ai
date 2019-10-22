from django.conf.urls import url
from . import views


urlpatterns = [
    url(r'^$', views.UserView.as_view()),
    url(r'^$', views.InputStockView.as_view()),
    url(r'^$', views.OutputStockView.as_view()),
]

