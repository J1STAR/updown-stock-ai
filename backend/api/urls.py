from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^stocks/$', views.stock_list),
    # url(r'^users/$', views.user_list),
    url(r'^stocks/(?P<pk>[a-zA-Z0-9]+)$', views.stock_detail)
]