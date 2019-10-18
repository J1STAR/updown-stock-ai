from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^$', TemplateView.as_view(template_name='index.html')),
    url(r'^api/', include('api.urls')),
]

