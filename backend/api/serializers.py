from rest_framework_mongoengine import serializers
from .models import *


class StockSerializer(serializers.DocumentSerializer):
    class Meta:
        model = Stock
        fields = ('pk', 'date', 'start_price', 'end_price', 'high_price', 'low_price', 'val')
