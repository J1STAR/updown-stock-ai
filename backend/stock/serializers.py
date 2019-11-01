from rest_framework_mongoengine import serializers
from .models import Stock


class StockSerializer(serializers.DocumentSerializer):
    class Meta:
        model = Stock
        fields = '__all__'
