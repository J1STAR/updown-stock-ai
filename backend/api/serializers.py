from rest_framework_mongoengine import serializers
from .models import User
from .models import InputStock
from .models import OutputStock


class UserSerializer(serializers.DocumentSerializer):
    class Meta:
        model = User
        fields = '__all__'


class InputStockSerializer(serializers.DocumentSerializer):
    class Meta:
        model = InputStock
        fields = '__all__'


class OutputStockSerializer(serializers.DocumentSerializer):
    class Meta:
        model = OutputStock
        fields = '__all__'
