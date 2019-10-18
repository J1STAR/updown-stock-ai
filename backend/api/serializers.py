from rest_framework_mongoengine import serializers
from .models import User

class UserSerializer(serializers.DocumentSerializer):
    class Meta:
        model = User
        fields = '__all__'