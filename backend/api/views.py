from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from .models import User
from .models import InputStock
from .models import OutputStock
from .serializers import UserSerializer
from .serializers import InputStockSerializer
from .serializers import OutputStockSerializer


class UserView(APIView):
    def get(self, request):
        serializer = UserSerializer(User.objects.all(), many=True)
        response = {"users": serializer.data}
        return Response(response, status=status.HTTP_200_OK)

    def post(self, request, format=None):
        data = request.data
        serializer = UserSerializer(data=data)
        if serializer.is_valid():
            user = User(**data)
            user.save()
            response = serializer.data
            return Response(response, status=status.HTTP_200_OK)


class InputStockView(APIView):
    def get(self, request):
        serializer = InputStockSerializer(InputStock.objects.all(), many=True)
        response = {"inputstock": serializer.data}
        return Response(response, status=status.HTTP_200_OK)

    def post(self, request, format=None):
        data = request.data
        serializer = InputStockSerializer(data=data)
        if serializer.is_valid():
            inputstock = InputStock(**data)
            inputstock.save()
            response = serializer.data
            return Response(response, status=status.HTTP_200_OK)


class OutputStockView(APIView):
    def get(self, request):
        serializer = OutputStockSerializer(OutputStock.objects.all(), many=True)
        response = {"outputstock": serializer.data}
        return Response(response, status=status.HTTP_200_OK)

    def post(self, request, format=None):
        data = request.data
        serializer = OutputStockSerializer(data=data)
        if serializer.is_valid():
            outputstock = OutputStock(**data)
            outputstock.save()
            response = serializer.data
            return Response(response, status=status.HTTP_200_OK)
