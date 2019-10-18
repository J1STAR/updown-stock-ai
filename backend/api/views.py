from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from .models import User
from .serializers import UserSerializer


class UserView(APIView):
    def get(self, request):
        serializer = UserSerializer(User.objects.all(), many=True)
        response = {"users": serializer.data}
        return Response(response, status=status.HTTP_200_OK)

    def post(self, request, format=None):
        data = request.data
        serializer = UserSerializer(data=data)
        if serializer.is_valid() :
            user = User(**data)
            user.save()
            response = serializer.data
            return Response(response, status=status.HTTP_200_OK)


