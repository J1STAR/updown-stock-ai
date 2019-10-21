from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

from .models import User
from .serializers import *

# Create your views here.


@api_view(['GET', 'POST'])
def user_list(request):
    """
    List  users, or create a new user.
    """
    if request.method == 'GET':
        data = []
        nextPage = 1
        previousPage = 1
        users = User.objects.all()
        page = request.GET.get('page', 1)
        paginator = Paginator(users, 10)
        try:
            data = paginator.page(page)
        except PageNotAnInteger:
            data = paginator.page(1)
        except EmptyPage:
            data = paginator.page(paginator.num_pages)

        serializer = UserSerializer(data,context={'request': request} ,many=True)
        if data.has_next():
            nextPage = data.next_page_number()
        if data.has_previous():
            previousPage = data.previous_page_number()

        return Response({'data': serializer.data , 'count': paginator.count, 'numpages' : paginator.num_pages, 'nextlink': '/api/users/?page=' + str(nextPage), 'prevlink': '/api/users/?page=' + str(previousPage)})

    elif request.method == 'POST':
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



class UserView(APIView):
    def get(self, request):
        serializer = UserSerializer(User.objects.all(), many=True)
        response = {"users": serializer.data}

        return Response(response, status=status.HTTP_200_OK)

    def post(self, request, format=None):
        data = request.data
        serializer = UserSerializer(data=data)

        if serializer.is_valid():
            poll = User(**data)
            poll.save()
            response = serializer.data

            return Response(response, status=status.HTTP_200_OK)
