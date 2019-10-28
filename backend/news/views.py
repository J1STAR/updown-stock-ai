from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from datetime import datetime
from bs4 import BeautifulSoup
import requests


# Create your views here.
class NewsView(APIView):
    def get(self, corp=None):
        news_data = requests.get("https://news.google.com/search?q=%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90&hl=ko&gl=KR&ceid=KR:ko")
        print(news_data.text)
        response = {"news": ""}
        return Response(response, status=status.HTTP_200_OK)