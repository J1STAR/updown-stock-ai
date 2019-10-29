from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from datetime import datetime
from bs4 import BeautifulSoup
import requests


# Create your views here.
class NewsView(APIView):
    def get(self, request, corp=None):
        news_data = requests.get("https://news.google.com/search?q={}&hl=ko&gl=KR&ceid=KR:ko".format(corp))

        soup = BeautifulSoup(news_data.text, 'html.parser')
        news_data_list = soup.select("c-wiz > div > div > div > div > main > c-wiz > div > div > div > article > h3 > a")

        news_list = []
        for news in news_data_list:
            link = news.get('href').replace(".", "https://news.google.com")
            title = news.text
            news_list.append({"link": link, "title": title})

        response = {"news": news_list}
        return Response(response, status=status.HTTP_200_OK)