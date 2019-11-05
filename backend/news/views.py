from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from datetime import datetime
from bs4 import BeautifulSoup
import requests
import urllib.parse


# Create your views here.
class NewsView(APIView):
    def get(self, request, corp=None):
        news_data = requests.get("https://finance.naver.com/item/news_news.nhn?code={}&page=&sm=entity_id.basic&clusterId=".format(urllib.parse.quote(corp)))

        soup = BeautifulSoup(news_data.text, 'html.parser')
        news_data_list = soup.select("body > div > table.type5 > tbody > tr > td.title > a")

        news_list = []
        for news in news_data_list:
            link = "https://finance.naver.com" + news.get('href')
            title = news.text
            news_list.append({"link": link, "title": title})

        response = {"news": news_list}
        return Response(response, status=status.HTTP_200_OK)