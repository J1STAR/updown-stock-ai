from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from .models import Stock, Corp, StockInfo
from .serializers import StockSerializer

from datetime import datetime
from bs4 import BeautifulSoup
import requests


# Create your views here.
class StockView(APIView):
    def get(self, request, page=None):
        if page is None:
            serializer = StockSerializer(Stock.objects.all(), many=True)
            response = {"stock": serializer.data}
        else:
            items_per_page = 20

            offset = (int(page) - 1) * items_per_page

            serializer = StockSerializer(Stock.objects.skip(offset).limit(items_per_page), many=True)
            response = {"stock": serializer.data}
        return Response(response, status=status.HTTP_200_OK)


class StockDetailView(APIView):
    def get(self, request, code=None):
        try:
            stock_detail = Stock.objects.get(code=code)
        except:
            return Response(status=status.HTTP_404_NOT_FOUND)

        serializer = StockSerializer(stock_detail)
        response = serializer.data
        return Response(response, status=status.HTTP_200_OK)

    def post(self, request, code=None):
        data = request.data

        serializer = None
        if type(data) == dict:
            serializer = StockSerializer(data=data)

            stock_info = data['stock_info'][0]
            stock_info['date'] = datetime.strptime(stock_info['date'], '%Y.%m.%d')

            if serializer.is_valid():
                stock_info_doc = StockInfo(**stock_info)

                try:
                    stock = Stock.objects.get(code=code)
                    stock.corp.stock_info.append(stock_info_doc)
                except:
                    corp = Corp(corp_name=data['corp_name'])
                    corp.stock_info.append(stock_info_doc)

                    stock = Stock(code=code, corp=corp)
                stock.save()

                response = serializer.data
        elif type(data) == list:
            corp = Corp(corp_name=data[0]['corp_name'])
            for doc in data:
                serializer = StockSerializer(data=doc)

                stock_info = doc['stock_info'][0]
                stock_info['date'] = datetime.strptime(stock_info['date'], '%Y.%m.%d')

                if serializer.is_valid():
                    stock_info_doc = StockInfo(**stock_info)

                    corp.stock_info.append(stock_info_doc)
            stock = Stock(code=code, corp=corp)
            stock.save()

            response = serializer.data

        return Response(None, status=status.HTTP_200_OK)


class StockBusinessView(APIView):
    def get(self, request):
        business_types_url = "https://finance.naver.com/sise/sise_group.nhn?type=upjong"
        res = requests.get(business_types_url)

        soup = BeautifulSoup(res.text, 'html.parser')
        raw_dataset = soup.select('#contentarea_left > table > tr:nth-child(n+4) > td:nth-child(1) > a')

        business_types = []
        for data in raw_dataset:
            business_type = {"name": data.text, "business_code": data.get('href').split('no=')[1]}
            business_types.append(business_type)

        return Response(business_types, status=status.HTTP_200_OK)


class StockCorpView(APIView):
    def get(self, request, business_code=None):
        corps_url = "https://finance.naver.com/sise/sise_group_detail.nhn?type=upjong&no=" + business_code
        res = requests.get(corps_url)

        soup = BeautifulSoup(res.text, 'html.parser')
        raw_dataset = soup.select('table.type_5 tr:nth-child(n+3) td:nth-child(1) a')

        corparations = []
        for data in raw_dataset:
            corp = {"name": data.text, "corp_code": data.get('href').split("code=")[1]}
            corparations.append(corp)

        return Response(corparations, status=status.HTTP_200_OK)
