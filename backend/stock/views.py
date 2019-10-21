from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from .models import Stock, Corp, StockInfo
from .serializers import StockSerializer

from datetime import datetime


# Create your views here.
class StockView(APIView):
    def get(self, request):
        print("STOCKLIST")
        serializer = StockSerializer(Stock.objects.all(), many=True)
        response = {"stock": serializer.data}
        return Response(response, status=status.HTTP_200_OK)

    def post(self, request, format=None):
        data = request.data
        stock_info = data['stock_info'][0]
        stock_info['date'] = datetime.strptime(stock_info['date'], '%Y.%m.%d')
        serializer = StockSerializer(data=data)
        if serializer.is_valid():
            stock_info = StockInfo(date=stock_info['date'],
                                   closing_price=stock_info['closing_price'],
                                   diff=stock_info['diff'],
                                   open_price=stock_info['open_price'],
                                   high_price=stock_info['high_price'],
                                   low_price=stock_info['low_price'],
                                   volumn=stock_info['volumn']
                                   )

            corp = Corp(corp_name=data['corp_name'])
            corp.stock_info.append(stock_info)

            stock = Stock(code=data['code'], corp=corp)
            stock.save()

            response = serializer.data
            return Response(response, status=status.HTTP_200_OK)


class StockDetailView(APIView):
    def get(self, request, code=None):
        print("STOCKDETAIL")
        try:
            stock_detail = Stock.objects.get(code=code)
        except:
            return Response(status=status.HTTP_404_NOT_FOUND)

        serializer = StockSerializer(stock_detail)
        response = serializer.data
        return Response(response, status=status.HTTP_200_OK)

    def post(self, request, code=None):
        return Response(None, status=status.HTTP_200_OK)
