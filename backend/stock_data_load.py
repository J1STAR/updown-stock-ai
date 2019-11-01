import requests
import pandas as pd
if __name__ == '__main__':
   res = requests.get("http://j1star.ddns.net:8000/stock/067630")
   data = res.json()
   stock_info_list = data['corp']['stock_info']
   # date open high low close volumn
   pre_data_list = []
   for stock_info in stock_info_list:
       pre_dataset = [stock_info['date'][:10], stock_info['open_price'], stock_info['high_price'], stock_info['low_price'], stock_info['closing_price'], stock_info['volumn']]
       pre_data_list.append(pre_dataset)
   df_stock_info = pd.DataFrame(pre_data_list, columns =['date', 'open', 'high', 'low', 'close', 'volumn'])
   print(df_stock_info)