import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import requests
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

from sklearn.preprocessing import MinMaxScaler

# 텐서플로우 시드 설정
tf.compat.v1.set_random_seed(777)

if __name__ == '__main__':
    # 데이터 불러오기
    res = requests.get("http://j1star.ddns.net:8000/stock/corp/240810")
    data = res.json()
    stock_info = data['corp']['stock_info']
    price = []
    date = []
    open_price = []
    high_price = []
    low_price = []
    closing_price = []
    volume = []
    for p in stock_info:
        price.append([p['date'][:10], p['open_price'], p['high_price'],
                      p['low_price'], p['closing_price'], p['volume']])
        date.append([p['date'][:10]])
        open_price.append([p['open_price']])
        high_price.append([p['high_price']])
        low_price.append([p['low_price']])
        closing_price.append([p['closing_price']])
        volume.append([p['volume']])
    df_stock_info = pd.DataFrame(price, columns=['date', 'open', 'high', 'low', 'close', 'volume'])

# 이전 몇개의 데이터를 바탕으로 예측하는가
look_back = 20
# 주식 정보를 날짜로 인덱싱
df_stock_info.set_index('date')
# 트레이닝 셋, 테스트 셋 만들기
# 70%를 트레이닝 셋, 나머지를 테스트 셋으로 사용
train_size = int(len(df_stock_info) * 0.7)
test_size = len(df_stock_info) - train_size
train_set = df_stock_info.loc[0:train_size, ['close', 'volume']]
test_set = df_stock_info.loc[train_size + 1:train_size + test_size, ['close', 'volume']]

# LSTM 적용을 위한 값 normalize
sc = MinMaxScaler()
train_sc = sc.fit_transform(train_set)
test_sc = sc.fit_transform(test_set)
df_train_sc = pd.DataFrame(train_sc, columns=['종가', '거래량'], index=train_set.index)
df_test_sc = pd.DataFrame(test_sc, columns=['종가', '거래량'], index=test_set.index)

print(df_train_sc)
print(df_test_sc)

for s in range(1, look_back+1):
    df_train_sc['{}일전 종가'.format(s)] = df_train_sc['종가'].shift(s)
    df_train_sc['{}일전 거래량'.format(s)] = df_train_sc['거래량'].shift(s)
    df_test_sc['{}일전 종가'.format(s)] = df_test_sc['종가'].shift(s)
    df_test_sc['{}일전 거래량'.format(s)] = df_test_sc['거래량'].shift(s)

X_train = df_train_sc.dropna().drop(['종가', '거래량'], axis=1)
Y_train = df_train_sc.dropna()[['종가']]

print(X_train)
print(Y_train)


ax = train_set.plot()
test_set.plot(ax=ax)
plt.legend(['train', 'test'])
plt.show()
