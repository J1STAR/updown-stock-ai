import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import requests
import keras.backend.tensorflow_backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler

# 텐서플로우 시드 설정
tf.compat.v1.set_random_seed(777)

if __name__ == '__main__':
    # 데이터 불러오기
    res = requests.get("http://j1star.ddns.net:8000/stock/corp/035720")
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
look_back = 5
# 주식 정보를 날짜로 인덱싱
df_stock_info.set_index('date')
# 트레이닝 셋, 테스트 셋 만들기
# 70%를 트레이닝 셋, 나머지를 테스트 셋으로 사용
train_size = int(len(df_stock_info) * 0.7)
test_size = len(df_stock_info) - train_size
train_set = df_stock_info.loc[0:train_size, ['open', 'high', 'low', 'close', 'volume']]
test_set = df_stock_info.loc[train_size + 1:train_size + test_size, ['open', 'high', 'low', 'close', 'volume']]


Y_train_set = df_stock_info.loc[0 + look_back:train_size, ['close']]
Y_test_set = df_stock_info.loc[train_size + 1 + look_back:train_size + test_size, ['close']]

# LSTM 적용을 위한 값 normalize
sc = MinMaxScaler()
train_sc = sc.fit_transform(train_set)
test_sc = sc.fit_transform(test_set)

sc2 = MinMaxScaler()

Y_train_set = sc2.fit_transform(Y_train_set)
Y_test_set = sc2.transform(Y_test_set)

df_train_sc = pd.DataFrame(train_sc, columns=['시작가', '고가', '저가', '종가', '거래량'], index=train_set.index)
df_test_sc = pd.DataFrame(test_sc, columns=['시작가', '고가', '저가', '종가', '거래량'], index=test_set.index)

for s in range(1, look_back+1):
    df_train_sc['{}일전 시작가'.format(s)] = df_train_sc['시작가'].shift(s)
    df_train_sc['{}일전 고가'.format(s)] = df_train_sc['고가'].shift(s)
    df_train_sc['{}일전 저가'.format(s)] = df_train_sc['저가'].shift(s)
    df_train_sc['{}일전 종가'.format(s)] = df_train_sc['종가'].shift(s)
    df_train_sc['{}일전 거래량'.format(s)] = df_train_sc['거래량'].shift(s)
    df_test_sc['{}일전 시작가'.format(s)] = df_test_sc['시작가'].shift(s)
    df_test_sc['{}일전 고가'.format(s)] = df_test_sc['고가'].shift(s)
    df_test_sc['{}일전 저가'.format(s)] = df_test_sc['저가'].shift(s)
    df_test_sc['{}일전 종가'.format(s)] = df_test_sc['종가'].shift(s)
    df_test_sc['{}일전 거래량'.format(s)] = df_test_sc['거래량'].shift(s)

X_train = df_train_sc.dropna().drop(['시작가', '고가', '저가', '종가', '거래량'], axis=1)
# Y_train = df_train_sc.dropna()[['종가']]
X_test = df_test_sc.dropna().drop(['시작가', '고가', '저가', '종가', '거래량'], axis=1)
# Y_test = df_test_sc.dropna()[['종가']]
X_train = X_train.values
# Y_train = Y_train.values
X_test = X_test.values
# Y_test = Y_test.values
Y_train = Y_train_set
Y_test = Y_test_set

X_train_t = X_train.reshape(X_train.shape[0], look_back * 5, 1)
X_test_t = X_test.reshape(X_test.shape[0], look_back * 5, 1)

# LSTM 모델 만들기
K.clear_session()

model = Sequential()
model.add(LSTM(40, input_shape=(look_back * 5, 1)))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train_t, Y_train, epochs=200, verbose=2, batch_size=30, validation_data=(X_test_t, Y_test))

training_loss = history.history["loss"]
test_loss = history.history["val_loss"]

epoch_count = range(1, len(training_loss) + 1)

# # 트레이닝, 테스트 셋 확인
# ax = train_set.plot()
# test_set.plot(ax=ax)
# plt.legend(['train', 'test'])
#
# # Loss율 계산
# plt.figure(2)
# plt.plot(epoch_count, training_loss, "r--")
# plt.plot(epoch_count, test_loss, "b-")
# plt.legend(["Training Loss", "Test Loss"])
# plt.xlabel("Epoch")
# plt.ylabel("Loss Score")

# 예측 결과
Y_pred = model.predict(X_test_t)
plt.figure(3)
count = range(1, len(Y_pred) + 1)
plt.plot(count, sc2.inverse_transform(Y_test), "r--")
plt.plot(count, sc2.inverse_transform(Y_pred), "b-")
plt.legend(["Y_test", "Y_pred"])

# 정답률 계산
count = 0
for val in range(1, len(Y_test)):
    test_val = Y_test[val]-Y_test[val-1]
    pred_val = Y_pred[val]-Y_test[val-1]
    if test_val > 0:
        test_val = 1
    else:
        test_val = -1
    if pred_val > 0:
        pred_val = 1
    else:
        pred_val = -1
    if test_val == pred_val:
        count += 1

print("정답률 : ", count/len(Y_test))

plt.show()
