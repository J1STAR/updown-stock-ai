import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from pandas.tseries.offsets import  MonthEnd

from sklearn.preprocessing import MinMaxScaler


from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import keras.backend.tensorflow_backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import optimizers

import requests

tf.compat.v1.set_random_seed(777)

if __name__ == '__main__':
   res = requests.get("http://j1star.ddns.net:8000/stock/corp/067280")
   data = res.json()
   stock_info_list = data['corp']['stock_info']
   # date open high low close volumn
   pre_data_list = []
   for stock_info in stock_info_list:
       pre_dataset = [stock_info['date'][:10], stock_info['open_price'], stock_info['high_price'], stock_info['low_price'], stock_info['closing_price'], stock_info['diff'], stock_info['volume']]
       pre_data_list.append(pre_dataset)
   df_stock_info = pd.DataFrame(pre_data_list, columns =['date', 'open', 'high', 'low', 'close', 'diff', 'volume'])
   df_stock_info = df_stock_info[df_stock_info.open != 0]
   # print(df_stock_info)

print(df_stock_info)

sequence_length=5

df_stock_info.set_index('date')
# split_date = pd.Timestamp('2011-01-01')
#
# # print(split_date)
#

maxlength = len(df_stock_info)
train = df_stock_info.loc[maxlength-(480+sequence_length):maxlength-121, ['close']]
test = df_stock_info.loc[maxlength-(120+sequence_length):, ['close']]

print(train)
print(test)

Y_train = df_stock_info.loc[maxlength-480:maxlength-121, ['diff']]
Y_test = df_stock_info.loc[maxlength-120:, ['diff']]
# print(test)
ax = train.plot()
# print(ax)
# ax2 = test.plot()
test.plot(ax=ax)
plt.xlabel('date')
plt.ylabel('diff')
plt.legend(['train', 'test'])

# print("학습데이터")
# print(train)
# print("테스트데이터")
# print(test)



sc = MinMaxScaler(feature_range=(-1,1))

train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)

sc2 = MinMaxScaler(feature_range=(-1,1))

Y_train = sc2.fit_transform(Y_train)

Y_test = sc2.transform(Y_test)

print("Y_train")
print(Y_train)





print("train_sc")
print(train_sc)

train_sc_df = pd.DataFrame(train_sc, columns=['종가'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['종가'], index=test.index)

# print("train_sc_df")
# print(train_sc_df)

for s in range(1, sequence_length+1):

    train_sc_df['{}일전 종가'.format(s)] = train_sc_df['종가'].shift(s)

    test_sc_df['{}일전 종가'.format(s)] = test_sc_df['종가'].shift(s)

print(train_sc_df)

X_train = train_sc_df.dropna().drop(['종가'], axis=1)
# Y_train = train_sc_df.dropna()[['전일비']]
# dropna()가 none이 포함되어있는 부분을 제외해버리기때문에 앞의 몇일이 짤린다.
print(X_train)
# print(Y_train)

X_test = test_sc_df.dropna().drop(['종가'], axis=1)
# Y_test = test_sc_df.dropna()[['전일비']]

# print(X_train)

X_train = X_train.values
# print(X_train)
X_test = X_test.values

# Y_train = Y_train.values
# print("Y_train")
# print(type(Y_train))
# print(Y_train)
# Y_test = Y_test.values

X_train_t = X_train.reshape(X_train.shape[0], sequence_length, 1)
X_test_t = X_test.reshape(X_test.shape[0], sequence_length, 1)

# print("최종 DATA")
# print(type(X_train_t))
# print(X_train_t)


# LSTM 모델 만들기
# Clear session
# 현재의 TF graph를 버리고 새로 만든다. 예전 모델, 레이어와의 충돌을 피한다.
K.clear_session()

model = Sequential() # Sequential Model
model.add(LSTM(40, input_shape=(sequence_length, 1)))# (timestep, feature)
# model.add(Dense(100))
# model.add(Dense(100))
model.add(Dense(1)) # output = 1

adam = optimizers.adam(learning_rate=0.0001)
model.compile(loss='mean_squared_error', optimizer='adam')

# loss를 모니터링해서 patience만큼 연속으로 loss률이 떨어지지 않으면 훈련을 멈춘다.
early_stop = [EarlyStopping(monitor='val_loss', patience=20, verbose=1), ModelCheckpoint(filepath='best_model_diff', monitor='val_loss', save_best_only=True)]

# history=model.fit(X_train_t, Y_train, epochs=100, batch_size=30, verbose=1, callbacks=[early_stop])

history = model.fit(X_train_t, Y_train, epochs=1000, verbose=2, batch_size=5, validation_data=(X_test_t, Y_test), callbacks=early_stop)

# Y_pred = model.predict(X_test_t)

training_loss = history.history["loss"]
test_loss = history.history["val_loss"]

print(type(training_loss))
#
epoch_count = range(1, len(training_loss)+1)
#
# plt.plot(epoch_count, training_loss, "r--")
# plt.plot(epoch_count, test_loss, "b-")
# plt.legend(["Training Loss", "Test loss"])
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()

plt.figure(2)

plt.plot(epoch_count, training_loss, "r--")

plt.plot(epoch_count, test_loss, "b-")

plt.legend(["Training Loss", "Test Loss"])

plt.xlabel("Epoch")
plt.ylabel("Loss Score")

Y_pred = model.predict(X_test_t)

best_model = load_model('best_model_diff')

Y_pred_best = best_model.predict(X_test_t)


plt.figure(3)

count= range(1, len(Y_pred)+1)
# print(Y_test)
# print(Y_pred)
plt.plot(count, sc2.inverse_transform(Y_test), "r--")
plt.plot(count, sc2.inverse_transform(Y_pred), "b-")

plt.legend(["Y_test", "Y_pred_by_diff"])

Y_pred = model.predict(X_test_t)

plt.figure(4)
#
# c = range(1, len(Y_train)+1)
# count= range(len(Y_train)+1, len(Y_train)+len(Y_pred)+1)
# ax = plt.plot(c, Y_train)
# plt.plot(count, Y_test)
# plt.plot(count, Y_pred_best)
count= range(1, len(Y_pred)+1)
plt.plot(count, sc2.inverse_transform(Y_test))
plt.plot(count, sc2.inverse_transform(Y_pred_best))
#
plt.legend(["Y_test", "Y_pred_best"])

plt.figure(5)

c = range(1, len(Y_train)+1)
count= range(len(Y_train)+1, len(Y_train)+len(Y_pred)+1)
ax = plt.plot(c, Y_train)
plt.plot(count, Y_test)
plt.plot(count, Y_pred_best)

plt.legend(["Y_train", "Y_test", "Y_pred_best"])


# print(Y_test)
# print(Y_pred)

count = 0
best_count = 0
for val in range(1, len(Y_test)):
    test_val = Y_test[val]-Y_test[val-1]
    pred_val = Y_pred[val]-Y_test[val-1]
    pred_best_val = Y_pred_best[val]-Y_test[val-1]
    if test_val > 0:
        test_val = 1
    else:
        test_val = -1
    if pred_val > 0:
        pred_val = 1
    else:
        pred_val = -1
    if pred_best_val > 0:
        pred_best_val = 1
    else:
        pred_best_val = -1
    if test_val == pred_best_val:
        best_count+=1

    if test_val == pred_val:
        count+=1

print("count = ", count)
print("총 개수 = ", len(Y_test))
print("모델 정답률 : ", count/len(Y_test))
print("베스트 모델 정답률 : ", best_count/len(Y_test))

plt.show()
