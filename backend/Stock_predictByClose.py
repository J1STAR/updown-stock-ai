import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from pandas.tseries.offsets import  MonthEnd

from sklearn.preprocessing import MinMaxScaler


from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend.tensorflow_backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import requests

tf.compat.v1.set_random_seed(777)
#컴퓨터 프로그램에서 발생하는 무작위 수는 사실 엄격한 의미의 무작위 수가 아니다.
# 어떤 특정한 시작 숫자를 정해 주면 컴퓨터가 정해진 알고리즘에 의해 마치 난수처럼
# 보이는 수열을 생성한다. 이런 시작 숫자를 시드(seed)라고 한다.

def data_standardization(x):
    x_np=np.asarray(x)
    return (x_np-x_np.mean())/x_np.std()

# 너무 작거나 너무 큰 값이 학습을 방해하는 것을 방지하고자 정규화한다
# x가 양수라는 가정하에 최소값과 최대값을 이용하여 0~1사이의 값으로 변환
# Min-Max scaling : (x-xmin)/(xmax-xmin)
#스케일링은 자료 집합에 적용되는 전처리 과정으로 모든 자료에 선형 변환을
# 적용하여 전체 자료의 분포를 평균 0, 분산 1이 되도록 만드는 과정이다.
def min_max_scaling(x):
    x_np=np.asarray(x)#행렬 복사
    return (x_np-x_np.min())/(x_np.max()-x_np.min()+1e-7)# 1e-7은 0으로 나누는 오류 예방차원

#정규화된 값을 원래의 값으로 되돌리는 함수
# 정규화하기 이전의 org_x값과 되돌리고 싶은 x를 입력하면
# 역정규화된 값을 리턴한다
def reverse_min_max_scaling(org_x,x):
    org_x_np=np.asarray(org_x)
    x_np=np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()







if __name__ == '__main__':
   res = requests.get("http://j1star.ddns.net:8000/stock/corp/005380")
   data = res.json()
   #REST 는 Representational State Transfer 의 약자로서,
   # 월드와이드웹(www) 와 같은 하이퍼미디어 시스템을 위한 소프트웨어 아키텍쳐
   #  중 하나의 형식입니다. REST 서버는 클라이언트로 하여금 HTTP 프로토콜을 사용해
   #  서버의 정보에 접근 및 변경을 가능케 합니다. 여기서 정보는 text, xml, json 등
   # 형식으로 제공되는데, 요즘 트렌드는 json
   stock_info_list = data['corp']['stock_info']
   # date open high low close volumn
   pre_data_list = []
   for stock_info in stock_info_list:
       pre_dataset = [stock_info['date'][:10], stock_info['open_price'],
                      stock_info['high_price'], stock_info['low_price'],
                      stock_info['closing_price'], stock_info['volume']]
       pre_data_list.append(pre_dataset)
   df_stock_info = pd.DataFrame(pre_data_list, columns =['date', 'open', 'high', 'low', 'close', 'volume'])
   #volumn : 거래량
   # print(df_stock_info)

   #drop out : 왠진 모르겠는데 잘 안된다

print(df_stock_info)

sequence_length=5  # 최근 몇일 거 보고 예측할 건지

df_stock_info.set_index('date')
# split_date = pd.Timestamp('2011-01-01')
#
# # print(split_date)
#

maxlength = len(df_stock_info)
train = df_stock_info.loc[maxlength-240:maxlength-61, ['close']]
#20이 1개월 최근 9개월 60일전까지 학습

#3달은 test 총 1달 240일
test = df_stock_info.loc[maxlength-60:, ['close']]

print(train)
print(test)



# print(test)
ax = train.plot()
# print(ax)
# ax2 = test.plot()
test.plot(ax=ax)
plt.legend(['train', 'test'])
# legend : 범례
# 책의 첫머리에 그 책의 내용, 사용법, 편수 방침 등에 대하여 설명한 글.
# 그림보여주는 matlab부분


# print("학습데이터")
# print(train)
# print("테스트데이터")
# print(test)



sc = MinMaxScaler()

train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)

# print("train_sc")
# print(train_sc)

train_sc_df = pd.DataFrame(train_sc, columns=['Scaled'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Scaled'], index=test.index)

# print("train_sc_df")
# print(train_sc_df)

# 시가 5일 종가 5일

for s in range(1, sequence_length+1):
    train_sc_df['shift_{}'.format(s)] = train_sc_df['Scaled'].shift(s)
    test_sc_df['shift_{}'.format(s)] = test_sc_df['Scaled'].shift(s)
#원래 있던 데이터 한단계씩 미뤄가지고   시가 5개 종가 5개 고가 5개  저가 5개


X_train = train_sc_df.dropna().drop('Scaled', axis=1)
Y_train = train_sc_df.dropna()[['Scaled']]
# dropna()가 none이 포함되어있는 부분을 제외해버리기때문에 앞의 몇일이 짤린다.
# 예를 들면 7일차 예측하려면 이전의 5일치가 2,3,4,5,6이어서 상관없지만
# 4알차 예측하려면 이전의 5일치 -1 0 1 2 3이어서
# -1 0 1 2 3 이어서 -1과 0을 자르는 작업
# print(X_train)
# print(Y_train)


# 2일치 1일치
X_test = test_sc_df.dropna().drop('Scaled', axis=1)
Y_test = test_sc_df.dropna()[['Scaled']]
#이전 5일치 종가 x 현재종가 hy

# print(X_train)

X_train = X_train.values
# print(X_train)
X_test = X_test.values

Y_train = Y_train.values
print("Y_train")
print(type(Y_train))
print(Y_train)
Y_test = Y_test.values

X_train_t = X_train.reshape(X_train.shape[0], sequence_length, 1)
X_test_t = X_test.reshape(X_test.shape[0], sequence_length, 1)
#행렬연산 모양 맞춰주려고


print("최종 DATA")
print(type(X_train_t))
print(X_train_t)


# LSTM 모델 만들기
# Clear session
# 현재의 TF graph를 버리고 새로 만든다. 예전 모델, 레이어와의 충돌을 피한다.
K.clear_session()

model = Sequential() # Sequential Model
model.add(LSTM(20, input_shape=(sequence_length, 1)))# (timestep, feature)
# model.add(Dense(20))
# model.add(LSTM(20))

# LSTM추가하는 경우의수
# 조기종료 drop out 0에서 1사이의 값 정하는 거 있고
# 가중치 규제는 빼자 batch size는 10-20, unit은 sequence length보다는 커야
# 몇일 전 ; 최근 12일꺼 확인






model.add(Dense(1)) # output = 1
# dense output 어떻게 나올지
model.compile(loss='mean_squared_error', optimizer='adam')
# loss로 계산을 해서 오차값을 어떻게 수정하는지의 알고리즘. 그 중에 adam이 보편적으로 괜찮다고 한다.


#unit : 노드개수 5일치보다 적음-
#가중치 랜덤



# loss를 모니터링해서 patience만큼 연속으로 loss률이 떨어지지 않으면 훈련을 멈춘다.
early_stop = [EarlyStopping(monitor='val_loss', patience=20, verbose=1),
ModelCheckpoint(filepath='best_model_close', monitor='val_loss', save_best_only=True)]
#

# history=model.fit(X_train_t, Y_train, epochs=100, batch_size=30, verbose=1, callbacks=[early_stop])

history = model.fit(X_train_t, Y_train, epochs=1000, verbose=2, batch_size=30, validation_data=(X_test_t, Y_test),
                    callbacks=early_stop)

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

best_model = load_model('best_model_close')

Y_pred_best = best_model.predict(X_test_t)


plt.figure(3)

count= range(1, len(Y_pred)+1)
# print(Y_test)
# print(Y_pred)
plt.plot(count, Y_test, "r--")
plt.plot(count, Y_pred, "b-")

plt.legend(["Y_test", "Y_pred_by_close"])

Y_pred = model.predict(X_test_t)

plt.figure(4)
#
plt.plot(count, Y_test, "r--")
plt.plot(count, Y_pred_best, "b-")
#
plt.legend(["Y_test", "Y_pred_best"])


print(Y_test)
print(Y_pred)

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
print("best_count = ", best_count)
print("베스트 모델 정답률 : ", best_count/len(Y_test))

plt.show()