import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from pandas.tseries.offsets import  MonthEnd

from sklearn.preprocessing import MinMaxScaler

#tanh가 default이다?

from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend.tensorflow_backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
#케라스는 파이썬으로 구현된 쉽고 간결한 딥러닝 라이브러리입니다.
# 딥러닝 비전문가라도 각자 분야에서 손쉽게 딥러닝 모델을 개발하고
# 활용할 수 있도록 케라스는 직관적인 API를 제공하고 있습니다.
# 내부적으로는 텐서플로우(TensorFlow), 티아노(Theano), CNTK 등의 딥러닝 전용 엔진이 구동되지만
# 케라스 사용자는 복잡한 내부 엔진을 알 필요는 없습니다. 직관적인 API로 쉽게 다층퍼셉트론 모델, 컨볼루션 신경망 모델,
# 순환 신경망 모델 또는 이를 조합한 모델은 물론 다중 입력 또는 다중 출력 등 다양한 구성을 할 수 있습니다.

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
   #dataframe은 pandas의 기본 자료구조로 2차원 배열 또는 리스트,
   #data table 전체를 포함하는 object이다.

   #volumn : 거래량
   # print(df_stock_info)

   #drop out : 왠진 모르겠는데 잘 안된다

print(df_stock_info)

sequence_length=5  # 최근 몇일 거 보고 예측할 건지
#한달이 20일 1주가 5일

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
#스케일링
#스케일링은 자료 집합에 적용되는 전처리 과정으로 모든 자료에 선형 변환을 적용하여
# 전체 자료의 분포를 평균 0, 분산 1이 되도록 만드는 과정이다.스케일링은 자료의
# 오버플로우(overflow)나 언더플로우(underflow)를 방지하고 독립 변수의 공분산
# 행렬의 조건수(condition number)를 감소시켜 최적화 과정에서의 안정성 및 수렴 속도를 향상시킨다.

#scikit-learn에서는 다음과 같은 스케일링 클래스를 제공한다.

#StandardScaler(X): 평균이 0과 표준편차가 1이 되도록 변환.
#RobustScaler(X): 중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환.
#MinMaxScaler(X): 최대값이 각각 1, 최소값이 0이 되도록 변환  -- >인자없을 때 default가 0 -1
#MaxAbsScaler(X): 0을 기준으로 절대값이 가장 큰 수가 1또는 -1이 되도록 변환




#minmaxscaler로  sc를 0-1로 만들어주고 .train을 거기 넣어주는 느낌. train이 작업완료되었으므로
#그 다음  test도 transform
train_sc = sc.fit_transform(train)#표준화
test_sc = sc.transform(test)
#서로 다른 정규분포 사이에 비교를 하거나, 특정 정규분포를 토대로 하여 통계적 추정 등의 분석작업을 해야 할 때,
# 필요에 따라 정규분포의 분산과 표준편차를 표준에 맞게 통일시키는 것. 정규분포의 치환적분이라고 보면 된다.

#표준화가 되지 않은 데이터는 비유하자면 늘어났다 줄어들었다 하는 자를 가지고 길이를 재는 것과도 같다.
#  게다가 서로 다른 단위체계를 가진 서로 다른 연구대상에 대해서도 분석의 호환이 안 된다.
# 그래서 표준적으로 사용할 수 있는 통계적 단위를 제안하여 그것에 자신의 "자" 를 일치시켜야 하는 것이다.
# 이 때 모두가 쓸 수 있는 단위로서 제안되는 것이 바로 표준 편차, 즉 시그마(sigma)이다.

#즉, 평균을 0으로, 표준 편차를 1로 만들어준다.


#1. Fit () : 메서드는 매개 변수 μ 및 σ를 계산하고 내부 개체로 저장합니다.

#2. Transform () : 이 계산 된 매개 변수를 사용하는 방법은 특정 데이터 집합에 변환을 적용합니다.

#3. Fit_transform () : 데이터 집합의 변환을 위해 fit () 및 transform () 메서드를 조인합니다.

#print("train_sc") #소수 8자리까지 표현하나봄
#print(train_sc)

train_sc_df = pd.DataFrame(train_sc, columns=['Scaled'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Scaled'], index=test.index)
#속성이 scaled 소수 7자리에서 반올림 하나봄

#print("train_sc_df")
#print(train_sc_df)

# 시가 5일 종가 5일
# ... 같은 truncation(잘림) 말고 다 보고싶을때
# pd.set_option('display.max_colwidth',-1) 해주기


for s in range(1, sequence_length+1):
    train_sc_df['shift_{}'.format(s)] = train_sc_df['Scaled'].shift(s)
    test_sc_df['shift_{}'.format(s)] = test_sc_df['Scaled'].shift(s)
#원래 있던 데이터 한단계씩 미뤄가지고   시가 5개 종가 5개 고가 5개  저가 5개
#여기선 종가로 했으므로 처음꺼 nan이고 그다음부턴
#   Scaled   shift_1   shift_2   shift_3   shift_4   shift_5
# 5630  0.289941       NaN       NaN       NaN       NaN       NaN
# 5631  0.289941  0.289941       NaN       NaN       NaN       NaN
# 5632  0.250493  0.289941  0.289941       NaN       NaN       NaN
# 5633  0.240631  0.250493  0.289941  0.289941       NaN       NaN
# 5634  0.181460  0.240631  0.250493  0.289941  0.289941     NaN
#이런식으로 한칸씩 밀게 된다.

print("train_sc_df")
print(train_sc_df)

X_train = train_sc_df.dropna().drop('Scaled', axis=1)
Y_train = train_sc_df.dropna()[['Scaled']]
# dropna()가 none이 포함되어있는 부분을 제외해버리기때문에 앞의 몇일이 짤린다.
# 예를 들면 7일차 예측하려면 이전의 5일치가 2,3,4,5,6이어서 상관없지만
# 4알차 예측하려면 이전의 5일치 -1 0 1 2 3이어서
# -1 0 1 2 3 이어서 -1과 0을 자르는 작업
print(X_train)
#      shift_1   shift_2   shift_3   shift_4   shift_5
#5635  0.181460  0.240631  0.250493  0.289941  0.289941
#5636  0.191321  0.181460  0.240631  0.250493  0.289941
#5637  0.171598  0.191321  0.181460  0.240631  0.250493
#5638  0.201183  0.171598  0.191321  0.181460  0.240631
#5639  0.201183  0.201183  0.171598  0.191321  0.181460

# 이런식으로 nan이 있는 것들이 제거된다.
print(Y_train)


# 2일치 1일치
X_test = test_sc_df.dropna().drop('Scaled', axis=1)
Y_test = test_sc_df.dropna()[['Scaled']]
#이전 5일치 종가 x 현재종가 hy

# print(X_train)

X_train = X_train.values
print('X_train.values를 대입한 X_train값')
print(X_train)
X_test = X_test.values

Y_train = Y_train.values
print("Y_train")
print(type(Y_train)) #<class 'numpy.ndarray'>
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
#sequential : 레이어들이 일렬로 쭉 나열된 형태
# model과 다른 것은 모델의 맨 앞에 input이 없다는 것
#sequential의 경우, 첫번째 레이어에서 input_shape에 input의 데이터 형
#태를 함께 넘겨준다.
model.add(LSTM(20, input_shape=(sequence_length, 1)))# (timestep, feature)
#https://tykimos.github.io/2017/04/09/RNN_Layer_Talk/

# ex) LSTM(20, input_shape=(sequence_length,1)) 뒤에 dense레이어도 필요
#첫번째 인자 : 메모리 셀의 개수(노드의 개수)입니다.@@@이걸 우리가 정할 수 있다.
#타임스텝(하나의 샘플에 포함된 시퀀스 개수) sequence_length개, 속성이 1개
# 매 샘플마다 sequence length개의 값을 입력
# 입력되는 종가 1개 당 하나의 인덱스 값을 입력하므로 속성이 한개
#input_dim : 입력 속성 수 입니다.

# ex) LSTM(3, input_dim=1)
#첫번째 인자 : 메모리 셀의 개수입니다.
#input_dim : 입력 속성 수 입니다.

# model.add(Dense(20))
# model.add(LSTM(20))


# LSTM추가하는 경우의수
# 조기종료 drop out 0에서 1사이의 값 정하는 거 있고
# 가중치 규제는 빼자 batch size는 10-20, unit은 sequence length보다는 커야
# 몇일 전 ; 최근 12일꺼 확인



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#1. 검증셋과 시험셋으로는 가중치 갱신이 일어나지 않습니다.
#2. 검증셋은 모델을 튜닝할 때 사용합니다. 하이퍼 파라미터를 찾는다던지,
# 학습을 중단할 시점을 찾는다던지요.
#3. 그럼 시험셋으로 모델을 튜닝하면 안되나요? 안됩니다. 검증셋은 어떻게
# 보면 객관적인 입장에서 본다는 의미이고, 시험셋은 그야말로 평가에 목적을 두고 있습니다.
# 시험셋에 맞추어 모델을 튜닝하게 되면, 시험은 잘 볼지 몰라도 실전에서는 잘 안될 수 있습니다.
'''

학생들이 스스로 학습 상태를 확인하고 학습 방법을 바꾸거나 학습을 중단하는 시점을 정할 수 없을까요? 
이를 위해서 검증셋이 필요합니다. 학습할 때는 모의고사 1회~4회만 사용하고, 모의고사 5회분을 검증셋으로
 두어 학습할 때는 사용하지 않습니다. 이 방식은 두 가지 효과를 얻을 수 있습니다. 첫번째로 학습 방법을
  바꾼 후 훈련셋으로 학습을 해보고 검증셋으로 평가해볼 수 있습니다. 검증셋으로 가장 높은 평가를 받은 
  학습 방법이 최적의 학습 방법이라고 생각하면 됩니다. 이러한 학습 방법을 결정하는 파라미터를 
  하이퍼파라미터(hyperparameter)라고 하고 최적의 학습 방법을 찾아가는 것을 하이퍼파라미터 튜닝이라고 합니다.

이 상태는 아직 학습이 덜 된 상태 즉 학습을 더 하면 성능이 높아질 가능성이 있는 상태입니다.
 이를 언더피팅(underfitting)이라고 합니다. 담임선생님 입장에서 학생들을 평생 반복 학습만 시킬 수 없으므로 
 (하교도 해야하고, 퇴근도 해야하고) 학생들의 학습 상태를 보면서 ‘아직 학습이 덜 되었으니 계속 반복하도록!’ 
 또는 ‘충분히 학습했으니 그만해도 돼’ 라는 판단을 내려야 합니다. 그 판단 기준이 무엇일까요? 
 에포크를 계속 증가시키다보면 더 이상 검증셋의 평가는 높아지지 않고 오버피팅이 되어 오히려 틀린 개수가 많아집니다. 
 이 시점이 적정 반복 횟수로 보고 학습을 중단합니다. 이를 조기종료(early stopping)이라고 합니다.

자동차 운전 시험으로 예를 들어보겠습니다. 학습은 운전학원에서 배우는 것이고, 검증은 친구한테 검증받는다고 가정하죠.
 시험은 정말 운전면허시험장에서 시험을 치는 것이구요. 실제로는 친구한테 검증을 받거나 운전면허시험장에서 시험칠 때 
 가장 많이 학습이 되지만, 원칙상 학습은 운전학원에서만 한다고 가정합니다. 이 때, 운전면허시험장에 맞추어 모델을 
 튜닝해버리면, 면허증은 딸 수 있어도, 실전엔 많이 약할 수 있겠죠? 친구 검증을 통해 충분히 모델을 튜닝한 다음, 
 시험을 치루는 것이 올바른 선택입니다.

제가 데이터 사이언스는 아니지만, 세가지 데이터셋 중에 가장 중요한 것을 선택하라면, 바로 "검증셋"일 겁니다. 
우리는 시험셋에 잘 평가받는 모델을 만드는 것이 목표가 아니라, 실전에도 잘 운용될 수 있는 모델을 만드는 것이 목표이니깐요.
'''
#dense에 대해선 여기 url 참고
#https://tykimos.github.io/2017/04/09/RNN_Layer_Talk/

model.add(Dense(1)) # output = 1
#ex)모델.add(keras.layers.Dense(1,input_shape(1,)))
#Dense는 완전연결계층으로 여러 레이어들중 하나입니다.
# 가장 대표적인 레이어이죠. w*x+b와 같은 계산을 수행하는 레이어입니다.
#  Dense의 첫번째 매개변수는 노드의 갯수입니다. 그리고 input_shape는 입력값의 모양입니다.
#  배열의 차원이라고 하면 이해가 쉬울 것 입니다.
# (1,)라고 되어있는 이유는 1개씩 여러번 해야하기에 정하지 않았다는 의미로 남겨둡니다.
# dense output 어떻게 나올지

#즉 lstm과 dense를 추가했으므로 LSTM레이어 1개와 Dense 레이어로 구성

model.compile(loss='mean_squared_error', optimizer='adam')
# loss로 계산을 해서 오차값을 어떻게 수정하는지의 알고리즘. 그 중에 adam이 보편적으로 괜찮다고 한다.


#unit : 노드개수 5일치보다 적음-
#가중치 랜덤



# loss를 모니터링해서 patience만큼 연속으로 loss률이 떨어지지 않으면 훈련을 멈춘다.
early_stop = [EarlyStopping(monitor='val_loss', patience=20, verbose=1),

#학습을 하면 loss 오차가 줄어야 over training
#조기종료 : drop out 가중치 제한  patience : loss율이 0.000  연속으로 몇번 떨어지지 않을떄까지 버티는 것 10 9 8 7 계속 10번 반복되면 개선불가 stop



ModelCheckpoint(filepath='best_model_close', monitor='val_loss', save_best_only=True)]
#

# history=model.fit(X_train_t, Y_train, epochs=100, batch_size=30, verbose=1, callbacks=[early_stop])
#verbose : ? print창이 뜬다
# history=model.fit(X_train_t, Y_train, epochs=100, batch_size=30, verbose=1, callbacks=[early_stop])
#epoch : 반복횟수 batch size : 에러가지고 가중치 개선 에러를 어느 주기마다 개선할건지 보는
#것  batch size가 1이면 하나의 케이스 오차 있으면 바로 적용 한번에 30개

history = model.fit(X_train_t, Y_train,
                    epochs=800, verbose=2, batch_size=10, validation_data=(X_test_t, Y_test),
                    callbacks=early_stop)
'''
fit 함수 인자로 shuffle 이라는 옵션이 있습니다. 
이 옵션을 활성화하면 랜덤하게 적용이 될 것 같아요~ 
두번째 질문에서는 shuffle은 매 에포크마다 처음에 한 번 섞어주는 것이기에 
한 에포크내에서 이미 사용된 배치가 또 사용되는 일은 없을 듯 합니다~

시계열 데이터므로 shuffle하면 안된다
'''

#unit batch epoch 이전 몇일 치 sequence length,
# Y_pred = model.predict(X_test_t)

'''
케라스에서 학습시킬 때 fit 함수를 사용합니다. 
이 함수의 반환 값으로 히스토리 객체를 얻을 수 있는데, 이 객체는 다음의 정보를 담고 있습니다.

매 에포크 마다의 훈련 손실값 (loss)
매 에포크 마다의 훈련 정확도 (acc)
매 에포크 마다의 검증 손실값 (val_loss)
매 에포크 마다의 검증 정확도 (val_acc)
'''
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