import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from pandas.tseries.offsets import  MonthEnd

from sklearn.preprocessing import MinMaxScaler


from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping


# 파일 읽어오기


df = pd.read_csv('cansim-0800020-eng-6674700030567901031.csv', skiprows=6, skipfooter=9, engine='python')

# print(df.head())

#  Jan-1991와 같은 형태를 1991-01-31과 같이 변형
#  월을 숫자로 바꿔주고 일을 MonthEnd() 를 사용해서 각 월의 마지막날이 되도록 세팅
df['Adjustments'] = pd.to_datetime(df['Adjustments'])+MonthEnd()
# df['Adjustments'] = pd.to_datetime(df['Adjustments'])

# print(df.head())

# df의 인덱스를 Adjustments로 세팅. plot과 같이 그래프 그릴대 X축기준이 된다.
# 인덱스가 되면서 value값으로는 취급이 안되는듯(아래 X_train.values 하면 Adjustment에 대한 정보는 빠진다.

df = df.set_index('Adjustments')



# print(df.head())

# df.plot()
# plt.show()


# split_date = pd.Timestamp('01-01-2011')
split_date = pd.Timestamp('2011-01-01')

# print(split_date)

train = df.loc[:split_date, ['Unadjusted']]
test = df.loc[split_date:, ['Unadjusted']]

# print(test)
ax = train.plot()
# print(ax)
# ax2 = test.plot()
test.plot(ax=ax)
plt.legend(['train', 'test'])

# plt.show()


# 변수 Scaling
sc = MinMaxScaler()

train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)

# print(train_sc)
# print(test_sc)
train_sc_df = pd.DataFrame(train_sc, columns=['Scaled'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Scaled'], index=test.index)

# print(train_sc_df.head())
# print(test_sc_df.head())

for s in range(1, 13):
    train_sc_df['shift_{}'.format(s)] = train_sc_df['Scaled'].shift(s)
    test_sc_df['shift_{}'.format(s)] = test_sc_df['Scaled'].shift(s)

# print(train_sc_df.head(13))

X_train = train_sc_df.dropna().drop('Scaled', axis=1)
Y_train = train_sc_df.dropna()[['Scaled']]

X_test = test_sc_df.dropna().drop('Scaled', axis=1)
Y_test = test_sc_df.dropna()[['Scaled']]

# print(X_train.head())
# print(Y_train.head())

X_train = X_train.values
# print("X_train.values")
# print(X_train)
X_test = X_test.values
Y_train = Y_train.values
Y_test = Y_test.values
# print(X_train.shape)
# print(X_train)
# print(Y_train.shape)
# print(Y_train)

X_train_t = X_train.reshape(X_train.shape[0], 12, 1)
X_test_t = X_test.reshape(X_test.shape[0], 12, 1)
# print("최종 DATA")
# print(X_train_t.shape)
# print(X_train_t)
# print(Y_train)

# LSTM 모델 만들기
# Clear session
# 현재의 TF graph를 버리고 새로 만든다. 예전 모델, 레이어와의 충돌을 피한다.
K.clear_session()
model = Sequential() # Sequential Model
model.add(LSTM(20, input_shape=(12, 1)))  # (timestep, feature)
model.add(Dense(1)) # output = 1
model.compile(loss='mean_squared_error', optimizer='adam')

# loss를 모니터링해서 patience만큼 연속으로 loss률이 떨어지지 않으면 훈련을 멈춘다.
early_stop = EarlyStopping(monitor='loss', patience=8, verbose=1)

# history=model.fit(X_train_t, Y_train, epochs=100, batch_size=30, verbose=1, callbacks=[early_stop])

history = model.fit(X_train_t, Y_train, epochs=1000, verbose=1, batch_size=10, validation_data=(X_test_t, Y_test), callbacks=[early_stop])

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


plt.figure(3)

count= range(1, len(Y_pred)+1)
# print(Y_test)
# print(Y_pred)
plt.plot(count, Y_test, "r--")
plt.plot(count, Y_pred, "b-")

plt.legend(["Y_test", "Y_pred"])

Y_pred = model.predict(X_test_t)

# plt.figure(4)
#
# plt.plot(count, Y_test, "r--")
# plt.plot(count, Y_pred, "b-")
#
# plt.legend(["Y_test", "Y_pred"])



plt.show()



# plt.show()
# print(Y_pred)