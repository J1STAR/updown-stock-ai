import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.core import Activation


import requests

corp_code = "005930"

stock_data = requests.get('http://j1star.ddns.net:8000/stock/corp/' + corp_code).json()
stock_data = stock_data['corp']['stock_info']
stock_data_df = pd.DataFrame(stock_data, dtype=np.float64)
stock_data_df = stock_data_df[(stock_data_df['open_price'] > 0) & (stock_data_df['high_price'] > 0) & (stock_data_df['closing_price'] > 0)]
stock_data_df["average"] = (stock_data_df['high_price'] + stock_data_df['low_price']) / 2
with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.expand_frame_repr',
                       False):  # more options can be specified also
    print(stock_data_df)


# closing_price high_price low_price volume
training_set = stock_data_df.iloc[:, [1, 4, 5, 6]].values
print("training set > ", training_set)

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
print(len(training_set_scaled))

seq_len = 50
sequence_length = seq_len + 1

result = []
for index in range(len(stock_data_df) - sequence_length):
    result.append(training_set_scaled[index: index + sequence_length])
result = np.array(result)
print("result > ", result)
print("resultshape > ", result.shape)

row = int(round(result.shape[0] * 0.9))
train = result[:row, :-1]
test = result[row:, :-1]
print(train.shape, test.shape)

x_train = np.reshape(train, (train.shape[0], train.shape[1], 4))
y_train = train[:, 0, 0]

x_test = np.reshape(test, (test.shape[0], test.shape[1], 4))
y_test = test[:, 0, 0]
print("x train > ", x_train)
print("y train > ", y_train)

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 4)))
model.add(Dropout(0.5))

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.5))

model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))

model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop')

model.summary()

model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=64,
          epochs=1)

pred = model.predict(x_test)
print(pred)

prepro_pred = []
for p in pred:
    prepro_pred.append([p[0], 0, 0, 0])
prepro_pred = np.array(prepro_pred)
inverse_pred = sc.inverse_transform(prepro_pred)
print(inverse_pred)

fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
# ax.plot(inverse_pred[:, [0]], label='Prediction')
ax.legend()
plt.show()