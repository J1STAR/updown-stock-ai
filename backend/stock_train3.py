import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.core import Activation
from sklearn.preprocessing import MinMaxScaler


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

high_prices = stock_data_df['high_price'].values
low_prices = stock_data_df['low_price'].values
mid_prices = (high_prices + low_prices) / 2

seq_len = 50
sequence_length = seq_len + 1

result = []
for index in range(len(mid_prices) - sequence_length):
    result.append(mid_prices[index: index + sequence_length])

normalized_data = []
for window in result:
    normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
    normalized_data.append(normalized_window)

result = np.array(normalized_data)
print("result > ", result.shape)


# split train and test data
row = int(round(result.shape[0] * 0.9))
train = result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]
print(x_train, y_train)

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]


model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))

model.add(LSTM(64, return_sequences=False))

model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop')

model.summary()

model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=64,
    epochs=1)

pred = model.predict(x_test)

fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()