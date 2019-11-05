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
# with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr',
#                        False):  # more options can be specified also
#     print(stock_data_df)

# closing_price high_price low_price volume
training_set = stock_data_df.iloc[:, [1, 4, 5, 6]].values
# print(training_set.shape)

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
print("TSS > ", training_set_scaled)
print("TSS > ", training_set_scaled.shape)

#
# X_train = []
# y_train = []
# for i in range(60, len(stock_data_df) - 120):
#     X_train.append(training_set_scaled[i-60:i, :])
#     y_train.append(training_set_scaled[i, 0])
# X_train, y_train = np.array(X_train), np.array(y_train)
#
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))
#
# real_stock_price = stock_data_df.iloc[-120:, [1]].values
# print("real stock price 120 day > ", real_stock_price[:, 0])
#
#
# regressor = Sequential()
#
# regressor.add(LSTM(units=60, return_sequences=True, input_shape=(X_train.shape[1], 4)))
# regressor.add(Dropout(0.2))
#
# regressor.add(LSTM(units=60, return_sequences=True))
# regressor.add(Dropout(0.2))
#
# regressor.add(LSTM(units=60, return_sequences=True))
# regressor.add(Dropout(0.2))
#
# regressor.add(LSTM(units=60))
# regressor.add(Dropout(0.2))
#
# regressor.add(Dense(units=1))
#
# regressor.compile(optimizer='adam', loss='mse')
#
# regressor.fit(X_train, y_train, epochs=20, batch_size=32)
#
# dataset_total = stock_data_df.iloc[:, [1, 4, 5, 6]]
# inputs = dataset_total.values
# print(inputs)
# inputs = inputs.reshape(-1, 4)
# inputs = sc.transform(inputs)
# print(inputs)
#
# X_test = []
# for i in range(len(inputs)-120, len(inputs)):
#     X_test.append(inputs[i-60:i, :])
# X_test = np.array(X_test)
# print(X_test)
# print(X_test.shape)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))
# predicted_stock_price = regressor.predict(X_test)
#
# print("predicted stock price > ", predicted_stock_price)
# prepro_predicted_stock_price = []
# for psp in predicted_stock_price:
#     prepro_predicted_stock_price.append([psp[0], 0, 0, 0])
# prepro_predicted_stock_price = np.array(prepro_predicted_stock_price)
# print(prepro_predicted_stock_price)
# predicted_stock_price = sc.inverse_transform(prepro_predicted_stock_price)
#
# plt.plot(real_stock_price[:], color='black', label='real')
# plt.plot(predicted_stock_price[:, 0], color='green', label='predict')
# plt.title('TATA Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('TATA Stock Price')
# plt.legend()
# plt.show()
#
# # # lstm_model.save('./stock/model/{}_model.h5'.format(corp_code))
