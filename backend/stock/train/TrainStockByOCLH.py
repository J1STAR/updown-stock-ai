import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from pandas.tseries.offsets import MonthEnd

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
from sklearn.externals import joblib
from multiprocessing import Pool

tf.compat.v1.set_random_seed(777)

def train_stock_close(corp_code):
    res = requests.get("http://j1star.ddns.net:8000/stock/corp/" + corp_code)
    data = res.json()
    stock_info_list = data['corp']['stock_info']

    type_list = ['open', 'low', 'high', 'close']
    for t in type_list:
        pre_data_list = []
        for stock_info in stock_info_list:
            pre_dataset = [stock_info['date'][:10], stock_info['open_price'], stock_info['high_price'],
                           stock_info['low_price'], stock_info['closing_price'], stock_info['volume']]
            pre_data_list.append(pre_dataset)
        df_stock_info = pd.DataFrame(pre_data_list, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df_stock_info = df_stock_info[df_stock_info.open != 0]

        sequence_length = 10

        df_stock_info.set_index('date')

        maxlength = len(df_stock_info)
        train = df_stock_info.loc[maxlength - 480 + sequence_length:maxlength - 121, [t]]
        test = df_stock_info.loc[maxlength - 120 + sequence_length:, [t]]

        sc = MinMaxScaler(feature_range=(0, 1))

        train_sc = sc.fit_transform(train)
        joblib.dump(sc, '../model/' + corp_code + '_{}_scaler.pkl'.format(t))
        test_sc = sc.transform(test)

        train_sc_df = pd.DataFrame(train_sc, columns=[t], index=train.index)
        test_sc_df = pd.DataFrame(test_sc, columns=[t], index=test.index)

        for s in range(1, sequence_length + 1):
            train_sc_df['{}일전 종가'.format(s)] = train_sc_df[t].shift(s)
            test_sc_df['{}일전 종가'.format(s)] = test_sc_df[t].shift(s)


        X_train = train_sc_df.dropna().drop(t, axis=1)
        Y_train = train_sc_df.dropna()[[t]]
        # dropna()가 none이 포함되어있는 부분을 제외해버리기때문에 앞의 몇일이 짤린다.

        X_test = test_sc_df.dropna().drop(t, axis=1)
        Y_test = test_sc_df.dropna()[[t]]

        X_train = X_train.values
        X_test = X_test.values

        Y_train = Y_train.values
        Y_test = Y_test.values

        tmp = len(Y_test) % sequence_length
        if tmp != 0:
            Y_test = Y_test[tmp:]

        X_test = X_test[tmp:]

        X_train_t = X_train.reshape(X_train.shape[0], sequence_length, 1)
        X_test_t = X_test.reshape(X_test.shape[0], sequence_length, 1)


        # LSTM 모델 만들기
        # Clear session
        # 현재의 TF graph를 버리고 새로 만든다. 예전 모델, 레이어와의 충돌을 피한다.
        K.clear_session()

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
        model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(1))

        adam = optimizers.adam(learning_rate=0.0001)
        model.compile(loss='mean_squared_error', optimizer=adam)

        # loss를 모니터링해서 patience만큼 연속으로 loss률이 떨어지지 않으면 훈련을 멈춘다.
        early_stop = [EarlyStopping(monitor='val_loss', patience=20, verbose=1),
                      ModelCheckpoint(filepath='../model/' + corp_code + '_best_model_{}.h5'.format(t), monitor='val_loss', save_best_only=True)]

        history = model.fit(X_train_t, Y_train, epochs=100, verbose=2, batch_size=10, validation_data=(X_test_t, Y_test),
                            callbacks=early_stop)


if __name__ == '__main__':
    # res = requests.get("http://j1star.ddns.net:8000/stock/corp")
    #
    # corp_list = res.json()['corp_code_list']

    corp_list = ['263750', '036570', '067160', '078340']

    pool = Pool(processes=1)
    pool.map(train_stock_close, corp_list)

