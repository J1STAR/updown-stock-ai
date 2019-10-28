import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from pandas.tseries.offsets import  MonthEnd

from sklearn.preprocessing import MinMaxScaler


from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import Dropout
import keras.backend.tensorflow_backend as K
from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from data_manager import preprocess, build_training_data

import requests

tf.compat.v1.set_random_seed(777)

if __name__ == '__main__':
    res = requests.get("http://j1star.ddns.net:8000/stock/corp/005930")
    data = res.json()
    stock_info_list = data['corp']['stock_info']
    # date open high low close volumn
    pre_data_list = []
    for stock_info in stock_info_list:
        pre_dataset = [stock_info['date'][:10], stock_info['open_price'], stock_info['high_price'], stock_info['low_price'], stock_info['closing_price'], stock_info['volume']]
        pre_data_list.append(pre_dataset)
    df_stock_info = pd.DataFrame(pre_data_list, columns =['date', 'open', 'high', 'low', 'close', 'volume'])
    # df_stock_info = df_stock_info[df_stock_info.open != 0]
    # print(df_stock_info)
    # print(df_stock_info)

    prep_data = preprocess(df_stock_info)
    stock_data = build_training_data(prep_data)

    # # 기간 필터링
    # training_data = training_data[(training_data['date'] >= '2016-01-01') &
    #                               (training_data['date'] <= '2016-12-31')]
    # 결측값 제거
    # training_data = stock_data.dropna()

    # 차트 데이터 분리
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
    chart_data = stock_data[features_chart_data]

    # 학습 데이터 분리
    features_training_data = [
        'date',
        'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 'volume_ma20_ratio',
        'close_ma60_ratio', 'volume_ma60_ratio',
        'close_ma120_ratio', 'volume_ma120_ratio'
    ]
    training_data = stock_data[features_training_data]


    # training_data.set_index('date')
    # print(training_data)

    maxlength = len(training_data)
    # print(maxlength)
    train = training_data.loc[maxlength-240:maxlength-61, features_training_data]
    test = training_data.loc[maxlength-60:, features_training_data]

    sc = MinMaxScaler()

    close_data_sc = sc.fit_transform(stock_data.loc[maxlength-240:, ['close']])

    Y_train = close_data_sc.loc[maxlength-239:maxlength-60, ['close']]
    close_test = close_data_sc.loc[maxlength-:, ['close']]

    train['close'] = close_data_sc.loc[:181, ['close']]
    test['close'] = close_data_sc.loc[181:, ['close']]

    print("test")
    print(train)
    print(test)

    ax = train.plot()
    # print(ax)
    # ax2 = test.plot()
    test.plot(ax=ax)
    plt.legend(['train', 'test'])
    # plt.show()


    # print("train_sc")
    # print(train_sc)

    train_sc_df = pd.DataFrame(train, columns=['close'], index=train.index)
    test_sc_df = pd.DataFrame(test, columns=['close'], index=test.index)

    # print("train_sc_df")
    # print(train_sc_df)

    # for s in range(1, sequence_length + 1):
    #     train_sc_df['{}일전 종가'.format(s)] = train_sc_df['종가'].shift(s)
    #     test_sc_df['{}일전 종가'.format(s)] = test_sc_df['종가'].shift(s)

    # print(train_sc_df)

    X_train = train.dropna().drop('date', axis = 1)
    Y_train = train.dropna()[['close']]
    Y_train = Y_train[1:]
    # dropna()가 none이 포함되어있는 부분을 제외해버리기때문에 앞의 몇일이 짤린다.

    X_test = test.dropna().drop('date', axis = 1)
    Y_test = test.dropna()[['close']]
    Y_train = Y_train.append(Y_test[0:1], ignore_index = True)
    X_test = X_test[:-1]
    Y_test = Y_test[1:]
    print(X_train)
    print(Y_train)

    X_train = X_train.values
    Y_train = Y_train.values

    X_test = X_test.values
    Y_test = Y_test.values
    # print(X_train)
    # print(Y_train)

    X_train_t = X_train.reshape(X_train.shape[0], 16, 1)
    X_test_t = X_test.reshape(X_test.shape[0], 16, 1)

    # print("최종 DATA")
    # print(type(X_train_t))
    # print(X_train_t)
    # print(X_test_t)

    # LSTM 모델 만들기
    # Clear session
    # 현재의 TF graph를 버리고 새로 만든다. 예전 모델, 레이어와의 충돌을 피한다.
    K.clear_session()

    model = Sequential() # Sequential Model
    model.add(LSTM(256, input_shape=(16, 1)))# (timestep, feature)
    # model.add(LSTM(256, input_shape=(sequence_length, 1), return_sequences=True, stateful=False))# (timestep, feature)
    # model.add(BatchNormalization())
    # model.add(LSTM(128, return_sequences=True, stateful=False))# (timestep, feature)
    # model.add(BatchNormalization())
    # model.add(LSTM(64, return_sequences=False, stateful=False))# (timestep, feature)
    # model.add(Dense(100))
    # model.add(Dense(100))
    # model.add(Dropout(0.3))
    model.add(Dense(1)) # output = 1
    model.compile(loss='mean_squared_error', optimizer='adam')

    # early_stop = [EarlyStopping(monitor='val_loss', patience=20, verbose=1), ModelCheckpoint(filepath='best_model_close', monitor='val_loss', save_best_only=True)]

    # history=model.fit(X_train_t, Y_train, epochs=100, batch_size=30, verbose=1, callbacks=[early_stop])

    # history = model.fit(X_train_t, Y_train, epochs=1000, verbose=2, batch_size=30, validation_data=(X_test_t, Y_test), callbacks=early_stop)
    history = model.fit(X_train_t, Y_train, epochs=1000, verbose=2, batch_size=1, validation_data=(X_test_t, Y_test))

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

    # best_model = load_model('best_model_close')

    # Y_pred_best = best_model.predict(X_test_t)


    plt.figure(3)

    count= range(1, len(Y_pred)+1)
    # print(Y_test)
    # print(Y_pred)
    Y_pred_re = np.copy(Y_pred)
    Y_pred_re[0:len(Y_pred)-1] = np.copy(Y_pred[1:len(Y_pred)])
    plt.plot(count, Y_test, "r--")
    plt.plot(count, Y_pred, "b-")

    plt.legend(["Y_test", "Y_pred_by_close"])

    Y_pred = model.predict(X_test_t)

    # plt.figure(4)
    # #
    #
    # Y_pred_re = np.copy(Y_pred_best)
    # Y_pred_re[0:len(Y_pred)-1] = np.copy(Y_pred_best[1:len(Y_pred)])
    # plt.plot(count, Y_test, "r--")
    # plt.plot(count, Y_pred_best, "b-")
    # #
    # plt.legend(["Y_test", "Y_pred_best"])


    print(Y_test)
    print(Y_pred)

    count = 0
    best_count = 0
    for val in range(1, len(Y_test)):
        test_val = Y_test[val]-Y_test[val-1]
        pred_val = Y_pred[val]-Y_test[val-1]
        # pred_best_val = Y_pred_best[val]-Y_test[val-1]
        if test_val > 0:
            test_val = 1
        else:
            test_val = -1
        if pred_val > 0:
            pred_val = 1
        else:
            pred_val = -1
        # if pred_best_val > 0:
        #     pred_best_val = 1
        # else:
        #     pred_best_val = -1
        # if test_val == pred_best_val:
        #     best_count+=1

        if test_val == pred_val:
            count+=1

    print("count = ", count)
    print("총 개수 = ", len(Y_test)-1)
    print("모델 정답률 : ", count/(len(Y_test)-1))
    # print("best_count = ", best_count)
    # print("베스트 모델 정답률 : ", best_count/(len(Y_test)-1))

    plt.show()