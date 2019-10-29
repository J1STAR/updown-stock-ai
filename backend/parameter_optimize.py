import pandas as pd
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib
import requests

# be able to save images on server
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy


# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df = df.drop(0)
    return df


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
    new_row = [x for x in X] + [yhat]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# evaluate the model on a dataset, returns RMSE in transformed units
def evaluate(model, raw_data, X, Y, scaler, offset, batch_size):
    # separate
    # X, y = scaled_dataset[:, 0:-1], scaled_dataset[:, -1]
    # reshape
    reshaped = X.reshape(len(X), 1, 5)
    # forecast dataset
    output = model.predict(reshaped, batch_size=batch_size)
    # invert data transforms on forecast
    predictions = list()
    for i in range(len(output)):
        yhat = output[i, 0]
        # invert scaling
        yhat = invert_scale(scaler, X[i], yhat)
        # invert differencing
        yhat = yhat + raw_data[i]
        # store forecast
        predictions.append(yhat)
    # report performance
    rmse = sqrt(mean_squared_error(raw_data[1:], predictions))
    return rmse


# fit an LSTM network to training data
# def fit_lstm(train, test, raw, scaler, batch_size, nb_epoch, neurons):
#     X, y = train[:, 0:-1], train[:, -1]
#     X = X.reshape(X.shape[0], 1, X.shape[1])
#     # prepare model
#     model = Sequential()
#     model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     # fit model
#     train_rmse, test_rmse = list(), list()
#     for i in range(nb_epoch):
#         model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
#         model.reset_states()
#         # evaluate model on train data
#         raw_train = raw[-(len(train) + len(test) + 1):-len(test)]
#         train_rmse.append(evaluate(model, raw_train, train, scaler, 0, batch_size))
#         model.reset_states()
#         # evaluate model on test data
#         raw_test = raw[-(len(test) + 1):]
#         test_rmse.append(evaluate(model, raw_test, test, scaler, 0, batch_size))
#         model.reset_states()
#     history = DataFrame()
#     history['train'], history['test'] = train_rmse, test_rmse
#     return history

def fit_lstm(X_train, Y_train, X_test, Y_test, raw, sequence_length, scaler, batch_size, nb_epoch, neurons):
    # X, y = train[:, 0:-1], train[:, -1]
    # X = X.reshape(X.shape[0], 1, X.shape[1])
    # prepare model
    X = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit model
    train_rmse, test_rmse = list(), list()
    for i in range(nb_epoch):
        model.fit(X, Y_train, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()
        # evaluate model on train data
        raw_train = raw[-(len(X_train) + len(X_test) + 1):-len(X_test)]
        # train_rmse.append(evaluate(model, raw_train, train, scaler, 0, batch_size))
        train_rmse.append(evaluate(model, raw_train, X_train, Y_train, scaler, 0, batch_size))
        model.reset_states()
        # evaluate model on test data
        raw_test = raw[-(len(X_test) + 1):]
        test_rmse.append(evaluate(model, raw_test, X_test, Y_test, scaler, 0, batch_size))
        model.reset_states()
    history = DataFrame()
    history['train'], history['test'] = train_rmse, test_rmse
    return history

def read_data():
    res = requests.get("http://j1star.ddns.net:8000/stock/corp/067280")
    data = res.json()
    stock_info_list = data['corp']['stock_info']
    # date open high low close volumn
    pre_data_list = []
    for stock_info in stock_info_list:
        pre_dataset = [stock_info['date'][:10], stock_info['open_price'], stock_info['high_price'],
                       stock_info['low_price'], stock_info['closing_price'], stock_info['volume']]
        pre_data_list.append(pre_dataset)
    df_stock_info = pd.DataFrame(pre_data_list, columns=['날짜', '시가', '고가', '저가', '종가', '거래량'])
    df_stock_info = df_stock_info[df_stock_info.종가 != 0]

    return df_stock_info

def train_test(data, train_length, test_length, sequence_length):
    data.set_index('날짜')
    maxlength = len(data)
    train = data.loc[maxlength - (train_length+test_length):maxlength - (test_length+1), ['종가']]
    test = data.loc[maxlength - (test_length+sequence_length):, ['종가']]
    return data.loc[maxlength-(train_length+test_length):,['종가']] ,train, test
def preprocessing(train, test, sequence_length):
    # 스케일링 0~1값으로 (train 데이터 기준)
    sc = MinMaxScaler(feature_range=(-1,1))
    train_sc = sc.fit_transform(train)
    test_sc = sc.transform(test)

    train_sc_df = pd.DataFrame(train_sc, columns=['종가'], index=train.index)
    test_sc_df = pd.DataFrame(test_sc, columns=['종가'], index=test.index)

    # print("train_sc_df")
    # print(train_sc_df)

    for s in range(1, sequence_length + 1):
        train_sc_df['{}일전 종가'.format(s)] = train_sc_df['종가'].shift(s)
        test_sc_df['{}일전 종가'.format(s)] = test_sc_df['종가'].shift(s)

    # print(train_sc_df)

    X_train = train_sc_df.dropna().drop('종가', axis=1)
    Y_train = train_sc_df.dropna()[['종가']]
    # dropna()가 none이 포함되어있는 부분을 제외해버리기때문에 앞의 몇일이 짤린다.
    # print(X_train)
    # print(Y_train)

    X_test = test_sc_df.dropna().drop('종가', axis=1)
    Y_test = test_sc_df.dropna()[['종가']]

    # print(X_train)

    X_train = X_train.values
    # print(X_train)
    X_test = X_test.values

    Y_train = Y_train.values
    # print("Y_train")
    # print(type(Y_train))
    # print(Y_train)
    Y_test = Y_test.values

    # X_train_t = X_train.reshape(X_train.shape[0], sequence_length, 1)
    # X_test_t = X_test.reshape(X_test.shape[0], sequence_length, 1)

    return X_train, Y_train, X_test, Y_test, sc

# run diagnostic experiments
def run():
    # load dataset
    # series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    data = read_data()
    sequence_length = 5
    raw_data, train, test = train_test(data, 720, 240, sequence_length)
    raw_data = raw_data.values
    raw_data = raw_data.reshape(1,len(raw_data))[0]
    X_train, Y_train, X_test, Y_test, scaler = preprocessing(train, test, sequence_length)
    print(train)
    print(test)
    # transform data to be stationary
    # raw_values = series.values
    # raw_values = df_stock_info.values
    # diff_values = difference(raw_values, 1)
    # transform data to be supervised learning
    # supervised = timeseries_to_supervised(diff_values, 1)
    # supervised_values = supervised.values
    # split data into train and test-sets
    # train, test = supervised_values[0:-12], supervised_values[-12:]
    # transform the scale of the data
    # scaler, train_scaled, test_scaled = scale(train, test)
    # fit and evaluate model
    # train_trimmed = train_scaled[2:, :]
    # config
    repeats = 5
    n_batch = 5
    n_epochs = 100
    n_neurons = 1
    # run diagnostic tests
    for i in range(repeats):
        history = fit_lstm(X_train, Y_train, X_test, Y_test, raw_data, sequence_length, scaler, n_batch, n_epochs, n_neurons)
        pyplot.plot(history['train'], color='blue')
        pyplot.plot(history['test'], color='orange')
        print('%d) TrainRMSE=%f, TestRMSE=%f' % (i, history['train'].iloc[-1], history['test'].iloc[-1]))
    pyplot.savefig('epochs_diagnostic.png')


# entry point
run()