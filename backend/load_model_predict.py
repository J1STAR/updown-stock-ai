from keras.models import load_model
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

BATCH_SIZE = 10
TIME_STEPS = 50
lr = 1e-3

corp_code = "005930"
stock_res = requests.get('http://j1star.ddns.net:8000/stock/corp/' + corp_code).json()
stock_res = stock_res['corp']['stock_info']
data = pd.DataFrame(stock_res)
data = data[(data['open_price'] > 0) & (data['high_price'] > 0) & (data['closing_price'] > 0)]
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr',
                       False):  # more options can be specified also
    print(data)

seq_len = 50
sequence_length = seq_len + 1

data_cols = ["high_price", "low_price", "closing_price", "volume"]
x = data.loc[-50:, data_cols].values
print(x)

min_max_scaler = MinMaxScaler()
min_max_scaler_closing = MinMaxScaler()

x_predict = min_max_scaler.fit_transform(x)
closing_predict = min_max_scaler_closing.fit_transform(data.loc[-50:, ["closing_price"]])

def build_timeseries(mat, y_col_index):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))

    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS + i]
        y[i] = mat[TIME_STEPS + i, y_col_index]
    print("length of time-series i/o", x.shape, y.shape)
    return x, y


def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0] % batch_size
    if (no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat


x_t, y_t = build_timeseries(x_predict, 2)
x_t = trim_dataset(x_t, BATCH_SIZE)
y_t = trim_dataset(y_t, BATCH_SIZE)
print(x_t)
print(y_t)

lstm_model = load_model('./stock/model/' + corp_code + '_model.h5')
res = lstm_model.predict(x_t, batch_size=BATCH_SIZE)

print(min_max_scaler_closing.inverse_transform(res))
