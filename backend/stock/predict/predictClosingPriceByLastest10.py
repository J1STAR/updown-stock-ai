import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.externals import joblib


def predict(code, data_list):
    dataset = []
    for data in data_list:
        dataset.append(dict(data))

    closing_df = pd.DataFrame(dataset, columns=['closing_price'])
    low_df = pd.DataFrame(dataset, columns=['low_price'])
    high_df = pd.DataFrame(dataset, columns=['high_price'])
    open_df = pd.DataFrame(dataset, columns=['open_price'])

    saved_closing_sc = joblib.load(filename='stock/model/' + code + '_close_scaler.pkl')
    saved_low_sc = joblib.load(filename='stock/model/' + code + '_low_scaler.pkl')
    saved_high_sc = joblib.load(filename='stock/model/' + code + '_high_scaler.pkl')
    saved_open_sc = joblib.load(filename='stock/model/' + code + '_open_scaler.pkl')

    scaled_closing_df = saved_closing_sc.transform(closing_df)
    scaled_low_df = saved_low_sc.transform(low_df)
    scaled_high_df = saved_high_sc.transform(high_df)
    scaled_open_df = saved_open_sc.transform(open_df)

    close_trained_model = keras.models.load_model('stock/model/' + code + '_best_model_close.h5')
    close_pred = close_trained_model.predict(np.array([scaled_closing_df]))

    low_trained_model = keras.models.load_model('stock/model/' + code + '_best_model_low.h5')
    low_pred = low_trained_model.predict(np.array([scaled_low_df]))

    high_trained_model = keras.models.load_model('stock/model/' + code + '_best_model_high.h5')
    high_pred = high_trained_model.predict(np.array([scaled_high_df]))

    open_trained_model = keras.models.load_model('stock/model/' + code + '_best_model_open.h5')
    open_pred = open_trained_model.predict(np.array([scaled_open_df]))

    return {
        "close": saved_closing_sc.inverse_transform(close_pred)[0][0],
        "low": saved_low_sc.inverse_transform(low_pred)[0][0],
        "high": saved_high_sc.inverse_transform(high_pred)[0][0],
        "open": saved_open_sc.inverse_transform(open_pred)[0][0]
    }
