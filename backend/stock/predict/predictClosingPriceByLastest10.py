import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.externals import joblib


def predict(code, data_list):
    dataset = []
    for data in data_list:
        dataset.append(dict(data))

    df = pd.DataFrame(dataset, columns=['closing_price'])
    print(df)

    saved_sc = joblib.load(filename='stock/model/' + code + '_close_scaler.pkl')
    scaled_df = saved_sc.transform(df)
    print(saved_sc.inverse_transform(scaled_df))

    trained_model = keras.models.load_model('stock/model/' + code + '_best_model_close.h5')
    pred = trained_model.predict(np.array([scaled_df]))

    return saved_sc.inverse_transform(pred)[0][0]
