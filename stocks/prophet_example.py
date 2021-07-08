# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:19:24 2021

@author: xixiu
"""

import datetime
import time
import pandas as pd
from pandas import Series,DataFrame
from Api import download_data
from prophet import Prophet
import matplotlib.pyplot as plt


config = {
    "alpha_vantage": {
        "function":"TIME_SERIES_INTRADAY",
        "key": "PR3XLLYLAN8V9CBY", # Claim your free API key here: https://www.alphavantage.co/support/#api-key
        "symbol": "AMZN",
        "outputsize": "full",
        "interval": "1min",
        "key_close": "4. close",
    },
    "data": {
        "window_size": 3,
        "train_split_size": 0.80,
    }, 
    "plots": {
        "xticks_interval": 10, # show a date every 90 days
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1, # since we are only using 1 feature, close price
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}

data_date, data_close_price, num_data_points, display_date_range = download_data(config)

df = {'ds': data_date,
      'y': data_close_price}
df = DataFrame(df)
df['y'] = (df['y'] - df['y'].mean()) / (df['y'].std())
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=8000, freq='min')
future.tail()
forecast = m.predict(future)

m.plot(forecast)
# m.plot_components(forecast)

x1 = forecast['ds']
y1 = forecast['yhat']
y2 = forecast['yhat_lower']
y3 = forecast['yhat_upper']
plt.plot(x1,y1)
plt.plot(x1,y2)
plt.plot(x1,y3)
plt.show()

