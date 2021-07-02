# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 16:08:03 2021

@author: xixiu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)


from Api import download_data


from matplotlib import rcParams
rcParams['font.family']='serif'
rcParams['font.sans-serif']=['Times New Roman']

config = {
    "alpha_vantage": {
        "function":"TIME_SERIES_INTRADAY",
        "key": "PR3XLLYLAN8V9CBY", # Claim your free API key here: https://www.alphavantage.co/support/#api-key
        "symbol": "TCEHY",
        "outputsize": "full",
        "interval": "5min",
        "key_close": "4. close",
    },
    "data": {
        "window_size": 3,
        "train_split_size": 0.80,
    },
    "plots": {
        "xticks_interval": 250, # show a date every 90 days
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
        "num_epoch": 50,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}


def tSA_loop(config):
    data_date, data_close_price, num_data_points, display_date_range = download_data(config)

    closing = data_close_price

    #result = adfuller(data_close_price)
    #print(result)

    stock_diff = pd.DataFrame(data_close_price) - pd.DataFrame(data_close_price).shift()
    stock_diff.dropna(inplace=True)

    #result = adfuller(stock_diff)
    #print(result)

    # # plt.figure()
    # # plt.plot(stock_diff, label = 'first difference of origin data')
    # # plt.legend(loc='upper right')
    # # plt.xlabel('Time point')
    # # plt.ylabel('Price/Time point')
    # # plt.show()

    fig2 = plt.figure()
    ax2_1 = fig2.add_subplot(121)
    ax2_2 = fig2.add_subplot(122)

    fig2 = plot_acf(stock_diff, lags=40, ax = ax2_1)
    ax2_1.set_title("ACF")
    # acf.show()

    fig2 = plot_pacf(stock_diff, lags=40, ax = ax2_2)
    ax2_2.set_title("PACF")
    # pacf.show()

    from statsmodels.tsa.arima_model import ARIMA
    model = ARIMA(stock_diff.values, order=(1, 1, 1))
    results_ARIMA = model.fit(disp=-1)

    # fig1 = plt.figure()
    # ax1_1 = fig1.add_subplot(211)
    # ax1_2 = fig1.add_subplot(212)

    # ax1_2.plot(stock_diff)
    # ax1_2.plot(results_ARIMA.fittedvalues, color='red', label = 'regression')
    # ax1_2.legend(loc='upper right')
    # ax1_2.set_xlabel('Time point')
    # ax1_2.set_ylabel('Price/Time point')

    # ax1_1.plot(stock_diff, label = 'first difference of origin data')
    # ax1_1.legend(loc='upper right')
    # ax1_1.set_xlabel('Time point')
    # ax1_1.set_ylabel('Price/Time point')



    predict_ts = results_ARIMA.predict()

    prediction = []
    yhat = closing[-1]
    for i in range(len(predict_ts)):
        yhat += predict_ts[i]
        prediction.append(yhat)

    prediction = prediction[0:int(0.3*len(prediction))]

    fig3, ax3 = plt.subplots(figsize=(1110/80, 500/80), dpi=80)
    ax3.plot(data_date, data_close_price, label = 'Regression price', color=config["plots"]["color_actual"])
    ax3.plot([i + len(closing) for i in range(len(prediction))], prediction, color = 'y', label = 'Predictions')
    xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation=30)
    plt.title(config["alpha_vantage"]["symbol"])
    ax3.legend(loc='upper left')
    ax3.set_xlabel('Time point')
    ax3.set_ylabel('Price')
    ax3.axvline(x=len(closing), color='r', linestyle='--')

    return fig3