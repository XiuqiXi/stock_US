# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 16:08:03 2021

@author: xixiu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from Api import download_data
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times New Roman']

config = {
    "alpha_vantage": {
        "function": "TIME_SERIES_INTRADAY",
        "key": "PR3XLLYLAN8V9CBY",
        "symbol": "AMZN",
        "outputsize": "full",
        "interval": "5min",
        "key_close": "4. close",
    },
    "data": {
        "window_size": 3,
        "train_split_size": 0.80,
    },
    "plots": {
        "xticks_interval": 250,  # show a date every 90 days
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1,  # since we are only using 1 feature, close price
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 50,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}


def on_move(event, point, annotation, po_annotation):
    visibility_changed = False
    for point, annotation in po_annotation:
        should_be_visible = (point.contains(event)[0] == True)
        if should_be_visible != annotation.get_visible():
            visibility_changed = True
            annotation.set_visible(should_be_visible)
    if visibility_changed:
        plt.draw()



data_date, data_close_price, num_data_points, display_date_range \
    = download_data(config)

closing = data_close_price

# result = adfuller(data_close_price)
# print(result)

stock_diff = pd.DataFrame(data_close_price) - \
    pd.DataFrame(data_close_price).shift()
stock_diff.dropna(inplace=True)

# result = adfuller(stock_diff)
# print(result)

# # plt.figure()
# # plt.plot(stock_diff, label = 'first difference of origin data')
# # plt.legend(loc='upper right')
# # plt.xlabel('Time point')
# # plt.ylabel('Price/Time point')
# # plt.show()

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

fig3, ax3 = plt.subplots(figsize=(1110/80, 500/80), dpi=80)
ax3.plot(data_date, data_close_price,
            label='Regression price',
            color=config["plots"]["color_actual"])
xticks = [data_date[i] if ((i % config["plots"]["xticks_interval"] == 0 and
            (num_data_points-i) > config["plots"]["xticks_interval"])
            or i == num_data_points-1) else None
            for i in range(num_data_points)]  # make x ticks nice
x = np.arange(0, len(xticks))
plt.xticks(x, xticks, rotation=30)
plt.title(config["alpha_vantage"]["symbol"])
ax3.legend(loc='upper left')
ax3.set_xlabel('Time point')
ax3.set_ylabel('Price')
ax3.axvline(x=len(closing), color='r', linestyle='--')
fig3.set_canvas(fig3.canvas.manager.canvas)

len_data_close_price = len(data_close_price)
x = range(len_data_close_price)
_data_close_price = [data_close_price[-1]]*len_data_close_price

line_x, = ax3.plot(data_date, data_close_price,
                    label='Regression price',
                    color=config["plots"]["color_actual"])
line_y = ax3.axvline(x=len_data_close_price-1, color='skyblue')

text0 = plt.text(len_data_close_price-1, data_close_price[-1], str(data_close_price[-1]), fontsize=10)

def scroll(event):
    axtemp = event.inaxes
    x_min, x_max = axtemp.get_xlim()
    fanwei_x = (x_max - x_min) / 10
    if event.button == 'up':
        axtemp.set(xlim=(x_min + fanwei_x, x_max - fanwei_x))
    elif event.button == 'down':
        axtemp.set(xlim=(x_min - fanwei_x, x_max + fanwei_x))
    fig3.canvas.draw_idle()

def motion(event):
    try:
        print(np.round(event.xdata))
        temp_numb = data_close_price[int(np.round(event.xdata.astype(np.float64)))]
        for i in range(len_data_close_price):
            _data_close_price[i] = temp_numb
        ax3[0].set_ydata(_data_close_price)
        ax3[1].set_xdata(event.xdata)
        text0.set_position((event.xdata, temp_numb))
        text0.set_text(str(temp_numb))
        fig3.canvas.draw_idle() # 绘图动作实时反映在图像上
    except:
        print(type(event.xdata))
        print(np.round(event.xdata.astype(np.float64)))
        temp = data_close_price[int(np.round(event.xdata.astype(np.float64)))]
        for i in range(len_data_close_price):
            _data_close_price[i] = temp
        # line_x.set_ydata(_data_close_price)
        line_y.set_xdata(event.xdata)
        text0.set_position((event.xdata, temp))
        text0.set_text(str(temp))
        fig3.canvas.draw_idle() # 绘图动作实时反映在图像上
        #pass

fig3.canvas.mpl_connect('scroll_event', scroll)
fig3.canvas.mpl_connect('motion_notify_event', motion)

plt.show()



