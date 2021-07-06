# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 17:24:38 2021

@author: xixiu
"""

import pandas_market_calendars as mcal
import datetime
import time
import pandas as pd
from Api import download_data
config = {
    "alpha_vantage": {
        "function":"TIME_SERIES_INTRADAY",
        "key": "PR3XLLYLAN8V9CBY", # Claim your free API key here: https://www.alphavantage.co/support/#api-key
        "symbol": "TSLA",
        "outputsize": "full",
        "interval": "5min",
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

def timestamp_to_date(time_stamp, format_string="%Y-%m-%d %H:%M:%S"):
    time_array = time.localtime(time_stamp)
    str_date = time.strftime(format_string, time_array)
    return str_date

def prediction_date(data_date, cutoff, interval):
    date = data_date
    date_day = []
    date_second = []
    for i in range(len(data_date)):
        date[i] = data_date[i].split(' ', 1)
        date_second.append(date[i][1])
        if i == 0:
            pass
        else:
            if date[i][1] == date[0][1]:
                break
    del date_second[-1]
    
    opening_time = datetime.datetime.strptime(date_second[0], '%H:%M:%S')
    closing_time = datetime.datetime.strptime(date_second[-1], '%H:%M:%S')
    
    current_date = data_date[-1].split(' ', 1)[0]
    current_second = date[i] = data_date[-1].split(' ', 1)[1]
    
    points = 10000
    
    index = 0
    current_time = datetime.datetime.strptime(data_date[-1], '%Y-%m-%d %H:%M:%S')
    current_time_30days_later = (current_time + datetime.timedelta(days=90))
    
    nyse = mcal.get_calendar('NYSE')
    trade_date = nyse.schedule(start_date=str(current_time), end_date=str(current_time_30days_later))
    
    prediction_time = []
    
    i = pd.date_range(data_date[-1], periods=points, freq=interval)
    all_point = pd.DataFrame({'index': list(range(points))}, index=i)
    time_points_range = all_point.between_time('04:05', '20:00')
    
    # for i in range(point):
    #     current_time = (current_time + datetime.timedelta(minutes=5))    
    #     prediction_time.append(current_time)
        
    # A = pd.DataFrame(prediction_time)
    # A.columns = ['date']
    # #A.set_index(['date'], inplace=True)
    # time_points_trading = A.set_index("date").between_time("04:05", "20:00")
    
    time_points_range = time_points_range.index.tolist()
    
    drop_index = []
    for i in range(len(time_points_range)):
        temp = datetime.datetime.strptime(str(time_points_range[i]), '%Y-%m-%d %H:%M:%S')
        temp = str(temp.year) + '-' + str(temp.month) + '-' + str(temp.day) + ' 12:00'
        if nyse.open_at_time(trade_date, pd.Timestamp(temp, tz='America/New_York')) == False:
            drop_index.append(i)
            
    time_points_range = pd.DataFrame(time_points_range)
    time_points_trading = time_points_range.drop(labels=drop_index,axis=0)
    time_points_trading.columns = ['date']
    time_points_trading = time_points_trading['date'].tolist()
    
    prediction_date = time_points_trading[0:cutoff]
    
    for i in range(len(prediction_date)):
        # prediction_date[i].to_pydatetime()
        prediction_date[i] = datetime.datetime.strftime(prediction_date[i],'%Y-%m-%d %H:%M:%S')
    
    return prediction_date

A = prediction_date(data_date, 300, config["alpha_vantage"]["interval"])


    


    
    
