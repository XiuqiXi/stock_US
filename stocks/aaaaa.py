import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
time=['09:38:21','09:37:41','09:37:16','09:37:11','09:36:46',
      '09:36:31','09:36:01','09:35:36','09:35:31','09:35:26',
      '09:35:06','09:34:46','09:34:06','09:33:41','09:33:36',
      '09:33:21','09:33:16','09:33:11','09:33:01','09:32:46',
      '09:32:36','09:32:26','09:32:16','09:32:01','09:31:46',
      '09:31:41','09:31:21','09:30:46','09:30:06','09:25:06']
price=[6.08,6.08,6.07,6.09,6.09,6.09,6.09,6.09,6.08,6.09,6.09,
       6.09,6.09,6.07,6.07,6.06,6.06,6.06,6.06,6.06,6.07,6.08,
       6.08,6.07,6.07,6.07,6.08,6.08,6.08,6.09 ]

xDict=dict(enumerate(time))
xValue=list(xDict.keys())

config = {
    "alpha_vantage": {
        "function":"TIME_SERIES_INTRADAY",
        "key": "PR3XLLYLAN8V9CBY", # Claim your free API key here: https://www.alphavantage.co/support/#api-key
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
        "num_epoch": 50,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}

num_data_points = len(xDict)

win = pg.GraphicsWindow()
bottomAxis = pg.AxisItem(orientation='bottom')
plot = win.addPlot(axisItems={'bottom': bottomAxis},name='price')
xtickts=[xDict.items()]
xtickts = [list(xDict.items())[i][1] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)]
xtickts = [dict(zip(range(len(xtickts)), xtickts)).items()]
print(type(xDict.items()), type(xtickts))
bottomAxis.setTicks(xtickts)
plot.plot(xValue,price)
if __name__ == '__main__':
   import sys
   if sys.flags.interactive != 1 or not hasattr( QtCore, 'PYQT_VERSION' ):
      pg.QtGui.QApplication.exec_()
