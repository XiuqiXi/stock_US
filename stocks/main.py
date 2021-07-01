# This Python file uses the following encoding: utf-8
import os
from pathlib import Path
import sys
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import random
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtCore import QFile
from PySide2.QtUiTools import QUiLoader
import pyqtgraph as pg

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5 import QtWidgets


from form import Ui_Main

def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombuffer("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    return image

class MyMainForm(QMainWindow, Ui_Main):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        #self.graphicsView = QtWidgets.QGraphicsView(self)
        self.Compute_button.clicked.connect(self.compute)

    def compute(self, parent=None):
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


        from Main_Loop import main_loop
        figure = main_loop(config)
        axes = figure.gca()

        canvas = FigureCanvas(figure)
        self.scene = QGraphicsScene()
        proxy_widget = self.scene.addWidget(canvas)

        self.graphicsView.resize(1110, 500)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.show()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MyMainForm()
    widget.show()
    sys.exit(app.exec_())
