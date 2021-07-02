# This Python file uses the following encoding: utf-8
import sys
import numpy as np
from matplotlib.backends.backend_qt5agg  import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QIcon
from form import Ui_Main
from PyQt5 import QtWidgets


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

    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        self.Compute_button.clicked.connect(self.compute)
        self.comboBox.activated.connect(self.comboBoxeActivated)
        self.setWindowTitle('US Stocks Predictions')
        self.setWindowIcon(QIcon('favicon.ico'))
        self.stock_code_text.setText("AMZN")
        self.window_size_text.setText("3")
        self.iteration_times_text.setText("50")
        self.show()

    def compute(self, parent=None):
        from RNN_Loop import rNN_loop
        from TSA_Loop import tSA_loop

        try:
            self.config["alpha_vantage"]["symbol"] = self.stock_code_text.toPlainText()
            self.config["data"]["window_size"] = self.window_size_text.toPlainText()
            self.config["training"]["num_epoch"] = self.iteration_times_text.toPlainText()
            figure = tSA_loop(self.config)
        except:
            self.show_message()
        else:
            canvas = FigureCanvas(figure)
            self.scene = QtWidgets.QGraphicsScene(self.graphicsView)
            proxy_widget = self.scene.addWidget(canvas)

            self.graphicsView.resize(1110, 500)
            self.graphicsView.setScene(self.scene)
            self.graphicsView.show()

    def show_message(self):
        QtWidgets.QMessageBox.critical(self, "Error", self.config["alpha_vantage"]["symbol"], QtWidgets.QMessageBox.Ok)

    def comboBoxeActivated(self, index):
        interval = str(self.comboBox.itemText(index))
        interval = interval.replace(' ','')
        print(interval)
        self.config["alpha_vantage"]["interval"] = interval


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MyMainForm()
    widget.show()
    sys.exit(app.exec_())
