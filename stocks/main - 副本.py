
# This Python file uses the following encoding: utf-8
import sys
import numpy as np
from matplotlib.backends.backend_qt5agg  import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QIcon
from form import Ui_Main
from PyQt5 import QtWidgets
import pyqtgraph as pg
from pyqtgraph.Point import Point
from RNN_Loop import rNN_loop
from TSA_Loop import tSA_loop

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

    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        self.Compute_button.clicked.connect(self.set_graph_ui)
        self.comboBox.activated.connect(self.comboBoxeActivated)
        self.setWindowTitle('US Stocks Predictions')
        self.setWindowIcon(QIcon('favicon.ico'))
        self.stock_code_text.setText("AMZN")
        self.window_size_text.setText("3")
        self.iteration_times_text.setText("50")
        # self.p1, self.p2 = self.set_graph_ui()
        self.show()

    def show_message(self):
        QtWidgets.QMessageBox.critical(self, "Error", self.config["alpha_vantage"]["symbol"], QtWidgets.QMessageBox.Ok)

    def comboBoxeActivated(self, index):
        interval = str(self.comboBox.itemText(index))
        interval = interval.replace(' ','')
        print(interval)
        self.config["alpha_vantage"]["interval"] = interval

    def set_graph_ui(self):
        try:
            self.config["alpha_vantage"]["symbol"] = self.stock_code_text.toPlainText()
            self.config["data"]["window_size"] = self.window_size_text.toPlainText()
            self.config["training"]["num_epoch"] = self.iteration_times_text.toPlainText()
            data_date, data_close_price, prediction = tSA_loop(self.config)
        except:
            self.show_message()
        else:
            pg.setConfigOptions(antialias=True)
            win = pg.GraphicsLayoutWidget()
            label = pg.LabelItem(justify='right')
            win.addItem(label)
            self.verticalLayout.addWidget(win)
            bottomAxis_1 = pg.AxisItem(orientation='bottom')
            bottomAxis_2 = pg.AxisItem(orientation='bottom')
            p1 = win.addPlot(row=1, col=0, axisItems={'bottom': bottomAxis_1},name='price')
            p2 = win.addPlot(row=2, col=0, axisItems={'bottom': bottomAxis_2},name='price')
            region = pg.LinearRegionItem()
            p2.addItem(region, ignoreBounds=True)
            p1.setAutoVisible(y=True)

            xDict=dict(enumerate(data_date))
            xValue=list(xDict.keys())
            xtickts=[xDict.items()]
            bottomAxis_1.setTicks(xtickts)
            bottomAxis_2.setTicks(xtickts)
            data1 = [x.timestamp() for x in data_date]
            data2 = data_close_price
            print(xValue)
            p1.plot(xValue, data2, pen="r")
            p2.plot(xValue, data2, pen="r")
            #p1.plot([i + len(data1) for i in range(len(prediction))], prediction, pen="w")
            #p2.plot([i + len(data1) for i in range(len(prediction))], prediction, pen="w")

            def update():
                region.setZValue(10)
                minX, maxX = region.getRegion()
                p1.setXRange(minX, maxX, padding=0)

            def updateRegion(window, viewRange):
                rgn = viewRange[0]
                region.setRegion(rgn)

            def mouseMoved(evt):
                pos = evt  ## using signal proxy turns original arguments into a tuple
                if p1.sceneBoundingRect().contains(pos):
                    mousePoint = vb.mapSceneToView(pos)
                    # 建议不用int，精度高时用float，这样可以显示横坐标的小数
                    index = int(mousePoint.x())
                    if index > 0 and index < len(data1):
                        label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>Price=%0.1f</span>" % (mousePoint.x(), data2[index]))
                    vLine.setPos(mousePoint.x())
                    hLine.setPos(mousePoint.y())

            region.sigRegionChanged.connect(update)
            p1.sigRangeChanged.connect(updateRegion)

            region.setRegion([int(0.3*len(data2)), int(0.4*len(data2))])

            vLine = pg.InfiniteLine(angle=90, movable=False)  # angle控制线相对x轴正向的相对夹角
            hLine = pg.InfiniteLine(angle=0, movable=False)
            p1.addItem(vLine, ignoreBounds=True)
            p1.addItem(hLine, ignoreBounds=True)

            vb = p1.vb
            proxy = pg.SignalProxy(p1.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)
            p1.scene().sigMouseMoved.connect(mouseMoved)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MyMainForm()
    widget.show()
    sys.exit(app.exec_())
