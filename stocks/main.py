
# This Python file uses the following encoding: utf-8
import sys
import numpy as np
from matplotlib.backends.backend_qt5agg  import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSignal
from form import Ui_Main
from trade_window import Ui_trade_window
import pyqtgraph as pg
from pyqtgraph.Point import Point
from Prediction_Date import prediction_date
from PyQt5.QtWidgets import QPushButton
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


current_date = ''
current_price = 0
data_date_prediction_child = []
prediction_child = []
number_point_prediction_child = 0

config_global = {
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
        "xticks_interval": 90, # show a date every 90 days
        "xticks_zoom_in": 100,
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


class MyMainForm(QMainWindow, Ui_Main):
    global config_global
    config =config_global

    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        self.Compute_button.clicked.connect(self.set_graph_ui)
        self.Compute_button_reset.clicked.connect(self.reset_ui)
        self.comboBox.activated.connect(self.comboBoxeActivated)
        self.setWindowTitle('US Stocks Predictions')
        self.setWindowIcon(QIcon('favicon.ico'))
        self.stock_code_text.setText("AMZN")
        self.window_size_text.setText("3")
        self.iteration_times_text.setText("50")
        self.show()

    def show_message(self):
        QtWidgets.QMessageBox.critical(self, "Error", self.config["alpha_vantage"]["symbol"], QtWidgets.QMessageBox.Ok)

    def comboBoxeActivated(self, index):
        interval = str(self.comboBox.itemText(index))
        interval = interval.replace(' ','')
        print(interval)
        self.config["alpha_vantage"]["interval"] = interval

    def set_graph_ui(self):
        global data_date_prediction_child
        global prediction_child
        global number_point_prediction_child
        global current_date
        global current_price

        try:
            self.config["alpha_vantage"]["symbol"] = self.stock_code_text.toPlainText()
            self.config["data"]["window_size"] = self.window_size_text.toPlainText()
            self.config["training"]["num_epoch"] = self.iteration_times_text.toPlainText()
            data_date, data_close_price, prediction, num_data_points= tSA_loop(self.config)
            current_date = data_date[-1]
            current_price = data_close_price[-1]
            prediction_data_date = prediction_date(data_date, len(prediction), self.config["alpha_vantage"]["interval"])
            data_date_prediction_child = prediction_data_date
            prediction_child = prediction
            data_date = data_date+prediction_data_date
            number_point_prediction_child = len(prediction)
            num_data_points = num_data_points+len(prediction)
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
            p1 = win.addPlot(row=1, col=0, axisItems={'bottom': bottomAxis_1}, name='price')
            p2 = win.addPlot(row=2, col=0, axisItems={'bottom': bottomAxis_2}, name='price')
            region = pg.LinearRegionItem()
            p2.addItem(region, ignoreBounds=True)
            p1.setAutoVisible(y=True)


            xDict=dict(enumerate(data_date))
            xValue=list(xDict.keys())
            xtickts=[xDict.items()]
            xtickts_1 = [list(xDict.items())[i][1] if ((i%self.config["plots"]["xticks_interval"]==0 and (num_data_points-i) > self.config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)]
            xtickts_1 = [dict(zip(range(len(xtickts_1)), xtickts_1)).items()]
            xtickts_2 = [list(xDict.items())[i][1] if ((i%self.config["plots"]["xticks_zoom_in"]==0 and (num_data_points-i) > self.config["plots"]["xticks_zoom_in"]) or i==num_data_points-1) else None for i in range(num_data_points)]
            xtickts_2 = [dict(zip(range(len(xtickts_2)), xtickts_2)).items()]
            bottomAxis_1.setTicks(xtickts_2)
            bottomAxis_2.setTicks(xtickts_1)
            data2 = np.append(data_close_price, prediction)
            data1 = xValue
            p1.plot(data1[:len(data_date) - len(prediction_data_date)], data2[:len(data_date) - len(prediction_data_date)], pen="r")
            p2.plot(data1[:len(data_date) - len(prediction_data_date)], data2[:len(data_date) - len(prediction_data_date)], pen="r")
            p1.plot(data1[len(data_date) - len(prediction_data_date):], data2[len(data_date) - len(prediction_data_date):], pen="w")
            p2.plot(data1[len(data_date) - len(prediction_data_date):], data2[len(data_date) - len(prediction_data_date):], pen="w")

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
                    index = int(mousePoint.x())
                    if index > 0 and index < len(data1):
                        label.setText("<span style='font-size: 12pt'>x=%s,   <span style='color: red'>Price=%0.1f</span>" % (data_date[index], data2[index]))
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

    def reset_ui(self):
        for i in range(self.verticalLayout.count()):
            self.verticalLayout.itemAt(i).widget().deleteLater()


class Trade_Window(QMainWindow,Ui_trade_window):
    global config_global
    config =config_global
    def __init__(self):
        super(Trade_Window, self).__init__()
        self.setupUi(self)
        self.buy_button.clicked.connect(self.msg1_buy)
        self.sell_button.clicked.connect(self.msg1_sell)
        self.label_current_time.setText(current_date)
        self.label_current_price.setText(str(current_price))

    def reset_ui(self):
        for i in range(self.verticalLayout.count()):
            self.verticalLayout.itemAt(i).widget().deleteLater()

    def msg1_buy(self):
        global current_date
        shares = self.TextEdit_share.toPlainText()
        QMessageBox.information(self,"Are you sure?","Are you sure to buy "+ shares + " shares at " + str(current_date),QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)

    def msg1_sell(self):
        global current_date
        shares = self.TextEdit_share.toPlainText()
        QMessageBox.information(self,"Are you sure?","Are you sure to sell "+ shares + " shares at " + str(current_date),QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)

    def mousePressEvent(self, evt):
        self.Point_list = []
        len_point_list = len(self.Point_list)/2
        for i in range(int(len_point_list)):
            painter.drawPoint(self.Point_list[i*2],self.Point_list[i*2+1])
            print("DrawPoint")
        print('AAAAA')

    def OPEN(self):
        self.show()

        global data_date_prediction_child
        global prediction_child
        global number_point_prediction_child
        global current_date
        global current_price

        self.label_current_time.setText(current_date)
        self.label_current_price.setText(str(current_price))

        data_date = data_date_prediction_child
        prediction = prediction_child
        num_data_points = number_point_prediction_child
        print(num_data_points)

        pg.setConfigOptions(antialias=True)
        win = pg.GraphicsLayoutWidget()
        self.verticalLayout.addWidget(win)
        bottomAxis_1 = pg.AxisItem(orientation='bottom')
        p1 = win.addPlot(axisItems={'bottom': bottomAxis_1}, name='price')
        region = pg.LinearRegionItem()
        p1.setAutoVisible(y=True)

        xDict=dict(enumerate(data_date))
        xValue=list(xDict.keys())
        xtickts=[xDict.items()]
        xtickts_1 = [list(xDict.items())[i][1] if ((i%self.config["plots"]["xticks_interval"]==0 and (num_data_points-i) > self.config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)]
        xtickts_1 = [dict(zip(range(len(xtickts_1)), xtickts_1)).items()]
        bottomAxis_1.setTicks(xtickts_1)
        data2 = prediction
        data1 = xValue
        p1.plot(data1, data2, pen="w")

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
                index = int(mousePoint.x())
                if index > 0 and index < len(data1):
                    self.label_selected_price.setText(str(format(data2[index], '.3f')))
                    self.label_selected_time.setText(data_date[index])
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
    main = MyMainForm()
    main.show()
    trade = Trade_Window()
    main.trade_button.clicked.connect(trade.OPEN)
    main.Compute_button_reset.clicked.connect(trade.reset_ui)
    sys.exit(app.exec_())
