# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'trade_window.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_trade_window(object):
    def setupUi(self, trade_window):
        trade_window.setObjectName("trade_window")
        trade_window.resize(1025, 540)
        self.centralwidget = QtWidgets.QWidget(trade_window)
        self.centralwidget.setObjectName("centralwidget")
        self.sell_button = QtWidgets.QPushButton(self.centralwidget)
        self.sell_button.setGeometry(QtCore.QRect(390, 490, 80, 35))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.sell_button.setFont(font)
        self.sell_button.setObjectName("sell_button")
        self.buy_button = QtWidgets.QPushButton(self.centralwidget)
        self.buy_button.setGeometry(QtCore.QRect(149, 490, 80, 35))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.buy_button.setFont(font)
        self.buy_button.setObjectName("buy_button")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 0, 1001, 351))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.stock_code_3 = QtWidgets.QLabel(self.centralwidget)
        self.stock_code_3.setGeometry(QtCore.QRect(760, 410, 73, 26))
        self.stock_code_3.setMaximumSize(QtCore.QSize(300, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.stock_code_3.setFont(font)
        self.stock_code_3.setTextFormat(QtCore.Qt.AutoText)
        self.stock_code_3.setAlignment(QtCore.Qt.AlignCenter)
        self.stock_code_3.setObjectName("stock_code_3")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(12, 379, 731, 81))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.stock_code = QtWidgets.QLabel(self.layoutWidget)
        self.stock_code.setMaximumSize(QtCore.QSize(300, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.stock_code.setFont(font)
        self.stock_code.setTextFormat(QtCore.Qt.AutoText)
        self.stock_code.setAlignment(QtCore.Qt.AlignCenter)
        self.stock_code.setObjectName("stock_code")
        self.gridLayout.addWidget(self.stock_code, 0, 0, 1, 1)
        self.label_current_price = QtWidgets.QLabel(self.layoutWidget)
        self.label_current_price.setMaximumSize(QtCore.QSize(300, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_current_price.setFont(font)
        self.label_current_price.setText("")
        self.label_current_price.setTextFormat(QtCore.Qt.AutoText)
        self.label_current_price.setAlignment(QtCore.Qt.AlignCenter)
        self.label_current_price.setObjectName("label_current_price")
        self.gridLayout.addWidget(self.label_current_price, 0, 1, 1, 1)
        self.stock_code_2 = QtWidgets.QLabel(self.layoutWidget)
        self.stock_code_2.setMaximumSize(QtCore.QSize(300, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.stock_code_2.setFont(font)
        self.stock_code_2.setTextFormat(QtCore.Qt.AutoText)
        self.stock_code_2.setAlignment(QtCore.Qt.AlignCenter)
        self.stock_code_2.setObjectName("stock_code_2")
        self.gridLayout.addWidget(self.stock_code_2, 0, 2, 1, 1)
        self.label_selected_price = QtWidgets.QLabel(self.layoutWidget)
        self.label_selected_price.setMaximumSize(QtCore.QSize(300, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_selected_price.setFont(font)
        self.label_selected_price.setText("")
        self.label_selected_price.setTextFormat(QtCore.Qt.AutoText)
        self.label_selected_price.setAlignment(QtCore.Qt.AlignCenter)
        self.label_selected_price.setObjectName("label_selected_price")
        self.gridLayout.addWidget(self.label_selected_price, 0, 3, 1, 1)
        self.stock_code_6 = QtWidgets.QLabel(self.layoutWidget)
        self.stock_code_6.setMaximumSize(QtCore.QSize(300, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.stock_code_6.setFont(font)
        self.stock_code_6.setTextFormat(QtCore.Qt.AutoText)
        self.stock_code_6.setAlignment(QtCore.Qt.AlignCenter)
        self.stock_code_6.setObjectName("stock_code_6")
        self.gridLayout.addWidget(self.stock_code_6, 1, 0, 1, 1)
        self.label_current_time = QtWidgets.QLabel(self.layoutWidget)
        self.label_current_time.setMaximumSize(QtCore.QSize(300, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_current_time.setFont(font)
        self.label_current_time.setText("")
        self.label_current_time.setTextFormat(QtCore.Qt.AutoText)
        self.label_current_time.setAlignment(QtCore.Qt.AlignCenter)
        self.label_current_time.setObjectName("label_current_time")
        self.gridLayout.addWidget(self.label_current_time, 1, 1, 1, 1)
        self.stock_code_8 = QtWidgets.QLabel(self.layoutWidget)
        self.stock_code_8.setMaximumSize(QtCore.QSize(300, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.stock_code_8.setFont(font)
        self.stock_code_8.setTextFormat(QtCore.Qt.AutoText)
        self.stock_code_8.setAlignment(QtCore.Qt.AlignCenter)
        self.stock_code_8.setObjectName("stock_code_8")
        self.gridLayout.addWidget(self.stock_code_8, 1, 2, 1, 1)
        self.label_selected_time = QtWidgets.QLabel(self.layoutWidget)
        self.label_selected_time.setMaximumSize(QtCore.QSize(300, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_selected_time.setFont(font)
        self.label_selected_time.setText("")
        self.label_selected_time.setTextFormat(QtCore.Qt.AutoText)
        self.label_selected_time.setAlignment(QtCore.Qt.AlignCenter)
        self.label_selected_time.setObjectName("label_selected_time")
        self.gridLayout.addWidget(self.label_selected_time, 1, 3, 1, 1)
        self.TextEdit_share = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.TextEdit_share.setGeometry(QtCore.QRect(840, 400, 171, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.TextEdit_share.setFont(font)
        self.TextEdit_share.setObjectName("TextEdit_share")
        trade_window.setCentralWidget(self.centralwidget)

        self.retranslateUi(trade_window)
        QtCore.QMetaObject.connectSlotsByName(trade_window)

    def retranslateUi(self, trade_window):
        _translate = QtCore.QCoreApplication.translate
        trade_window.setWindowTitle(_translate("trade_window", "MainWindow"))
        self.sell_button.setText(_translate("trade_window", "Sell"))
        self.buy_button.setText(_translate("trade_window", "Buy"))
        self.stock_code_3.setText(_translate("trade_window", "Shares"))
        self.stock_code.setText(_translate("trade_window", "Current Price"))
        self.stock_code_2.setText(_translate("trade_window", "Selected Price"))
        self.stock_code_6.setText(_translate("trade_window", "Current Time"))
        self.stock_code_8.setText(_translate("trade_window", "Selected Time"))