# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Main(object):
    def setupUi(self, Main):
        Main.setObjectName("Main")
        Main.resize(1191, 699)
        self.stock_code = QtWidgets.QLabel(Main)
        self.stock_code.setGeometry(QtCore.QRect(50, 550, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.stock_code.setFont(font)
        self.stock_code.setTextFormat(QtCore.Qt.AutoText)
        self.stock_code.setAlignment(QtCore.Qt.AlignCenter)
        self.stock_code.setObjectName("stock_code")
        self.sample_frequency_label = QtWidgets.QLabel(Main)
        self.sample_frequency_label.setGeometry(QtCore.QRect(30, 620, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.sample_frequency_label.setFont(font)
        self.sample_frequency_label.setTextFormat(QtCore.Qt.AutoText)
        self.sample_frequency_label.setAlignment(QtCore.Qt.AlignCenter)
        self.sample_frequency_label.setObjectName("sample_frequency_label")
        self.stock_code_text = QtWidgets.QTextEdit(Main)
        self.stock_code_text.setGeometry(QtCore.QRect(200, 550, 241, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.stock_code_text.setFont(font)
        self.stock_code_text.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.stock_code_text.setObjectName("stock_code_text")
        self.advance_setting_button = QtWidgets.QPushButton(Main)
        self.advance_setting_button.setGeometry(QtCore.QRect(930, 620, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.advance_setting_button.setFont(font)
        self.advance_setting_button.setObjectName("advance_setting_button")
        self.Compute_button = QtWidgets.QPushButton(Main)
        self.Compute_button.setGeometry(QtCore.QRect(930, 540, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.Compute_button.setFont(font)
        self.Compute_button.setObjectName("Compute_button")
        self.comboBox = QtWidgets.QComboBox(Main)
        self.comboBox.setGeometry(QtCore.QRect(250, 620, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.window_size_text = QtWidgets.QTextEdit(Main)
        self.window_size_text.setGeometry(QtCore.QRect(620, 550, 241, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.window_size_text.setFont(font)
        self.window_size_text.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.window_size_text.setObjectName("window_size_text")
        self.window_size_label = QtWidgets.QLabel(Main)
        self.window_size_label.setGeometry(QtCore.QRect(470, 550, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.window_size_label.setFont(font)
        self.window_size_label.setTextFormat(QtCore.Qt.AutoText)
        self.window_size_label.setAlignment(QtCore.Qt.AlignCenter)
        self.window_size_label.setObjectName("window_size_label")
        self.iteration_times_text = QtWidgets.QTextEdit(Main)
        self.iteration_times_text.setGeometry(QtCore.QRect(620, 610, 241, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.iteration_times_text.setFont(font)
        self.iteration_times_text.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.iteration_times_text.setObjectName("iteration_times_text")
        self.iiteration_times_label = QtWidgets.QLabel(Main)
        self.iiteration_times_label.setGeometry(QtCore.QRect(450, 610, 171, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.iiteration_times_label.setFont(font)
        self.iiteration_times_label.setTextFormat(QtCore.Qt.AutoText)
        self.iiteration_times_label.setAlignment(QtCore.Qt.AlignCenter)
        self.iiteration_times_label.setObjectName("iiteration_times_label")
        self.graphicsView = QtWidgets.QGraphicsView(Main)
        self.graphicsView.setGeometry(QtCore.QRect(60, 30, 1021, 461))
        self.graphicsView.setObjectName("graphicsView")

        self.retranslateUi(Main)
        QtCore.QMetaObject.connectSlotsByName(Main)

    def retranslateUi(self, Main):
        _translate = QtCore.QCoreApplication.translate
        Main.setWindowTitle(_translate("Main", "stocks"))
        self.stock_code.setText(_translate("Main", "Stock Code"))
        self.sample_frequency_label.setText(_translate("Main", "Sample Frequency"))
        self.advance_setting_button.setText(_translate("Main", "Advance Settins"))
        self.Compute_button.setText(_translate("Main", "Compute"))
        self.comboBox.setItemText(0, _translate("Main", "1 min"))
        self.comboBox.setItemText(1, _translate("Main", "15 min"))
        self.comboBox.setItemText(2, _translate("Main", "30 min"))
        self.comboBox.setItemText(3, _translate("Main", "60 min"))
        self.window_size_label.setText(_translate("Main", "Window Size"))
        self.iiteration_times_label.setText(_translate("Main", "Iteration Times"))