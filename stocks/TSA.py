# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 16:08:03 2021

@author: xixiu
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import seaborn as sns
import statsmodels.tsa.api as smt
from statsmodels.tsa.ar_model import AR   
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import statsmodels.api as sm
from scipy import  stats
from statsmodels.graphics.api import qqplot

from matplotlib import rcParams
rcParams['font.family']='serif'
rcParams['font.sans-serif']=['Times New Roman']



X = pd.read_csv("#GOOGLE5.csv", names=['date','time','opening','max', 'min', 'closing', 'turnover']) 
X = X.dropna()

X['Full date'] = X['date']+' '+X['time']

print(np.where(X['Full date'].str.contains("2021.06.23 19:55")==True))

cutoff_1 = np.where(X['Full date'].str.contains("2021.06.23 19:55")==True)[0]
cutoff_1 = int(cutoff_1)

cutoff_2 = np.where(X['Full date'].str.contains("2021.06.25 19:55")==True)[0]
cutoff_2 = int(cutoff_2)

Part_1 = X[:cutoff_1]
Part_2 = X[cutoff_1:cutoff_2]
Part_3 = X[cutoff_2:]

# weekday = X[:weekend]
# nextweek = X[weekend+1:]
# nextweek = nextweek.dropna()

X = Part_1

time = X['date']+' '+X['time']

index = len(time)

time = pd.to_datetime(time,format='%Y.%m.%d %H:%M')

time = np.linspace(1, index, index) 

opening = X['opening'].values
closing = X['closing'].values

# for i in range(0, index):
#     dateOne = np.zeros([2])
#     dateOne[0] = i
#     dateOne[1] = i
#     priceOne = np.zeros([2])
#     priceOne[0] = opening[i]
#     priceOne[1] = closing[i]
#     if closing[i] > opening[i]:
#         plt.plot(dateOne, priceOne, 'r', lw=1)
#     else:
#         plt.plot(dateOne, priceOne, 'g', lw=1)
# plt.xlabel("date")
# plt.ylabel("price")
# plt.show()


result = adfuller(X['closing'])
print(result)

stock_diff = X['closing'] - X['closing'].shift()
stock_diff.dropna(inplace=True)

result = adfuller(stock_diff)
print(result)

# plt.figure()
# plt.plot(stock_diff, label = 'first difference of origin data')
# plt.legend(loc='upper right')
# plt.xlabel('Time point')
# plt.ylabel('Price/Time point')
# plt.show()

fig2 = plt.figure()
ax2_1 = fig2.add_subplot(121)
ax2_2 = fig2.add_subplot(122)

fig2 = plot_acf(stock_diff, lags=40, ax = ax2_1)
ax2_1.set_title("ACF")
# acf.show()

fig2 = plot_pacf(stock_diff, lags=40, ax = ax2_2)
ax2_2.set_title("PACF")
# pacf.show()

#ACF and PACF plots:
# from statsmodels.tsa.stattools import acf, pacf
# lag_acf = acf(stock_diff, nlags=20)
# lag_pacf = pacf(stock_diff, nlags=20, method='ols')
# #Plot ACF: 
# plt.subplot(121) 
# plt.plot(lag_acf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(stock_diff)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(stock_diff)),linestyle='--',color='gray')
# plt.title('Autocorrelation Function')

# #Plot PACF:
# plt.subplot(122)
# plt.plot(lag_pacf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(stock_diff)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(stock_diff)),linestyle='--',color='gray')
# plt.title('Partial Autocorrelation Function')
# plt.tight_layout()

# arma_mod80 = smt.AR(stock_diff).fit(maxlag=30, ic='aic', trend='nc')
# print(arma_mod80.aic,arma_mod80.bic,arma_mod80.hqic)
# resid = arma_mod80.resid
# print(sm.stats.durbin_watson(arma_mod80.resid))
# print(stats.normaltest(resid))
# fig = plt.figure()
# ax = fig.add_subplot(111)
# fig = qqplot(resid, line='q', ax=ax, fit=True)
# plt.show()

# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(resid.squeeze(), lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
# plt.show()

# fig=plt.figure(figsize=(15,7))
# fit = arma_mod80.predict(0, 1100)
# plt.plot(range(1100),fit[:1100],label='predict')
# plt.legend(loc=4)
# plt.show()

# model = ARMA(stock_diff, order=(26, 5)) 
# result_arma = model.fit( disp=-1, method='css')

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(stock_diff.values, order=(1, 1, 1))  
results_ARIMA = model.fit(disp=-1)

fig1 = plt.figure()
ax1_1 = fig1.add_subplot(211)
ax1_2 = fig1.add_subplot(212)

ax1_2.plot(stock_diff)
ax1_2.plot(results_ARIMA.fittedvalues, color='red', label = 'regression')
ax1_2.legend(loc='upper right')
ax1_2.set_xlabel('Time point')
ax1_2.set_ylabel('Price/Time point')

ax1_1.plot(stock_diff, label = 'first difference of origin data')
ax1_1.legend(loc='upper right')
ax1_1.set_xlabel('Time point')
ax1_1.set_ylabel('Price/Time point')



predict_ts = results_ARIMA.predict()

prediction = []
yhat = closing[-1]
for i in range(len(predict_ts)):
    yhat += predict_ts[i]
    prediction.append(yhat)    

# plt.figure(facecolor='white')
# diff_recover_1.plot(color='blue', label='Predict')
# plt.legend(loc='best')
# plt.show()

# 234 338
# 410 983

prediction = prediction[0:213]

fig3, ax3 = plt.subplots()

validation = Part_2['closing'].values

real_stock = Part_3['closing'].values

plt.plot(range(len(closing)), closing, label = 'Regression price')
plt.plot([i + len(closing) for i in range(len(validation))], validation, color = 'g', label = 'Real price')
plt.plot([i + len(closing) + len(validation) for i in range(len(real_stock))], real_stock, color = 'g')
plt.plot([i + len(closing) for i in range(len(prediction))], prediction, color = 'y', alpha=0)
plt.legend(loc='upper left')
plt.xlabel('Time point')
plt.ylabel('Price')
plt.axvline(x=len(closing), color='r', linestyle='--')
plt.axvline(x=len(closing)+len(validation), color='r', linestyle='--')

plt.text(57, 330, "6.21~6.24")
plt.text(250, 330, "6.24~6.25")
plt.text(400, 330, "6.28")

ax3.set_xlim = (0, len(closing)+len(validation)+len(prediction))
# ax3.set_xlim = (0, 3000)

ax3.axvspan(len(closing), len(closing)+len(validation), color="pink", alpha=0.2)
ax3.axvspan(len(closing)+len(validation), len(closing)+len(validation)+len(prediction), color="yellow", alpha=0.2)

X = Part_1.append(Part_2)

stock_diff = X['closing'] - X['closing'].shift()
stock_diff.dropna(inplace=True)

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(stock_diff.values, order=(1, 1, 1))  
results_ARIMA = model.fit(disp=-1)

predict_ts = results_ARIMA.predict()

temp1 = len(prediction)
temp2 = X['closing'].values

prediction = []
yhat = temp2[-1]
for i in range(len(predict_ts)):
    yhat += predict_ts[i]
    prediction.append(yhat)
    
plt.plot([i + len(closing)+ len(validation) for i in range(len(prediction))], prediction, color = 'b', label = 'Updated predicted price')
plt.legend(loc='upper left')



