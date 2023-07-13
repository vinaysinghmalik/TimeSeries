import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from datetime import datetime
pd.options.display.float_format = '{:.2f}'.format
from itertools import combinations
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA as ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
data = pd.read_csv('AirPassengers.csv')
data.head()
data.shape
data.columns
data.info()
data.describe()
data['Date'] = pd.to_datetime(data['Month'])
data = data.drop(columns = 'Month')
data = data.set_index('Date')
data = data.rename(columns = {'#Passengers':'Passengers'})
data.head()
data.info()
plt.figure(figsize = (15,5))
data['Passengers'].plot();
dec = sm.tsa.seasonal_decompose(data['Passengers'],period = 12, model = 'multiplicative').p
plt.show()
data_diff = data.diff()
data_diff = data_diff.dropna()
dec = sm.tsa.seasonal_decompose(data_diff,period = 12).plot()
plt.show()
def test_stationarity(timeseries):
 #Determing rolling statistics
 MA = timeseries.rolling(window=12).mean()
 MSTD = timeseries.rolling(window=12).std()
 #Plot rolling statistics:
 plt.figure(figsize=(15,5))
 orig = plt.plot(timeseries, color='blue',label='Original')
 mean = plt.plot(MA, color='red', label='Rolling Mean')
 std = plt.plot(MSTD, color='black', label = 'Rolling Std')
 plt.legend(loc='best')
 plt.title('Rolling Mean & Standard Deviation')
 plt.show(block=False)
 #Perform Dickey-Fuller test:
 print('Results of Dickey-Fuller Test:')
 dftest = adfuller(timeseries, autolag='AIC')
 dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Numbe
 for key,value in dftest[4].items():
 dfoutput['Critical Value (%s)'%key] = value
 print(dfoutput)
 def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
 if not isinstance(y, pd.Series):
 y = pd.Series(y)
 
 with plt.style.context(style): 
 fig = plt.figure(figsize=figsize)
 layout = (2, 2)
 ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
 acf_ax = plt.subplot2grid(layout, (1, 0))
 pacf_ax = plt.subplot2grid(layout, (1, 1))
 
 y.plot(ax=ts_ax)
 p_value = sm.tsa.stattools.adfuller(y)[1]
 ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_va
 smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
 smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
 plt.tight_layout()
 plt.figure(figsize = (15,5))
data['Passengers'].plot();
 dec = sm.tsa.seasonal_decompose(data['Passengers'],period = 12, model = 'multiplicative').p
plt.show()
test_stationarity(data['Passengers'])
data_diff = data.diff()
data_diff = data_diff.dropna()
dec = sm.tsa.seasonal_decompose(data_diff,period = 12).plot()
plt.show()
test_stationarity(data_diff)
 tsplot(data['Passengers'])
tsplot(data_diff['Passengers'])                                                                                
model = ARIMA(data['Passengers'],order = (2,1,2))
model_fit = model.fit()
print(model_fit.summary())
size = int(len(data) - 30)
train, test = data['Passengers'][0:size], data['Passengers'][size:len(data)]
print('\t ARIMA MODEL : In- Sample Forecasting \n')
history = [x for x in train]
predictions = []
for t in range(len(test)):
 
 model = ARIMA(history, order=(2,1,2))
 model_fit = model.fit(disp = 0)
 
 output = model_fit.forecast()
 yhat = output[0]
 predictions.append(float(yhat))
 
 obs = test[t]
 history.append(obs)
 
 print('predicted = %f, expected = %f' % (yhat, obs))
  predictions_series = pd.Series(predictions, index = test.index)
fig,ax = plt.subplots(nrows = 1,ncols = 1,figsize = (15,5))
plt.subplot(1,1,1)
plt.plot(data['Passengers'],label = 'Expected Values')
plt.plot(predictions_series,label = 'Predicted Values');
plt.legend(loc="upper left")
plt.show()
error = np.sqrt(mean_squared_error(test,predictions))
print('Test RMSE: %.4f' % error)
from pandas.tseries.offsets import DateOffset
future_dates = [data.index[-1] + DateOffset(weeks = x) for x in range(0,49)]
# New dataframe for storing the future values
df1 = pd.DataFrame(index = future_dates[1:],columns = data.columns)
forecast = pd.concat([data,df1])
forecast['ARIMA_Forecast_Function'] = np.NaN
forecast['ARIMA_Predict_Function'] = np.NaN
forecast.head()  
ARIMA_history_f = [x for x in train]
f1 = []
for t in range(len(df1)):
 
 model = ARIMA(ARIMA_history_f, order = (2,1,2))
 model_fit = model.fit(disp=0)
 
 output = model_fit.forecast()[0][0]
 
 ARIMA_history_f.append(output)
 f1.append(output)
 
for i in range(len(f1)):
 forecast.iloc[144 + i,1] = f1[i]
forecast.tail()
forecast[['Passengers','ARIMA_Forecast_Function']].plot(figsize = (12,8));
ARIMA_history_p = [x for x in train]
f2 = []
for t in range(len(df1)):
 
 model = ARIMA(ARIMA_history_p, order = (2,1,2))
 model_fit = model.fit(disp=0)
 
 output = model_fit.predict(start = len(ARIMA_history_p),end = len(ARIMA_history_p),typ
 
 ARIMA_history_p.append(output)
 f2.append(output)
 
for i in range(len(f2)):
 forecast.iloc[144 + i,2] = f2[i]
forecast.tail()
 forecast[['Passengers','ARIMA_Predict_Function']].plot(figsize = (12,8));
 sum(f1) == sum(f2)
 data_diff_seas = data_diff.diff(12)
data_diff_seas = data_diff_seas.dropna()
dec = sm.tsa.seasonal_decompose(data_diff_seas,period = 12)
dec.plot()
plt.show() 
test_stationarity(data_diff_seas['Passengers'])
tsplot(data_diff_seas['Passengers'])
model = sm.tsa.statespace.SARIMAX(data['Passengers'],order = (2,1,2),seasonal_order = (0,1,
model_fit = model.fit()
print(model_fit.summary())
size = int(len(data) - 30)
train, test = data['Passengers'][0:size], data['Passengers'][size:len(data)]
print('\t SARIMA MODEL : In - Sample Forecasting \n')
history = [x for x in train]
predictions = []
for t in range(len(test)):
 
 model = sm.tsa.statespace.SARIMAX(history,order = (2,1,2),seasonal_order = (0,1,1,12))
 model_fit = model.fit()
 
 output = model_fit.forecast()
 yhat = output[0]
 predictions.append(float(yhat))
 
 obs = test[t]
 history.append(obs)
 
 print('predicted = %f, expected = %f' % (yhat, obs))
 predictions_series = pd.Series(predictions, index = test.index)
fig,ax = plt.subplots(nrows = 1,ncols = 1,figsize = (15,5))
plt.subplot(1,1,1)
plt.plot(data['Passengers'],label = 'Expected Values')
plt.plot(predictions_series,label = 'Predicted Values');
plt.legend(loc="upper left")
plt.show()
error = np.sqrt(mean_squared_error(test,predictions))
print('Test RMSE: %.4f' % error)
                                                                                      
