import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima

# Load the data
file_path = 'data/prices.txt'
data = pd.read_csv(file_path, sep=r"\s+", header=None)

# Visualize the Data
plt.figure(figsize=(10, 6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.plot(data.iloc[:, 0])  # Assuming first column as  prices
plt.title('Stock price time series')
plt.show()


# Test for Stationarity
def test_stationarity(timeseries):
    rolmean = timeseries.rolling(5).mean()
    rolstd = timeseries.rolling(5).std()

    plt.plot(timeseries, color='yellow', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)

    print("Results of Dickey-Fuller Test")
    dftest = adfuller(timeseries, autolag='AIC')
    output = pd.Series(dftest[0:4],
                       index=['Test Statistic', 'p-value', 'No. of Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        output[f'Critical Value ({key})'] = value
    print(output)


# Select the instrument's prices
time_series = data.iloc[:, 33]

# Test stationarity
test_stationarity(time_series)

# Difference the series to make it stationary
time_series_diff = time_series.diff().dropna()

# Test stationarity again
test_stationarity(time_series_diff)

# Plot ACF and PACF
fig, ax = plt.subplots(1, 2, figsize=(18, 4))
plot_acf(time_series_diff, ax=ax[0], lags=20, title="Autocorrelation Function")
plot_pacf(time_series_diff, ax=ax[1], lags=20, title="Partial Autocorrelation Function")
plt.show()

# Seasonal Decomposition
result = seasonal_decompose(time_series, model='multiplicative', period=20)
fig = result.plot()
fig.set_size_inches(16, 9)
plt.show()

# Log Transformation
df_log = np.log(time_series)
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()

plt.plot(df_log, color='yellow', label='Original')
plt.plot(moving_avg, color='red', label='Rolling Mean')
plt.plot(std_dev, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Log Transformed Rolling Mean and Standard Deviation')
plt.show()

# Splitting Data into Train and Test Sets
train_data, test_data = df_log[3:int(len(df_log) * 0.9)], df_log[int(len(df_log) * 0.9):]
plt.figure(figsize=(10, 6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.plot(df_log, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()
plt.show()

# Automated ARIMA Model Selection
model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                             test='adf', max_p=3, max_q=3, m=1, d=None,
                             seasonal=False, start_P=0, D=0, trace=True,
                             error_action='ignore', suppress_warnings=True,
                             stepwise=True)
print(model_autoARIMA.summary())

# Plotting Diagnostics
model_autoARIMA.plot_diagnostics(figsize=(15, 8))
plt.show()

# Fitting the ARIMA Model Manually
model = ARIMA(train_data, order=(1, 1, 1))
fitted = model.fit()
print(fitted.summary())

# Forecasting and Plotting the Results
# 预测
forecast_results = fitted.forecast(len(test_data), alpha=0.05)  # 仅假设返回预测值

# 使用预测结果
fc = forecast_results

# 手动计算置信区间（仅示例，需要根据实际情况调整）
if not hasattr(forecast_results, 'conf_int'):
    # 计算标准误差，这需要您知道或估算预测的标准差
    se = np.std(fc) / np.sqrt(len(fc))  # 这是一个简化的估计示例
    conf_interval = 1.96 * se  # 95% 置信区间的临界值

    # 计算置信区间的上下界
    lower_series = fc - conf_interval
    upper_series = fc + conf_interval
    conf = np.vstack([lower_series, upper_series]).T  # 转置以匹配常用格式
else:
    conf = forecast_results.conf_int()

# 绘图或其他处理
plt.figure(figsize=(12, 5))
plt.plot(train_data.index, train_data, label='Training Data')
plt.plot(test_data.index, test_data, color='blue', label='Actual Stock Price')
plt.plot(fc.index, fc, color='orange', label='Predicted Stock Price')
if conf is not None:
    plt.fill_between(fc.index, conf[:, 0], conf[:, 1], color='k', alpha=0.1)

plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend(loc='upper left')
plt.show()