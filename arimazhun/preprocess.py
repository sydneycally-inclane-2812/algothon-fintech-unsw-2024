import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
file_path = 'prices.txt'
data = pd.read_csv(file_path, sep="\s+", header=None)
print("Data shape:", data.shape)
print(data.head())
plt.figure(figsize=(14, 14))
for i in range(data.shape[1]):
    plt.plot(data[i], label=f'Stock {i+1}')


plt.title('Price Time Series for 50 Stocks Over 500 Days')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# 假设我们将股票分为5组，每组10只
num_groups = 5
stocks_per_group = 10

# 创建子图布局
fig = make_subplots(rows=num_groups, cols=1)

# 添加每组股票的数据
for i in range(num_groups):
    for j in range(stocks_per_group):
        stock_index = i * stocks_per_group + j
        fig.add_trace(
            go.Scatter(x=list(range(data.shape[0])), y=data[stock_index], mode='lines', name=f'Stock {stock_index+1}'),
            row=i+1, col=1
        )

# 更新图表布局
fig.update_layout(height=1200, title_text="Stock Prices Grouped by Subset", showlegend=False)
fig.show()


# 假设data是一个DataFrame，包含了50只股票的500天价格数据
# 计算每只股票的波动率（标准差）
volatility = data.std()

# 分组：低风险（低于第33百分位数），中风险（33到67百分位数），高风险（高于第67百分位数）
low_risk = volatility[volatility <= volatility.quantile(0.33)]
mid_risk = volatility[(volatility > volatility.quantile(0.33)) & (volatility <= volatility.quantile(0.67))]
high_risk = volatility[volatility > volatility.quantile(0.67)]

# 打印分组结果
print("Low Risk Stocks:", low_risk.index)
print("Mid Risk Stocks:", mid_risk.index)
print("High Risk Stocks:", high_risk.index)

# 计算RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

rsi_values = data.apply(calculate_rsi)

# 获取最后一天的RSI作为当前RSI
current_rsi = rsi_values.iloc[-1]

# 分组：低RSI（<30），中RSI（30-70），高RSI（>70）
low_rsi = current_rsi[current_rsi < 30]
mid_rsi = current_rsi[(current_rsi >= 30) & (current_rsi <= 70)]
high_rsi = current_rsi[current_rsi > 70]

# 打印分组结果
print("Low RSI Stocks:", low_rsi.index)
print("Mid RSI Stocks:", mid_rsi.index)
print("High RSI Stocks:", high_rsi.index)