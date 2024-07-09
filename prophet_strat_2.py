import numpy as np
import pandas as pd
from prophet import Prophet
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, DiscreteAllocation
import matplotlib.pyplot as plt

# Try to incorporate 4 trading strategy in the algorithm-threshold optimization
# reference: https://github.com/gardnmi/prophet/blob/master/Stock%20Price%20Forecast%20with%20Prophet.ipynb
# Load data from prices_750.txt
data = np.loadtxt('data/prices_750.txt')
dates = pd.date_range(start='1980-01-01', periods=data.shape[0], freq='D')
price_data = pd.DataFrame(data, index=dates, columns=[f'Stock_{i}' for i in range(data.shape[1])])

# Define parameters
commission_rate = 0.0010  # Commission rate
position_limit = 10000  # Position limit
initial_cash = 10000  # Initial cash

# Calculate percent change for each column
percent_changes = price_data.pct_change().fillna(0)
price_data['Percent Change'] = percent_changes.mean(axis=1)

# Initialize columns for strategies
price_data['Hold'] = (1 + percent_changes).cumprod().mean(axis=1)

# Prophet prediction function
def get_predictions(data_with_date, nins):
    all_predictions = pd.DataFrame(index=data_with_date['ds'])
    for i in range(nins):
        stock_name = f'Stock_{i}'  # Correct column name
        train = data_with_date[['ds', stock_name]].copy()
        train.columns = ['ds', 'y']
        model = Prophet(changepoint_range=0.9)
        model.fit(train)
        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)
        all_predictions[stock_name] = forecast['yhat']
    return all_predictions

# Add date column
price_data['ds'] = price_data.index
nins = data.shape[1]
all_predictions = get_predictions(price_data, nins)
price_data['yhat'] = all_predictions.mean(axis=1)
price_data['Prophet'] = ((price_data['yhat'].shift(-1) > price_data['yhat']).shift(1) * (price_data['Percent Change']) + 1).cumprod()

# Prophet Thresh strategy
price_data['y'] = price_data.iloc[:, :-5].mean(axis=1)  # Mock actual prices
price_data['yhat_lower'] = price_data['yhat'] * 0.95  # Mock lower bound
price_data['Prophet Thresh'] = ((price_data['y'] > price_data['yhat_lower']).shift(1) * (price_data['Percent Change']) + 1).cumprod()

# Seasonality strategy
seasonality_flag = pd.Series(~price_data.index.month.isin([8, 9]), index=price_data.index)
price_data['Seasonality'] = ((seasonality_flag.shift(1)) * (price_data['Percent Change']) + 1).cumprod()

# Function to apply trading constraints
def apply_constraints(position, current_price):
    dollar_volume = np.abs(position * current_price).sum()
    commission = dollar_volume * commission_rate
    position = np.clip(position, -position_limit / current_price, position_limit / current_price)
    return position, commission

# Portfolio optimization function
npos = np.zeros(nins)
counter_port = 0

def make_portfolio(prcSoFar):
    global npos, counter_port
    prcHistSoFar = prcSoFar.transpose()
    df_prcHistSoFar = pd.DataFrame(prcHistSoFar)
    mu = get_predictions(df_prcHistSoFar, df_prcHistSoFar.shape[1])
    S = risk_models.exp_cov(df_prcHistSoFar)
    ef = EfficientFrontier(mu, S, weight_bounds=(-0.02, 0.02))
    weights = ef.max_sharpe()
    latest_prices = df_prcHistSoFar.iloc[-1]
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000*50)
    alloc, leftover = da.greedy_portfolio()
    all_assets = df_prcHistSoFar.columns.tolist()
    full_alloc = {asset: alloc.get(asset, 0) for asset in all_assets}
    rpos = np.array(list(full_alloc.values()))
    if counter_port % 50 == 0:
        npos = np.vstack((npos, rpos))
    else:
        currentPos = npos[-2, :] - 2 * npos[-1, :]
    counter_port += 1
    return currentPos

# Simulate trading with constraints and portfolio optimization
def simulate_trading(strategy):
    cash = initial_cash
    position = np.zeros(nins)
    global counter_port
    for date, row in price_data.iterrows():
        if date != price_data.index[0]:
            current_price = row[:-5]
            position, commission = apply_constraints(position, current_price)
            cash -= commission
        if counter_port % 50 == 0:
            prcSoFar = price_data.loc[:date, price_data.columns[:-5]].values
            position = make_portfolio(prcSoFar)
        position_change = row[strategy] * initial_cash / nins
        position += position_change
    return cash + (position * price_data.iloc[-1, :-5]).sum()

# Simulate each strategy
hold_result = simulate_trading('Hold')
prophet_result = simulate_trading('Prophet')
prophet_thresh_result = simulate_trading('Prophet Thresh')
seasonality_result = simulate_trading('Seasonality')

# Print final values
print(f"Hold = {hold_result:.2f}")
print(f"Prophet = {prophet_result:.2f}")
print(f"Prophet Thresh = {prophet_thresh_result:.2f}")
print(f"Seasonality = {seasonality_result:.2f}")

# Plot the strategies
plt.figure(figsize=(10, 6))
price_data.dropna().set_index('ds')[['Hold', 'Prophet', 'Prophet Thresh', 'Seasonality']].plot()
plt.title('Cumulative Returns of Different Strategies')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.legend()
plt.show()
