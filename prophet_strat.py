import numpy as np
import pandas as pd
import ipykernel
import pypfopt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import DiscreteAllocation
from prophet import Prophet



npos = np.zeros(50)
counter_port = 0

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    (nins, nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)
    return make_portfolio(prcSoFar)

position = np.zeros(50)
data_with_date = pd.DataFrame()
all_predictions = pd.DataFrame() # Includes data up till now and predictions for the next 10 days
upper_bounds = pd.Series(50)
lower_bounds = pd.Series(50)
date_start = '2021-01-01'

def get_predictions(price_so_far):
    nt, nins = price_so_far.shape
    global position, data_with_date, all_predictions, date_start, upper_bounds, lower_bounds
    data_with_date = price_so_far.copy()
    data_with_date.columns =  [int(i) for i in range(nins)]
    data_with_date['ds'] = pd.date_range(date_start, periods=len(data_with_date), freq='D')
    for i in range(nins):
        train = data_with_date[['ds', i]].copy()
        train.columns = ['ds', 'y']
        model = Prophet(
            changepoint_range=0.9,
        )
        # model.add_seasonality(name='monthly', period=250, fourier_order=100, prior_scale=10)
        # model.add_seasonality(name='weekly', period=100, fourier_order=80, prior_scale=7)
        
        model.fit(train)
        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)
        all_predictions[i] = forecast['yhat']
        expected_returns = all_predictions.iloc[-1]
    return expected_returns

def make_portfolio(prcSoFar):
    global npos
    global currentPos
    global counter_port
    if counter_port%50 == 1 or counter_port == 0:
        prcHistSoFar = prcSoFar
        prcHistSoFar = prcHistSoFar.transpose()
        df_prcHistSoFar = pd.DataFrame(prcHistSoFar)
        mu = get_predictions(df_prcHistSoFar)
        S = risk_models.exp_cov(df_prcHistSoFar)
        ef = EfficientFrontier(mu, S, weight_bounds=(-0.02, 0.02))
        weights = ef.max_sharpe()
        latest_prices = df_prcHistSoFar.iloc[-1]  
        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000*50)
        alloc, leftover = da.greedy_portfolio()
        all_assets = df_prcHistSoFar.columns.tolist()
        full_alloc = {asset: alloc.get(asset, 0) for asset in all_assets}
        rpos = np.array(list(full_alloc.values()))
        npos = np.vstack((npos, rpos))
        current_pos = (npos[-2,:] - 2*rpos)
    else:
        current_pos = npos[-2,:] - 2*npos[-1,:]
    counter_port += 1
    return current_pos