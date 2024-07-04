import numpy as np
import pandas as pd
from prophet import Prophet

# number_of_ins = 50
# days_wait_before_retrain = 10

# # Position data
# current_pos = np.zeros(number_of_ins)
# previous_pos = np.zeros(number_of_ins)

# # Maybe save all of the previous data directly in this file?

# # Stock statistics


# # Save predictions n days ahead
# date_starting = '2023-07-01'
# days_saving_ahead = 5
# data_now_and_ahead = pd.DataFrame() # Should this have date information?


# # Function to check if date data is alright


# def getPositionWithProphet(price_so_far):
#     '''
#         [WORK IN PROGRESS]
#         An attempt at a trading algorithm using the Prophet library.
#         1. Grab existing data
#         2. Iterate through each stock and fit the model
#         3. Predict, number of days ahead proportional to the number of days of data 
#         4. Use the prediction to determine the position
#     '''
#     global current_pos, previous_pos, data_now_and_ahead
#     (nins, nt) = price_so_far.shape # nins is the number of instruments and nt is the number of days of data
#     price_so_far = pd.DataFrame(price_so_far) # Convert to DataFrame for easier manipulation
    
#     # Adding 
#     # Parameters
    
#     # Check data, if first run then add to data_now_and_ahead
#     # Else, update data_now_and_ahead to current row
#     if data_now_and_ahead.shape[0] == 0:
#         cols = [int(i) for i in range(50)]
#         data_now_and_ahead = pd.DataFrame(price_so_far, columns=cols)
#         date_range = pd.date_range(start=date_starting, periods=price_so_far.shape[1], freq='D')
#         data_now_and_ahead['ds'] = date_range
#     else: # Update data_now_and_ahead
#         data_now_and_ahead.iloc[:nt, :-1] = price_so_far
#         data_now_and_ahead['ds'] = pd.date_range(start=date_starting, periods=price_so_far.shape[1], freq='D')
    
#     if nt < 10: #TODO fix, as theres always the first 500
#         return current_pos

#     if nt % days_wait_before_retrain == 0: # Repredict the trend every 10 days
#         # Iterate through each stock
#         # Assemble the right data (ds, y)
#         # Train the model
#         # Predict n days ahead
#         # Save prediction, maybe upper/lower bounds as well?
#         # Use prediction to determine position
#         for stock in range(nins):
#             data = pd.DataFrame()
#             data['ds'] = data_now_and_ahead['ds']
#             data['y'] = data_now_and_ahead[stock]
#             model = Prophet(
#                 seasonality_mode='multiplicative',
#                 changepoint_prior_scale=0.01,
#                 seasonality_prior_scale=10,
#                 n_changepoints=100,
#                 changepoint_range=0.9,
#             )
#             model.add_seasonality(name='monthly', period=250, fourier_order=100, prior_scale=10)
#             model.add_seasonality(name='weekly', period=100, fourier_order=100, prior_scale=10)
#             model.fit(data)
#             future = model.make_future_dataframe(periods=days_saving_ahead)
#             forecast = model.predict(future)
#             # Save the prediction
#             data_now_and_ahead[stock] = forecast['yhat']

#     else: # Implement generic trading strategy
#         current_pos = [int(1) for i in range(nins)]
#     return current_pos

position = np.zeros(50)
data_with_date = pd.DataFrame()
all_predictions = pd.DataFrame() # Includes data up till now and predictions for the next 10 days
upper_bounds = pd.Series(50)
lower_bounds = pd.Series(50)
date_start = '2021-01-01'

def getPosition(price_so_far):
    nt, nins = price_so_far.shape
    global position, data_with_date, all_predictions, date_start, upper_bounds, lower_bounds
    data_with_date = price_so_far.copy()
    data_with_date.columns =  [int(i) for i in range(nins)]
    data_with_date['ds'] = pd.date_range(date_start, periods=len(data_with_date), freq='D')

    # Parameters
    predict_cycle = 10 # Number of days before retraining the model

    # Stage 1: Train the model and predict when necessary
    if len(price_so_far) % predict_cycle == 1: # TODO: tune this for runtime optimization
        for i in range(nins):
            train = data_with_date[['ds', i]].copy()
            train.columns = ['ds', 'y']
            model = Prophet(
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.01,
                seasonality_prior_scale=7,
                n_changepoints=120,
                changepoint_range=1,
            )
            model.add_seasonality(name='monthly', period=250, fourier_order=100, prior_scale=10)
            model.add_seasonality(name='weekly', period=100, fourier_order=100, prior_scale=10)
            
            model.fit(train)
            future = model.make_future_dataframe(periods=10)
            forecast = model.predict(future)
            all_predictions[i] = forecast['yhat']
            # TODO: add code to update upper_bounds and lower_bounds
    
    else: # Update data with current values, TODO raise flag if anomaly detected
        all_predictions = pd.concat([data_with_date, all_predictions.iloc[nt:]], axis = 'rows')

    for i in range(50):
        avg = all_predictions[i].mean()
        if data_with_date[i].iloc[-1] > avg:
            position[i] = 1
        else:
            position[i] = -1

    # for i in range(50):
    #     train = data_with_date[i, 'ds']
    #     train.columns = ['y', 'ds']
    #     model = Prophet()
    #     model.fit(train)
    #     future = model.make_future_dataframe(periods=10)
    #     forecast = model.predict(future)
    #     all_predictions[i] = forecast['yhat']
    return position