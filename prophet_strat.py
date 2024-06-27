import numpy as np
import pandas as pd
from prophet import Prophet



nInst = 50
currentPos = np.zeros(nInst)
training_data = pd.read_csv(r'data/prices0to250.csv')
prediction = training_data.copy()



def getPositionWithProphet(prcSoFar):
    '''
        [WORK IN PROGRESS]
        An attempt at a trading algorithm using the Prophet library.
        1. Grab existing data
        2. Iterate through each stock and fit the model
        3. Predict, number of days ahead proportional to the number of days of data 
        4. Use the prediction to determine the position
    '''
    global currentPos, training_data, prediction
    # Parameters


    (nins, nt) = prcSoFar.shape # nins is the number of instruments and nt is the number of days of data
    for i in range(nins):
        # Create a dataframe with the date and price
        ...
    return currentPos