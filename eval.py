import numpy as np
import pandas as pd
from simple_strat import getMyPosition as getPosition

nInst = 0 # Number of stocks/instruments
nt = 0 # Number of trading periods/days
commRate = 0.0010
dlrPosLimit = 10000 # Max total position


def loadPrices(fn):
    '''
        Load prices from the specified file.
        Returns the prices as a transpose of a numpy array
        Updates the global variables nInst and nt
    '''
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T


pricesFile = "data/prices.txt" # Specify the file name here
prcAll = loadPrices(pricesFile) # Load the prices
print("Loaded %d instruments for %d days" % (nInst, nt))


def calcPL(prcHist):
    '''
        Main function for testing the trading strategy
        Calculating P&L for a given price history
        Input:
            prcHist: a 2D numpy array with stock prices
        Returns:
        (plmu, ret, plstd, annSharpe, totDVolume, newPosOrig)
            plmu: Mean P&L
            ret: Total Return (plmu / totDVolume)
            plstd: Standard Deviation of P&L, volatility or risk associated with the strategy's daily returns. Higher is riskier
            annSharpe: Annualized Sharpe ratio, calculated as mean(P&L) / std(P&L) * sqrt(250). Higher is better
                Read more: https://www.investopedia.com/terms/s/sharperatio.asp
                Represents the excess return per unit of risk in an annualized context
            totDVolume: Total Dollar Volume traded over the entire duration of the price history
    '''
    cash = 0 # Initial available cash
    curPos = np.zeros(nInst) # Current position, initializes an 1*nInst array of nInst stocks with 0s
    totDVolume = 0 # Total Dollar Volume traded
        # totDVolumeSignal = 0 # TDV for signal trading, unused
        # totDVolumeRandom = 0 # Total Dollar Volume for random trading, unused
    value = 0 # Current portfolio value
    todayPLL = [] # P&L for each day, values TODO find out what values are stored here
    (_, nt) = prcHist.shape # Get the number of trading days
    for t in range(250, 501): 
        # Iterate over the trading days from 250 to 500
        prcHistSoFar = prcHist[:, :t] # Get the price history until day t
        global newPosOrig # New position to take
        newPosOrig = getPosition(prcHistSoFar) # Get position for the team's code
        print()
        curPrices = prcHistSoFar[:, -1]
        posLimits = np.array([int(x) for x in dlrPosLimit / curPrices]) # Calculate number of shares that can be bought for each stock, staying under the limit
        newPos = np.clip(newPosOrig, -posLimits, posLimits)
            # np.clip: Restrict the values in an array to stay within a given range
            # Here, the range is -posLimits to posLimits
            # This is to ensure that the position taken does not exceed the dollar position limit
        deltaPos = newPos - curPos # Change in position
        dvolumes = curPrices * np.abs(deltaPos) # Dollar volume of the trade
        
        # Calculate the total dollar volume traded
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume
        
        comm = dvolume * commRate # Commission for the trade
        cash -= curPrices.dot(deltaPos) + comm # Updates the cash after the trade
        curPos = np.array(newPos) # Updates the current position
        posValue = curPos.dot(curPrices) # Current position value
        todayPL = cash + posValue - value
        todayPLL.append(todayPL)
        value = cash + posValue
        ret = 0.0
        if (totDVolume > 0):
            ret = value / totDVolume
        print("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" %
              (t, value, todayPL, totDVolume, ret))
    pll = np.array(todayPLL)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(250) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume)


(meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll)
score = meanpl - 0.1*plstd
print("=====")
print("mean(PL): %.1lf" % meanpl)
print("return: %.5lf" % ret)
print("StdDev(PL): %.2lf" % plstd)
print("annSharpe(PL): %.2lf " % sharpe)
print("totDvolume: %.0lf " % dvol)
print("Score: %.2lf" % score)
#print("Test new position: ", newPosOrig)