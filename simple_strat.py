# get expected returns
# divide by variance => [2, 5]
# multiply by 10000 -> money spent on each stock
# check purchase limit ()
# divide by stock price -> number of shares
# add to previous position
# return current position

import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)
    random_int_array = np.random.randint(-10, 5, size=50)
    rpos = np.array([int(x) for x in random_int_array])
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos