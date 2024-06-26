
import numpy as np

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)


def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape # nins is the number of instruments and nt is the number of days of data
    if (nt < 2):
        return np.zeros(nins)
    # Grabbing the last price and the second last price, calculate log return
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
    lNorm = np.sqrt(lastRet.dot(lastRet))
    lastRet /= lNorm # Normalizing the log return, so all values adds up to 1

    rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
        # rpos is the position we want to take, which is proportional to the normalized log return
        # 5000 is the money available to the instruments.
        # divide by prcSoFar[:, -1] to calculate the number of shares we want to buy/sell
    currentPos = np.array([int(x) for x in currentPos+rpos])
        # Updates the current position by adding the new position we want to take
    return currentPos

# Model building

