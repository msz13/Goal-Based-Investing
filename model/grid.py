import numpy as np

def WMax(t,W0, infusions, meanMax,stdMin,stdMax):
    valueOfInfusions = 0
    for i in range(t):
        valueOfInfusions += infusions[i]*np.exp((meanMax - (stdMin**2)/2)*(t-i) + (3*stdMax*np.sqrt(t-i)))
                                                           
    return W0*np.exp((meanMax-(stdMin**2/2))*t + 3*stdMax*np.sqrt(t)) + valueOfInfusions

def WMin(t, W0, infusions, goal, meanMin, stdMin, stdMax):
    valueOfInfusions = 0
    for i in range(t):
        valueOfInfusions += (infusions[i]-goal)*np.exp((meanMin - (stdMin**2)/2)*(t-i) - (3*stdMax*np.sqrt(t-i)))

    return W0*np.e**((meanMin-stdMax**2/2)*t - 3*stdMax*np.sqrt(t)) + valueOfInfusions


def deductE(row, logW0):
    diff = row - logW0
    e = diff[diff >= 0].min()
    return row - e

def generateGrid(W0, T, iMax, infusions, goals, minMean, minStd, maxMean, maxStd) ->np.array:
    grid = np.zeros((T+2,iMax))
    logW0 = np.log(W0)
    grid[0,:] = logW0
    Wmin = 1
    for t in range(1,T+1):
        cmax = goals.get(t)[:,0].max() if (goals.get(t) is not None) else 0 
        wMin = WMin(t,W0,infusions,cmax, minMean,minStd,maxStd)
        wMin = Wmin if np.all(wMin < Wmin) else wMin
        wMax = WMax(t,W0, infusions, maxMean,minStd, maxStd)
        row = np.linspace(np.log(wMin),np.log(wMax),iMax)
        row = deductE(row,logW0)
        grid[t] = row
    return np.exp(grid)
