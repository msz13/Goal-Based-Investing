import numpy as np
from model._utilities import Goals

def WMax(t: int,W0: int, infusions: int, meanMax: float ,stdMax: float, stdMin: float):
    
    drift = meanMax-(stdMin**2)/2
    vioalitility = 3*stdMax
    
    valueOfInitialWelath = W0*np.exp(drift*t + vioalitility*np.sqrt(t))
    
    valueOfInfusions = 0
    for p in range(t): 
        valueOfInfusions += infusions[p]*np.exp(drift*(t-p-1) + vioalitility*np.sqrt(t-p-1))
                                                              
    return np.round(valueOfInitialWelath + valueOfInfusions,0)

def WMin(t, W0: int, infusions: int, max_goal_cost: int, meanMin, stdMin, stdMax):
    valueOfInfusions = 0
    drift = meanMin-(stdMax**2)/2
    vioalitility = 3*stdMax

    valueOfInitialWelath = W0*np.e**(drift*t - vioalitility*np.sqrt(t))

    for p in range(t): 
        valueOfInfusions += (infusions[p]-max_goal_cost[p])*np.exp(drift*(t-p-1) - vioalitility*np.sqrt(t-p-1))
    
    return  np.round(valueOfInitialWelath + valueOfInfusions)


def __deductE(row, logW0):
    diff = row - logW0
    e = diff[diff >= 0].min()
    return row - e

def generateGrid(W0, iMax, infusions, goals: Goals, minMean, minStd, maxMean, maxStd) ->np.array:
    T = goals.get_investment_period()
    grid = np.zeros((T,iMax))
    logW0 = np.log(W0)
    grid[0,:] = logW0
    Wmin = 1
    for t in range(1,T):
        cmax = goals.get_highest_cost_for_time(t)
        wMin = WMin(t,W0,infusions,cmax, minMean,minStd,maxStd)
        wMin = Wmin if np.all(wMin < Wmin) else wMin
        wMax = WMax(t,W0, infusions, maxMean,minStd, maxStd)
        row = np.linspace(np.log(wMin),np.log(wMax),iMax)
        row = __deductE(row,logW0)
        grid[t] = row
    return np.exp(grid)
