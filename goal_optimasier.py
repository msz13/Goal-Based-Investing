import numpy as np
from scipy.stats import norm

def WMax(t,W0, meanMax,stdMin,stdMax):
    return W0*np.e**((meanMax-(stdMin**2/2))*t + 3*stdMax*np.sqrt(t))
def WMin(t, W0, meanMin, stdMin, stdMax):
    return W0*np.e**((meanMin-stdMax**2/2)*t - 3*stdMax*np.sqrt(t))
def Wi(i, imax, Wmin, Wmax):
    return np.log(Wmin)+((i-1)/(imax-1))*(np.log(Wmax) - np.log(Wmin))
def deductE(row, logW0):
    diff = row - logW0
    e = diff[diff >= 0].min()
    return row - e

def __prob2(W0, W1, mean, std, h):
    return norm.pdf((np.log(W1/W0)-(mean-0.5*std**2)*h)/(std*np.sqrt(h)))

def generateGrid(W0, T, iMax, minMean, minStd, maxMean, maxStd):
    grid = np.zeros((T,iMax))
    logW0 = np.log(W0)
    for t in range(1,T+1):
        #iter = (Wi(i,iMax, WMin(t,W0, minMean,minStd,maxStd),WMax(t,W0, maxMean,minStd, maxStd)) for i in range(1,iMax+1))
        iter = np.linspace(np.log(WMin(t,W0, minMean,minStd,maxStd)),np.log(WMax(t,W0, maxMean,minStd, maxStd)),iMax)
        row = np.fromiter(iter, float)
        row = deductE(row,logW0)
        grid[t-1] = row
    return np.exp(grid)

def calculateTransitionPropabilities(W0, W1: np.array, mean,std, h):
    p = np.fromiter((__prob2(W0,W,mean,std,h) for W in W1),np.float64)
    return p/p.sum()

def __calculate_values(portfolios,propabilities)

