import numpy as np
from scipy.stats import norm
import pandas as pd

def WMax(t,W0, infusions, meanMax,stdMin,stdMax):
    valueOfInfusions = 0
    for i in range(t):
        valueOfInfusions += infusions[i]*np.exp((meanMax - (stdMin**2)/2)*(t-i) + (3*stdMax*np.sqrt(t-i)))
                                                           
    return W0*np.exp((meanMax-(stdMin**2/2))*t + 3*stdMax*np.sqrt(t) + valueOfInfusions)
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

def generateGrid(W0, T, iMax, infusions, minMean, minStd, maxMean, maxStd) ->np.array:
    grid = np.zeros((T,iMax))
    logW0 = np.log(W0)
    for t in range(1,T+1):
        row = np.linspace(np.log(WMin(t,W0, minMean,minStd,maxStd)),np.log(WMax(t,W0, infusions, maxMean,minStd, maxStd)),iMax)
        row = deductE(row,logW0)
        grid[t-1] = row
    return np.exp(grid)

def calculateTransitionPropabilities(portfolioMeasures, W0, W1, h=1):
    p = __prob2(W0,W1,portfolioMeasures[0], portfolioMeasures[1],1)
    return p/p.sum()

def calculateTransitionPropabilitiesForAllPorfolios(portfolioMeasures, W0, W1, h=1):
    return []

def reachedGoal(W, goal=160):
    reachedGoal = W >= goal
    return reachedGoal.astype(int)

def calculateTransitionPropabilitiesForAllPorfolios(portfolioMeasures, WT, WT1, h=1):
    i = len(WT1)
    probabilities = np.zeros((i,len(portfolioMeasures),i),np.float64)
    for i in range(i):
        probabilities[i] = np.apply_along_axis(calculateTransitionPropabilities,1,portfolioMeasures, W0=WT, W1=WT1[i])
    return probabilities


def get_strategies(V):
    sums = V.sum(2)
    maxes = np.amax(sums,1)
    porfolio_ids = np.argmax(sums,1)
    return porfolio_ids, maxes




class InvestmentPlanner:
       
    def set_params(self, T: int, W0: float, infusion: float, infusionInterval: float, portfolios: np.ndarray):
        iMax = 5
        infusions = np.full(T,infusion)       
        self.grid = generateGrid(W0, T, iMax, infusions, portfolios[0,0], portfolios[0,1], portfolios[-1,0], portfolios[-1,1] )
    