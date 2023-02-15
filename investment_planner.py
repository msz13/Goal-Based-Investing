import numpy as np
from scipy.stats import norm
import pandas as pd

def WMax(t,W0, infusions, meanMax,stdMin,stdMax):
    valueOfInfusions = 0
    for i in range(t):
        valueOfInfusions += infusions[i]*np.exp((meanMax - (stdMin**2)/2)*(t-i) + (3*stdMax*np.sqrt(t-i)))
                                                           
    return W0*np.exp((meanMax-(stdMin**2/2))*t + 3*stdMax*np.sqrt(t)) + valueOfInfusions

def WMin(t, W0, infusions, goals, meanMin, stdMin, stdMax):
    valueOfInfusions = 0
    for i in range(t+1):
        valueOfInfusions += (infusions[i]-goals[i])*np.exp((meanMin - (stdMin**2)/2)*(t-i) - (3*stdMax*np.sqrt(t-i)))

    return W0*np.e**((meanMin-stdMax**2/2)*t - 3*stdMax*np.sqrt(t)) + valueOfInfusions


def deductE(row, logW0):
    diff = row - logW0
    e = diff[diff >= 0].min()
    return row - e

def __prob2(W0, W1, mean, std, h):
    return norm.pdf((np.log(W1/W0)-(mean-0.5*std**2)*h)/(std*np.sqrt(h)))

def generateGrid(W0, T, iMax, infusions, goals, minMean, minStd, maxMean, maxStd) ->np.array:
    grid = np.zeros((T,iMax))
    logW0 = np.log(W0)
    Wmin = 1
    for t in range(1,T+1):
        wMin = WMin(t,W0,infusions,goals, minMean,minStd,maxStd)
        wMin = Wmin if wMin < Wmin else wMin
        wMax = WMax(t,W0, infusions, maxMean,minStd, maxStd)
        row = np.linspace(np.log(wMin),np.log(wMax),iMax)
        row = deductE(row,logW0)
        grid[t-1] = row
    return np.exp(grid)

def calculateTransitionPropabilities(portfolioMeasures, W0, W1, h=1):
    p = __prob2(W0,W1,portfolioMeasures[0], portfolioMeasures[1],1)
    return p/p.sum()


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
       
    def set_params(self, T: int, W0: float, infusion: float, infusionInterval: float, goals: np.array, portfolios: np.ndarray):
        iMax = 8
        infusions = np.full(T+1,infusion)   
        infusions[0] = 0    
        self.grid = generateGrid(W0, T, iMax, infusions, goals, portfolios[0,0], portfolios[0,1], portfolios[-1,0], portfolios[-1,1] )
    