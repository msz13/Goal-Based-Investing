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

def __prob2(W0, W1, mean, std, h):
    return norm.pdf((np.log(W1/W0)-(mean-0.5*std**2)*h)/(std*np.sqrt(h)))

def __prob(W0, W1, mean, std, Infusion, Cost, h):
    return norm.pdf((np.log(W1/(W0+Infusion+Cost))-(mean-0.5*std**2)*h)/(std*np.sqrt(h)))

def calculateTransitionPropabilities(portfolioMeasures, W0: int, W1: np.array, infusions, costs, h=1):
    mean = portfolioMeasures[0]
    std = portfolioMeasures[1]
    p = norm.pdf((np.log(W1/(W0+infusions-costs))-(mean-0.5*std**2)*h)/(std*np.sqrt(h)))
    return p/p.sum()


def reachedGoal(W, goal=160):
    reachedGoal = W >= goal
    return reachedGoal.astype(int)

def calculateValuesForLastPeriod(W: np.array, k: np.array):
    values = np.zeros((len(k), len(W)))
    for i in range(len(k)):
        values[i] = np.where(W >= k[i,0], k[i,1], 0 )
    return np.amax(values, axis=0)

def calculateValues(W: np.array, k: np.array):
    return


def calculateTransitionPropabilitiesForAllPorfolios(portfolioMeasures, WT: np.array, WT1: np.array, infusions, costs, h=1):
    i = len(WT1)
    probabilities = np.zeros((i,len(portfolioMeasures),i),np.float64)
    for i in range(i):
        probabilities[i] = np.apply_along_axis(calculateTransitionPropabilities,1,portfolioMeasures, W0=WT[i], W1=WT1, infusions=infusions, costs=costs, h=1)
    return probabilities


def get_portfolios_strategies(VT1, propabilities):
    V = VT1 * propabilities
    sums = V.sum(2)
    maxes = np.amax(sums,1)
    portfolios_ids = np.argmax(sums,1)
    chosen_propabilities = np.take_along_axis(propabilities,np.expand_dims(portfolios_ids,axis=(0,1)),1)
    return portfolios_ids, maxes, chosen_propabilities

def __get_value_index(WT, wealth_value):
    difference = np.absolute(WT - wealth_value)
    index = np.argmin(difference)
    return index


def get_goals_strategies(goals, infusion, Wt, Wt1, VTK1, portfolios):
    k = len(goals)
    i = len(Wt)

    probabilities = calculateTransitionPropabilitiesForAllPorfolios(portfolios,Wt,Wt1,infusion,0)
    portfolios_strategies, VTk0, chosen_propabilities = get_portfolios_strategies(VTK1,probabilities)

    porfolios_strategies = np.zeros((k, i))
    propabilities_kc = np.zeros((k+1, i, len(Wt1)))
    values = np.zeros((k+1, i)) 
    Wtc = - np.tile(Wt,(k,1)) - np.repeat(goals[:,0],i).reshape((k,i))
    Wtc[Wtc < 0] = 0
    
    values[0] = VTk0
    propabilities_kc[0] = chosen_propabilities[:,0,:]
    
    for k in range(k):
        for i in range(i):
            if Wt[i] > goals[k][0]:
                value_index = __get_value_index(Wt, Wtc[k,i])
                portfolio_strategy = portfolios_strategies[value_index]
                probabilities = calculateTransitionPropabilities(portfolios[portfolio_strategy],Wt[i],Wt1,0,0)
                values[k+1,i] = (probabilities * VTK1).sum()+ goals[k,1]
                porfolios_strategies[k,i] = portfolio_strategy
                propabilities_kc[k+1,i] = probabilities
                             
                          
   
    strategies = values.argmax(0)
    return strategies, portfolios_strategies, values, propabilities_kc
   

# REFACTOR obliczyc vector wt-kc, zamienić minusowe liczby na zero
# dla każdej wartości wiekszej od zero wziac odpowiadnia strategie porfolio, obliczyc transition prob
# obliczxyc values
# trzeba zmienic strategie porfolio na te ktore wynikja z goals



''' def generateGlidePath(W0, goal, T, portfolioMeasures):
    iMax = 475
    grid = generateGrid(W0,T,iMax,meanMin,stdMin,meanMax,stdMax)
    strategies = np.zeros((T,iMax))
    V = np.zeros((T,iMax))
    probabilitiesT = np.zeros((T,iMax,iMax))

    indexOf100 = np.where(grid[1]==100)

    V[T-1] = reachedGoal(grid[T-1],goal)   

    for t in range(T-2,0,-1):
        probabilities = calculateTransitionPropabilitiesForAllPorfolios(portfolioMeasures,grid[t],grid[t+1])
        VT = V[t+1] * probabilities        
        porfolios_ids, VT_max = get_strategies(VT)
        V[t] = VT_max  
        strategies[t] = porfolios_ids  
        chosen_propabilities = np.take_along_axis(probabilities,np.expand_dims(porfolios_ids,axis=(0,1)),1)
        probabilitiesT[t] = chosen_propabilities[:,0,:]

    return strategies, grid '''



class InvestmentPlanner:
       
    def set_params(self, T: int, W0: float, infusion: float, infusionInterval: float, goals: np.array, portfolios: np.ndarray):
        self.iMax = 500
        infusions = np.full(T+1,infusion)   
        infusions[0] = 0    
        self.grid = generateGrid(W0, T, self.iMax, infusions, goals[:,0], portfolios[0,0], portfolios[0,1], portfolios[-1,0], portfolios[-1,1] )

        self._strategies = np.zeros((T,self.iMax))
        V = np.zeros((T,self.iMax))
        self.probabilitiesT = np.zeros((T,self.iMax,self.iMax))
       
        for t in range(T-2,0,-1):
            transition_probabilities = calculateTransitionPropabilitiesForAllPorfolios(portfolios,self.grid[t],self.grid[t+1], infusions[t], 0)
            porfolios_ids, VT_max_k0, k0_probabilities = get_portfolios_strategies(V[t+1],transition_probabilities)
            goal_ids, VT_max = get_goals_strategies(goals,self.grid[t], VT_max_k0)
            V[t] = VT_max  
            self._strategies[t] = porfolios_ids  
            self.probabilitiesT[t] = k0_probabilities[:,0,:]

        self._calculate_cumulative_propabilities(T, self.probabilitiesT)
    
    def _calculate_cumulative_propabilities(self, T, probabilitiesT):
        inputPropabilities = probabilitiesT
        sums = inputPropabilities.sum(1)
        cumulativeProbabilities = np.ones((T,self.iMax))
        cumulativeProbabilities[0] = sums[1]
        T=8
        for t in range(2,T):
            cumulativeProbabilities[t-1] = cumulativeProbabilities[t-2]*sums[t]
        self.propabilities = cumulativeProbabilities
         
        
    @property    
    def glide_paths(self):
        return self._strategies.T
    
   

    

   
    