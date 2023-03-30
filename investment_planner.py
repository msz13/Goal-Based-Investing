import numpy as np
from scipy.stats import norm
import pandas as pd

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
    grid = np.zeros((T+1,iMax))
    logW0 = np.log(W0)
    grid[0,:] = logW0
    Wmin = 1
    for t in range(1,T+1):
        cmax = goals[t-1][:,0].max() if (len(goals[t]) >1) else 0 
        wMin = WMin(t,W0,infusions,cmax, minMean,minStd,maxStd)
        wMin = Wmin if np.all(wMin < Wmin) else wMin
        wMax = WMax(t,W0, infusions, maxMean,minStd, maxStd)
        row = np.linspace(np.log(wMin),np.log(wMax),iMax)
        row = deductE(row,logW0)
        grid[t] = row
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

def __calculateWtc(WT, goals_costs, infusion):        
    k = len(goals_costs)
    cf = goals_costs - infusion
    Wtc = np.tile(WT,(k,1)) - cf.reshape((k,1))
    return Wtc

def calculateTransitionPropabilitiesForGoals(WTc, Wt1, portfolios_wtc, h=1):
    i0 = WTc.shape[1]
    i1 = len(Wt1)    
    k = WTc.shape[0]
    
    portfolios_measures = portfolios_wtc.reshape(k*i0,2)  
    b = (portfolios_measures[:,0]-0.5*(portfolios_measures[:,1]**2))*h
    b = b.reshape(k*i0,1)
    c = portfolios_measures[:,1]*np.sqrt(h)
    c = c.reshape(k*i0,1)
    result = np.zeros((k*i0,i1))

    Wtc = WTc.reshape((k*i0,1))
    Wt1k = np.tile(Wt1, (k*i0,1))
    
    np.divide(Wt1k, Wtc, out=result, where=Wtc>0)
    np.log(result,out=result, where=result>0)
    result = np.where(result > 0, (result - b)/c, 0)
    result = norm.pdf(result)
    result = np.divide(result,np.expand_dims(result.sum(1), axis=1),where=result>0)
    result = result.reshape(k,i0,i1) 
    
    return result


def reachedGoal(W, goal=160):
    reachedGoal = W >= goal
    return reachedGoal.astype(int)

def calculateValuesForLastPeriod(W: np.array, k: np.array):
    values = np.zeros((len(k), len(W)))
    for i in range(len(k)):
        values[i] = np.where(W >= k[i,0], k[i,1], 0 )
    return np.amax(values, axis=0)



def calculateTransitionPropabilitiesForAllPorfolios(portfolios,WT,WT1,infusions, h=1):
    l = len(portfolios)
    i = len(WT)
    b = (portfolios[:,0]-0.5*portfolios[:,1]**2)*h
    bi = np.repeat(b, i).reshape(l*i,1)
    c = portfolios[:,1]*np.sqrt(h)
    ci = np.repeat(c, i).reshape(l*i,1)
   
    Wt1 = np.tile(WT1, (l*i,1))
    Wt = np.tile(WT,(l,1)).reshape(i*l,1)+infusions
    result = norm.pdf((np.log(Wt1/Wt)- bi)/ci).reshape(l,i,len(WT1))
    return np.divide(result, result.sum(), where=result.sum()>0) #result/result.sum(2).reshape(l,i,1)



def get_portfolios_strategies(VT1, probabilities):
    Vt = VT1 * probabilities
    sums = Vt.sum(2)
    maxes = np.amax(sums,0)
    portfolios_ids = np.argmax(sums,0)    
    chosen_propabilities = np.take_along_axis(probabilities.transpose(1,0,2),portfolios_ids.reshape(len(VT1),1,1),1).squeeze(1)

    return portfolios_ids, maxes, chosen_propabilities

def __get_porfolios_strategy_for_wealth_values(WT, Wtc, porfolios_strategies):
    k = Wtc.shape[0]
    i = Wtc.shape[1]
    wc = Wtc.reshape(k*i,1)
    wg = np.tile(WT,(k*i,1))
    difference = np.absolute(wg - wc)
    index = np.argmin(difference, axis=1)
    index = index.reshape(k,i)
    return np.take(porfolios_strategies,index)

'''
Wt: wealth in time t
Wt1: wealth in time t1
'''

def get_goals_strategies(goals, infusion, Wt, Wt1, VTK1, portfolios, h=1): 
    k = len(goals)
    i = len(Wt)
           
    probabilities = calculateTransitionPropabilitiesForAllPorfolios(portfolios,Wt,Wt1,infusion,1)
    portfolios_strategies, VTk0, chosen_propabilities = get_portfolios_strategies(VTK1,probabilities)
    
    if(goals.shape[0] <= 1):
        return np.zeros(i), portfolios_strategies, VTk0, chosen_propabilities

    probabilities_kc = np.zeros((k, i, len(Wt1)))
    values = np.zeros((k, i)) 
      
    values[0] = VTk0
    probabilities_kc[0] = chosen_propabilities    
    
    Wtc = __calculateWtc(Wt,goals[1:,0],infusion)

    goal_porfolio_strategies = __get_porfolios_strategy_for_wealth_values(Wt,Wtc,portfolios_strategies)
    portfolios_measures = np.take(portfolios, goal_porfolio_strategies, axis=0)
    
    probabilities_kc[1:] = calculateTransitionPropabilitiesForGoals(Wtc, Wt1, portfolios_measures)
    values[1:] = (probabilities_kc[1:] * VTK1).sum(2)+ np.expand_dims(goals[1:,1],1)                         
                        
    strategies = values.argmax(0)
    chosen_goal_propabilities = np.take_along_axis(probabilities_kc,np.expand_dims(strategies,axis=(0,1)),1)
    
    return strategies, portfolios_strategies, values.max(0), np.squeeze(chosen_goal_propabilities)
    
   

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
         
        self.grid = generateGrid(W0, T, self.iMax, infusions, goals, portfolios[0,0], portfolios[0,1], portfolios[-1,0], portfolios[-1,1])

        self._portfolio_strategies = np.zeros((T,self.iMax))
        self._goal_strategies = np.zeros((T,self.iMax))
        V = np.zeros((T,self.iMax))
        V[-1] = calculateValuesForLastPeriod(self.grid[-1],goals[-1])
        self.probabilitiesT = np.zeros((T,self.iMax, self.iMax))
       
        for t in range(T-1,-1,-1):
            #goal_strategies, portfolio_strategies, values, probabilities = get_goals_strategies(goals[t], infusions[t], self.grid[t], self.grid[t+1], V[t+1], portfolios)
            goal_strategies, portfolio_strategies, values, probabilities = get_goals_strategies(goals[t], infusions[1], self.grid[1], self.grid[1+1], V[1+1], portfolios)
            print(t)
            print(probabilities.shape)
            #V[t] = values 
            #self._portfolio_strategies[t] = portfolio_strategies
            #print(self.grid[t].shape) 
            #print(self.grid[t-1].shape)              
            #self.probabilitiesT[t] = probabilities            
            #self._goal_strategies[t] = goal_strategies
                        
        #self._calculate_cumulative_propabilities(T, self.probabilitiesT)
    
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
        return self._portfolio_strategies.T
    
'''  def goal_probabilities(self):
        T = self.propabilities.shape[0]

        for t in range(0,T):
            goals = np.unique(self.goal_strategies[goal_strategies > 0])
        result = np.zeros(len(goals))
        for k in goals:
        k_index = np.where(goal_strategies == k)
        result[k-1]  = np.take(probabilites, k_index).sum()       
    
    return result '''

    
    
   

    

   
    