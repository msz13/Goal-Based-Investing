import numpy as np
from numba import jit
from model.propabilities import calculateTransitionPropabilitiesForAllPorfolios, calculateTransitionPropabilitiesForGoals, _calculate_cumulative_propabilities
from model.grid import generateGrid

def convert_goals_to_k(goals):
    result = {}

    for goal in goals:
        result[goal['time']] = np.array([[goal['cost'], goal['utility']]]) 

    return result

def __calculateWtc(WT, goals_costs, infusion):        
    k = len(goals_costs)
    cf = goals_costs - infusion
    Wtc = np.tile(WT,(k,1)) - cf.reshape((k,1))
    return Wtc

def reachedGoal(W, goal=160):
    reachedGoal = W >= goal
    return reachedGoal.astype(int)

def calculateValuesForLastPeriod(W: np.array, k: np.array):
    values = np.zeros((len(k), len(W)))
    goal_strategies = np.zeros(len(W))
    for i in range(len(k)):
        values[i] = np.where(W >= k[i,0], k[i,1], 0)
    return np.amax(values, axis=0), np.argmax(values, axis=0)


def get_portfolios_strategies(VT1, probabilities):
    Vt = VT1 * probabilities
    sums = np.round(Vt.sum(2),4)
    maxes = np.amax(sums,0)
    portfolios_ids = np.argmax(sums,0)    
    chosen_propabilities = np.take_along_axis(probabilities.transpose(1,0,2),portfolios_ids.reshape(len(VT1),1,1),1).squeeze(1)

    return portfolios_ids, maxes, chosen_propabilities

def get_goals_strategies(propabilites, goal_utilities, VT1):
    utilities = np.vstack(([0],np.expand_dims(goal_utilities,1)))
    VT = np.sum(propabilites * VT1,axis=2) + utilities
    result_V = np.nanmax(VT,0) #np.expand_dims(utilities,1) #VT #np.nanmax(VT,axis=0)
    result_a = np.nanargmax(VT,0)
    return result_V, result_a

#@jit(nopython=True)
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
#@jit(nopython=True)
def calculateBelmanForT(goals, infusion, Wt, Wt1, VTK1, portfolios, h=1): 
    i = len(Wt)
           
    probabilities = calculateTransitionPropabilitiesForAllPorfolios(portfolios,Wt,Wt1,infusion,1)
    portfolios_strategies, VTk0, chosen_propabilities = get_portfolios_strategies(VTK1,probabilities)
    
    if(goals is None):
        return np.zeros(i), portfolios_strategies, VTk0, chosen_propabilities
    
    k = len(goals)
    probabilities_kc = np.zeros((k+1, i, len(Wt1)))
    values = np.zeros((k, i)) 
      
    #values[0] = VTk0
    probabilities_kc[0] = chosen_propabilities    
    
    Wtc = __calculateWtc(Wt,goals[:,0],infusion)

    goal_portfolio_strategies = __get_porfolios_strategy_for_wealth_values(Wt,Wtc,portfolios_strategies)
    portfolios_measures = np.take(portfolios, goal_portfolio_strategies, axis=0)
    
    probabilities_kc[1:] = calculateTransitionPropabilitiesForGoals(Wtc, Wt1, portfolios_measures)

    values, goal_strategies = get_goals_strategies(probabilities_kc,goals[:,1],VTK1)

    goal_portfolio_strategies = np.vstack((portfolios_strategies, goal_portfolio_strategies))
    
    ''' values[1:],  = (probabilities_kc[1:] * VTK1).sum(2)+ np.expand_dims(goals[:,1],1)                         
                        
    strategies = values.argmax(0) '''

    chosen_goal_propabilities = np.zeros((i,len(Wt1)))

    for i in range(i):
        chosen_goal_propabilities[i,:] = probabilities_kc[goal_strategies[i],i]

    chosen_portfolios_strategies = np.zeros(6)
    for i in range(i):
        chosen_portfolios_strategies[i] = goal_portfolio_strategies[goal_strategies[i],i] 
    
    return goal_strategies, chosen_portfolios_strategies, values, np.squeeze(chosen_goal_propabilities)


def calculateBelman(grid,goals, portfolios):
    T = grid.shape[0]
    i = grid.shape[1]

    goal_strategies = np.zeros((T,i))
    portfolios_strategies = np.zeros((T,i))
    probabilities = np.zeros((T,i,i))
    V = np.zeros((T,i))
    V[-1] = 0

    for t in range(T-2,-1,-1):
        goal_strategies[t], portfolios_strategies[t], V[t], probabilities[t] = calculateBelmanForT(goals.get(t),0,grid[t], grid[t+1], V[t+1], portfolios)    
    
    return goal_strategies, portfolios_strategies, probabilities
   

class InvestmentPlanner:
        
    def set_params(self, T: int, W0: float, infusion: float, infusionInterval: float, goals: dict, portfolios: np.ndarray):
        self.iMax = T*40        

        infusions = np.full(T+1,infusion)   
        self.k_dict = convert_goals_to_k(goals)
         
        self.grid = generateGrid(W0, T, self.iMax, infusions, self.k_dict, portfolios[0,0], portfolios[0,1], portfolios[-1,0], portfolios[-1,1])

        self._portfolio_strategies = np.zeros((T,self.iMax))
        self._goal_strategies = np.zeros((T+1,self.iMax))
        self.probabilitiesT = np.zeros((T,self.iMax, self.iMax))
        
        self._goal_strategies, self._portfolio_strategies, self.probabilitiesT = calculateBelman(self.grid, self.k_dict, portfolios)
                        
        self.cum_propabilities =  _calculate_cumulative_propabilities(self.probabilitiesT)
             
        
    @property    
    def glide_paths(self):
        return self._portfolio_strategies.T
    

    def get_goal_propabilities(self):
        t_goals = self.k_dict.keys()
        goal_propabilities = np.zeros((t,1))
        for t in t_goals:
            goal_propabilities[t] = 1
        return goal_propabilities


    
    
   

    

   
    