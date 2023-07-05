import numpy as np
from scipy.stats import norm

def __prob2(W0, W1, mean, std, h):
    return norm.pdf((np.log(W1/W0)-(mean-0.5*std**2)*h)/(std*np.sqrt(h)))

def __prob(W0, W1, mean, std, Infusion, Cost, h):
    return norm.pdf((np.log(W1/(W0+Infusion+Cost))-(mean-0.5*std**2)*h)/(std*np.sqrt(h)))


def calculateTransitionPropabilitiesForAllPorfolios(portfolios,WT,WT1,infusions, h=1):
    l = len(portfolios)
    i = len(WT)
    portfolios = np.array(portfolios)
    b = (portfolios[:,0]-0.5*portfolios[:,1]**2)*h
    bi = np.repeat(b, i).reshape(l*i,1)
    c = portfolios[:,1]*np.sqrt(h)
    ci = np.repeat(c, i).reshape(l*i,1)
   
    Wt1 = np.tile(WT1, (l*i,1))
    Wt = np.tile(WT,(l,1)).reshape(i*l,1)+infusions
    #Wt1 = np.tile(WT1, (l*i,1))+infusions
    propabilities = norm.pdf((np.log(Wt1/Wt)- bi)/ci).reshape(l,i,len(WT1))
    result = np.zeros_like(propabilities)
    sums = np.expand_dims(propabilities.sum(2), axis=2)      
    np.divide(propabilities, sums, out=result, where=sums>0)
    #result = propabilities    
    return result



#@jit(nopython=True)
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

    _WTc = np.where(WTc > 0, WTc, np.nan).reshape((k*i0,1))
    Wt1k = np.tile(Wt1, (k*i0,1))

    result = (np.log(Wt1k/_WTc)-b)/c
    result = norm.pdf(result)
    result = np.divide(result,np.expand_dims(result.sum(1), axis=1)) 
    result = result.reshape(k,i0,i1)  
    
    return result


def select_probabilities_for_chosen_strategies(probabilities,portfolios_strategies, goal_strategies, goal_costs, grid):
    i = portfolios_strategies.shape[0]
    selected_probabilities = np.zeros((i,i))


    for it in range(i):
        probs = 0
        if goal_strategies[it]== 0:
            s = int(portfolios_strategies[it])
            probs =  probabilities[s,it]                    
        else:
            c = grid[it] - goal_costs[goal_strategies[it]-1]
            probs = np.zeros(i)
            for itc in range(i):
                probs[itc] = np.interp(c,grid,selected_probabilities[:,itc])                
        selected_probabilities[it] = probs


    return selected_probabilities


def calculate_cumulative_propabilities(probabilities, goals_strategies, W0index):
        i = probabilities[0].shape[1]
        T = probabilities.shape[0] 
                
        result = {}

        cumulativeProbabilities = np.zeros((T+1, i))
        cumulativeProbabilities[0,W0index] = 1

        for t in range(1,T+1):
            for it in range(i):
                cumulativeProbabilities[t,it] = cumulativeProbabilities[t-1] @ probabilities[t-1,:,it]
        
        for t in range(1,T):
            goal_ids = np.unique(goals_strategies[t])
            goals_probs = {}
            for goal in goal_ids:
                goals_probs[goal] = np.round(np.sum(cumulativeProbabilities[t],where=goals_strategies[t]==goal),3)
            result[t] = goals_probs
        
        return result


