import numpy as np
from scipy.stats import norm

def __prob2(W0, W1, mean, std, h):
    return norm.pdf((np.log(W1/W0)-(mean-0.5*std**2)*h)/(std*np.sqrt(h)))

def __prob(W0, W1, mean, std, Infusion, Cost, h):
    return norm.pdf((np.log(W1/(W0+Infusion+Cost))-(mean-0.5*std**2)*h)/(std*np.sqrt(h)))


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
    #return  result.sum(2)    
    return np.divide(result, np.expand_dims(result.sum(2), axis=2), where=result.sum()>0) 




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


def _calculate_cumulative_propabilities(probabilitiesT):
        i = probabilitiesT[0].shape[1]
        T = probabilitiesT.shape[0]
        cumulativeProbabilities = np.zeros((T, probabilitiesT[0].shape[1]))
        cumulativeProbabilities[0,:] = 1
        cumulativeProbabilities[1] = probabilitiesT[0,0]
        for t in range(2, T):
                for it in range(0,i):                      
                        cumulativeProbabilities[t,it] = np.sum(probabilitiesT[t-1,:,it]*cumulativeProbabilities[t-1,it])
        return cumulativeProbabilities


