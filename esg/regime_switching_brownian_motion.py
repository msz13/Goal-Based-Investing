from pyesg import stochastic_process, utils
from pyesg.utils import Array
from scipy import stats
import numpy as np
from typing import Dict


from independent_lognormal_model import IndependentLogNormal

#TODO uzyc funkcji to array dla zmiennych, dodac random state


class RegimeSwitching():
    """
    models: stochastic process models
    probs: probability regime swtching, m x n matrix, 
        m - number of regimes, 
        n - number of regimes -1 
    """

    def __init__(self, models, probs: np. ndarray) -> None:
        super().__init__()
        self.models = models        
        self.probs = probs

    @property
    def regimes_idx(self):
        return np.arange(len(self.probs))
    
        
    def next_regime(self, current_regimes):
        
        regimes = np.unique(current_regimes)
        current_regimes = np.atleast_1d(current_regimes)
        n_scenarios = len(current_regimes)
        next_regimes = np.zeros(n_scenarios, dtype=np.int64)

        for r in regimes:
            idx = np.argwhere(current_regimes == r).flatten()
            random = self.random_regimes(n_scenarios=len(idx),probs=self.probs[r])
            next_regimes[idx] = random            
        
        return next_regimes 
    
    def step(self, x0, current_regimes, dt=1):

        current_regimes = np.atleast_1d(current_regimes)
        x0 = np.atleast_1d(x0)

        next_regimes = self.next_regime(current_regimes=current_regimes)

        result = np.zeros(len(current_regimes))
        
        for model in np.unique(next_regimes):
            idx = np.argwhere(next_regimes == model).flatten()
            x = x0[idx]
            result[idx] = self.models[model].step(x,dt)      
       
        return result
    
    def scenarios_regimes(self, current_regime, n_steps):

        current_regime = np.atleast_1d(current_regime)
        n_scenarios = len(current_regime)
        regimes = np.zeros((n_steps+1, n_scenarios, ),dtype=np.int32)
        regimes[0] = current_regime

        for r in range(1, n_steps+1):
            regimes[r] = self.next_regime(regimes[r-1])
        
        return regimes
    
    def scenarios(self, initial_values, current_regimes, dt, n_steps, n_scenarios):
        
        if (np.isscalar(current_regimes)):
            current_regimes = np.full(n_scenarios,current_regimes)

        if (isinstance(initial_values, int)):
            initial_values = np.full(n_scenarios,initial_values)

        regimes = self.scenarios_regimes(current_regimes,n_steps)

        result = np.zeros((n_steps+1, n_scenarios))
        result[0] = initial_values

        for step in range(1, n_steps+1):
            result[step] = self.step(result[step-1],regimes[step-1],dt)

        return result.T

    def random_regimes(self, n_scenarios,probs):
        return np.random.choice(self.regimes_idx,size=n_scenarios,p=probs)
    
    @classmethod
    def example(cls) -> "RegimeSwitching":
        return cls(models=[IndependentLogNormal(0.07,0.15),IndependentLogNormal(0.1,0.22)], probs=[[0.65, 0.35], [0.06, 0.94]])


