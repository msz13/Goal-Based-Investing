from pyesg import stochastic_process
from pyesg.utils import Array
from scipy import stats
import numpy as np
from typing import Dict

class IndependentLogNormal(stochastic_process.StochasticProcess):
     
    def __init__(self, mu: float, sigma: float) -> None:
         super().__init__()
         self.mu = mu
         self.sigma = sigma

    def coefs(self) -> Dict[str, float]:
        return dict(mu=self.mu, sigma=self.sigma)
    
    def _apply(self, x0: np.ndarray, dx: np.ndarray) -> np.ndarray:
        return x0 * np.exp(dx)

    def _drift(self, x0: np.ndarray) -> np.ndarray:       
        return np.full_like(x0, self.mu, dtype=np.float64)

    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        return np.full_like(x0, self.sigma, dtype=np.float64)

    @classmethod
    def example(cls) -> "IndependentLogNormal":
        return cls(mu=0.05, sigma=0.2)


class RegimeSwitching():

    def __init__(self, mu: np.ndarray, sigma: np.ndarray, probs: np. ndarray) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.probs = probs
    
    def coefs(self) -> Dict[str, float]:
        return dict(mu=self.mu, sigma=self.sigma, probs = self.probs)

    def _apply(self, x0: np.ndarray, dx: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def _drift(self, x0: Array) -> np.ndarray:
        raise NotImplementedError()
    
    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def step(self,x0,dt,previous_state, random_state):

        return super().step(x0,dt,random_state)
    
    def next_regime(self, current_regime):
        #np.random.seed(seed)

        next_regime = None
        random = self.random()
        probs = self.probs[current_regime]
        upper_bound = np.array([[probs]])
        upper_bound = np.insert(upper_bound,-1,[1])
        lower_bound = np.array([0])
        lower_bound = np.insert(lower_bound,0,[0])

        """ if current_regime == 0:
            next_regime = 0 if random <= probs else 1
        
        else:
            next_regime = 1 if random <= probs else 0 """
                
        next_regime = np.where(np.all([lower_bound < random, random<= upper_bound]))
        return next_regime

    def random(self):
        return np.random.uniform(0,1.01)
    
    @classmethod
    def example(cls) -> "RegimeSwitching":
        return cls(mu=[0.05, 0.09], sigma=[0.02, 0.15], probs=[0.65, 0.70])
