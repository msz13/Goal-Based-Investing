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


class RegimeSwitching(stochastic_process.StochasticProcess):

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
    
    @classmethod
    def example(cls) -> "RegimeSwitching":
        return cls(mu=[0.05, 0.09], sigma=[0.02, 0.15], probs=[0.70, 0.30])
