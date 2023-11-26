from typing import Dict
import numpy as np
from pyesg import stochastic_process


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