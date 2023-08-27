from pyesg import stochastic_process
from scipy import stats

class RegimeSwithingWienerProcess(stochastic_process.StochasticProcess):

    def __init__(self, dim: int = 1, mu, sig, p ) -> None:
        super().__init__(dim, dW= stats.norm)


    def step(self,x0,dt,previous_state, random_state):

        return super().step(x0,dt,random_state)
