import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.statespace.tools import (
    constrain_stationary_univariate, unconstrain_stationary_univariate)

""" 

Z = [1, 1]
states = [t, c]

F = [rhot, 0
    0 rhoc ]

"""


class ARTrendCycle(sm.tsa.statespace.MLEModel):
    def __init__(self, y_t):
        
        super(ARTrendCycle, self).__init__(
            endog=y_t, k_states=2, initialization="diffuse"
        )

        self.ssm["design"] = np.ones(2)
        self.ssm["selection"] = np.eye(self.k_states)
        self.ssm["transition"] = np.eye(self.k_states)

        self.positive_parameters = slice(3,4)
      

    @property
    def param_names(self):
        return ["trend_intercept", "rho.trend", "rho.cycle", "trend_error", "cycle.error"]

    @property
    def start_params(self):
        """
        Defines the starting values for the parameters
        The linear regression gives us reasonable starting values for the constant
        d and the variance of the epsilon error
        """
        params = np.r_[0.02, 0.2, 0.2, 0.01, 0.01]
        return params

    def transform_params(self, unconstrained):
        constrained = unconstrained.copy()
        constrained[self.positive_parameters] = constrained[self.positive_parameters]**2
        constrained[1] = constrain_stationary_univariate(constrained[1:2])
        constrained[2] = constrain_stationary_univariate(constrained[1:2])
        return constrained

    def untransform_params(self, constrained):
        unconstrained = constrained.copy()
        unconstrained[self.positive_parameters] = unconstrained[self.positive_parameters]**0.5
        unconstrained[1] = unconstrain_stationary_univariate(constrained[1:2])
        unconstrained[2] = unconstrain_stationary_univariate(constrained[1:2])
        return unconstrained

    def update(self, params, **kwargs):
        params = super(ARTrendCycle, self).update(params, **kwargs)

        self["state_intercept", 0, 0] = params[0]
        self["transition", 0,0] = params[1]
        self["transition", 1,1] = params[2]
        self["state_cov"] = np.diag(params[3:5])