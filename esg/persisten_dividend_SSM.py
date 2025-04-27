import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.tools import (
    constrain_stationary_univariate, unconstrain_stationary_univariate, constrain_stationary_multivariate)



class PersistentDividendModel(sm.tsa.statespace.MLEModel):
    def __init__(self, endog, k_states, k_posdef):
        """
        Parameters:
        - endog: array-like, shape (n_obs, n_series)
        - k_states: number of states
        - k_posdef: dimension of the state innovation vector
        """

        measurement_var = np.var(endog, axis=0) 
        state_var = np.array([4900, 4900, 4900, measurement_var[3], measurement_var[2], measurement_var[4], measurement_var[0], measurement_var[1], 0, 0, 0, 0, 0])

        # Initialize the base class (MLEModel)
        super().__init__(endog, k_states=k_states, k_posdef=k_posdef, initialization='known', initial_state=np.zeros(13), initial_state_cov=np.diag(state_var))

        self.nobs, self.n_series = self.endog.shape

        # Design matrix Z: shape (n_series, k_states)
        self['design'] = self.buildH(40, 0.967, np.full(5, .5),np.full(5, .5))

        # Transition matrix T: shape (k_states, k_states)
        self['transition'] = self.buildF(np.full(5, .5), np.full(5, .5))

        # Selection matrix R: shape (k_states, k_posdef)
        self['selection'] = np.ones((k_states, k_posdef))

        # State covariance matrix Q: shape (k_posdef, k_posdef)
        self['state_cov'] = self.buildQ(np.ones(36))

        # Observation covariance matrix H: shape (n_series, n_series)
        self['obs_cov'] = np.eye(self.n_series)

        self.coef_params = slice(0,10)
        self.obs_cov_params = slice(10,15)
        self.state_cov_params = slice(15,None)


    @property
    def start_params(self):
        """
        Initial guess of parameters.
        """
        ar = np.full(10, .5)
        measurement_var = np.var(self.endog, axis=0) * 0.1
        state_var = np.array([measurement_var[3], measurement_var[2], measurement_var[4], measurement_var[3], measurement_var[2], measurement_var[4], measurement_var[0], measurement_var[1]]) * 9
        chol_state_cov = np.linalg.cholesky(np.diag(state_var))

        return np.concatenate((ar, measurement_var, chol_state_cov[np.tril_indices(8)]))


    def update(self, params, **kwargs):
        """
        Update the parameters in the state space matrices given `params`.
        """

        params = np.array(params)
        # Design matrix Z: shape (n_series, k_states)
        self['design'] = self.buildH(40, 0.967, params[0:5] , params[5:10])

        # Transition matrix T: shape (k_states, k_states)
        self['transition'] = self.buildF(params[0:5] , params[5:10])

        # Observation covariance matrix H: shape (n_series, n_series)
        self['obs_cov'] = np.diag(params[10:15])

        # State covariance matrix Q: shape (k_posdef, k_posdef)
        self['state_cov'] = self.buildQ(params[self.state_cov_params])

   

    def transform_params(self, unconstrained):
        """
        Map unconstrained parameters to constrained space.
        E.g., variances must be positive.
        """
        constrained = unconstrained.copy() 
        constrained[0:10] = self.constrain_box(unconstrained[0:10], -.9999, .9999)
        #constrained[self.obs_cov_params] = constrained[self.obs_cov_params] ** 2


        return constrained

    def untransform_params(self, constrained):
        """
        Map constrained parameters back to unconstrained space.
        """
        unconstrained = constrained.copy()
        unconstrained[0:10] = self.unconstrain_box(constrained[0:10], -.9999, .9999)
        #unconstrained[10:] = unconstrained[10:] ** 0.5

        return unconstrained
        
    @property
    def param_names(self):
        """
        Names for each parameter for reference.
        """
        
        state_names = ["dp", "rp", "πp", "da", "ra", "πa", "ea", "τa"]
        observations_names = ["p", "il", "is", "d", "π"]

        # Equivalent of Julia's state_names[::4] (i.e., every 4th element starting at index 0)
        ar_names = [f"θ{lag}_{v}" for lag in range(1, 3) for v in state_names[3:]]
        measurement_sigmas = [f"σ_{n}" for n in observations_names]
        states_sigmas = self._generate_cov_names(state_names)

        return ar_names + measurement_sigmas + states_sigmas
    
    def _generate_cov_names(self, var_names):
        """
        Generate names for the lower triangular elements of the covariance matrix.

        Parameters:
        - var_names: list of str, variable names

        Returns:
        - list of str, names for lower triangular entries
        """
        n = len(var_names)
        names = []
        for i in range(n):
            for j in range(i + 1):
                if i == j:
                    names.append(f"{var_names[i]}")
                else:
                    names.append(f"{var_names[i]}_{var_names[j]}")
        return names
    
    def constrain_box(self, unconstrained, lb, ub):
         
        constrained = lb + (ub - lb) / (1 + np.exp(-unconstrained))

        return constrained
    
    def unconstrain_box(self, constrained, lb, ub):
         
        unconstrained = -np.log((ub - lb) / (constrained - lb) - 1)

        return unconstrained

    
    def buildH(self, n, rho, theta1, theta2):
        Hp = np.array([
            1 / (1 - rho),
            -1 / (1 - rho),
            0,
            theta1[0] / (1 - rho * theta1[0]),
            -theta1[1] / (1 - rho * theta1[1]),
            0,
            -theta1[3] / (1 - rho * theta1[3]),
            0,
            theta2[0] / (1 - rho * theta2[0]),
            -theta2[1] / (1 - rho * theta2[1]),
            0,
            -theta2[3] / (1 - rho * theta2[3]),
            0
        ])

        Hin = np.array([
            0,
            1,
            1,
            0,
            (1 / n) * ((1 - theta1[1]**n) / (1 - theta1[1])) * theta1[1],
            (1 / n) * ((1 - theta1[2]**n) / (1 - theta1[2])) * theta1[2],
            0,
            (1 / n) * ((1 - theta1[4]**(n - 1)) / (1 - theta1[4])) * theta1[4],
            0,
            (1 / n) * ((1 - theta2[1]**n) / (1 - theta2[1])) * theta2[1],
            (1 / n) * ((1 - theta2[2]**n) / (1 - theta2[2])) * theta2[2],
            0,
            (1 / n) * ((1 - theta2[4]**(n - 1)) / (1 - theta2[4])) * theta2[4]
        ])

        Hi = np.array([
            0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0
        ])

        Hd = np.array([
            1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0
        ])

        Hpi = np.array([
            0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
        ])

        H = np.vstack([Hp, Hin, Hi, Hd, Hpi])
        return H
    
    def buildF(self, theta1, theta2):
        I3 = np.eye(3)
        I5 = np.eye(5)
        Z35 = np.zeros((3, 5))
        Z53 = np.zeros((5, 3))
        Z55 = np.zeros((5, 5))
        D1 = np.diag(theta1)
        D2 = np.diag(theta2)

        top = np.hstack((I3, Z35, Z35))
        middle = np.hstack((Z53, D1, D2))
        bottom = np.hstack((Z53, I5, Z55))

        return np.vstack((top, middle, bottom))
    
    def buildQ(self, params):
        result = np.zeros((13, 13))
        L = np.zeros((8,8), np.float64)
        L[np.tril_indices(8)] = params 
        cov_matrix = L @ L.T
        result[:8, :8] = cov_matrix
        return result