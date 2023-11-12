from scipy.stats import invgamma, norm
import numpy as np

class GibsSamplerGBM():
    def __init__(self, returns, init_sigma, n_iter=10000) -> None:
        self.returns = returns
        self.returns_mu = returns.mean()
        self.T = len(self.returns)
        self.n_iter = n_iter
        self.mu_dist = np.zeros(self.n_iter+1)
        self.sigma_dist = np.zeros(self.n_iter+1)
        self.sigma_dist[0] = init_sigma
         

    def run(self):
        for i in range(1,self.n_iter):
            self.mu_dist[i] = self.__normal__(self.sigma_dist[i-1]**2)
            self.sigma_dist[i] = self.__inv_gamma__(self.mu_dist[i])
            
    def __normal__(self,  sigma2):
        return norm.rvs(loc = self.returns_mu, scale = sigma2/self.T,size=1)
    
    def __inv_gamma__(self, mu):
        B = (((self.returns - mu)**2).sum())*0.5
        a = (self.T/2+1)
        mean = B/(a-1)
        return invgamma.rvs(a, loc=mean,scale=B,size=1)**0.5 
