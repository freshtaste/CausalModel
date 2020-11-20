import numpy as np
from causalmodel.potentialoutcome import POdata


class DesignBase(object):
    
    
    def __init__(self, params, covariate=None, balance=False):
        self.params = params
        self.X = covariate
        self.balance = balance
    
    def est_via_obs(self, Z):
        pass
    
    
    def draw(self, n):
        pass
    
    
def get_balance(Z, X):
    n = len(Z)
    data = POdata(np.zeros(n), Z, X)
    m1 = np.mean(data.Xt,axis=0)
    m0 = np.mean(data.Xc,axis=0)
    cov = np.cov(X.T)
    m = (data.nt*data.nc/n)*(m1-m0).dot(cov).dot(m1-m0)
    return m