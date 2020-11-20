import numpy as np
from .base import DesignBase, get_balance

class Bernoulli(DesignBase):
    
    
    def __init__(self, treated_prob=0.5, covariate=None, balance=False):
        super(self.__class__, self).__init__(treated_prob, covariate, balance)
    
    
    def est_via_obs(self, Z):
        return np.mean(Z)
    
    
    def draw(self, n):
        pass