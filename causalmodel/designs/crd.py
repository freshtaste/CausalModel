import numpy as np
from .base import DesignBase, get_balance

class CRD(DesignBase):
    
    
    def __init__(self, treated_ratio=0.5, covariate=None, balance=False):
        super(self.__class__, self).__init__(treated_ratio, covariate, balance)
    
    
    def est_via_obs(self, Z):
        return np.mean(Z)
    
    
    def draw(self, n):
        pass
