import numpy as np
from .base import DesignBase, get_balance

class CRD(DesignBase):
    
    
    def __init__(self, treated_ratio=0.5, covariate=None, balance=False, 
                 eps=0.1, max_iter=1000):
        super(self.__class__, self).__init__(treated_ratio, covariate, 
                                    balance, eps, max_iter)
        self.template = None
    
    
    def get_params_via_obs(self, Z):
        self.params = np.mean(Z)
        
    
    def draw(self, n):
        self._get_template(n)
        if self.balance:
            if self.covariate is None:
                raise RuntimeError("covariate must be provided if balance is True.")
            else:
                Z = self._draw_via_balance(n)
        else:
            Z = self._draw(n)
        return Z
    
    
    def _get_template(self, n):
        nt = int(self.params*n)
        self.template = np.zeros(n)
        self.template[:nt] = 1
        
    
    def _draw(self, n):
        Z = np.random.permutation(self.template)
        return Z
    
    
    def _draw_via_balance(self, n):
        balance = 1
        count = 0
        while balance > self.eps and count < self.max_iter:
            Z = self._draw(n)
            balance = get_balance(Z, self.covariate)
            count += 1
        return Z