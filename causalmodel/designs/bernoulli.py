import numpy as np
from .crd import CRD

class Bernoulli(CRD):
    
    
    def __init__(self, treated_prob=0.5, covariate=None, balance=False,
                 eps=0.1, max_iter=1000):
        super(CRD, self).__init__(treated_prob, covariate, 
                                    balance, eps, max_iter)
    
    
    def _get_template(self, n):
        self.template = None
        
    
    def _draw(self, n):
        Z = np.random.choice([0,1],n)
        return Z
