import numpy as np
from scipy.stats import norm
from model import Model
from result import Result
import warnings


class PotentialOutcome(Model):
    
    
    def __init__(self, Y, Z, X):
        
        self.data = POdata(Y, Z, X)
        self.propensity = None
        self.result = {'Average Treatment Effect': None,
                       'Standard Error': None,
                       'z': None,
                       'p-value': None,
                       '95% Confidence Interval': None
                       }
        super(self.__class__, self).__init__(self.data, self.result)
        self.eps = 1e-4
        
        
    def estimate(self):
        pass
    
    
    def est_via_ipw(self, LearningModel, propensity=None):
        # Parse in learning model for propensity score: Z ~ X (binary classfication)
        prop_model = LearningModel()
        prop_model.fit(self.data.X, self.data.Z)
        if propensity:
            self.propensity = propensity
        else:
            self.propensity = prop_model.insample_proba()
        # Compute Average Treatment Effect (ATE)
        #ate = 1/self.data.n*(np.sum(self.Yt/self.propensity[self.data.idx_t])
        #                    -np.sum(self.Yc/self.propensity[self.data.idx_c]))
        num_bad_prop = np.sum((self.propensity*(1-self.propensity)) == 0)
        if num_bad_prop > 0:
            self.propensity[self.propensity == 0] += self.eps
            self.propensity[self.propensity == 1] -= self.eps
            warnings.warn("Propensity scores has {} number of 0s or 1s.".format(
                num_bad_prop))
        G =  ((self.data.Z - self.propensity) * self.data.Y) / (self.propensity*(1-self.propensity))
        self.result['Average Treatment Effect'] = np.mean(G)
        # Compute Standard Error and etc
        self.result['Standard Error'] = np.sqrt(np.var(G) / (len(G)-1))
        self.result['z'] = self.result['Average Treatment Effect']/self.result['Standard Error']
        self.result['p-value'] = (1 - norm.cdf(self.result['z']))*2
        self.result['95% Confidence Interval'] = (self.result['Average Treatment Effect'] - 
                   1.96 * self.result['Standard Error'], self.result['Average Treatment Effect'] +
                   1.96 * self.result['Standard Error'])
        return self.result
    
    
class POdata(object):
    
    
    def __init__(self, Y, Z, X):
        self.Y = Y
        self.Z = Z
        self.X = X
        if self.verify_data():
            self.n = self.get_n()
            self.idx_t = self.Z == 1
            self.idx_c = self.Z == 0
            self.Yc = self.get_Yc()
            self.Yt = self.get_Yt()
        else:
            import logging
            logging.error("The data provided should be ndarray of the same length")
    
    def get_n(self):
        return len(self.Y)
    
    
    def get_Yc(self):
        return self.Y[self.idx_c]
    
    
    def get_Yt(self):
        return self.Y[self.idx_t]
    
    
    def verify_data(self):
        if not (isinstance(self.X, np.ndarray) \
                and isinstance(self.Y, np.ndarray) \
                and isinstance(self.Z, np.ndarray)):
            return False
        if not len(self.Y) == len(self.Z) == len(self.X):
            return False
        return True
