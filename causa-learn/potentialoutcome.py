import numpy as np
from model import Model
from learning import LearningModel


class PotentialOutcome(Model):
    
    
    def __init__(self, Y, Z, X):
        
        self.data = POdata(Y, Z, X)
        self.propensity = None
        self.result = {'Average Treatment Effect': None,
                       'Standard Error': None,
                       'z': None,
                       'p-value:': None,
                       '95% Confidence Interval': None
                       }
        super(self.__class__, self).__init__(self.data)
        
        
    def estimate(self):
        pass
    
    
    def est_via_ipw(self, learning_model, propensity=None):
        # Parse in learning model for propensity score: Z ~ X (binary classfication)
        prop_model = LearningModel(learning_model)
        if propensity:
            self.propensity = propensity
        else:
            self.propensity = prop_model.insample_predict()
        # Compute Average Treatment Effect (ATE)
        ate = 1/self.data.n*(np.sum(self.Yt/self.propensity[self.data.idx_t])
                            -np.sum(self.Yc/self.propensity[self.data.idx_c]))
        # Compute Standard Error
        se = None
        # 
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
                or isinstance(self.Y, np.ndarray) \
                or isinstance(self.Z, np.ndarray)):
            return False
        if not len(self.Y) == len(self.Z) == len(self.X):
            return False
        return True
