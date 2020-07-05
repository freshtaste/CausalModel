import numpy as np
from model import Model
from Learning import LearningModel


class PotentialOutcome(Model):
    
    
    def __init__(self, Y, Z, X):
        
        self.data = POdata(Y, Z, X)
        self.propensity = None
        self.result = None
        
        
    def estimate(self):
        pass
    
    
    def est_via_ipw(self, learning_model):
        prop_model = LearningModel(learning_model)
        self.propensity = prop_model.insample_predict()
        estimates = 1/self.data.n*(np.sum(self.data.Y[self.data.Z == 1])
                                  -np.sum(self.data.Y[self.data.Z == 0]))
        result = None
        return result
    
    
class POdata(object):
    
    
    def __init__(self, Y, Z, X):
        self.Y = Y
        self.Z = Z
        self.X = X
        if self.verify_data():
            self.n = self.get_n()
            self.Yc = self.get_Yc()
            self.Yt = self.get_Yt()
    
    
    def get_n(self):
        return len(self.Y)
    
    
    def get_Yc(self):
        return self.Y[self.Z == 0]
    
    
    def get_Yt(self):
        return self.Y[self.Z == 1]
    
    
    def verify_data(self):
        if not len(self.Y) == len(self.Z) == len(self.X):
            return False
        if (type(self.Y) is not np.ndarray):
            return False
        if (type(self.Z) is not np.ndarray):
            return False
        if (type(self.X) is not np.ndarray):
            return False
        return True