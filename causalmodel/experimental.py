import numpy as np
from potentialoutcome import PotentialOutcome
from result import Result


class Experimental(PotentialOutcome):
    
    def __init__(self, Y, Z, X=None):
        if X is None:
            X = np.ones(Y.shape)
        super(self.__class__, self).__init__(Y,Z,X)
        
    
    def est_via_dm(self, se_type="Neyman"):
        ate = difference_in_mean(self.data)
        if se_type == "Neyman":
            pass
        elif se_type == "Fisher":
            pass
        else:
            raise TypeError("se_type is either Neyman or Fisher.")
    
    
    def est_via_strata(self):
        pass
    
    
    def est_via_ancova(self):
        pass
    
    
    def fisher(self, test_stats, assignment, n=1000):
        ate = test_stats(self.data.Z)
        for i in range(n):
            Z_s = assignment()
    
    
    @staticmethod
    def ReM(X):
        pass
    
    

def difference_in_mean(data):
    return np.mean(data.Yt) - np.mean(data.Yc)