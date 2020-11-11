import numpy as np
from result import Result
from scipy.stats import norm


class PotentialOutcome(object):
    
    
    def __init__(self, Y, Z, X):
        
        self.data = POdata(Y, Z, X)
        self.result = None
    
    
    def __repr__(self):
        """
        Return the string representation of the model object
        :return:
        """
        return "{}".format(self.__class__.__name__)
    
    
    def estimate(self):
        pass
    
    
    def _get_results(self, ate, se):
        self.result = Result(average_treatment_effect=ate,
                             standard_error=se,
                             z=ate/se,
                             p_value=((1 - norm.cdf(ate/se))*2),
                             confidence_interval=(ate - 1.96*se, ate+1.96*se))
        return self.result
    
    
class POdata(object):
    
    
    def __init__(self, Y, Z, X):
        self.Y = Y
        self.Z = Z
        self.X = X
        if self.verify_yzx():
            self.n = self.get_n()
            self.idx_t = self.Z == 1
            self.idx_c = self.Z == 0
            self.nc = np.sum(self.idx_c)
            self.nt = np.sum(self.idx_t)
            self.Yc = self.get_Yc()
            self.Yt = self.get_Yt()
            self.Xc = self.get_Xc()
            self.Xt = self.get_Xt()
            
    
    def get_n(self):
        return len(self.Y)
    
    
    def get_Yc(self):
        return self.Y[self.idx_c]
    
    
    def get_Yt(self):
        return self.Y[self.idx_t]
    
    
    def get_Xc(self):
        return self.X[self.idx_c]
    
    
    def get_Xt(self):
        return self.X[self.idx_t]
    
    
    def verify_yzx(self):
        if not (isinstance(self.X, np.ndarray) \
                and isinstance(self.Y, np.ndarray) \
                and isinstance(self.Z, np.ndarray)):
            raise RuntimeError("Incorrect input type for Y, Z or X. Should be np.ndarray.")
            return False
        if not len(self.Y) == len(self.Z) == len(self.X):
            raise RuntimeError("Incorrect input shape for Y, Z or X. Should have n rows.")
            return False
        if not len(self.Y.shape) == len(self.Z.shape) == 1:
            raise RuntimeError("Incorrect input shape for Y or Z. Should be (n,).")
            return False
        if not len(self.X.shape) == 2:
            raise RuntimeError("Incorrect input shape for X. Should be n by k.")
            return False
        return True
