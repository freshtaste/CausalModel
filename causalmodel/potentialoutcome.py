import numpy as np


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
    
    
class POdata(object):
    
    
    def __init__(self, Y, Z, X):
        self.Y = Y
        self.Z = Z
        self.X = X
        if self.verify_data():
            self.n = self.get_n()
            self.idx_t = self.Z == 1
            self.idx_c = self.Z == 0
            self.nc = np.sum(self.idx_c)
            self.nt = np.sum(self.idx_t)
            self.Yc = self.get_Yc()
            self.Yt = self.get_Yt()
            self.Xc = self.get_Xc()
            self.Xt = self.get_Xt()
        else:
            import logging
            logging.error("The data provided should be ndarray of the same length")
            
    
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
    
    
    def verify_data(self):
        if not (isinstance(self.X, np.ndarray) \
                and isinstance(self.Y, np.ndarray) \
                and isinstance(self.Z, np.ndarray)):
            return False
        if not len(self.Y) == len(self.Z) == len(self.X):
            return False
        return True
