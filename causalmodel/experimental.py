import numpy as np
from potentialoutcome import PotentialOutcome


class Experimental(PotentialOutcome):
    
    def __init__(self, Y, Z, X=None, design="CRD"):
        if X is None:
            X = np.ones(Y.shape)
        super(self.__class__, self).__init__(Y,Z,X)
        self.design = self.load_design(design)
        self.stats = None
        self.cal_stats = None
        
    
    def est_via_dm(self):
        self.cal_stats = lambda Z: np.mean(self.data.Y[Z==1]) - np.mean(self.data.Y[Z==0])
        self.stats = self.cal_stats(self.data.Z)
        v1 = np.var(self.data.Yt)
        v2 = np.var(self.data.Yc)
        se = np.sqrt(v1/self.data.nt + v2/self.data.nc)
        return self._get_results(self.stats, se)
        
    
    def est_via_strata(self):
        pass
    
    
    def est_via_ancova(self):
        pass
    
    
    def test_via_fisher(self, n=1000):
        T_s = np.zeros(n)
        for i in range(n):
            Z_s = self.design()
            T_s[i] = self.cal_stats(Z_s)
        pval = min(np.mean(T_s > self.stats), np.mean(T_s < self.stats))
        return pval
            
    
    def load_design(self, design):
        if design == "CRD":
            return lambda : np.random.choice(self.data.Z, self.data.n, replace=False)
        if design == "Bernoulli":
            p1 = np.mean(self.data.Z)
            return lambda : np.random.choice([0,1], self.data.n, p=[1-p1, p1])
        if callable(design):
            if self.verify_design(design):
                return design
            else:
                raise RuntimeError("{} is not a valid assignment function.".format(design))
        else:
            raise RuntimeError("{} is not a valid assignment function.".format(design))
    
    
    def verify_design(self, design):
        for i in range(10):
            zs = design()
            if set(zs) != set(self.data.Z) or len(zs) != len(self.data.Z):
                return False
        return True
    
    
    @staticmethod
    def ReM(X):
        pass
    
    

