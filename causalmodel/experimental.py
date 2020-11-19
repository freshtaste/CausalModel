import numpy as np
from potentialoutcome import PotentialOutcome
import statsmodels.api as sm 


class Experimental(PotentialOutcome):
    
    def __init__(self, Y, Z, X=None, design="CRD"):
        if X is None:
            X = np.ones(Y.shape)
        super(self.__class__, self).__init__(Y,Z,X)
        self.design = self.load_design(design)
        self.stats = None
        self.cal_stats = None
        
        
    def estimate(self):
        return self.est_via_dm()
    
    
    def est_via_dm(self):
        self.cal_stats = lambda Z: np.mean(self.data.Y[Z==1]) - np.mean(self.data.Y[Z==0])
        self.stats = self.cal_stats(self.data.Z)
        ate, se = self.cal_dm(self.data.Z, self.data.Y)
        return self._get_results(ate, se)
        
    
    def est_via_strata(self, strata):
        if len(strata) != self.data.n:
            raise RuntimeError("input doesn't have the same length as the data.")
        if type(strata) != np.ndarray:
            raise TypeError("input must be numpy array.")
        ate_list = list()
        se_list = list()
        for l in set(strata):
            w = np.mean(strata==l)
            ate_s, se_s = self.cal_dm(self.data.Z[strata==l], self.data.Y[strata==l])
            ate_list.append(ate_s*w)
            se_list.append((se_s**2)*(w**2))
        ate = np.sum(ate_list)
        se = np.sqrt(np.sum(se_list))
        return self._get_results(ate, se)

    
    def est_via_ancova(self):
        Z = self.data.Z.reshape(-1,1)
        regressor = np.concatenate((np.ones((self.data.n,1)), Z, 
                              self.data.X, self.data.X * Z), axis=1)
        ols = sm.OLS(self.data.Y, regressor).fit()
        ate = ols.params[1]
        se = ols.HC0_se[1]
        return self._get_results(ate, se)
    
    
    def test_via_fisher(self, n=1000):
        T_s = np.zeros(n)
        for i in range(n):
            Z_s = self.design()
            T_s[i] = self.cal_stats(Z_s)
        pval = min(np.mean(T_s > self.stats), np.mean(T_s < self.stats))
        return pval
    
    
    def cal_dm(self, Z, Y):
        ate = np.mean(Y[Z==1]) - np.mean(Y[Z==0])
        v1 = np.var(Y[Z==1])
        v2 = np.var(Y[Z==0])
        se = np.sqrt(v1/np.sum(Z==1)+ v2/np.sum(Z==0))
        return ate, se
            
    
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
    
    
    
def rerandomization(X, design, criteria, threshold, max_iter=1000):
    Z = design()
    while criteria(Z, X) > threshold:
        Z = design()
    return Z
