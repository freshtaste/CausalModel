import numpy as np
from scipy.stats import norm
from model import Model
from result import Result
import warnings
from LearningModels import LogisticRegression, OLS


class PotentialOutcome(Model):
    
    
    def __init__(self, Y, Z, X):
        
        self.data = POdata(Y, Z, X)
        self.propensity = None
        self.treated_pred = None
        self.control_pred = None
        self.result = None
        super(self.__class__, self).__init__(self.data, self.result)
        self.eps = 1e-4
    
    
    def est_propensity(self, PropensityModel):
        # Estiamte propensity score with learning model for propensity score: 
        # Z ~ X (binary classfication)
        prop_model = PropensityModel()
        prop_model.fit(self.data.X, self.data.Z)
        return prop_model.insample_proba()
    
        
    def estimate(self):
        return self.est_via_ipw(LogisticRegression)
    
    
    def est_via_ipw(self, PropensityModel, propensity=None, normalize=True):
        if propensity is not None:
            self.propensity = propensity
        else:
            self.propensity = self.est_propensity(PropensityModel)
            
        self._fix_propensity()
        # Compute Average Treatment Effect (ATE)
        w1 = self.data.Z/self.propensity
        w0 = (1-self.data.Z)/(1-self.propensity)
        if normalize:
            G = w1 * self.data.Y/(np.sum(w1)/self.data.n) - w0 * self.data.Y/(np.sum(w0)/self.data.n)
        else:
            G = w1 * self.data.Y - w0 * self.data.Y 
        
        ate = np.mean(G)
        se = np.sqrt(np.var(G) / (len(G)-1))
        return self._get_results(ate, se)
    
    
    def est_via_aipw(self, OutcomeModel, PropensityModel, treated_pred=None, 
                     control_pred=None, propensity=None):
        # compute conditional mean and propensity score
        if treated_pred is not None:
            self.treated_pred = treated_pred
        else:
            treated_model = OutcomeModel()
            treated_model.fit(self.data.Xt, self.data.Yt)
            self.treated_pred = treated_model.predict(self.data.X)
            
        if control_pred is not None:
            self.control_pred = control_pred
        else:
            control_model = OutcomeModel()
            control_model.fit(self.data.Xc, self.data.Yc)
            self.control_pred = control_model.predict(self.data.X)
            
        if propensity is not None:
            self.propensity = propensity
        else:
            self.propensity = self.est_propensity(PropensityModel)
            
        self._fix_propensity()
        # Compute Average Treatment Effect (ATE)
        G = (self.treated_pred - self.control_pred 
             + self.data.Z * (self.data.Y - self.treated_pred)/ self.propensity 
             - (1 - self.data.Z) * (self.data.Y - self.control_pred)/ (1-self.propensity))
        
        ate = np.mean(G)
        se = np.sqrt(np.var(G) / (len(G)-1))
        return self._get_results(ate, se)
    
    
    def est_via_matching(self, num_matches, num_matches_for_var, bias_adj=False):
        
        Xt, Yt, Xc, Yc = self.data.Xt, self.data.Yt, self.data.Xc, self.data.Yc
        nt, nc, n = self.data.nt, self.data.nc, self.data.n
        M, J = num_matches, num_matches_for_var

        # standardizing the covariates and construct the matrix of all differences
        sd_Xt, sd_Xc = np.sqrt(np.var(Xt, axis=0)), np.sqrt(np.var(Xc, axis=0))
        Xt_scaled, Xc_scaled = Xt/sd_Xt, Xc/sd_Xc
        X_diff = Xt_scaled[:,np.newaxis] - Xc_scaled

        # compute Nt by Nc matrix of distances and find top m indices
        X_dist = np.sum(X_diff**2, axis=2)
        match_for_t, match_for_c = np.argpartition(X_dist, M)[:,:M], \
            np.argpartition(X_dist.T, M)[:,:M]
        Yhat_c, Yhat_t = np.mean(Yt[match_for_c],axis=1), np.mean(Yc[match_for_t],axis=1)
        ITT_t, ITT_c = Yt - Yhat_t, Yhat_c - Yc
        Yhat1, Yhat0 = np.append(Yt, Yhat_c), np.append(Yhat_t, Yc)
        
        atc, att = ITT_c.mean(), ITT_t.mean()
        ate = (nc/n)*atc + (nt/n)*att

        if bias_adj:
            mu0 = OLS().fit(Xc, Yc)
            mu1 = OLS().fit(Xt, Yt)
            mu0_t, mu0_c = mu0.predict(Xt), mu0.predict(Xc)
            mu1_t, mu1_c = mu1.predict(Xt), mu1.predict(Xc)
            match_for_0t = np.mean(mu0_c[match_for_t],axis=1)
            match_for_1c = np.mean(mu1_t[match_for_c],axis=1)
            BM = np.sum(mu0_t - match_for_0t)/n - np.sum(mu1_c - match_for_1c)/n
            ate -= BM
        
        # estimate variance
        Km = np.zeros(n)
        for row in match_for_c:
            Km[row] += 1
        for row in match_for_t:
            Km[row+nt] += 1
        
        # 1. match treated to treated, control to control
        X_diff_t = Xt_scaled[:,np.newaxis] - Xt_scaled
        X_diff_c = Xc_scaled[:,np.newaxis] - Xc_scaled
        X_dist_t = np.sum(X_diff_t**2, axis=2)
        X_dist_c = np.sum(X_diff_c**2, axis=2)
        match_tt, match_cc = np.argpartition(X_dist_t, J)[:,:J], \
            np.argpartition(X_dist_c, J)[:,:J]
        Yhat_cc, Yhat_tt = np.mean(Yc[match_cc],axis=1), np.mean(Yt[match_tt],axis=1)
        Y, Y_close = np.append(Yt, Yc), np.append(Yhat_tt, Yhat_cc)
        # 2. estimate conditional variance
        sigmaXW = J/(J+1)*((Y - Y_close)**2)
        # 3. compute variance
        V1 = (Yhat1 - Yhat0 - ate)**2
        V2 = ((Km/M)**2 + (2*M-1)/M * Km/M)*sigmaXW
        V = np.mean(V1 + V2)
        
        se = np.sqrt(V/n)
        return self._get_results(ate, se)
        
    
    
    def _fix_propensity(self):
        if self.propensity is not None:
            num_bad_prop = np.sum((self.propensity*(1-self.propensity)) == 0)
            if num_bad_prop > 0:
                self.propensity[self.propensity == 0] += self.eps
                self.propensity[self.propensity == 1] -= self.eps
                warnings.warn("Propensity scores has {} number of 0s or 1s."
                              .format(num_bad_prop))
    
    
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