import numpy as np
from statsmodels.api import OLS as LinearRegression
from .potentialoutcome import POdata
from .observational import Observational
from .LearningModels import LogisticRegression, MultiLogisticRegression, OLS
from collections import Counter


class Clustered(Observational):
    """Estimate casual effects under clustered interference setting. """
    
    def __init__(self, Y, Z, X, cluster_label, cluster_feature=None, n_moments=1, 
            prop_idv_model=LogisticRegression(), prop_neigh_model=MultiLogisticRegression(), 
            n_matches=10, subsampling_match=2000):
        super(Observational, self).__init__(Y,Z,X)
        self.data = ClusterData(Y, Z, X, cluster_label, cluster_feature, n_moments)
        self.prop_idv_model = prop_idv_model
        self.prop_neigh_model = prop_neigh_model
        self.n_matches = n_matches
        self.subsampling_match = subsampling_match
        
    
    def est_via_ols(self):
        y = np.zeros(self.data.n)
        regressor = np.zeros(self.data.n, 2+self.data.covariate_dims)
        idx = 0
        for k,v in self.data.data_by_size:
            Y, Z, G, Xc, labels = v
            y[idx:idx+len(Y)] = Y
            regressor[idx:idx+len(Y), 0] = Z
            regressor[idx:idx+len(Y), 1] = G*Z
            regressor[idx:idx+len(Y), 2:] = Xc
        ols_model = LinearRegression(y, regressor)
        ols_model.fit()
        
    
    def est_via_ipw(self):
        return self._est()
    
    
    def est_via_aipw(self):
        return self._est(method='aipw')
            
        
    def _est(self, method='ipw'):
        sizes = sorted(list(self.data.data_by_size.keys()))
        size_max = max(sizes)
        total_result = {}
        for size in sizes:
            Mn = len(self.data.data_by_size[size][0])/size
            pn = Mn/self.data.M
            result = self.est_subsample(size, method)
            taug = result['beta(g)']
            Vg = result['se']**2*Mn
            total_result[size] = (pn, taug, Vg)
        ret = {'beta(g)': np.zeros(size_max), 'se': np.zeros(size_max)}
        for g in range(size_max):
            key_vals = np.array([[pn, taug[g], Vg[g]] for n, (pn, taug, Vg)
                        in total_result.items() if g < n])
            w =  key_vals[:,0]/np.sum(key_vals[:,0])
            taug_n = key_vals[:,1]
            Vg_n = key_vals[:,2]
            ret['beta(g)'][g] = w.dot(taug_n)
            Vg1 = Vg_n.dot(w**2/key_vals[:,0])
            ret['se'][g] = np.sqrt(Vg1/self.data.M)
        return ret
        
    
    def est_propensity(self, Z, G, Xc):
        idv = self.prop_idv_model.fit(Xc, Z)
        prop_idv = idv.insample_proba()
        neigh = self.prop_neigh_model.fit(Xc, G)
        prop_neigh = neigh.predict_proba(Xc)
        return prop_idv, prop_neigh
    
    
    def est_subsample(self, size, method='ipw'):
        Y, Z, G, Xc, labels = self.data.data_by_size[size]
        N = len(Y)
        prop_idv, prop_neigh = self.est_propensity(Z, G, Xc)
        result = {'beta(g)': np.zeros(size), 'se': np.zeros(size)}
        linear_model = OLS()
        for g in range(size):
            # fit outcome model for aipw
            linear_model.fit(Xc[(G==g) & (Z==1)], Y[(G==g) & (Z==1)])
            mu1g = linear_model.predict(Xc)
            linear_model.fit(Xc[(G==g) & (Z==0)], Y[(G==g) & (Z==0)])
            mu0g = linear_model.predict(Xc)
            if method == 'ipw':
                result['beta(g)'][g] = self._ipw_formula(Y, Z, G, prop_idv, prop_neigh, g)
            elif method == 'aipw':
                result['beta(g)'][g] = self._aipw_formula(Y, Z, G, prop_idv, prop_neigh, g, mu1g, mu0g)
            else:
                raise RuntimeError("Incorrect input of estimation method.")
            if self.subsampling_match <= N:
                sub = np.random.choice(np.arange(N), 
                            self.subsampling_match, replace=False)
            else:
                sub = slice(None)
            Vg = self.variance_via_matching(Y[sub], Z[sub], Xc[sub], 
                                prop_idv[sub], prop_neigh[sub,g], size, G[sub] == g)
            result['se'][g] = np.sqrt(Vg/(N/size))
        return result
    
    
    def _ipw_formula(self, Y, Z, G, prop_idv, prop_neigh, g):
        N = len(Y)
        w1 = (G == g) * Z /(prop_neigh[:,g]*prop_idv) 
        w0 = (G == g) * (1 - Z)/(prop_neigh[:,g]*(1-prop_idv))
        arr = Y * w1/(np.sum(w1)/N) - Y * w0/(np.sum(w0)/N)
        beta_g = np.mean(arr)
        return beta_g
    
    
    def _aipw_formula(self, Y, Z, G, prop_idv, prop_neigh, g, mu1g, mu0g):
        N = len(Y)
        w1 = (G == g) * Z /(prop_neigh[:,g]*prop_idv) 
        w0 = (G == g) * (1 - Z)/(prop_neigh[:,g]*(1-prop_idv))
        arr = (Y - mu1g) * w1/(np.sum(w1)/N) - (Y - mu0g) * w0/(np.sum(w0)/N) + mu1g - mu0g
        beta_g = np.mean(arr)
        return beta_g
               
    
    def variance_via_matching(self, Y, Z, Xc, q, pg, size, idx_g):
        idx1 = Z == 1
        Y1_g = Y[idx1 & idx_g]
        Y0_g = Y[(~idx1) & idx_g]
        Xc1_g = Xc[idx1 & idx_g]
        Xc0_g = Xc[(~idx1) & idx_g]
        # Standardize Xc
        Xc_s = Xc/np.sqrt(np.var(Xc, axis=0))
        X1 = Xc1_g/np.sqrt(np.var(Xc1_g, axis=0))
        X0 = Xc0_g/np.sqrt(np.var(Xc0_g, axis=0))
        # Matching for all units
        match_t = self.mat_match_mat(Xc_s, X1, self.n_matches)
        match_c = self.mat_match_mat(Xc_s, X0, self.n_matches)
        beta = np.mean(Y1_g[match_t], axis=1) - np.mean(Y0_g[match_c], axis=1)
        var_t = np.var(Y1_g[match_t], axis=1)
        var_c = np.var(Y0_g[match_c], axis=1)
        # calculate Vg
        arr = var_t/(pg*q) + var_c/(pg*(1-q)) + (beta - np.mean(beta))**2
        Vg = np.mean(arr)/size
        return Vg
    

class ClusterData(POdata):
    
    def __init__(self, Y, Z, X, cluster_label, cluster_feature, n_moments):
        self.Y = Y
        self.Z = Z
        self.X = X
        self.cluster_label = cluster_label
        self.M = len(set(cluster_label))
        self.cluster_feature = cluster_feature
        self.n_moments = n_moments
        if self.verify_clusters():
            self.n = len(Y)
            self.data_by_size = self.split_by_size()
        
    
    def split_by_size(self):
        """
        Split the data into {size: (Y, Z, G, Xc, labels)}
        """
        sizes = Counter(self.cluster_label)
        cluster_size = np.array([sizes[c] for c in self.cluster_label])
        arg = np.argsort(cluster_size)
        cluster_size_sorted = cluster_size[arg]
        data_by_size = {}
        idx1 = 0
        for idx2 in range(self.n):
            if cluster_size_sorted[idx2] != cluster_size_sorted[idx1] or idx2 == self.n-1:
                if idx2 == self.n - 1:
                    idx2 = self.n
                idx = arg[idx1:idx2]
                if self.cluster_feature:
                    cluster_feature_i = self.cluster_feature[idx]
                else:
                    cluster_feature_i = None
                data_by_size[cluster_size_sorted[idx1]] = self.get_final_tuple(self.Y[idx], 
                        self.Z[idx], self.X[idx], self.cluster_label[idx],
                        cluster_feature_i, cluster_size_sorted[idx1])
                idx1 = idx2
        return data_by_size
    
    
    def get_final_tuple(self, Y, Z, X, cluster_label, cluster_feature, size):
        """
            Process the X and cluster_feature to obtain the full covariates matrix.
            Compute G, # of treated neighbours.
        """
        arg = np.argsort(cluster_label)
        Y = Y[arg]
        Z = Z[arg]
        X = X[arg]
        cluster_label = cluster_label[arg]
        X = X.reshape(int(len(Y)/size), size, X.shape[1])
        if self.n_moments > 0:
            mu = np.mean(X, axis=1)
            f1 = np.repeat(np.expand_dims(mu, axis=1), size, axis=1)
            X_centered = X - f1
            X = np.append(X, f1, axis=2)
            for i in range(1, self.n_moments):
                mi = np.mean(np.power(X_centered, self.n_moments+1), axis=1)
                fi = np.repeat(np.expand_dims(mi, axis=1), len(X),axis=1)
                X = np.append(X, fi, axis=2)
        X = X.reshape(len(Y), X.shape[2])
        if cluster_feature:
            cluster_feature = cluster_feature[arg]
            X = np.append(X, cluster_feature, axis=2)
        Z = Z.reshape(int(len(Y)/size), size)
        G = np.repeat(np.expand_dims(np.sum(Z, axis=1), axis=1), size, axis=1) - Z
        G = G.reshape(len(Y)).astype(int)
        Z = Z.reshape(len(Y)).astype(int)
        self.covariate_dims = X.shape[1]
        return (Y, Z, G, X, cluster_label)
        
    
    def verify_clusters(self):
        if not self.verify_yzx():
            return False
        if not (isinstance(self.cluster_label, np.ndarray) \
                and isinstance(self.n_moments, int)):
            raise RuntimeError("Incorrect input type for cluster_label(np.ndarray) or n_moments(int).")
            return False
        if not len(self.cluster_label) == len(self.X):
            raise RuntimeError("Incorrect input shape for cluster_label. Should have n rows.")
            return False
        if self.n_moments < 0:
            raise RuntimeError("Incorrect value for n_moments. Should be positive integers.")
            return False
            return False
        if not len(self.cluster_label.shape) == 1:
            raise RuntimeError("Incorrect shape for cluster_label. Should be (n,).")
            return False
        if self.cluster_feature:
            if isinstance(self.cluster_feature, np.ndarray):
                raise RuntimeError("Incorrect input type for cluster_feature(np.ndarray)")
                return False
            if not len(self.cluster_feature) == len(self.X):
                raise RuntimeError("Incorrect input shape for cluster_feature. Should have n rows.")
                return False
            if not len(self.cluster_feature.shape) == 2:
                raise RuntimeError("Incorrect shape for cluster_feature. Should be n by m.")
        return True
    
    
    def all_same(self, A):
        diff = A - np.mean(A,axis=0)
        if np.abs(diff).sum() == 0:
            return True
        else:
            return False
