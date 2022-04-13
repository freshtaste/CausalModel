import warnings

import numpy as np
from statsmodels.api import OLS as LinearRegression
from .potentialoutcome import POdata
from .observational import Observational
from .LearningModels import LogisticRegression, MultiLogisticRegression, OLS


class Clustered(Observational):
    """Estimate casual effects under clustered interference setting. """
    
    def __init__(self, Y, Z, X, cluster_labels, group_labels, cluster_feature=None, n_moments=1, 
            prop_idv_model=LogisticRegression(), prop_neigh_model=MultiLogisticRegression(), 
            n_matches=10, subsampling_match=2000, categorical_Z=True):
        super(Observational, self).__init__(Y,Z,X)
        self.data = ClusterData(Y, Z, X, cluster_labels, group_labels, cluster_feature,
                                n_moments, categorical_Z)
        self.prop_idv_model = prop_idv_model
        self.prop_neigh_model = prop_neigh_model
        self.n_matches = n_matches
        self.subsampling_match = subsampling_match
        
    
    def est_via_ols(self):
        y = np.zeros(self.data.units)
        regressor = np.zeros((self.data.units, 2+self.data.covariate_dims))
        size_max = max(list(self.data.data_by_size.keys()))
        idx = 0
        for k,v in self.data.data_by_size.items():
            Y, Z, G, Xc, labels = v
            y[idx:idx+len(Y)] = Y
            regressor[idx:idx+len(Y), 0] = Z
            regressor[idx:idx+len(Y), 1] = G*Z
            regressor[idx:idx+len(Y), 2:] = Xc
            idx += len(Y)
        ols_model = LinearRegression(y, regressor)
        result = ols_model.fit()
        ret = {'beta(g)': np.zeros(size_max), 'se': np.zeros(size_max)}
        cov_HC0 = result.cov_HC0
        for g in range(size_max):
            ret['beta(g)'][g] = result.params[0] + result.params[1]*g
            test_arr = np.array([1,g])
            ret['se'][g] = np.sqrt(test_arr.dot(cov_HC0[:2,:2]).dot(test_arr))
        return ret
    
    
    def est_via_dml(self, outcome_model=OLS(), treatment_model=OLS()):
        Y = np.zeros(self.data.units)
        Xc = np.zeros((self.data.units, self.data.covariate_dims))
        Z = np.zeros(self.data.units)
        G = np.zeros(self.data.units)
        Labels = np.zeros(self.data.units)
        size_max = max(list(self.data.data_by_size.keys()))
        idx = 0
        for k,v in self.data.data_by_size.items():
            y, z, g, xc, labels = v
            Y[idx:idx+len(y)] = y
            Xc[idx:idx+len(y)] = xc
            Z[idx:idx+len(y)] = z
            G[idx:idx+len(y)] = g
            Labels[idx:idx+len(y)] = labels
            idx += len(y)
        outcome_reg = outcome_model.fit(Xc, Y)
        treatment_reg = treatment_model.fit(Xc, Z)
        y_res = Y - outcome_reg.insample_predict()
        z_res = Z - treatment_reg.insample_predict()
        data = ClusterData(y_res, z_res, np.zeros((self.data.units, self.data.X.shape[1])), 
                           Labels, self.data.cluster_feature, self.data.n_moments,
                           False)
        z_g_res = np.zeros((self.data.units, 2))
        y_res = np.zeros(self.data.units)
        idx = 0
        for k,v in data.data_by_size.items():
            y, z, g, xc, labels = v
            y_res[idx:idx+len(y)] = y
            z_g_res[idx:idx+len(y),0] = z
            z_g_res[idx:idx+len(y),1] = g*z
        ols_model = LinearRegression(y_res, z_g_res)
        result = ols_model.fit()
        ret = {'beta(g)': np.zeros(size_max), 'se': np.zeros(size_max)}
        cov_HC0 = result.cov_HC0
        for g in range(size_max):
            ret['beta(g)'][g] = result.params[0] + result.params[1]*g
            test_arr = np.array([1,g])
            ret['se'][g] = np.sqrt(test_arr.dot(cov_HC0[:2,:2]).dot(test_arr))
        return ret
            
    
    def est_via_ipw(self):
        return self._est()
    
    
    def est_via_aipw(self):
        return self._est(method='aipw')
            
        
    def _est(self, method='ipw'):
        group_structs = sorted(list(self.data.data_by_group_struct.keys()))
        group_struct_num = len(group_structs)
        total_result = {}
        for group_struct in group_structs:
            size = np.sum(group_struct)
            Mn = len(self.data.data_by_group_struct[group_struct][0])/size
            pn = Mn/self.data.M
            result = self.est_subsample(group_struct, method)
            taug = result['beta(g)']
            Vg = result['se']**2*Mn
            total_result[size] = (pn, taug, Vg)
        ret = {'beta(g)': np.zeros(group_struct_num), 'se': np.zeros(group_struct_num)}
        # FIXME: weighted sum?
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
        neigh = self.prop_neigh_model.fit(Xc, G)    # assuming Z is independent
        prop_neigh = neigh.predict_proba(Xc)
        return prop_idv, prop_neigh
    
    def encode_G(self, G, group_struct):
        weight = np.append(1, np.cumprod(np.array(group_struct[:-1]) + 1))
        return np.sum(G * weight, axis=1)

    def decode_G(self, G_encoded, group_struct):
        weight = np.append(1, np.cumprod(np.array(group_struct[:-1]) + 1))
        G_encoded = np.copy(G_encoded)
        G_rev = []
        for w in reversed(weight):
            G_rev.append(G_encoded // w)
            G_encoded %= w
        return np.vstack(tuple(reversed(G_rev))).T
    
    def est_subsample(self, group_struct, method='ipw'):
        Y, Z, G, Xc, cluster_labels, group_labels = self.data.data_by_group_struct[group_struct]
        G_encoded = self.encode_G(G, group_struct)
        prop_idv, prop_neigh = self.est_propensity(Z, G_encoded, Xc)
        G_count = np.prod(np.array(group_struct)+1)
        result = {'beta(g)': np.zeros(G_count), 'se': np.zeros(G_count)}
        linear_model = OLS()
        for g_encoded in range(G_count):
            g = self.decode_G(g_encoded, group_struct)
            # fit outcome model for aipw
            mask1 = np.all(G==g, axis=1) & (Z==1)
            mask0 = np.all(G==g, axis=1) & (Z==0)
            if not np.any(mask1) or not np.any(mask0):
                # we are left with no sample
                warnings.warn(f"Skipping g={g} due to absence of sample")
                continue

            linear_model.fit(Xc[mask1], Y[mask1])
            mu1g = linear_model.predict(Xc)
            linear_model.fit(Xc[mask0], Y[mask0])
            mu0g = linear_model.predict(Xc)
            if method == 'ipw':
                result['beta(g)'][g_encoded] = self._ipw_formula(Y, Z, G, prop_idv, prop_neigh, g, g_encoded)
            elif method == 'aipw':
                result['beta(g)'][g_encoded] = self._aipw_formula(Y, Z, G, prop_idv, prop_neigh, g, g_encoded, mu1g, mu0g)
            else:
                raise ValueError("Incorrect input of estimation method.")

            # if self.subsampling_match <= self.data.units:
            #     sub = np.random.choice(np.arange(self.data.units), 
            #                 self.subsampling_match, replace=False)
            # else:
            #     sub = slice(None)
            # Vg = self.variance_via_matching(Y[sub], Z[sub], Xc[sub], 
            #                     prop_idv[sub], prop_neigh[sub,g], size, G[sub] == g)
            # result['se'][g] = np.sqrt(Vg/(self.data.units/size))

        return result
    
    
    def _ipw_formula(self, Y, Z, G, prop_idv, prop_neigh, g, g_encoded):
        N = len(Y)
        w1 = np.all(G==g, axis=1) * Z /(prop_neigh[:,g_encoded]*prop_idv) 
        w0 = np.all(G==g, axis=1) * (1 - Z)/(prop_neigh[:,g_encoded]*(1-prop_idv))
        arr = Y * w1/(np.sum(w1)/N) - Y * w0/(np.sum(w0)/N)
        beta_g = np.mean(arr)
        return beta_g
    
    
    def _aipw_formula(self, Y, Z, G, prop_idv, prop_neigh, g, g_encoded, mu1g, mu0g):
        N = len(Y)
        w1 = np.all(G==g, axis=1) * Z /(prop_neigh[:,g_encoded]*prop_idv) 
        w0 = np.all(G==g, axis=1) * (1 - Z)/(prop_neigh[:,g_encoded]*(1-prop_idv))
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
    
    def __init__(self, Y, Z, X, cluster_labels, group_labels,
            cluster_feature, n_moments, categorical_Z=True):
        self.Y = Y
        self.Z = Z
        self.X = X
        self.cluster_labels = cluster_labels
        self.M = len(set(cluster_labels))
        self.group_labels = group_labels
        self.cluster_feature = cluster_feature
        self.n_moments = n_moments
        self.categorical_Z = categorical_Z
        self.units = len(Y)
        if self.verify_clusters():
            self.data_by_group_struct = self.split_by_group_struct()
        
    
    def split_by_group_struct(self):
        """
        Split the data into {group_struct: (Y, Z, G, Xc, labels)}
        """
        group_structs = np.zeros((np.max(self.cluster_labels)+1, np.max(self.group_labels)+1), dtype=int)
        for c, g in zip(self.cluster_labels, self.group_labels):
            group_structs[c][g] += 1

        cluster_group_structs = group_structs[self.cluster_labels]
        arg = np.lexsort(cluster_group_structs.T[::-1])
        cluster_group_structs_sorted = cluster_group_structs[arg]
        data_by_group_struct = {}
        idx1 = 0
        for idx2 in range(self.units):
            if not np.all(cluster_group_structs_sorted[idx2] == cluster_group_structs_sorted[idx1]) \
                    or idx2 == self.units:
                if idx2 == self.units - 1:
                    idx2 = self.units
                idx = arg[idx1:idx2]
                if self.cluster_feature:
                    cluster_feature_i = self.cluster_feature[idx]
                else:
                    cluster_feature_i = None

                group_struct = tuple(cluster_group_structs_sorted[idx1])
                data_by_group_struct[group_struct] = self.get_final_tuple(self.Y[idx], 
                        self.Z[idx], self.X[idx], self.cluster_labels[idx], self.group_labels[idx],
                        cluster_feature_i, group_struct)

                idx1 = idx2

        return data_by_group_struct
    
    
    def get_final_tuple(self, Y, Z, X, cluster_labels, group_labels, cluster_feature, group_struct):
        """
        Process the X and cluster_feature to obtain the full covariates matrix.
        Compute G, # of treated neighbours.
        """
        arg = np.argsort(cluster_labels)
        Y = Y[arg]
        Z = Z[arg]
        X = X[arg]
        cluster_labels = cluster_labels[arg]
        group_labels = group_labels[arg]

        nunit_per_cluster = np.sum(group_struct)
        ngroup_per_cluster = len(group_struct)
        units, k = X.shape
        clusters = units//nunit_per_cluster
        X = X.reshape(units//nunit_per_cluster, nunit_per_cluster, k)
        if self.n_moments > 0:
            mu = np.mean(X, axis=1, keepdims=True)
            f1 = np.repeat(mu, nunit_per_cluster, axis=1)
            X_centered = X - f1
            X = np.append(X, f1, axis=2)
            for i in range(1, self.n_moments):
                mi = np.mean(np.power(X_centered, i+1), axis=1, keepdims=True)
                fi = np.repeat(mi, len(X), axis=1)
                X = np.append(X, fi, axis=2)
        X = X.reshape(units, k * (self.n_moments+1))

        if cluster_feature:
            cluster_feature = cluster_feature[arg]
            X = np.append(X, cluster_feature, axis=2)

        # get G, number of treated neighbours
        ngroup_per_cluster = len(group_struct)
        Z_onehot = np.zeros((clusters, nunit_per_cluster, ngroup_per_cluster))
        cluster_labels_idx = np.cumsum(np.append(0, np.diff(cluster_labels)) > 0)
        Z_onehot[cluster_labels_idx, np.tile(np.arange(nunit_per_cluster), clusters), group_labels] = Z
        G_plus = np.sum(Z_onehot, axis=1, keepdims=True)
        G_stack = G_plus - Z_onehot
        G = G_stack.reshape(units, ngroup_per_cluster)

        if self.categorical_Z:
            G = G.astype(int)
            Z = Z.astype(int)
        self.covariate_dims = k
        return Y, Z, G, X, cluster_labels, group_labels
        
    
    def verify_clusters(self):
        if not self.verify_yzx():
            return False

        if not (isinstance(self.cluster_labels, np.ndarray)
                and isinstance(self.group_labels, np.ndarray)
                and isinstance(self.n_moments, int)):
            raise ValueError("Incorrect input type for cluster_labels, group_labels (np.ndarray) or n_moments (int).")
        if not self.cluster_labels.shape == (self.units, ):
            raise ValueError("Incorrect shape for cluster_labels. Should be `(units, )`.")
        if not self.group_labels.shape == (self.units, ):
            raise ValueError("Incorrect shape for group_labels. Should be `(units, )`.")
        if not self.n_moments >= 0:
            raise ValueError("Incorrect value for n_moments. Should be a non-negative integer.")

        if self.cluster_feature:
            if not isinstance(self.cluster_feature, np.ndarray):
                raise ValueError("Incorrect input type for cluster_feature (np.ndarray)")
            if not len(self.cluster_feature) == self.units:
                raise ValueError("Incorrect input shape for cluster_feature. Should have `units` rows.")
            if not len(self.cluster_feature.shape) == 2:
                raise ValueError("Incorrect shape for cluster_feature. Should be `(units, *)`.")

        return True
    
    
    def all_same(self, A):
        diff = A - np.mean(A,axis=0)
        if np.abs(diff).sum() == 0:
            return True
        else:
            return False
