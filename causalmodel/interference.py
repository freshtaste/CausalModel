import numpy as np
from .potentialoutcome import POdata
from .observational import Observational
from .LearningModels import LogisticRegression, MultiLogisticRegression, OLS


class Clustered(Observational):
    """Estimate casual effects under clustered interference setting. """
    
    def __init__(self, Y, Z, X, cluster_labels, group_labels, ingroup_labels, cluster_feature=None, n_moments=1, 
            prop_idv_model=LogisticRegression(), prop_neigh_model=MultiLogisticRegression(solver='saga'),
            n_matches=100, subsampling_match=2000, categorical_Z=True):
        super(Observational, self).__init__(Y, Z, X)
        self.data = ClusterData(Y, Z, X, cluster_labels, group_labels, ingroup_labels,
                cluster_feature, n_moments, categorical_Z)
        self.prop_idv_model = prop_idv_model
        self.prop_neigh_model = prop_neigh_model
        self.n_matches = n_matches
        self.subsampling_match = subsampling_match

    
    def est_via_ipw(self):
        return self._est()
    
    
    def est_via_aipw(self):
        return self._est(method='aipw')
            
        
    def _est(self, method='ipw'):
        group_structs = sorted(list(self.data.data_by_group_struct.keys()))
        total_result = [{} for _ in range(len(group_structs[0]))]   # all group_struct should have the same length
        for group_struct in group_structs:
            individuals_in_cluster = np.sum(group_struct)
            Mn = len(self.data.data_by_group_struct[group_struct][0])/individuals_in_cluster
            pn = Mn/self.data.M
            result = self.est_subsample(group_struct, method)
            for j in range(len(group_struct)):
                taug = result[j]['beta(g)']
                Vg = result[j]['se']**2*Mn
                total_result[j][group_struct] = pn, taug, Vg

        max_group_struct = np.maximum.reduce(group_structs)
        ret = []
        G_count = np.prod(max_group_struct+1)
        for j in range(len(group_structs[0])):
            ret_j = {
                    'beta(g)': np.zeros(max_group_struct+1),
                    'se': np.zeros(max_group_struct+1)
                    } 

            for G_encoded in range(G_count):
                g = self.decode_G(G_encoded, max_group_struct)
                key_vals = []
                for gg, (pn, taug, Vg) in total_result[j].items():
                    if np.all(gg >= g) and not np.all(gg == g):
                        encoded = self.encode_G(g, gg)
                        key_vals.append((pn, taug[encoded].item(), Vg[encoded].item()))
                key_vals = np.array(key_vals)

                if key_vals.size == 0:
                    continue

                key = tuple(g.squeeze(axis=0))
                w = key_vals[:, 0]/np.sum(key_vals[:, 0])

                taug_n = key_vals[:, 1]
                invalid = np.isnan(taug_n)  # certain group_struct is invalid because e.g. no such samples
                taug_n[invalid] = 0
                w[invalid] = 0
                if np.sum(w) > 0:
                    w /= np.sum(w)

                ret_j['beta(g)'][key] = w.dot(taug_n)

                Vg_n = key_vals[:, 2]

                # Equality doesn't hold when np.any(idx_g) is True
                # but variance_via_matching returns np.nan
                assert np.all(invalid <= np.isnan(Vg_n))

                Vg_n[invalid] = 0
                Vg1 = Vg_n.dot(w**2/key_vals[:, 0])
                ret_j['se'][key] = np.sqrt(Vg1/self.data.M)

            ret.append(ret_j)

        return ret
        

    def est_propensity(self, Z, G_encoded, X_g):
        idv = self.prop_idv_model.fit(X_g, Z)
        prop_idv = idv.insample_proba()

        # NOTE: this is the performance bottleneck.
        # It takes forever to converge when we have a lot of groups
        neigh = self.prop_neigh_model.fit(X_g, G_encoded)   # Z is not in the regressor since we assume Z_i is independent of Z_j for all i != j
        prop_neigh = neigh.predict_proba(X_g)

        return prop_idv, prop_neigh


    def est_subsample(self, group_struct, method='ipw'):
        Y, Z, G, Xc, cluster_labels, group_labels, ingroup_labels = self.data.data_by_group_struct[group_struct]
        M = len(set(cluster_labels))
        N = len(Y)

        G_encoded = self.encode_G(G, group_struct)
        Xc_s = Xc/np.sqrt(np.var(Xc, axis=0))
        X_g = np.column_stack([Xc_s, group_labels]) # the group label tells us the number of neighbours, so it's helpful for estimating the number of treated neighbours
        prop_idv, prop_neigh = self.est_propensity(Z, G_encoded, X_g)

        G_count = np.prod(np.array(group_struct)+1)
        result = [{'beta(g)': np.zeros(G_count), 'se': np.zeros(G_count)}
                for _ in range(len(group_struct))]
        linear_model = OLS()
        for g_encoded in range(G_count):
            g = self.decode_G(g_encoded, group_struct)
            # fit outcome model for aipw
            mask1 = np.all(G==g, axis=1) & (Z==1)
            mask0 = np.all(G==g, axis=1) & (Z==0)
            if not np.any(mask1) or not np.any(mask0):
                for j in range(len(group_struct)):
                    # no such samples
                    result[j]['beta(g)'][g_encoded] = np.nan
                continue

            if method == 'ipw':
                for j in range(len(group_struct)):
                    mask = group_labels == j

                    gg = g.copy()
                    gg[0, j] += 1
                    if np.max(gg - group_struct) > 0:
                        # it's impossible to have more treated people than all people in each group (assuming I am treated)
                        result[j]['beta(g)'][g_encoded] = np.nan
                    else:
                        result[j]['beta(g)'][g_encoded] = self._ipw_formula(Y[mask], Z[mask], G[mask], prop_idv[mask], prop_neigh[mask], g, g_encoded)

            elif method == 'aipw':
                linear_model.fit(Xc[mask1], Y[mask1])
                mu1g = linear_model.predict(Xc)
                linear_model.fit(Xc[mask0], Y[mask0])
                mu0g = linear_model.predict(Xc)
                for j in range(len(group_struct)):
                    mask = group_labels == j

                    gg = g.copy()
                    gg[0, j] += 1
                    if np.max(gg - group_struct) > 0:
                        # it's impossible to have more treated people than all people in each group (assuming I am treated)
                        result[j]['beta(g)'][g_encoded] = np.nan
                    else:
                        result[j]['beta(g)'][g_encoded] = self._aipw_formula(Y[mask], Z[mask], G[mask], prop_idv[mask], prop_neigh[mask], g, g_encoded, mu1g[mask], mu0g[mask])

            else:
                raise ValueError("Incorrect input of estimation method.")


            for j, individuals_in_group in enumerate(group_struct):
                sub = np.arange(N)
                sub = sub[group_labels == j]
                if self.subsampling_match < sub.size:
                    sub = np.random.choice(sub, self.subsampling_match, replace=False)

                idx_g = np.all(G[sub] == g, axis=1)
                if not np.any(idx_g):
                    # no such samples
                    result[j]['se'][g_encoded] = np.nan
                    continue

                Vg = self.variance_via_matching(Y[sub], Z[sub], Xc[sub], ingroup_labels[sub],
                                    prop_idv[sub], prop_neigh[sub, g_encoded], individuals_in_group, idx_g)
                result[j]['se'][g_encoded] = np.sqrt(Vg/M)

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
               
    
    def variance_via_matching(self, Y, Z, Xc, ingroup_labels, q, pg, size, idx_g):
        idx1 = Z == 1
        Y1_g = Y[idx1 & idx_g]
        Y0_g = Y[(~idx1) & idx_g]
        Xc1_g = Xc[idx1 & idx_g]
        Xc0_g = Xc[(~idx1) & idx_g]

        if Xc1_g.shape[0] < 2 or Xc0_g.shape[0] < 2:
            # we need at least 2 samples to calculate the variance
            return np.nan

        # Standardize Xc
        Xc_s = Xc/np.sqrt(np.var(Xc, axis=0))
        X1 = Xc1_g/np.sqrt(np.var(Xc1_g, axis=0))
        X0 = Xc0_g/np.sqrt(np.var(Xc0_g, axis=0))

        # Match for all units
        match_t = self.mat_match_mat(Xc_s, X1, self.n_matches)
        match_c = self.mat_match_mat(Xc_s, X0, self.n_matches)
        var_t = np.var(Y1_g[match_t], axis=1)
        var_c = np.var(Y0_g[match_c], axis=1)

        # Calculate Vg
        expectations = np.empty(size)
        for i in range(size):
            # The i-th individual in the j-th group
            mask = ingroup_labels == i
            arr = var_t[mask]/(pg[mask]*q[mask]) + var_c[mask]/(pg[mask]*(1-q[mask]))
            expectations[i] = np.mean(arr)

        beta = np.mean(Y1_g[match_t], axis=1) - np.mean(Y0_g[match_c], axis=1)
        beta_j = [beta[ingroup_labels == i] for i in range(size)]
        min_len = min(len(bi) for bi in beta_j)
        beta_mat = np.row_stack([bi[:min_len] for bi in beta_j])
        cov = np.cov(beta_mat)

        Vg = np.sum(expectations)/size**2 + np.sum(cov)/size**2
        return Vg
    

    def encode_G(self, G, group_struct):
        """
        Encode the neighbourhood structure G to an integer.

        >>> encode_G(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([2, 3]))
        [0 3 1 4]
        """
        weight = np.append(1, np.cumprod(np.array(group_struct[:-1], dtype=int) + 1))
        return np.sum(G * weight, axis=1)


    def decode_G(self, G_encoded, group_struct):
        """
        Decode the neighbourhood structure G from an integer.

        >>> decode_G(np.array([0, 3, 1, 4]), np.array([2, 3]))
        array([[0, 0],
               [0, 1],
               [1, 0],
               [1, 1]])
        """
        weight = np.append(1, np.cumprod(np.array(group_struct[:-1], dtype=int) + 1))
        G_encoded = np.copy(G_encoded)
        G_rev = []
        for w in reversed(weight):
            G_rev.append(G_encoded // w)
            G_encoded %= w
        return np.vstack(tuple(reversed(G_rev))).T
    

class ClusterData(POdata):
    def __init__(self, Y, Z, X, cluster_labels, group_labels, ingroup_labels,
            cluster_feature, n_moments, categorical_Z=True):
        self.Y = Y
        self.Z = Z
        self.X = X
        self.cluster_labels = cluster_labels
        self.M = len(set(cluster_labels))
        self.group_labels = group_labels
        self.ingroup_labels = ingroup_labels
        self.cluster_feature = cluster_feature
        self.n_moments = n_moments
        self.categorical_Z = categorical_Z
        self.units = len(Y)
        if self.verify_clusters():
            self.data_by_group_struct = self.split_by_group_struct()
        
    
    def split_by_group_struct(self):
        """
        Split the data into {group_struct: (Y, Z, G, Xc, labels)},
        where group_struct is the number of people from each group
        within a cluster
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
                    or idx2 == self.units - 1:
                if idx2 == self.units - 1:
                    idx2 = self.units
                idx = arg[idx1:idx2]
                if self.cluster_feature:
                    cluster_feature_i = self.cluster_feature[idx]
                else:
                    cluster_feature_i = None

                group_struct = tuple(cluster_group_structs_sorted[idx1])
                data_by_group_struct[group_struct] = self.get_final_tuple(self.Y[idx], 
                        self.Z[idx], self.X[idx], self.cluster_labels[idx], self.group_labels[idx], self.ingroup_labels[idx],
                        cluster_feature_i, group_struct)

                idx1 = idx2

        return data_by_group_struct
    
    
    def get_final_tuple(self, Y, Z, X, cluster_labels, group_labels, ingroup_labels, cluster_feature, group_struct):
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
        ingroup_labels = ingroup_labels[arg]

        nunit_per_cluster = np.sum(group_struct)
        units, k = X.shape
        clusters = units//nunit_per_cluster
        X_augmented = X = X.reshape(clusters, nunit_per_cluster, k)

        if self.n_moments > 0:
            for j in range(len(group_struct)):
                # average within each group
                mask = (group_labels == j).reshape(clusters, nunit_per_cluster)
                mask = np.tile(mask[:, :, np.newaxis], k)
                Xj = np.ma.masked_where(mask, X)

                # first-order plain moment, a.k.a. mean
                m1 = np.mean(Xj, axis=1, keepdims=True)
                f1 = np.repeat(m1, nunit_per_cluster, axis=1)
                X_augmented = np.append(X_augmented, f1, axis=2)

                # second- and higher order central moments
                X_centered = X - f1
                for p in range(1, self.n_moments):
                    mp = np.mean(np.power(X_centered, p+1), axis=1, keepdims=True)
                    fp = np.repeat(mp, nunit_per_cluster, axis=1)
                    X_augmented = np.append(X_augmented, fp, axis=2)

        X = X_augmented.reshape(units, k * (1+self.n_moments*len(group_struct)))

        if cluster_feature:
            cluster_feature = cluster_feature[arg]
            X = np.append(X, cluster_feature, axis=2)

        # get G, number of treated neighbours
        ngroup = len(group_struct)
        Z_onehot = np.zeros((clusters, nunit_per_cluster, ngroup))
        cluster_labels_idx = np.cumsum(np.append(0, np.diff(cluster_labels)) > 0)
        Z_onehot[cluster_labels_idx, np.tile(np.arange(nunit_per_cluster), clusters), group_labels] = Z
        G_plus = np.sum(Z_onehot, axis=1, keepdims=True)
        G_stack = G_plus - Z_onehot
        G = G_stack.reshape(units, ngroup)

        if self.categorical_Z:
            G = G.astype(int)
            Z = Z.astype(int)
        self.covariate_dims = k
        return Y, Z, G, X, cluster_labels, group_labels, ingroup_labels
        
    
    def verify_clusters(self):
        if not self.verify_yzx():
            return False

        if not (isinstance(self.cluster_labels, np.ndarray)
                and isinstance(self.group_labels, np.ndarray)
                and isinstance(self.ingroup_labels, np.ndarray)
                and isinstance(self.n_moments, int)):
            raise ValueError("Incorrect input type for cluster_labels, group_labels, ingroup_labels (np.ndarray) or n_moments (int).")
        if not self.cluster_labels.shape == (self.units, ):
            raise ValueError("Incorrect shape for cluster_labels. Should be `(units, )`.")
        if not self.group_labels.shape == (self.units, ):
            raise ValueError("Incorrect shape for group_labels. Should be `(units, )`.")
        if not self.ingroup_labels.shape == (self.units, ):
            raise ValueError("Incorrect shape for ingroup_labels. Should be `(units, )`.")
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

