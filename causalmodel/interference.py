import numpy as np
from potentialoutcome import PotentialOutcome, POdata
from LearningModels import LogisticRegression, MultiLogisticRegression


class Clustered(PotentialOutcome):
    
    def __init__(self, Y, Z, X, cluster_label, cluster_feature=None, n_moments=1, 
            prop_idv_model=LogisticRegression(), prop_neigh_model=MultiLogisticRegression(), n_matches=1):
        super(self.__class__, self).__init__(Y,Z,X)
        self.data = ClusterData(Y, Z, X, cluster_label, cluster_feature, n_moments)
        self.prop_idv_model = prop_idv_model
        self.prop_neigh_model = prop_neigh_model
        self.n_matches = n_matches
            
        
    def est_via_ipw(self):
        pass
    
    
    def est_propensity(self, Z, G, Xc):
        idv = self.prop_idv_model.fit(Xc, Z)
        prop_idv = idv.insample_proba()
        neigh = self.prop_neigh_model.fit(Xc, G)
        prop_neigh = neigh.insample_proba()
        return prop_idv, prop_neigh
    
    
    
        
    

class ClusterData(POdata):
    
    def __init__(self, Y, Z, X, cluster_label, cluster_feature, n_moments):
        self.Y = Y
        self.Z = Z
        self.X = X
        self.cluster_label = cluster_label
        self.cluster_feature = cluster_feature
        self.n_moments = n_moments
        if self.verify_clusters():
            self.data_by_size = self.split_by_size()
        
    
    # TO DO: Can we optimize?
    def split_by_size(self):
        """
        Split the data into {size: (Y, Z, G, Xc, labels)}
        """
        clusters = set(self.cluster_label)
        data_by_size = {}
        for c in clusters:
            idx = self.cluster_label == c
            size = np.sum(idx)
            if self.cluster_feature:
                cluster_feature_c = self.cluster_feature[idx]
                if not self.all_same(cluster_feature_c):
                    raise RuntimeWarning("Please make sure cluster_features are\
                                    the same with cluster {}.".format(c))
            else:
                cluster_feature_c = None
            nt = np.sum(self.Z[idx])
            data_c = [self.Y[idx], self.Z[idx], nt - self.Z[idx], self.get_covariates(
                self.X[idx], cluster_feature_c, self.n_moments),
                self.cluster_label[idx]]
            if data_by_size.get(size) is None:
                data_by_size[size] = data_c
            else:
                for i, v in enumerate(data_by_size.get(size)):
                    data_by_size[size][i] = np.append(v, data_c[i], axis=0)
        return data_by_size
    
    
    def get_covariates(self, X, cluster_feature, n_moments):
        """
        Append cluster_feature to X and generate n_moments for as additional 
        cluster features

        Parameters
        ----------
        X : np.ndarray
            (n, k) shape matrix.
        cluster_feature : np.ndarray or None
            (n, m) shape matrix.
        n_moments : int
            Number of moments to include as additional features.

        Returns
        -------
        Xc : np.ndarray
            Full matrix of features.

        """
        if n_moments > 0:
            mu = np.mean(X, axis=0)
            f1 = np.repeat(np.expand_dims(mu, axis=0), len(X),axis=0)
            X = np.append(X, f1, axis=1)
            X_centered = X - mu
            for i in range(1, n_moments):
                mi = np.mean(np.power(X_centered, n_moments+1), axis=0)
                fi = np.repeat(np.expand_dims(mi, axis=0), len(X),axis=0)
                X = np.append(X, fi, axis=1)
        if cluster_feature:
            X = np.append(X, cluster_feature, axis=1)
        return X
        
    
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
    