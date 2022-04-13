import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_data(N=10000, k=2, tau=10):
    """
    Get the classic (Y,Z,X) random samples.

    """
    X = np.random.multivariate_normal(np.zeros(k), np.diag(np.ones(k)), N)
    prob = sigmoid(X.dot(np.linspace(-1,1,k)))
    Z = np.zeros(N)
    unif = np.random.uniform(0,1,N)
    Z[unif < prob] = 1
    Y = tau*Z + X.dot(np.linspace(-1,1,k)) + np.random.normal(0,1,N)
    return Y, Z, X


def get_data_continuous(N=10000, k=2, tau=10):
    """
    Get the classic (Y,Z,X) random samples with Z being continuous

    """
    X = np.random.multivariate_normal(np.zeros(k), np.diag(np.ones(k)), N)
    Z = X.dot(np.linspace(-1,1,k)) + np.random.normal(0,1,N)
    Y = tau*Z + X.dot(np.linspace(-1,1,k)) + np.random.normal(0,1,N)
    return Y, Z, X


def get_fixed_cluster(clusters=10000, ngroup_per_cluster=2, nunit_per_group=3,
                      k=2, tau=1, gamma=np.array((0.1, 0.01, 0.001)),
                      label_start=0):
    """
    Get data for fixed cluster size.

    Parameters
    ----------
    clusters : int, optional
        Number of clusters. The default is 10000.
    ngroup_per_cluster : int, optional
        Number of groups within each cluster. The default is 2.
    nunit_per_group : int, optional
        Number of units within each group. The default is 3.
    k : int, optional
        Number of features for each unit. The default is 2.
    tau : float, optional
        Amount of direct treatment effect. The default is 1.
    gamma : tuple whose length equals `nunit_per_group`, optional
        Amount of spillover effect. The default is (0.1, 0.01, 0.001).
    label_start : int, optional
        The beginning index for cluster labels. The default is 0.
    """
    if len(gamma) != ngroup_per_cluster:
        raise ValueError(f"len(gamma) = {len(gamma)} != {nunit_per_group} = nunit_per_group")
    nunit_per_cluster = ngroup_per_cluster * nunit_per_group
    units = clusters * nunit_per_cluster
    # get clustering labels
    cluster_labels = label_start + np.repeat(np.arange(clusters), nunit_per_cluster)
    # get group labels
    group_labels = np.tile(np.repeat(np.arange(ngroup_per_cluster), nunit_per_group), clusters)
    # get covariates
    X = 0.1 * np.random.multivariate_normal(np.zeros(k), np.eye(k), (clusters, nunit_per_cluster))
    # average within each cluster
    Xcmean = np.mean(X, axis=1, keepdims=True)
    Xc = np.empty((clusters, nunit_per_cluster, 2*k))
    Xc[:, :, :k] = X
    Xc[:, :, k:] = Xcmean
    X = X.reshape((units, k))
    Xc = Xc.reshape((units, 2*k))
    # get treatment
    Z = np.zeros(units)
    prop_idv = sigmoid(Xc.dot(np.linspace(-1,1,2*k)))
    unif = np.random.uniform(0, 1, units)
    Z[unif < prop_idv] = 1
    # get G, number of treated neighbours
    Z_onehot = np.zeros((clusters, nunit_per_cluster, ngroup_per_cluster))
    Z_onehot[cluster_labels-label_start, np.tile(np.arange(nunit_per_cluster), clusters), group_labels] = Z
    G_plus = np.sum(Z_onehot, axis=1, keepdims=True)
    G_stack = G_plus - Z_onehot
    G = G_stack.reshape(units, ngroup_per_cluster)
    # get outcome Y
    epsilon = np.random.normal(0, 1, units)
    Y = tau*Z + Xc.dot(np.linspace(-1,1,2*k)) + (G@gamma) *Z + epsilon
    sub = np.random.choice(np.arange(units), units, replace=False)
    return Y[sub], Z[sub], X[sub], cluster_labels[sub], group_labels[sub], G[sub], Xc[sub]


def get_clustered_data(
        clusters_list=[5000, 5000, 2000],
        ngroup_per_cluster=3,
        nunit_per_group_list=[7, 11, 13]):
    """
    Get data for varying cluster sizes

    Parameters
    ----------
    clusters_list : list, optional
        List of # clusters. The default is [5000,5000,2000].
    ngroup_per_cluster : int, optional
        The common # groups within each cluster. All clusters must share the same `ngroup_per_cluster` value. The default is 3.
    nunit_per_group_list : list, optional
        List of # units within each cluster. The default is [7,11,13].

    Returns
    -------
    list
        Output data [Y, Z, X, labels].

    """
    label_start_list = [0, *np.cumsum(clusters_list)[:-1]]
    # FIXME: if `ngroup_per_cluster` is not 3, we need to pass in `gamma`
    zipped = list(zip(*[list(get_fixed_cluster(clusters, ngroup_per_cluster, nunit_per_group,
                                               label_start=label_start)) 
                        for clusters, nunit_per_group, label_start
                        in zip(clusters_list, nunit_per_group_list, label_start_list)]))
    return [np.concatenate(Vs) for Vs in zipped]

