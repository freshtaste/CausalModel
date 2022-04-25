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


def get_fixed_cluster(clusters=10000, group_struct=(2, 3, 4),
                      k=2, tau=1, gamma=None, label_start=0):
    """
    Get data for fixed cluster size.

    Parameters
    ----------
    clusters : int, optional
        Number of clusters. The default is 10000.
    group_struct : tuple, optional
        Group structure within each cluster. For example, (2, 3, 4)
        means the each cluster has 2 units in the first group, 3
        units in the second group, and 4 units in the third group.
    k : int, optional
        Number of features for each unit. The default is 2.
    tau : float, optional
        Amount of direct treatment effect. The default is 1.
    gamma : tuple, optional
        Amount of spillover effect. Should have the exact same
        length as `group_struct`. The default is 0.1 for each
        group.
    label_start : int, optional
        The beginning index for cluster labels. The default is 0.
    """
    if gamma is None:
        gamma = 0.1 * np.ones(len(group_struct))
    elif len(group_struct) != len(gamma):
        raise ValueError(f"len(group_struct) = {len(group_struct)} != {len(gamma)} = len(gamma)")
    nunit_per_cluster = np.sum(group_struct)
    units = clusters * nunit_per_cluster
    # get clustering labels
    cluster_labels = label_start + np.repeat(np.arange(clusters), nunit_per_cluster)
    # get group labels
    group_labels_within_cluster = np.repeat(np.arange(len(group_struct)), group_struct)
    group_labels = np.tile(group_labels_within_cluster, clusters)
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
    ngroup_per_cluster = len(group_struct)
    Z_onehot = np.zeros((clusters, nunit_per_cluster, ngroup_per_cluster))
    Z_onehot[cluster_labels-label_start, np.tile(np.arange(nunit_per_cluster), clusters), group_labels] = Z
    G_plus = np.sum(Z_onehot, axis=1, keepdims=True)
    G_stack = G_plus - Z_onehot
    G = G_stack.reshape(units, ngroup_per_cluster)
    # get outcome Y
    epsilon = np.random.normal(0, 1, units)
    Y = tau*Z + Xc.dot(np.linspace(-1,1,2*k)) + (G@gamma) * Z + epsilon
    sub = np.random.choice(np.arange(units), units, replace=False)
    return Y[sub], Z[sub], X[sub], cluster_labels[sub], group_labels[sub], G[sub], Xc[sub]


def get_clustered_data(
        clusters_list=[5000, 5000, 2000],
        group_struct_list=[(2, 3, 4), (3, 4, 5), (4, 5, 6)],
        tau=1,
        gamma=(5, 0, 1)):
    """
    Get data for varying cluster sizes

    Parameters
    ----------
    clusters_list : list, optional
        List of # clusters. The default is [5000, 5000, 2000].
    group_struct_list : list of tuple, optional
        The group structure of each cluster. The default is
        [(2, 3, 4), (3, 4, 5), (4, 5, 6)].
    tau : float, optional
        Amount of direct treatment effect. This attribute is
        common to all clusters. The default is 1.
    gamma_list: tuple, optional
        The group spillover effect of each group. This attribute
        is common to all clusters. The default is (5, 0, 1).

    Returns
    -------
    list
        Output data [Y, Z, X, labels].

    """
    label_start_list = [0, *np.cumsum(clusters_list[:-1])]
    zipped = list(zip(*[list(get_fixed_cluster(clusters, group_struct, tau=tau, gamma=gamma, label_start=label_start))
        for clusters, group_struct, label_start
        in zip(clusters_list, group_struct_list, label_start_list)]))
    return [np.concatenate(Vs) for Vs in zipped]

