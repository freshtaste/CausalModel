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


def get_fixed_cluster(N=30000, M=10000, k=2, tau=1, gamma=0.1, label_start=0):
    """
    Get data for fixed cluster size.

    Parameters
    ----------
    N : int, optional
        Number of units. The default is 30000.
    M : int, optional
        Number of clusters. The default is 10000.
    k : int, optional
        Number of features. The default is 2.
    tau : float, optional
        Amount of direct treatment effect. The default is 1.
    gamma : float, optional
        Amount of spillover effect. The default is 0.1.
    label_start : int, optional
        The beginning index for cluster labels. The default is 0.

    """
    n = int(N/M)
    # get clustering labels
    labels = np.array(list(np.arange(M))*n) + label_start
    labels = np.sort(labels)
    # get covariates
    X = np.random.multivariate_normal(np.zeros(k), np.diag(np.ones(k)), (M, n))*0.1
    Xc = np.zeros((M, n, k*2))
    Xcmean = np.mean(X, axis=1).reshape((M, 1, k))
    Xc[:,:,:k] = X.reshape((M, n, k))
    Xc[:,:,k:] = np.repeat(Xcmean, n, axis=1)
    X = X.reshape(N, k)
    Xc = Xc.reshape(N, k*2)
    # get treatment
    Z = np.zeros(N)
    prop_idv = sigmoid(Xc.dot(np.linspace(-1,1,2*k)))
    unif = np.random.uniform(0,1,N)
    Z[unif < prop_idv] = 1
    # get G, number of treated neighbours
    Z_stack = Z.reshape(M, n)
    G_plus = np.repeat(np.sum(Z_stack, axis=1).reshape(M, 1), n, axis=1)
    G_stack = G_plus - Z_stack
    G = G_stack.reshape(N)
    # get outcome Y
    epsilon = np.random.normal(0, 1, N)
    Y = tau*Z + Xc.dot(np.linspace(-1,1,2*k)) + gamma * np.sqrt(G) *Z + epsilon
    sub = np.random.choice(np.arange(N), N, replace=False)
    return Y[sub], Z[sub], X[sub], labels[sub]


def get_clustered_data(Ms=[5000,5000,2000], ns=[2,3,5]):
    """
    Get data for varying cluster sizes

    Parameters
    ----------
    Ms : list, optional
        List of # clusters. The default is [5000,5000,2000].
    ns : list, optional
        List of the corresponding cluster size. The default is [2,3,5].

    Returns
    -------
    list
        Output data [Y, Z, X, labels].

    """
    ls = [0,* np.cumsum(Ms)[0:-1]]
    zipped = list(zip(*[list(get_fixed_cluster(ns[i]*M, M, label_start=ls[i])) 
                        for i, M in enumerate(Ms)]))
    return [np.concatenate(Vs) for Vs in zipped]

