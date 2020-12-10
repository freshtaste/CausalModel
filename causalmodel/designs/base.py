import numpy as np
from causalmodel.potentialoutcome import POdata


class DesignBase(object):
    """Base Class for design classes."""
    
    def __init__(self, params, covariate=None, balance=False,
                 eps=0.1, max_iter=1000):
        """
        Initialization of design classes requires a parameter. The rest of the
        optional parameters are meant for covariate balancing with rerandomization.

        """
        self.params = params
        self.X = covariate
        self.balance = balance
        self.eps = eps
        self.max_iter = max_iter
    
    
    def get_params_via_obs(self, Z):
        """Estimate the design parameter from one realization"""
        pass
    
    
    def draw(self, n):
        """Randomly draw treatment allocations"""
        pass
    
    
def get_balance(Z, X):
    """
    Helper function to calculate balancing criterion. 
    See: "Kari Lock Morgan & Donald B. Rubin (2015) Rerandomization to Balance 
    Tiers of Covariates, Journal of the American Statistical Association, 
    110:512, 1412-1421, DOI: 10.1080/01621459.2015.1079528"

    Parameters
    ----------
    Z : numpy.ndarray
        Treatment vector.
    X : numpy.ndarray
        Covariates.

    Returns
    -------
    m : float
        balancing criterion.

    """
    n = len(Z)
    pseudoY = np.zeros(n)
    data = POdata(pseudoY, Z, X)
    m1 = np.mean(data.Xt,axis=0)
    m0 = np.mean(data.Xc,axis=0)
    cov = np.cov(X.T)
    m = (data.nt*data.nc/n)*(m1-m0).dot(cov).dot(m1-m0)
    return m