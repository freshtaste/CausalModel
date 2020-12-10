import numpy as np
import warnings
from .base import DesignBase, get_balance


class CRD(DesignBase):
    """Completely Randomized Design"""
    
    
    def __init__(self, treated_ratio=0.5, covariate=None, balance=False, 
                 eps=0.1, max_iter=1000):
        """CRD design requires an input of treated ratio. The rest of the parameters
        are meant for covariate balancing via rerandomization"""
        super(self.__class__, self).__init__(treated_ratio, covariate, 
                                    balance, eps, max_iter)
        self.template = None
    
    
    def get_params_via_obs(self, Z):
        """Estimate the ratio of treated units with sample mean"""
        self.params = np.mean(Z)
        
    
    def draw(self, n):
        """
        Draw a trial of allocation of size n.

        Parameters
        ----------
        n : int
            Number of units.

        Returns
        -------
        Z : numpy.ndarray
            Treatment vector.

        """
        self._get_template(n)
        if self.balance:
            if self.X is None:
                raise RuntimeError("covariate must be provided if balance is True.")
            else:
                Z = self._draw_via_balance(n)
        else:
            Z = self._draw(n)
        return Z
    
    
    def _get_template(self, n):
        """Helper method to create a sample treatment vector for randomization."""
        nt = int(self.params*n)
        self.template = np.zeros(n)
        self.template[:nt] = 1
        
    
    def _draw(self, n):
        """draw a trial"""
        Z = np.random.permutation(self.template)
        return Z
    
    
    def _draw_via_balance(self, n):
        """draw a randomized trial via covariate balancing"""
        balance = 1
        count = 0
        while balance > self.eps and count < self.max_iter:
            Z = self._draw(n)
            balance = get_balance(Z, self.X)
            count += 1
        if balance > self.eps:
            warnings.warn("Exceed maximum iterations without converging.")
        return Z