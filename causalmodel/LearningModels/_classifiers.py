from sklearn.linear_model import LogisticRegression as Logit
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class LogisticRegression(Logit):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.X = None
        self.y = None
    
    
    def fit(self, X, y):
        if set(y) == {0, 1}:
            self.X = X
            self.y = y
            return super(LogisticRegression, self).fit(X, y)
        else:
            raise RuntimeError("Input independent variable (y) is not binary!")
    
    
    def predict_proba(self, X):
        sklearn_out = super(LogisticRegression, self).predict_proba(X)
        return sklearn_out[:,1]
    
    
    def insample_proba(self):
        return self.predict_proba(self.X)
    
    
class MultiLogisticRegression(Logit):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.X = None
        self.y = None
        
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        return super(MultiLogisticRegression, self).fit(X, y)
    
        
    def insample_proba(self):
        sklearn_out = self.predict_proba(self.X)
        return sklearn_out[np.arange(len(sklearn_out)), self.y]
    

class RandomForest(RandomForestClassifier):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.X = None
        self.y = None
        
    
    def fit(self, X, y):
        if set(y) == {0, 1}:
            self.X = X
            self.y = y
            return super(RandomForestClassifier, self).fit(X, y)
        else:
            raise RuntimeError("Input independent variable (y) is not binary!") 
            
    
    def predict_proba(self, X):
        sklearn_out = super(RandomForestClassifier, self).predict_proba(X)
        return sklearn_out[:,1]
    
    
    def insample_proba(self):
        return self.predict_proba(self.X) 
    