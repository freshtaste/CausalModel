from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class OLS(LinearRegression):
    
    def __init__(self, **kwargs):
        self.X = None
        self.y = None
        super().__init__(**kwargs)
        
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        return super(OLS, self).fit(X, y)
    
    
    def predict(self, X):
        return super(OLS, self).predict(X)
    
    
    def insample_predict(self):
        return self.predict(self.X)
    
    
class RandomForest(RandomForestRegressor):
    
    def __init__(self, **kwargs):
        self.X = None
        self.y = None
        super().__init__(**kwargs)
        
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        return super(RandomForest, self).fit(X, y)
    
    
    def predict(self, X):
        return super(RandomForest, self).predict(X)
    
    
    def insample_predict(self):
        return self.predict(self.X)