from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor as RF


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
    
    
class RandomForestRegressor(RF):
    
    def __init__(self, **kwargs):
        self.X = None
        self.y = None
        super().__init__(**kwargs)
        
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        return super(RandomForestRegressor, self).fit(X, y)
    
    
    def predict(self, X):
        return super(RandomForestRegressor, self).predict(X)
    
    
    def insample_predict(self):
        return self.predict(self.X)