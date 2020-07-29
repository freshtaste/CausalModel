from sklearn.linear_model import LogisticRegression


class LogisticRegression(LogisticRegression):
    
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

