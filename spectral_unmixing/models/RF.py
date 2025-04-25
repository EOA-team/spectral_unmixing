from sklearn.ensemble import RandomForestRegressor

class RFRegressor:
    def __init__(self, **kwargs):
        
        self.params = {**kwargs}
        self.model = RandomForestRegressor(**self.params)

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return self.params
