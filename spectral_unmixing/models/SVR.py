from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class SVRegressor:
    def __init__(self, **kwargs):
        default_params = {
            'C': 1.0,
            'epsilon': 0.1,
            'kernel': 'rbf',
            'gamma': 'scale'
        }
        self.params = {**default_params, **kwargs}
        self.model = make_pipeline(StandardScaler(), SVR(**self.params))

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return self.params
