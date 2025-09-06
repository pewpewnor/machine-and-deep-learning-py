import numpy as np


class MinMaxScaler:
    """
    Transformed values will always be between 0 and 1
    """

    def fit(self, X):
        X = np.array(X)
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        return self

    def transform(self, X):
        X = np.array(X)
        return (X - self.min) / (self.max - self.min)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
