import numpy as np


class StandardScaler:
    """
    Transformed values will usually end up between -3 and 3
    """

    def fit(self, X):
        X = np.array(X)
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True)
        self.std[self.std == 0] = 1.0  # to avoid divide by 0
        return self

    def transform(self, X):
        X = np.array(X)
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)
