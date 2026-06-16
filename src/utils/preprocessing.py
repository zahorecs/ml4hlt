import numpy as np


class StandardScaler:
    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class PCA:
    """
    PCA via SVD. Keeps enough components to explain
    `variance_threshold` fraction of total variance.
    """

    def __init__(self, variance_threshold: float = 0.95):
        self.variance_threshold = variance_threshold

    def fit(self, X: np.ndarray) -> "PCA":
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        explained = s ** 2
        explained /= explained.sum()
        cumulative = np.cumsum(explained)
        self.n_components_ = int(np.searchsorted(cumulative, self.variance_threshold)) + 1
        self.components_   = Vt[:self.n_components_]
        self.explained_variance_ratio_ = explained[:self.n_components_]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)