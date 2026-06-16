import numpy as np
from scipy.special import expit
from scipy import linalg


class BayesianGaussianClassifier:
    """
    Bayesian generative classifier with Gaussian class-conditionals.

    Models each class as a multivariate Gaussian:
        p(x | y=k) = N(x | mu_k, Sigma_k)

    with a Gaussian-Wishart conjugate prior on (mu_k, Sigma_k),
    giving closed-form posteriors.

    For classification:
        p(y=1 | x) proportional to p(x | y=1) * p(y=1)

    The class prior p(y=1) is set from the training data (empirical Bayes)
    but can be overridden.

    Parameters
    ----------
    prior_strength : float
        Strength of the prior on the mean (kappa_0 in the
        Normal-Wishart parameterisation). Higher = stronger
        regularisation toward the prior mean. Default 1.0.
    shared_covariance : bool
        If True, fit a single pooled covariance (LDA-style).
        If False, fit separate covariances per class (QDA-style).
        Default True (more stable with limited data).
    """

    def __init__(
        self,
        prior_strength: float = 1.0,
        shared_covariance: bool = True,
    ):
        self.prior_strength   = prior_strength
        self.shared_covariance = shared_covariance

        self.mu_    = {}
        self.sigma_ = {}
        self.log_prior_ = {}
        self.classes_   = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianGaussianClassifier":
        n, d = X.shape
        self.classes_ = np.unique(y)

        for k in self.classes_:
            Xk = X[y == k]
            nk = len(Xk)

            mu_0    = np.zeros(d)
            kappa_0 = self.prior_strength

            kappa_n = kappa_0 + nk
            mu_n    = (kappa_0 * mu_0 + nk * Xk.mean(axis=0)) / kappa_n

            self.mu_[k]       = mu_n
            self.log_prior_[k] = np.log(nk / n)

        if self.shared_covariance:
            S = np.zeros((d, d))
            for k in self.classes_:
                Xk   = X[y == k]
                diff = Xk - self.mu_[k]
                S   += diff.T @ diff
            nu_0    = d + 2
            psi_0   = np.eye(d)
            sigma   = (psi_0 + S) / (n + nu_0 - d - 1)
            for k in self.classes_:
                self.sigma_[k] = sigma
        else:
            for k in self.classes_:
                Xk   = X[y == k]
                nk   = len(Xk)
                diff = Xk - self.mu_[k]
                S    = diff.T @ diff
                nu_0    = d + 2
                psi_0   = np.eye(d)
                self.sigma_[k] = (psi_0 + S) / (nk + nu_0 - d - 1)

        return self

    def _log_likelihood(self, X: np.ndarray, k: int) -> np.ndarray:
        mu    = self.mu_[k]
        sigma = self.sigma_[k]
        d     = X.shape[1]

        try:
            L        = linalg.cholesky(sigma, lower=True)
            log_det  = 2 * np.sum(np.log(np.diag(L)))
            diff     = X - mu
            alpha    = linalg.solve_triangular(L, diff.T, lower=True)
            maha     = np.sum(alpha ** 2, axis=0)
        except linalg.LinAlgError:
            sigma_reg = sigma + 1e-6 * np.eye(d)
            L         = linalg.cholesky(sigma_reg, lower=True)
            log_det   = 2 * np.sum(np.log(np.diag(L)))
            diff      = X - mu
            alpha     = linalg.solve_triangular(L, diff.T, lower=True)
            maha      = np.sum(alpha ** 2, axis=0)

        return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_posts = np.column_stack([
            self._log_likelihood(X, k) + self.log_prior_[k]
            for k in self.classes_
        ])
        log_posts -= log_posts.max(axis=1, keepdims=True)
        probs      = np.exp(log_posts)
        probs     /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_signal_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        signal_idx = list(self.classes_).index(1)
        return probs[:, signal_idx]