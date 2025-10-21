from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
from make_dataset import make_data_separated
from matplotlib import pyplot as plt

try:
    from scipy.spatial.distance import cdist
except ImportError:
    cdist = None

Array = np.ndarray

def zscore(X: Array) -> Tuple[Array, Array, Array]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mu)/sd, mu, sd

def pairwise_sq_dists_numpy(X1: Array, X2: Array) -> Array:
    X1s = np.sum(X1**2, axis=1, keepdims=True)
    X2s = np.sum(X2**2, axis=1, keepdims=True).T
    return np.maximum(X1s + X2s - 2 * X1 @ X2.T, 0.0)

def K_radial(u: Array, name: str) -> Array:
    if name == "rbf":
        return np.exp(-0.5*u**2)
    if name == "epanechnikov":
        w = 0.75*(1 - u**2)
        return np.where(u <= 1, np.maximum(w, 0.0), 0.0)
    if name == "tricube":
        w = (1 - np.abs(u)**3)**3
        return np.where(np.abs(u) <= 1, w, 0.0)
    if name == "laplace":
        return 0.5*np.exp(-np.abs(u))
    if name == "uniform":
        return 0.5*(np.abs(u) <= 1).astype(float)
    if name == "triangular":
        return np.maximum(1 - np.abs(u), 0.0)
    raise ValueError(f"Unknown kernel '{name}'")

def loo_from_hat(y: Array, yhat: Array, sdiag: Array, eps: float = 1e-12) -> Array:
    den = 1 - np.clip(sdiag, 0.0, 1.0 - eps)
    return (yhat - sdiag*y) / den

def rmse(a: Array, b: Array) -> float: 
    return float(np.sqrt(np.mean((a - b)**2)))

def mae(a: Array, b: Array) -> float:
    return float(np.mean(np.abs(a - b)))

@dataclass
class NadarayaWatsonRegressor:
    kernel: Literal["rbf","epanechnikov","tricube","laplace","uniform","triangular"] = "rbf"
    h: float = 1.0
    per_feature_bandwidth: Optional[Array] = None
    standardize: bool = True
    truncate: Optional[float] = None

    X_: Optional[Array] = None
    y_: Optional[Array] = None
    mu_: Optional[Array] = None
    sd_: Optional[Array] = None
    _train_dists: Optional[Array] = None

    def _prep(self, X: Array) -> Array:
        X = np.atleast_2d(X)
        return (X - self.mu_) / self.sd_ if self.standardize else X

    def fit(self, X: Array, y: Array) -> "NadarayaWatsonRegressor":
        X = np.atleast_2d(X); y = np.asarray(y).reshape(-1)
        if self.standardize:
            self.X_, self.mu_, self.sd_ = zscore(X)
        else:
            self.X_, self.mu_, self.sd_ = X, np.zeros((1, X.shape[1])), np.ones((1, X.shape[1]))
        self.y_ = y
        return self

    # Calculate the distance matrix. In training case, it is the distance matrix between X_train and X_train.
    def _get_distances(self, Xq: Array, X: Array) -> Array:
        if self.per_feature_bandwidth is not None:
            if cdist is None: raise ImportError("SciPy is required for per_feature_bandwidth.")
            h_sq = np.asarray(self.per_feature_bandwidth).flatten()**2
            return cdist(Xq, X, metric='seuclidean', V=h_sq)
        else:
            if cdist is not None:
                return cdist(Xq, X, metric='euclidean')
            return np.sqrt(pairwise_sq_dists_numpy(Xq, X))

    def _weights(self, Xq: Array, h: Optional[float] = None, _dists: Optional[Array] = None) -> Array:
        h = h if h is not None else self.h
        Xq_prep = self._prep(np.atleast_2d(Xq))

        if self.per_feature_bandwidth is not None:
            u = self._get_distances(Xq_prep, self.X_)
        else:
            dists = _dists if _dists is not None else self._get_distances(Xq_prep, self.X_)
            u = dists / (h + 1e-12)

        W = K_radial(u, self.kernel)
        if self.truncate is not None:
            W = np.where(u <= self.truncate, W, 0.0)
        
        denom = W.sum(axis=1, keepdims=True) + 1e-12
        return W / denom

    def predict(self, Xq: Array) -> Array:
        W = self._weights(Xq)
        return W @ self.y_

    def smoother_matrix(self, h: Optional[float] = None) -> Tuple[Array, Array]:
        if self._train_dists is None:
            self._train_dists = self._get_distances(self.X_, self.X_)
        W = self._weights(self.X_, h=h, _dists=self._train_dists)
        return W, W.diagonal()

    def predict_train(self) -> Array:
        W, _ = self.smoother_matrix()
        return W @ self.y_

    def loo_predict_train(self) -> Array:
        W, sdiag = self.smoother_matrix()
        yhat = W @ self.y_
        return loo_from_hat(self.y_, yhat, sdiag)

    def select_bandwidth(self, h_grid: Array) -> float:
        best_h, best_cv = self.h, np.inf
        self._train_dists = self._get_distances(self.X_, self.X_)

        for h_val in h_grid:
            W, sdiag = self.smoother_matrix(h=float(h_val))
            yhat = W @ self.y_
            yloo = loo_from_hat(self.y_, yhat, sdiag)
            cv = rmse(self.y_, yloo)
            if cv < best_cv:
                best_cv, best_h = cv, float(h_val)
        
        self.h = best_h
        self._train_dists = None
        return best_h

    def score(self, kind: Literal["mse","mae"]="mse", loo: bool=False) -> float:
        ypred = self.loo_predict_train() if loo else self.predict_train()
        fn = rmse if kind=="mse" else mae
        return fn(self.y_, ypred)

@dataclass
class KernelRegression:
    kernel: str = "rbf"
    h: float = 1.0
    standardize: bool = True
    per_feature_bandwidth: Optional[Array] = None

    _model: Optional[NadarayaWatsonRegressor] = None

    def fit(self, X: Array, y: Array) -> "KernelRegression":
        self._model = NadarayaWatsonRegressor(
            kernel=self.kernel, h=self.h,
            per_feature_bandwidth=self.per_feature_bandwidth,
            standardize=self.standardize
        ).fit(X, y)
        return self

    def predict(self, Xq: Array) -> Array:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        return self._model.predict(Xq)

    def select_bandwidth(self, h_grid: Array) -> float:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        return self._model.select_bandwidth(h_grid)

    def score(self, kind: Literal["mse","mae"]="mse", loo: bool=False) -> float:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        return self._model.score(kind=kind, loo=loo)
        
    def __getattr__(self, name):
        if self._model is not None and hasattr(self._model, name):
            return getattr(self._model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

if __name__ == "__main__":
    n_train = 450
    n_test = 50
    X_train, y_train, X_test, y_test = make_data_separated(n_train=n_train, n_test=n_test, seed=0)

    # n_trains = 200
    # n_tests = 50
    # n_samples = n_trains + n_tests
    # theta = np.linspace(0, 2 * np.pi, n_samples)
    # r = 1.0
    # X = np.column_stack((r * np.cos(theta), r * np.sin(theta)))

    # noise_level = 0.5
    # X_noisy = X + noise_level * np.random.randn(*X.shape)

    # X_train, y_train, X_test, y_test = make_data_separated(n_train=n_trains, n_test=n_tests, n_features=1 ,seed=0, X=X[:, 0], y=X[:, 1])
    
    kr = KernelRegression(kernel="rbf", h=0.8, standardize=True).fit(X_train, y_train)

    Xz, mu, sd = zscore(X_train)
    if cdist:
        dists = cdist(*[Xz]*2)
    else:
        dists = np.sqrt(pairwise_sq_dists_numpy(*[Xz]*2))
    
    med = np.median(dists[np.triu_indices(n_train,1)])
    grid = np.logspace(np.log10(med*0.25), np.log10(med*25), 25)

    h_star = kr.select_bandwidth(grid)
    print(f"Selected h = {h_star:.3f} | Final Train MSE={kr.score():.4f} | Final LOO MSE={kr.score(loo=True):.4f}")

    print(f"Test MSE={rmse(y_test, kr.predict(X_test)):.4f}")

    Xq = np.linspace(X_train.min(), X_train.max(), 400)[:, None]
    Xqz = (Xq - mu)/sd
    yq_pred = kr.predict(Xqz)


    plt.figure(figsize=(8, 5))
    plt.plot(Xq[:, 0], yq_pred, label=f"NW (RBF), h={h_star:.3f}")
    plt.scatter(X_train[:, 0], y_train, s=25, color="red", alpha=0.8, label="train")
    plt.scatter(X_test[:, 0], y_test, s=25, color="blue", alpha=0.7, label="test")

    plt.xlabel("x")
    plt.ylabel("y / prediction")
    plt.title(f"Kernel smoother with LOO-selected bandwidth (best LOO MSE = {kr.score(loo=True):.3f})")
    plt.legend()
    plt.show()