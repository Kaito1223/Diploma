import numpy as np
from typing import Optional, Tuple, List
from scipy.linalg import solve
from make_dataset import make_data_separated
import matplotlib.pyplot as plt
import itertools
try:
    from scipy.spatial.distance import cdist
except ImportError:
    cdist = None

def _multi_index_tuples(d: int, max_total_degree: int):
    result = [(0,)*d]
    while max_total_degree > 0:
        for dividers in itertools.combinations(range(d + max_total_degree - 1), d - 1):
            last = -1
            index = []
            for bar in dividers + (d + max_total_degree - 1,):
                index.append(bar - last - 1)
                last = bar
            result.append(tuple(index))
        max_total_degree -= 1
    return list(reversed(result))

def _poly_features(X, alphas):
    """Vectorized polynomial feature generation."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, d = X.shape
    T = np.empty((n, len(alphas)), dtype=float)
    for j, alpha in enumerate(alphas):
        T[:, j] = np.prod(X ** alpha, axis=1)
    return T

def _phi_polyharmonic(r, k, eps=1e-15):
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    rk = np.power(r, k, where=(r > 0), out=np.zeros_like(r))
    if k % 2 == 1:
        out = rk
    else:
        out = rk * np.log(r + eps)
        out[r == 0] = 0.0
    return out

def _pairwise_dists(X, Y):
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    X2 = np.sum(X * X, axis=1)[:, None]
    Y2 = np.sum(Y * Y, axis=1)[None, :]
    D2 = np.maximum(X2 + Y2 - 2.0 * (X @ Y.T), 0.0)
    return np.sqrt(D2, out=D2)


class TPSRegressor:
    def __init__(self, lambda_=1.0, m=2, include_polynomial=True, standardize=False, jitter=1e-12):
        self.lambda_ = float(lambda_)
        self.m = int(m)
        self.include_polynomial = bool(include_polynomial)
        self.standardize = bool(standardize)
        self.jitter = float(jitter)
        self.X_ = None
        self.c_ = None
        self.beta_ = None
        self.alphas_ = None
        self.k_ = None
        self.x_mean_ = None
        self.x_scale_ = None
        self.d_ = None
        self.q_ = None
        self.n_ = None
        self.K_ = None

    def _check_shapes(self, X, y, sample_weight):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X[:, None]
        y = np.asarray(y, dtype=float)
        if y.ndim == 1: y = y[:, None]
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=float).reshape(-1)
            if w.shape[0] != X.shape[0]:
                raise ValueError("sample_weight must be length n_samples.")
        else:
            w = None
        return X, y, w

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        X, y, w = self._check_shapes(X, y, sample_weight)
        n, d = X.shape
        self.n_, self.d_ = n, d

        if self.standardize:
            self.x_mean_ = X.mean(axis=0, keepdims=True)
            self.x_scale_ = X.std(axis=0, keepdims=True)
            self.x_scale_ = np.where(self.x_scale_ == 0, 1.0, self.x_scale_)
            Xs = (X - self.x_mean_) / self.x_scale_
        else:
            self.x_mean_ = np.zeros((1, d))
            self.x_scale_ = np.ones((1, d))
            Xs = X.copy()
        self.X_ = Xs

        k = 2 * self.m - d
        if k <= 0:
            raise ValueError(f"Order m={self.m} too small for d={d}; require m > d/2 so k=2m-d>0.")
        self.k_ = k

        if cdist is not None:
            r = cdist(Xs, Xs, metric='euclidean')
        else:
            r = _pairwise_dists(Xs, Xs)

        K = _phi_polyharmonic(r, k)
        K[np.arange(n), np.arange(n)] += self.jitter
        self.K_ = K
        if self.include_polynomial:
            self.alphas_ = _multi_index_tuples(d, self.m - 1)
            T = _poly_features(Xs, self.alphas_)
            self.q_ = T.shape[1]
        else:
            self.alphas_ = []
            T = np.zeros((n, 0), dtype=float)
            self.q_ = 0

        W = np.ones(n) if w is None else w

        A = np.empty((n + self.q_, n + self.q_), dtype=float)
        A[:n, :n] = (W[:, None] * self.K_) + self.lambda_ * np.eye(n)
        A[:n, n:] = (W[:, None] * T)
        A[n:, :n] = T.T
        A[n:, n:] = 0.0

        rhs = np.vstack([(W[:, None] * y), np.zeros((self.q_, y.shape[1]))])

        try:
            sol = solve(A, rhs, assume_a='sym')
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)

        self.c_ = sol[:n, :]
        self.beta_ = sol[n:, :] if self.q_ > 0 else None
        return self

    def _build_design_new(self, Xnew):
        Xnew = np.asarray(Xnew, dtype=float)
        if Xnew.ndim == 1: Xnew = Xnew[:, None]
        
        Xs = (Xnew - self.x_mean_) / self.x_scale_

        if cdist is not None:
            r = cdist(Xs, self.X_, metric='euclidean')
        else:
            r = _pairwise_dists(Xs, self.X_)

        Knew = _phi_polyharmonic(r, self.k_)
        
        if self.q_ and self.beta_ is not None:
            Tnew = _poly_features(Xs, self.alphas_)
        else:
            Tnew = np.zeros((Xs.shape[0], 0), dtype=float)
        return Knew, Tnew

    def predict(self, Xnew):
        if self.X_ is None:
            raise RuntimeError("Call fit() before predict().")
        Knew, Tnew = self._build_design_new(Xnew)
        yhat = Knew @ self.c_
        if self.q_ and self.beta_ is not None:
            yhat += Tnew @ self.beta_
        return yhat.squeeze()


    def __repr__(self):
        return (f"TPSRegressor(lambda_={self.lambda_}, m={self.m}, "
                f"include_polynomial={self.include_polynomial}, standardize={self.standardize})")
    
if __name__ == '__main__':
    X_train, y_train, X_test, y_test = make_data_separated(
        n_train=500,
        n_test=80,
        n_features=1,
        seed=42,
        distribution='uniform',
        dist_params={'low': -np.pi, 'high': np.pi}
    )

    lambda_reg = 9
    print(f"Fitting TPS Regressor with lambda = {lambda_reg}...")
    model = TPSRegressor(lambda_=lambda_reg, m=2, standardize=True)
    model.fit(X_train, y_train)

    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred)**2)
    print(f"  -> Mean Squared Error on Test Set: {mse:.4f}")

    print("Creating 2D visualization...")
    fig, ax = plt.subplots(figsize=(10, 6))

    x_grid = np.linspace(X_train.min(), X_train.max(), 500).reshape(-1, 1)
    
    y_smooth = model.predict(x_grid)

    ax.plot(x_grid, y_smooth, label='TPS Smoothed Curve', color='green', linewidth=2.5, zorder=3)

    ax.scatter(X_train, y_train, label='Training Data', color='blue', alpha=0.7, zorder=2)
    ax.scatter(X_test, y_test, label='Test Data', color='red', marker='x', s=50, zorder=2)
    
    ax.set_xlabel('$X$')
    ax.set_ylabel('$y$')
    ax.set_title('Thin Plate Spline Smoothing in 1D')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()