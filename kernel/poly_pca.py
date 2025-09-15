# poly_pca.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
from math import lgamma
import itertools
import numpy as np

def sample_data(p: int, n: int, seed: Optional[int] = None) -> np.ndarray: 
    rng = np.random.default_rng(seed) 
    return rng.standard_normal((p, n))

def eigh_sorted(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of a symmetric matrix with eigenvalues sorted descending."""
    w, V = np.linalg.eigh(A)
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]


def degree_d_multi_indices(d: int, p: int) -> List[Tuple[int, ...]]:
    """
    Stars and bars algorithm.
    All nonnegative integer p-tuples (j1,...,jp) that sum to d.
    Returned in reverse lexicographic order.
    """
    result = []
    for dividers in itertools.combinations(range(d + p - 1), p - 1):
        last = -1
        index = []
        for bar in dividers + (d + p - 1,):
            index.append(bar - last - 1)
            last = bar
        result.append(tuple(index))
    return list(reversed(result))


def multinomial_coefficients_sqrt(d: int, j_vec: Tuple[int, ...]) -> float:
    """sqrt( d! / (j1! ... jp!) ) using log-gamma for stability."""
    log_d = lgamma(d + 1)
    log_j = sum(lgamma(j_k + 1) for j_k in j_vec)
    return float(np.sqrt(np.exp(log_d - log_j)))


def empirical_moment(X: np.ndarray, moments_vec: np.ndarray) -> float:
    """Empirical mixed moment: E[ prod_k X_k^{m_k} ] for X shape (p, n)."""
    return float(np.mean(np.prod(X ** moments_vec.reshape(-1, 1), axis=0)))


def feature_matrix(
    X: np.ndarray,
    d: int,
    exponents: List[Tuple[int, ...]],
    bin_sqrts: List[float],
    const: Optional[float] = None,
) -> np.ndarray:
    """
    Explicit polynomial features (degree d with sqrt-multinomial weights).
    If const is not None and nonzero, we treat the last coordinate as fixed sqrt(const).
    Returns Phi with shape (d_p, n).
    """
    p, n = X.shape
    d_p = len(exponents)
    Phi = np.empty((d_p, n), dtype=float)

    has_const = (const is not None) and (const != 0.0)
    sqrt_u = float(const) ** 0.5 if has_const else 0.0

    for r, (j, bin_sqrt) in enumerate(zip(exponents, bin_sqrts)):
        j_arr = np.array(j)
        if has_const:
            jx = j_arr[:-1]                  # powers for the original p features
            jl = int(j_arr[-1])              # power on the fixed last coord
            prod = np.prod(X ** jx.reshape(-1, 1), axis=0) * (sqrt_u ** jl)
        else:
            prod = np.prod(X ** j_arr.reshape(-1, 1), axis=0)
        Phi[r] = bin_sqrt * prod
    return Phi

@dataclass
class KernelCenterer:
    """
    Double-center a Gram/Kernel matrix.

    After .fit(K_train):
      - transform(K_train) returns centered train Gram (n x n)
      - transform(K_cross) returns centered cross-kernel (m x n)
    """
    n_samples_: Optional[int] = None
    col_mean_: Optional[np.ndarray] = None  # (1, n)
    row_mean_: Optional[np.ndarray] = None  # (n, 1)
    grand_mean_: Optional[float] = None

    def fit(self, K: np.ndarray) -> "KernelCenterer":
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("Training Gram matrix must be square (n x n).")
        self.n_samples_ = K.shape[0]
        self.col_mean_ = K.mean(axis=0, keepdims=True)
        self.row_mean_ = K.mean(axis=1, keepdims=True)
        self.grand_mean_ = float(K.mean())
        return self

    def transform(self, K: np.ndarray) -> np.ndarray:
        if self.n_samples_ is None:
            raise RuntimeError("Call fit(...) before transform(...).")
        n = self.n_samples_
        if K.shape == (n, n):
            # training Gram
            return K - self.col_mean_ - self.row_mean_ + self.grand_mean_
        if K.ndim == 2 and K.shape[1] == n:
            # cross-kernel (m x n)
            m = K.shape[0]
            row_mean_star = K.mean(axis=1, keepdims=True)  # (m, 1)
            return (
                K
                - np.ones((m, 1)) @ self.col_mean_
                - row_mean_star @ np.ones((1, n))
                + self.grand_mean_
            )
        raise ValueError(f"Incompatible shape for K: expected (n,n) or (m,{n}), got {K.shape}.")

def moment_matrix(
    d: int,
    p: int,
    X: np.ndarray,
    exponents: List[Tuple[int, ...]],
    const: Optional[float] = None,
) -> np.ndarray:
    """
    M_{rs} = E[ phi_r(X) phi_s(X) ] for degree-d monomials with sqrt multinomial weights.
    If const is provided, the augmented last coordinate equals sqrt(const),
    so moments acquire a factor const^{(last_power_sum)/2}.
    Size: d_p x d_p.
    """
    d_p = len(exponents)
    M = np.zeros((d_p, d_p), dtype=float)

    has_const = (const is not None) and (const != 0.0)
    sqrt_u = float(const) ** 0.5 if has_const else 0.0

    for r, i_vec in enumerate(exponents):
        bin_i = multinomial_coefficients_sqrt(d, i_vec)
        i_vec = np.array(i_vec)
        for s, j_vec in enumerate(exponents):
            bin_j = multinomial_coefficients_sqrt(d, j_vec)
            j_vec = np.array(j_vec)

            if has_const:
                # starred moment: drop last index, multiply by u^{(i_last+j_last)/2}
                moments_vec = i_vec[:-1] + j_vec[:-1]
                last_pow = int(i_vec[-1] + j_vec[-1])
                M[r, s] = bin_i * bin_j * empirical_moment(X, moments_vec) * (sqrt_u ** last_pow)
            else:
                moments_vec = i_vec + j_vec
                M[r, s] = bin_i * bin_j * empirical_moment(X, moments_vec)
    return M

# def moment_matrix(d, p, X, exponents, const=None):
#     # X: (p, n), exponents: array shape (m, q) where q=p or p+1
#     X = np.asarray(X, float)
#     exps = np.asarray(exponents, int)
#     p, n = X.shape
#     m, q = exps.shape

#     has_const = (const is not None) and (const != 0.0)
#     sqrt_u = float(const)**0.5 if has_const else 1.0
#     if has_const:
#         exps_core = exps[:, :-1]
#         exps_last = exps[:, -1]
#     else:
#         exps_core = exps
#         exps_last = np.zeros(m, dtype=int)

#     w = np.array([multinomial_coefficients_sqrt(d, e) for e in exps], dtype=float)

#     t_max = int(exps_core.max())

#     X_pows = np.empty((p, t_max+1, n), dtype=float)
#     X_pows[:, 0, :] = 1.0
#     for t in range(1, t_max+1):
#         X_pows[:, t, :] = X_pows[:, t-1, :] * X

#     Phi = np.ones((m, n), dtype=float)
#     for k in range(p):
#         Phi *= X_pows[k, exps_core[:, k], :]

#     Phi *= w[:, None] * (sqrt_u ** exps_last)[:, None]

#     return (Phi @ Phi.T) / float(n)

class MomentPolynomialPCA:
    """
    Moment-based polynomial PCA with proper feature-space centering.

    Data orientation: X has shape (p, n) — features n samples.

    Parameters
    ----------
    degree : int
        Polynomial degree d.
    const : Optional[float]
        If None or 0.0 → homogeneous polynomial (no constant).
        If u (e.g. 5.0) → use (x^T y + u)^degree via an augmented fixed coord sqrt(u).
    n_components : Optional[int]
        Number of components to keep (defaults to all positive eigenpairs).
    tol : float
        Eigenvalue threshold; <= tol are dropped.
    score_norm : {'eig', 'sqrt', 'none'}
        How to scale scores.
    """

    def __init__(
        self,
        degree: int,
        n_components: Optional[int] = None,
        tol: float = 1e-12,
        score_norm: str = "eig",
        const: Optional[float] = None,
    ):
        self.degree = int(degree)
        self.n_components = n_components
        self.tol = float(tol)
        self.score_norm = score_norm
        self.const = const

        # Fitted attributes
        self.p_: Optional[int] = None
        self.n_: Optional[int] = None
        self.exponents_: Optional[List[Tuple[int, ...]]] = None
        self.bin_sqrts_: Optional[List[float]] = None
        self.M_: Optional[np.ndarray] = None
        self.m_feat_: Optional[np.ndarray] = None  # (d_p,)
        self.eigvals_: Optional[np.ndarray] = None
        self.eigvecs_: Optional[np.ndarray] = None
        self.scores_: Optional[np.ndarray] = None  # (m, n)

    def _scale_scores(self, VtPhi: np.ndarray) -> np.ndarray:
        if self.score_norm == "eig":
            denom = self.eigvals_[:, None]
        elif self.score_norm == "sqrt":
            denom = np.sqrt(self.eigvals_)[:, None]
        elif self.score_norm == "none":
            denom = 1.0
        else:
            raise ValueError("score_norm must be {'eig','sqrt','none'}.")
        return VtPhi / denom

    def fit(self, X: np.ndarray) -> "MomentPolynomialPCA":
        if X.ndim != 2:
            raise ValueError("X must have shape (p, n).")
        self.p_, self.n_ = X.shape

        p_eff = self.p_ + (1 if (self.const is not None and self.const != 0.0) else 0)

        # Degree-d feature spec
        self.exponents_ = degree_d_multi_indices(self.degree, p_eff)
        self.bin_sqrts_ = [multinomial_coefficients_sqrt(self.degree, j) for j in self.exponents_]

        # Raw moment matrix
        self.M_ = moment_matrix(self.degree, self.p_, X, self.exponents_, const=self.const)

        # Feature means m_r = E[phi_r(X)]
        has_const = (self.const is not None) and (self.const != 0.0)
        sqrt_u = float(self.const) ** 0.5 if has_const else 0.0
        m_vals = []
        for j, b in zip(self.exponents_, self.bin_sqrts_):
            arr = np.array(j)
            if has_const:
                jl = int(arr[-1])
                jx = arr[:-1]
                val = b * empirical_moment(X, jx) * (sqrt_u ** jl)
            else:
                val = b * empirical_moment(X, arr)
            m_vals.append(val)
        self.m_feat_ = np.array(m_vals)

        # Centered feature covariance
        C_phi = self.M_ - np.outer(self.m_feat_, self.m_feat_)

        eigvals, eigvecs = eigh_sorted(C_phi)
        keep = eigvals > self.tol
        eigvals, eigvecs = eigvals[keep], eigvecs[:, keep]

        if self.n_components is not None:
            m = min(self.n_components, eigvals.size)
            eigvals, eigvecs = eigvals[:m], eigvecs[:, :m]

        self.eigvals_, self.eigvecs_ = eigvals, eigvecs

        # Training scores
        Phi = feature_matrix(X, self.degree, self.exponents_, self.bin_sqrts_, const=self.const)
        Phi_c = Phi - self.m_feat_[:, None]
        self.scores_ = self._scale_scores(self.eigvecs_.T @ Phi_c)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.eigvals_ is None or self.eigvecs_ is None:
            raise RuntimeError("Call fit(...) before transform(...).")
        if X.ndim != 2 or X.shape[0] != self.p_:
            raise ValueError(f"X must have shape (p={self.p_}, n_new).")
        Phi = feature_matrix(X, self.degree, self.exponents_, self.bin_sqrts_, const=self.const)
        Phi_c = Phi - self.m_feat_[:, None]
        return self._scale_scores(self.eigvecs_.T @ Phi_c)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).scores_
    


class PolynomialKernelPCA:
    """
    KPCA with polynomial kernel: k(x, y) = (x^T y + const)^degree.

    Data orientation: X has shape (p, n).

    Parameters
    ----------
    degree : int
    const : Optional[float]
        None/0.0 → (x^T y)^degree.  Otherwise (x^T y + const)^degree.
    n_components : Optional[int]
    center_kernel : bool
        Double-center the Gram matrix (recommended=True).
    tol : float
    score_norm : {'eig','sqrt','none'}
        Score scaling.
    divide_eigs_by_n : bool
        If True, divide eigenvalues by n (helps matching moment-PCA scaling).
    """

    def __init__(
        self,
        degree: int,
        const: Optional[float] = None,
        n_components: Optional[int] = None,
        center_kernel: bool = True,
        tol: float = 1e-12,
        score_norm: str = "eig",
        divide_eigs_by_n: bool = False,
    ):
        self.degree = int(degree)
        self.const = const
        self.n_components = n_components
        self.center_kernel = center_kernel
        self.tol = float(tol)
        self.score_norm = score_norm
        self.divide_eigs_by_n = divide_eigs_by_n

        # Fitted attributes
        self.p_: Optional[int] = None
        self.n_: Optional[int] = None
        self.X_train_: Optional[np.ndarray] = None
        self.centerer_: Optional[KernelCenterer] = None
        self.Kc_: Optional[np.ndarray] = None
        self.eigvals_: Optional[np.ndarray] = None
        self.eigvecs_: Optional[np.ndarray] = None
        self.scores_: Optional[np.ndarray] = None

    def _gram(self, X: np.ndarray) -> np.ndarray:
        c = 0.0 if (self.const is None) else float(self.const)
        return (X.T @ X + c) ** self.degree

    def _cross_gram(self, X_new: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        c = 0.0 if (self.const is None) else float(self.const)
        return (X_new.T @ X_train + c) ** self.degree

    def _scale_scores(self, VtK: np.ndarray) -> np.ndarray:
        if self.score_norm == "eig":
            denom = self.eigvals_[:, None]
        elif self.score_norm == "sqrt":
            denom = np.sqrt(self.eigvals_)[:, None]
        elif self.score_norm == "none":
            denom = 1.0
        else:
            raise ValueError("score_norm must be {'eig','sqrt','none'}.")
        return VtK / denom

    def fit(self, X: np.ndarray) -> "PolynomialKernelPCA":
        if X.ndim != 2:
            raise ValueError("X must have shape (p, n).")
        self.p_, self.n_ = X.shape
        self.X_train_ = X.copy()

        K = self._gram(self.X_train_)  # (n, n)
        if self.center_kernel:
            self.centerer_ = KernelCenterer().fit(K)
            Kc = self.centerer_.transform(K)
        else:
            self.centerer_ = None
            Kc = K

        eigvals, eigvecs = eigh_sorted(Kc)
        if self.divide_eigs_by_n:
            eigvals = eigvals / self.n_

        keep = eigvals > self.tol
        eigvals, eigvecs = eigvals[keep], eigvecs[:, keep]

        if self.n_components is not None:
            m = min(self.n_components, eigvals.size)
            eigvals, eigvecs = eigvals[:m], eigvecs[:, :m]

        self.Kc_ = Kc
        self.eigvals_, self.eigvecs_ = eigvals, eigvecs

        # Training scores
        self.scores_ = self._scale_scores(self.eigvecs_.T @ Kc)
        return self

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        if self.eigvals_ is None or self.eigvecs_ is None:
            raise RuntimeError("Call fit(...) before transform(...).")
        if X_new.ndim != 2 or X_new.shape[0] != self.p_:
            raise ValueError(f"X_new must have shape (p={self.p_}, m).")

        K_star = self._cross_gram(X_new, self.X_train_)  # (m, n)
        if self.center_kernel and self.centerer_ is not None:
            Kc_star = self.centerer_.transform(K_star)
        else:
            Kc_star = K_star

        return self._scale_scores(self.eigvecs_.T @ Kc_star)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).scores_

def T_statistic(scores: np.ndarray) -> np.ndarray: 
    """Sum of squares across components per sample (m, n) -> (n,).""" 
    return np.sum(scores ** 2, axis=0)

def Q_statistic(scores: np.ndarray, m: int) -> np.ndarray: 
    """Residual variance after first m components.""" 
    total_var = np.sum(scores ** 2, axis=0) 
    retained_var = np.sum(scores[:m, :] ** 2, axis=0) 
    return total_var - retained_var

if __name__ == "__main__":
    p, n, d = 2, 10, 2
    rng = np.random.default_rng(42)
    X = rng.standard_normal((p, n))
    kpca = PolynomialKernelPCA(
        degree=d,
        const=None,
        n_components=3,
        center_kernel=True,
        score_norm="eig",
        divide_eigs_by_n=True,
    )
    Z_k = kpca.fit_transform(X)
    print("KPCA scores (train):\n", Z_k)


    mpca = MomentPolynomialPCA(
        degree=d,
        n_components=3,
        tol=1e-12,
        score_norm="eig",
        const=None,
    )
    Z_m = mpca.fit_transform(X)
    print("Moment Poly-PCA scores (train):\n", Z_m)

    # Optional: compare element-wise ratios
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = Z_k / Z_m
    print("Element-wise ratio (KPCA / Moment-PCA):\n", ratio)
