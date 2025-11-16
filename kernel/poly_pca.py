from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import itertools
from math import lgamma

Array = np.ndarray

def rbf_kernel(Z1: Array, Z2: Array, sigma: float = 1.0) -> Array:
    """RBF kernel: k(z,z') = exp(-||z-z'||^2 / (2 sigma^2))"""
    Z1_sq = np.sum(Z1**2, axis=1, keepdims=True)
    Z2_sq = np.sum(Z2**2, axis=1, keepdims=True).T
    cross = Z1 @ Z2.T
    dist2 = Z1_sq + Z2_sq - 2.0 * cross
    return np.exp(-dist2 / (2.0 * sigma * sigma))

def poly_kernel(Z1: Array, Z2: Array, degree: int = 2, c0: float = 0.0) -> Array:
    """Polynomial kernel: k(z,z') = (z.z' + c0)^degree"""
    return (Z1 @ Z2.T + c0) ** degree

@dataclass
class KernelConfig:
    kind: str
    params: Dict[str, float]

    def kernel_fn(self) -> Callable[[Array, Array], Array]:
        if self.kind == "rbf":
            sigma = float(self.params.get("sigma", 1.0))
            return lambda A, B: rbf_kernel(A, B, sigma=sigma)
        elif self.kind == "poly":
            degree = int(self.params.get("degree", 2))
            c0 = float(self.params.get("c0", 0.0))
            return lambda A, B: poly_kernel(A, B, degree=degree, c0=c0)
        else:
            raise ValueError(f"Unknown kernel kind: {self.kind}")

@dataclass
class KPCAResult:
    """Result object for the Kernel (Gram) PCA method."""
    Z_train: Array
    K: Array
    Kc: Array
    Q: Array
    lambdas: Array
    A_m: Array
    bar_k: Array
    bar_K: float
    m: int
    kernel_cfg: KernelConfig

def center_gram(K: Array) -> Tuple[Array, Array, float]:
    """Center a Gram matrix K."""
    n = K.shape[0]
    ones_n = np.ones((n, n)) / n
    bar_k = K.mean(axis=1)
    bar_K = K.mean()
    Kc = K - ones_n @ K - K @ ones_n + ones_n @ K @ ones_n
    return Kc, bar_k, bar_K

def kpca_fit(Z: Array,
             kernel_cfg: KernelConfig,
             m: Optional[int] = None,
             evr_target: Optional[float] = None) -> KPCAResult:
    """
    Fits Kernel PCA (Gram matrix method).
    Expects Z in (n_samples, n_features) format.
    """
    Z = np.asarray(Z, float)
    n, p = Z.shape
    kfun = kernel_cfg.kernel_fn()
    K = kfun(Z, Z)

    Kc, bar_k, bar_K = center_gram(K)

    eigvals, eigvecs = np.linalg.eigh(Kc)
    idx = np.argsort(eigvals)[::-1] 
    lambdas = eigvals[idx]
    Q = eigvecs[:, idx]

    eps = 1e-12
    pos = lambdas > eps
    lambdas_pos = lambdas[pos]
    Q_pos = Q[:, pos]
    m_eff: int
    if evr_target is not None:
        csum = np.cumsum(lambdas_pos)
        total = csum[-1] if csum.size > 0 else 0.0
        if total <= 0:
            m_eff = 0
        else:
            m_eff = int(np.searchsorted(csum / total, evr_target) + 1)
        m_eff = max(0, min(m_eff, lambdas_pos.size))
    else:
        m_eff = m if m is not None else min(10, lambdas_pos.size)

    if m_eff == 0:
        A_m = np.zeros((n, 0))
    else:
        A_m = Q_pos[:, :m_eff] / np.sqrt(lambdas_pos[:m_eff])

    return KPCAResult(Z_train=Z, K=K, Kc=Kc, Q=Q, lambdas=lambdas,
                      A_m=A_m, bar_k=bar_k, bar_K=bar_K, m=m_eff, kernel_cfg=kernel_cfg)

def eigh_sorted(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of a symmetric matrix with eigenvalues sorted descending."""
    w, V = np.linalg.eigh(A)
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]

def degree_d_multi_indices(d: int, p: int) -> List[Tuple[int, ...]]:
    """Stars and bars algorithm for p-tuples summing to d."""
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

def empirical_moment(X_np: np.ndarray, moments_vec: np.ndarray) -> float:
    """E[ prod_k X_k^{m_k} ] for X_np shape (n, p)."""
    return float(np.mean(np.prod(X_np ** moments_vec.reshape(1, -1), axis=1)))

def feature_matrix(
    X_np: np.ndarray,
    d: int,
    exponents: List[Tuple[int, ...]],
    bin_sqrts: List[float],
    const: Optional[float] = None,
) -> np.ndarray:
    """Explicit polynomial features. (n, p) -> (d_p, n)."""
    X = X_np.T # Transpose to (p, n) for logic
    p, n = X.shape
    d_p = len(exponents)
    Phi = np.empty((d_p, n), dtype=float)

    has_const = (const is not None) and (const != 0.0)
    sqrt_u = float(const) ** 0.5 if has_const else 0.0

    for r, (j, bin_sqrt) in enumerate(zip(exponents, bin_sqrts)):
        j_arr = np.array(j)
        if has_const:
            jx = j_arr[:-1]
            jl = int(j_arr[-1])
            prod = np.prod(X ** jx.reshape(-1, 1), axis=0) * (sqrt_u ** jl)
        else:
            prod = np.prod(X ** j_arr.reshape(-1, 1), axis=0)
        Phi[r] = bin_sqrt * prod
    return Phi

def moment_matrix(
    d: int,
    p: int,
    X_np: np.ndarray,
    exponents: List[Tuple[int, ...]],
    const: Optional[float] = None,
) -> np.ndarray:
    """M_{rs} = E[ phi_r(X) phi_s(X) ] for X_np (n, p)."""
    d_p = len(exponents)
    M = np.zeros((d_p, d_p), dtype=float)

    has_const = (const is not None) and (const != 0.0)
    sqrt_u = float(const) ** 0.5 if has_const else 0.0

    for r, i_vec in enumerate(exponents):
        bin_i = multinomial_coefficients_sqrt(d, i_vec)
        i_vec_arr = np.array(i_vec)
        for s, j_vec in enumerate(exponents):
            bin_j = multinomial_coefficients_sqrt(d, j_vec)
            j_vec_arr = np.array(j_vec)

            if has_const:
                moments_vec = i_vec_arr[:-1] + j_vec_arr[:-1]
                last_pow = int(i_vec_arr[-1] + j_vec_arr[-1])
                M[r, s] = bin_i * bin_j * empirical_moment(X_np, moments_vec) * (sqrt_u ** last_pow)
            else:
                moments_vec = i_vec_arr + j_vec_arr
                M[r, s] = bin_i * bin_j * empirical_moment(X_np, moments_vec)
    return M

@dataclass
class MomentPolynomialPCA:
    """Moment-based polynomial PCA. Expects X in (n, p) format."""
    degree: int
    const: Optional[float] = None
    n_components: Optional[int] = None
    tol: float = 1e-12
    score_norm: str = "eig"
    Z_train_: Optional[Array] = None
    evr_target: Optional[float] = 0.95
    # Fitted attributes
    p_: Optional[int] = None
    n_: Optional[int] = None
    exponents_: Optional[List[Tuple[int, ...]]] = None
    bin_sqrts_: Optional[List[float]] = None
    M_: Optional[np.ndarray] = None
    m_feat_: Optional[np.ndarray] = None
    eigvals_: Optional[np.ndarray] = None
    eigvecs_: Optional[np.ndarray] = None
    scores_: Optional[np.ndarray] = None
    m: int = 0

    def _scale_scores(self, VtPhi: np.ndarray) -> np.ndarray:
        if self.eigvals_ is None: return VtPhi
        if self.score_norm == "eig": denom = self.eigvals_[:, None]
        elif self.score_norm == "sqrt": denom = np.sqrt(self.eigvals_)[:, None]
        elif self.score_norm == "none": denom = 1.0
        else: raise ValueError("score_norm must be {'eig','sqrt','none'}.")
        return VtPhi / denom

    def fit(self, X_np: np.ndarray) -> "MomentPolynomialPCA":
        if X_np.ndim != 2:
            raise ValueError("X must have shape (n, p).")
        self.n_, self.p_ = X_np.shape
        
        self.Z_train_ = X_np

        p_eff = self.p_ + (1 if (self.const is not None and self.const != 0.0) else 0)

        self.exponents_ = degree_d_multi_indices(self.degree, p_eff)
        self.bin_sqrts_ = [multinomial_coefficients_sqrt(self.degree, j) for j in self.exponents_]

        self.M_ = moment_matrix(self.degree, self.p_, X_np, self.exponents_, const=self.const)

        has_const = (self.const is not None) and (self.const != 0.0)
        sqrt_u = float(self.const) ** 0.5 if has_const else 0.0
        m_vals = []
        for j, b in zip(self.exponents_, self.bin_sqrts_):
            arr = np.array(j)
            if has_const:
                jl = int(arr[-1])
                jx = arr[:-1]
                val = b * empirical_moment(X_np, jx) * (sqrt_u ** jl)
            else:
                val = b * empirical_moment(X_np, arr)
            m_vals.append(val)
        self.m_feat_ = np.array(m_vals)

        C_phi = self.M_ - np.outer(self.m_feat_, self.m_feat_)

        eigvals, eigvecs = eigh_sorted(C_phi)
        keep = eigvals > self.tol
        eigvals, eigvecs = eigvals[keep], eigvecs[:, keep]

        m_eff: int
        if self.n_components is not None:
            m_eff = min(self.n_components, eigvals.size)
        elif self.evr_target is not None:
            csum = np.cumsum(eigvals)
            total = csum[-1] if csum.size > 0 else 0.0
            if total <= 0:
                m_eff = 0
            else:
                m_eff = int(np.searchsorted(csum / total, self.evr_target) + 1)
            m_eff = max(0, min(m_eff, eigvals.size))
        else:
            m_eff = eigvals.size
        
        self.m = m_eff
        self.eigvals_, self.eigvecs_ = eigvals[:m_eff], eigvecs[:, :m_eff]

        Phi = feature_matrix(X_np, self.degree, self.exponents_, self.bin_sqrts_, const=self.const)
        Phi_c = Phi - self.m_feat_[:, None]
        self.scores_ = self._scale_scores(self.eigvecs_.T @ Phi_c)
        return self

    def transform(self, X_np: np.ndarray) -> np.ndarray:
        if self.eigvals_ is None:
            raise RuntimeError("Call fit(...) before transform(...).")
        if X_np.ndim != 2 or X_np.shape[1] != self.p_:
            raise ValueError(f"X must have shape (n_new, p={self.p_}).")
        
        Phi = feature_matrix(X_np, self.degree, self.exponents_, self.bin_sqrts_, const=self.const)
        Phi_c = Phi - self.m_feat_[:, None]
        return self._scale_scores(self.eigvecs_.T @ Phi_c)