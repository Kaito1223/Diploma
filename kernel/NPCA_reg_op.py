from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import math
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

Array = np.ndarray


def rbf_kernel(Z1: Array, Z2: Array, sigma: float = 1.0) -> Array:
    """RBF kernel on joint vectors Z; k(z,z') = exp(-||z-z'||^2 / (2 sigma^2))"""
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    Z1_sq = np.sum(Z1**2, axis=1, keepdims=True)  # (n1,1)
    Z2_sq = np.sum(Z2**2, axis=1, keepdims=True).T  # (1,n2)
    cross = Z1 @ Z2.T
    dist2 = Z1_sq + Z2_sq - 2.0 * cross
    K = np.exp(-dist2 / (2.0 * sigma * sigma))
    return K


def poly_kernel(Z1: Array, Z2: Array, degree: int = 2, c0: float = 0.0) -> Array:
    """Polynomial kernel: k(z,z') = (z.z' + c0)^degree"""
    return (Z1 @ Z2.T + c0) ** degree


@dataclass
class KernelConfig:
    kind: str  # "rbf" or "poly"
    params: Dict[str, float]  # e.g., {"sigma": 0.5} or {"degree": 3, "c0": 0.0}

    def kernel_fn(self) -> Callable[[Array, Array], Array]:
        if self.kind == "rbf":
            sigma = float(self.params.get("sigma", 1.0))
            return lambda A, B: rbf_kernel(A, B, sigma=sigma)
        elif self.kind == "poly":
            degree = int(self.params.get("degree", 2))
            # default c0 = 0.0 per your spec
            c0 = float(self.params.get("c0", 0.0))
            return lambda A, B: poly_kernel(A, B, degree=degree, c0=c0)
        else:
            raise ValueError(f"Unknown kernel kind: {self.kind}")


@dataclass
class KPCAResult:
    # Stored training info
    Z_train: Array                   # (n, p)
    mean_Z: Array                    # (p,)
    K: Array                         # uncentered (n, n)
    Kc: Array                        # centered (n, n)
    Q: Array                         # eigenvectors (n, n)
    lambdas: Array                   # eigenvalues (n,)
    A_m: Array                       # (n, m) columns q_j/sqrt(lambda_j)
    bar_k: Array                     # per-row mean of K (n,)
    bar_K: float                     # global mean of K
    m: int                           # retained components
    kernel_cfg: KernelConfig         # kernel config


def center_gram(K: Array) -> Tuple[Array, Array, float]:
    """Center a Gram matrix: Kc = K - 1K - K1 + 1K1, with 1 = (1/n)11^T.
    Returns Kc, per-row mean bar_k (n,), and global mean bar_K (scalar)."""
    n = K.shape[0]
    ones_n = np.ones((n, n)) / n
    bar_k = K.mean(axis=1)  # (n,)
    bar_K = K.mean()
    Kc = K - ones_n @ K - K @ ones_n + ones_n @ K @ ones_n
    return Kc, bar_k, bar_K


def kpca_fit(Z: Array,
             kernel_cfg: KernelConfig,
             m: Optional[int] = None,
             evr_target: Optional[float] = None) -> KPCAResult:
    """Fit KPCA on joint data Z (n,p).
       - Choose m directly OR via evr_target (explained variance ratio in feature space)."""
    Z = np.asarray(Z, float)
    n, p = Z.shape
    kfun = kernel_cfg.kernel_fn()
    K = kfun(Z, Z)  # (n,n)

    Kc, bar_k, bar_K = center_gram(K)

    eigvals, eigvecs = np.linalg.eigh(Kc)
    idx = np.argsort(eigvals)[::-1] 
    lambdas = eigvals[idx]
    Q = eigvecs[:, idx]

    # Decide m
    eps = 1e-12
    pos = lambdas > eps
    lambdas_pos = lambdas[pos]
    Q_pos = Q[:, pos]
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

    # Build A_m = [ q_j / sqrt(lambda_j) ]
    if m_eff == 0:
        A_m = np.zeros((n, 0))
    else:
        A_m = Q_pos[:, :m_eff] / np.sqrt(lambdas_pos[:m_eff])

    mean_Z = Z.mean(axis=0)
    return KPCAResult(Z_train=Z, mean_Z=mean_Z, K=K, Kc=Kc, Q=Q, lambdas=lambdas,
                      A_m=A_m, bar_k=bar_k, bar_K=bar_K, m=m_eff, kernel_cfg=kernel_cfg)



def centered_kernel_row(model: KPCAResult, z: Array) -> Tuple[Array, float]:
    """Compute centered kernel row k_c(z) and k_c(z,z) vs training set."""
    Ztr = model.Z_train
    kfun = model.kernel_cfg.kernel_fn()
    k_row = kfun(z[None, :], Ztr).ravel()  # (n,)
    row_mean = k_row.mean()
    k_c_row = k_row - row_mean - model.bar_k + model.bar_K

    # self term
    k_self = kfun(z[None, :], z[None, :])[0, 0]
    k_c_self = k_self - 2 * row_mean + model.bar_K
    return k_c_row, k_c_self


def kpca_scores(model: KPCAResult, z: Array) -> Array:
    """Scores t(z) in feature space of size m."""
    if model.m == 0:
        return np.zeros((0,))
    k_c_row, _ = centered_kernel_row(model, z)
    t = model.A_m.T @ k_c_row 
    return t


def kpca_feature_error(model: KPCAResult, z: Array) -> float:
    """Feature-space orthogonal squared error: E_phi(z) = k_c(z,z) - ||t(z)||^2."""
    if model.m == 0:
        _, k_c_self = centered_kernel_row(model, z)
        return float(k_c_self)
    k_c_row, k_c_self = centered_kernel_row(model, z)
    t = model.A_m.T @ k_c_row
    return float(k_c_self - np.dot(t, t))


def kpca_train_mse_feature(model: KPCAResult) -> float:
    """Training mean feature-space error using eigen shortcut:
       E_phi(z_i) = sum_{j>m} lambda_j * Q_{ij}^2"""
    n = model.Kc.shape[0]
    if model.m == 0:
        return float(model.lambdas.mean())
    residual_lambdas = model.lambdas[model.m:]
    Q_res = model.Q[:, model.m:]
    # Per-point errors: sum_j residual_lambda_j * Q_{ij}^2
    E_per = (Q_res**2) @ residual_lambdas
    return float(E_per.mean())


def kpca_test_mse_feature(model: KPCAResult, Z_test: Array) -> float:
    errs = [kpca_feature_error(model, z) for z in Z_test]
    return float(np.mean(errs))


def evaluate_kernels(Z_train: Array,
                     Z_test: Optional[Array] = None,
                     kernel_choice: str = "all",
                     poly_degrees: Union[str, List[int]] = "all",
                     rbf_sigmas: Union[str, List[float]] = [0.25, 0.5, 1.0, 2.0],
                     m: Optional[int] = None,
                     evr_target: Optional[float] = 0.95) -> pd.DataFrame:
    """
    Evaluate multiple kernels/configs and return a DataFrame of feature-space MSEs.
    - kernel_choice: "rbf", "poly", or "all"
    - poly_degrees: list of ints or "all" -> [2,3,4]
    - rbf_sigmas: list of floats or "all" -> [0.25, 0.5, 1.0, 2.0]
    - choose m directly OR via evr_target (explained variance in feature space)
    """
    if poly_degrees == "all":
        poly_degrees = [1, 2, 3, 4]
    if rbf_sigmas == "all":
        rbf_sigmas = [0.25, 0.5, 1.0, 2.0]

    configs: List[KernelConfig] = []
    if kernel_choice in ("poly", "all"):
        for d in poly_degrees:  # c0 fixed to 0.0 per your request
            configs.append(KernelConfig(kind="poly", params={"degree": int(d), "c0": 0.0}))
    if kernel_choice in ("rbf", "all"):
        for s in rbf_sigmas:
            configs.append(KernelConfig(kind="rbf", params={"sigma": float(s)}))

    rows = []
    for cfg in configs:
        model = kpca_fit(Z_train, cfg, m=m, evr_target=evr_target)
        train_mse = kpca_train_mse_feature(model)
        if Z_test is not None:
            test_mse = kpca_test_mse_feature(model, Z_test)
        else:
            test_mse = np.nan
        rows.append({
            "kernel": cfg.kind,
            "params": cfg.params,
            "m": model.m,
            "train_feature_MSE": train_mse,
            "test_feature_MSE": test_mse
        })

    df = pd.DataFrame(rows)
    return df



def make_synthetic_xy(n_train: int = 80, n_test: int = 80, seed: int = 0) -> Tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    n_total = int(n_train) + int(n_test)

    x = np.linspace(-2*np.pi, 2*np.pi, n_total)
    y = np.sin(x) + 0.05 * np.random.randn(n_total)
    Z = np.column_stack((x, y))

    idx = rng.permutation(n_total)
    Z = Z[idx]
    split = int(0.8 * n_total)
    Z_train = Z[:split]
    Z_test = Z[split:]
    return Z_train, Z_test


def _torch_kernel_from_cfg(cfg: KernelConfig):
    if cfg.kind == "rbf":
        sigma = float(cfg.params.get("sigma", 1.0))
        inv2sigma2 = 1.0 / (2.0 * sigma * sigma)
        def k(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            A_sq = (A*A).sum(dim=1, keepdims=True)
            B_sq = (B*B).sum(dim=1, keepdims=True).T
            dist2 = A_sq + B_sq - 2.0 * (A @ B.T)
            return torch.exp(-dist2 * inv2sigma2)
        return k
    elif cfg.kind == "poly":
        degree = int(cfg.params.get("degree", 2))
        c0 = float(cfg.params.get("c0", 0.0))
        def k(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            return (A @ B.T + c0) ** degree
        return k
    else:
        raise ValueError("Unknown kernel kind for torch")


class TorchProjector:
    """Optimize y for given x by minimizing feature-space residual E_phi(x,y)."""
    def __init__(self, model: KPCAResult, device: Optional[str] = None):
        self.model = model
        self.device = torch.device(device) if device else torch.device('cpu')

        self.Ztr = torch.tensor(model.Z_train, dtype=torch.float32, device=self.device)   # (n,p)
        self.A_m = torch.tensor(model.A_m, dtype=torch.float32, device=self.device)       # (n,m)
        self.bar_k = torch.tensor(model.bar_k, dtype=torch.float32, device=self.device)   # (n,)
        self.bar_K = torch.tensor([model.bar_K], dtype=torch.float32, device=self.device) # (1,)
        self.k_torch = _torch_kernel_from_cfg(model.kernel_cfg)

        self.n = self.Ztr.shape[0]
        self.p = self.Ztr.shape[1]
        self.m = self.A_m.shape[1]

        self.mean_y = float(model.Z_train[:, -1].mean())

    def E_phi(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """E_phi(x,y) = k_c(z,z) - ||A_m^T k_c(z)||^2,  z=(x,y)."""
        assert x.ndim == 1 and y.ndim == 0
        z = torch.cat([x, y.view(1)], dim=0).unsqueeze(0)  # (1,p)

        k_row = self.k_torch(z, self.Ztr).ravel()        # (n,)
        row_mean = k_row.mean()
        k_c_row = k_row - row_mean - self.bar_k + self.bar_K

        if self.m == 0:
            proj_energy = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        else:
            t = (self.A_m.T @ k_c_row)
            proj_energy = (t * t).sum()

        k_self = self.k_torch(z, z)[0, 0]
        k_c_self = k_self - 2.0 * row_mean + self.bar_K[0]

        return k_c_self - proj_energy

    def predict_y(self,
                  x_np: Array,
                  y_init: Optional[float] = None,
                  lr: float = 0.05,
                  steps: int = 300,
                  restarts: int = 1,
                  init_perturb: float = 0.5) -> float:
        """Minimize E_phi(x,y) over y with AdamW, return y_hat (pre-image space)."""
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)

        base_init = self.mean_y if y_init is None else float(y_init)
        best_y = None
        best_val = float('inf')

        for r in range(restarts):
            y0 = base_init + (np.random.randn() * init_perturb if restarts > 1 else 0.0)
            y = torch.tensor([y0], dtype=torch.float32, device=self.device, requires_grad=True)
            opt = torch.optim.AdamW([y], lr=lr)

            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                loss = self.E_phi(x, y[0])
                loss.backward()
                opt.step()

            val = float(self.E_phi(x, y[0]).item())
            if val < best_val:
                best_val = val
                best_y = float(y.item())

        return best_y


def evaluate_kernels_with_torch_argmin(Z_train: Array,
                                       Z_test: Array,
                                       kernel_choice: str = "all",
                                       poly_degrees: Union[str, List[int]] = "all",
                                       rbf_sigmas: Union[str, List[float]] = [0.25, 0.5, 1.0, 2.0],
                                       m: Optional[int] = None,
                                       evr_target: Optional[float] = 0.95,
                                       lr: float = 0.05,
                                       steps: int = 300) -> pd.DataFrame:
    """For each kernel config: fit KPCA, then Torch-minimize y for each test x, report pre-image space MSE(ŷ,y) and mean feature-space residual at optimum."""
    if poly_degrees == "all":
        poly_degrees = [2, 3, 4]
    if rbf_sigmas == "all":
        rbf_sigmas = [0.25, 0.5, 1.0, 2.0]

    configs: List[KernelConfig] = []
    if kernel_choice in ("poly", "all"):
        for d in poly_degrees:
            configs.append(KernelConfig(kind="poly", params={"degree": int(d), "c0": 0.0}))
    if kernel_choice in ("rbf", "all"):
        for s in rbf_sigmas:
            configs.append(KernelConfig(kind="rbf", params={"sigma": float(s)}))

    rows = []
    results = []
    X_te = Z_test[:, :-1]
    y_true = Z_test[:, -1]

    for cfg in configs:

        model = kpca_fit(Z_train, cfg, m=m, evr_target=evr_target)
        projector = TorchProjector(model)

        y_pred = np.array([projector.predict_y(x, lr=lr, steps=steps) for x in X_te], dtype=float)

        mse_y = float(np.mean((y_pred - y_true) ** 2))

        with torch.no_grad():
            residuals = []
            for x, yhat in zip(X_te, y_pred):
                e = projector.E_phi(torch.tensor(x, dtype=torch.float32),
                                    torch.tensor(yhat, dtype=torch.float32))
                residuals.append(float(e.item()))
        mean_residual = float(np.mean(residuals))

        rows.append({
            "kernel": cfg.kind,
            "params": cfg.params,
            "m": model.m,
            "MSE_yhat_vs_y": mse_y,
            "mean_feature_residual": mean_residual
        })

        results.append({
            "kernel": cfg.kind,
            "pred": y_pred
        })

    return pd.DataFrame(rows), results


Z_train, Z_test = make_synthetic_xy()

df_results = evaluate_kernels(
    Z_train,
    Z_test,
    kernel_choice="all",
    poly_degrees="all",
    rbf_sigmas="all",
    m=None,
    evr_target=0.99
)
print("Feature-space orthogonal MSE (train/test):")
print(df_results)

df_argmin, results = evaluate_kernels_with_torch_argmin(
    Z_train, Z_test,
    kernel_choice="all",
    poly_degrees="all",
    rbf_sigmas="all",
    m=None,
    evr_target=0.99,
    lr=0.05,
    steps=300
)
print("\nInput-space MSE on y (argmin over y) and residual at optimum:")
print(df_argmin)

# print("Predictions for each kernel config:")
# for res in results:
#     print(f"Kernel: {res['kernel']}, Predictions ŷ: {res['pred']}")
