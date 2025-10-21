from __future__ import annotations
from sklearn.datasets import make_circles

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import math
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from make_dataset import make_data_combined

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
    return float(np.maximum(0.0, k_c_self - np.dot(t, t)))


def kpca_train_rmse_feature(model: KPCAResult) -> float:
    """Training mean feature-space error using eigen shortcut:
       E_phi(z_i) = sum_{j>m} lambda_j * Q_{ij}^2"""
    n = model.Kc.shape[0]
    if model.m == 0:
        return float(np.sqrt(model.lambdas.mean(), 0))
    residual_lambdas = model.lambdas[model.m:]
    Q_res = model.Q[:, model.m:]

    E_per = (Q_res**2) @ residual_lambdas
    mse = E_per.mean()

    return float(np.sqrt(np.maximum(0.0, mse)))

def kpca_test_rmse_feature(model: KPCAResult, Z_test: Array) -> float:
    errs = [kpca_feature_error(model, z) for z in Z_test]
    mse = np.mean(errs)
    return float(np.sqrt(np.maximum(0.0, mse)))

def fit_all_kernels(Z_train: Array,
                    kernel_choice: str = "all",
                    poly_degrees: Union[str, List[int]] = "all",
                    rbf_sigmas: Union[str, List[float]] = "all",
                    m: Optional[int] = None,
                    evr_target: Optional[float] = 0.95) -> List[KPCAResult]:
    """
    Fits one model for each kernel config and returns a list of fitted models.
    """
    if poly_degrees == "all": poly_degrees = [1, 2, 3, 4]
    if rbf_sigmas == "all": rbf_sigmas = [0.25, 0.5, 1.0, 2.0]
    
    configs: List[KernelConfig] = []
    if kernel_choice in ("poly", "all"):
        for d in poly_degrees:
            configs.append(KernelConfig(kind="poly", params={"degree": int(d), "c0": 0.0}))
    if kernel_choice in ("rbf", "all"):
        for s in rbf_sigmas:
            configs.append(KernelConfig(kind="rbf", params={"sigma": float(s)}))

    models = []
    print(f"Fitting {len(configs)} models...")
    for cfg in configs:
        print(f"  Fitting {cfg.kind} with params: {cfg.params}")
        model = kpca_fit(Z_train, cfg, m=m, evr_target=evr_target)
        models.append(model)
    print("Model fitting complete.")
    return models

def evaluate_models(models: List[KPCAResult],
                    Z_test: Optional[Array] = None) -> pd.DataFrame:
    """
    Evaluates a list of pre-fitted models on new data.
    Calculates train/test feature-space RMSE.
    """
    print(f"Evaluating feature-space RMSE for {len(models)} models...")
    rows = []
    for model in models:
        cfg = model.kernel_cfg
        
        train_rmse = kpca_train_rmse_feature(model)
        
        if Z_test is not None:
            test_rmse = kpca_test_rmse_feature(model, Z_test)
        else:
            test_rmse = np.nan
        
        rows.append({
            "kernel": cfg.kind,
            "params": cfg.params,
            "m": model.m,
            "train_feature_RMSE": train_rmse,
            "test_feature_RMSE": test_rmse
        })
    return pd.DataFrame(rows)

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
    def __init__(self, model: KPCAResult, k:int = 0, lambda_knn: float = 0.0, device: Optional[str] = None):
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

        self.k = k
        self.lambda_knn = lambda_knn

        self.Xtr = torch.tensor(model.Z_train[:, :-1], dtype=torch.float32, device=self.device) # (n, p-1)
        self.ytr = torch.tensor(model.Z_train[:, -1], dtype=torch.float32, device=self.device) # (n,)

        self.mean_y = float(model.Z_train[:, -1].mean())
        
    def E_phi_batch(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        b = X.shape[0]
        Z = torch.cat([X, Y], dim=1)

        k_matrix = self.k_torch(Z, self.Ztr)
        row_means = k_matrix.mean(dim=1, keepdim=True)

        k_c_matrix = k_matrix - row_means - self.bar_k + self.bar_K

        if self.m == 0:
            proj_energy = torch.zeros(b, dtype=torch.float32, device=self.device)
        else:
            T = (self.A_m.T @ k_c_matrix.T).T
            proj_energy = (T * T).sum(dim=1)

        k_selfs = torch.diag(self.k_torch(Z, Z))

        k_c_selfs = k_selfs - 2.0 * row_means.squeeze() + self.bar_K[0]

        return k_c_selfs - proj_energy
    
    def _find_knn_neighbors(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist2_matrix = torch.cdist(X, self.Xtr, p=2)**2
        
        neighbor_dists_sq, neighbor_indices = torch.topk(
            dist2_matrix + 1e-8, 
            k=self.k, 
            dim=1, #since the cdist return the distance matrix, value in row i is the distance between X[i] and all Xtr. dim=1 search for row
            largest=False
        )

        y_neighbors = self.ytr[neighbor_indices]
        
        weights = 1.0 / neighbor_dists_sq
        weights = weights / weights.sum(dim=1, keepdims=True)
        
        return y_neighbors, weights
    
    def R_knn_batch(self, 
                    Y: torch.Tensor, 
                    y_neighbors: torch.Tensor, 
                    weights: torch.Tensor) -> torch.Tensor:
        Y_expanded = Y.expand(-1, self.k)
        
        diff_sq = (Y_expanded - y_neighbors) ** 2
        weighted_diff_sq = weights * diff_sq
        
        R_knn = weighted_diff_sq.sum(dim=1)
        return R_knn
    
    def total_loss_batch(self, 
                         X: torch.Tensor, 
                         Y: torch.Tensor,
                         y_neighbors: Optional[torch.Tensor],
                         weights: Optional[torch.Tensor]) -> torch.Tensor:
        E_phi = self.E_phi_batch(X, Y)
        
        if self.lambda_knn == 0.0 or self.k == 0:
            return E_phi
            
        R_knn = self.R_knn_batch(Y, y_neighbors, weights)
        return E_phi + self.lambda_knn * R_knn

    def predict_y_batch(self,
                        X_np: Array,
                        y_init: Optional[Array] = None,
                        lr: float = 0.05,
                        steps: int = 300,
                        restarts: int = 1,
                        init_perturb: float = 0.5) -> Array:
        
        b = X_np.shape[0]
        X = torch.tensor(X_np, dtype=torch.float32, device=self.device)

        best_Y = torch.zeros(b, 1, dtype=torch.float32, device=self.device)
        best_vals = torch.full((b,), float('inf'), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.k > 0 and self.lambda_knn > 0.0:
                y_neighbors_precomputed, weights_precomputed = self._find_knn_neighbors(X)
            else:
                y_neighbors_precomputed, weights_precomputed = None, None
                
        for r in range(restarts):
            if y_init is not None:
                base_init = torch.tensor(y_init, dtype=torch.float32, device=self.device).view(-1, 1)
            else:
                base_init = torch.full((b, 1), self.mean_y, dtype=torch.float32, device=self.device)
            
            if restarts > 1:
                perturbation = torch.randn(b, 1, device=self.device) * init_perturb
                Y0 = base_init + perturbation
            else:
                Y0 = base_init

            Y = Y0.clone().detach().requires_grad_(True)
            opt = torch.optim.AdamW([Y], lr=lr)

            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                loss = self.total_loss_batch(X, Y, 
                                             y_neighbors_precomputed, 
                                             weights_precomputed).sum()
                loss.backward()
                opt.step()

            with torch.no_grad():
                current_vals = self.E_phi_batch(X, Y)
                is_better = current_vals < best_vals
                best_vals[is_better] = current_vals[is_better]
                best_Y[is_better] = Y[is_better]

        return best_Y.squeeze(dim = 1).cpu().numpy()    

def predict(models: List[KPCAResult],
            X_test: Array,
            y_test: Optional[Array] = None,
            k: int = 0,
            lambda_knn: float = 0.0,
            batch_size: Optional[int] = None,
            lr: float = 0.05,
            steps: int = 300) -> pd.DataFrame:

    rows = []
    results = []
    # X_te = Z_test[:, :-1]
    # y_true = Z_test[:, -1]

    for model in models:
        cfg = model.kernel_cfg
        print(f"Predicting with: {cfg.kind}, params: {cfg.params}, k={k}, lambda={lambda_knn}")
        projector = TorchProjector(model, k=k, lambda_knn=lambda_knn)

        if batch_size is None or batch_size >= len(X_test):
            y_pred = projector.predict_y_batch(X_test, lr=lr, steps=steps)
        else:
            # Process in mini-batches
            y_pred_parts = []
            num_samples = len(X_test)
            for i in range(0, num_samples, batch_size):
                #print(f"  Processing batch {i//batch_size + 1}...")
                X_batch = X_test[i : i + batch_size]
                y_pred_batch = projector.predict_y_batch(X_batch, lr=lr, steps=steps)
                y_pred_parts.append(y_pred_batch)
            
            # Combine the results from all batches
            y_pred = np.concatenate(y_pred_parts)
        if y_test is not None:
            rmse_y = np.sqrt(float(np.mean((y_pred - y_test) ** 2)))
        else:
            rmse_y = None

        with torch.no_grad():
            x_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32).view(-1, 1)
            residuals = projector.E_phi_batch(x_tensor, y_pred_tensor)
            mean_residual = float(residuals.mean().item())

        rows.append({
            "kernel": cfg.kind,
            "params": cfg.params,
            "m": model.m,
            "RMSE_yhat_vs_y": rmse_y,
            "mean_feature_residual": mean_residual
        })
        results.append({
            "kernel": cfg.kind,
            "params": cfg.params,
            "pred": y_pred
        })

    return pd.DataFrame(rows), results
 

if __name__ == "__main__":
    Z_train, Z_test = make_data_combined(n_train=450, n_test=50, n_features=1 ,seed=0, noise_level=0.15)
    Z_train_clean, Z_test_clean = make_data_combined(n_train=450, n_test=50, n_features=1 ,seed=0, noise_level=0.0)

    X_train = Z_train[:, :-1]
    y_train = Z_train[:, -1]

    Xq = np.linspace(X_train.min(), X_train.max(), 400)[:, None]

    X_test = Z_test[:, :-1]
    y_test = Z_test[:, -1]

    fitted_models = fit_all_kernels(
        Z_train,
        kernel_choice="all",
        poly_degrees=[1, 2, 3],
        rbf_sigmas="all",
        m=None,
        evr_target=0.9
    )

    # df_results = evaluate_models(
    #     fitted_models,
    #     Z_test,
    # )

    # df_results_clean = evaluate_models(
    #     fitted_models,
    #     Z_test_clean,
    # )

    df_results, _ = predict(
        fitted_models,
        X_test=X_test,
        y_test=y_test,
        k=6,
        lambda_knn=0.0,
        batch_size=None,
        lr=0.05,
        steps=300
    )

    df_results_clean, _ = predict(
        fitted_models,
        X_test=Z_test_clean[:, :-1],
        y_test=Z_test_clean[:, -1],
        k=6,
        lambda_knn=0.0,
        batch_size=None,
        lr=0.05,
        steps=300
    )

    print("Feature-space orthogonal RMSE (train/test):")
    print(df_results)

    print("Feature-space orthogonal RMSE clean (train/test):")
    print(df_results_clean)

    print("Robustness score:")
    print(df_results_clean["RMSE_yhat_vs_y"] / df_results["RMSE_yhat_vs_y"])

    # X_train_clean = Z_train_clean[:, :-1]
    # y_train_clean = Z_train_clean[:, -1]
    # Xq = np.linspace(X_train.min(), X_train.max(), 400)[:, None]

    # X_test_clean = Z_test_clean[:, :-1]
    # y_test = Z_test_clean[:, -1]

    # uncomment for prediction and plotting
    # df_argmin, results = predict(
    #     fitted_models, Xq,
    #     k=6,
    #     lambda_knn=0.0,
    #     batch_size=None,
    #     lr=0.05,
    #     steps=300
    # )
    # print("\nInput-space RMSE on y (argmin over y) and residual at optimum:")
    # print(df_argmin)

    # yq_preds = { (row['kernel'], str(row['params'])): row['pred'] for row in results }

    # for (kernel, params), yq_pred in yq_preds.items():
    #     plt.figure(figsize=(8, 5))

    #     plt.plot(Xq[:, 0], yq_pred, label=f"Prediction: {kernel}, {params}")

    #     plt.scatter(X_train[:, 0], y_train, s=25, color="red", alpha=0.8, label="train")
    #     plt.scatter(X_test[:, 0], y_test, s=25, color="blue", alpha=0.7, label="test")

    #     plt.xlabel("x")
    #     plt.ylabel("y / prediction")
    #     plt.title(f"NPCA Regression with {kernel}")
    #     plt.legend()

    # plt.show()