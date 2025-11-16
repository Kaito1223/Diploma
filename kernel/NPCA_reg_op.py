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

import poly_pca
from poly_pca import KPCAResult, KernelConfig, MomentPolynomialPCA

Array = np.ndarray

def centered_kernel_row(model: KPCAResult, z: Array) -> Tuple[Array, float]:
    """Compute centered kernel row k_c(z) and k_c(z,z) vs training set."""
    Ztr = model.Z_train
    kfun = model.kernel_cfg.kernel_fn()
    k_row = kfun(z[None, :], Ztr).ravel()  # (n,)
    row_mean = k_row.mean()
    k_c_row = k_row - row_mean - model.bar_k + model.bar_K

    k_self = kfun(z[None, :], z[None, :])[0, 0]
    k_c_self = k_self - 2 * row_mean + model.bar_K
    return k_c_row, k_c_self

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
        mse = model.lambdas.mean()
        return float(np.sqrt(np.maximum(0.0, mse)))
    
    residual_lambdas = model.lambdas[model.m:]
    Q_res = model.Q[:, model.m:]

    E_per = (Q_res**2) @ residual_lambdas
    mse = E_per.mean()

    return float(np.sqrt(np.maximum(0.0, mse)))

def kpca_test_rmse_feature(model: KPCAResult, Z_test: Array) -> float:
    errs = [kpca_feature_error(model, z) for z in Z_test]
    mse = np.mean(errs)
    return float(np.sqrt(np.maximum(0.0, mse)))

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


class TorchProjector_Kernel:
    """Optimizes y for a KPCAResult (Kernel/Gram) model."""
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

class TorchProjector_Moment:
    """Optimizes y for a MomentPolynomialPCA model."""
    def __init__(self, model: MomentPolynomialPCA, k:int = 0, lambda_knn: float = 0.0, device: Optional[str] = None):
        self.model = model
        self.device = torch.device(device) if device else torch.device('cpu')

        self.p = model.p_
        self.m = model.m
        self.d = model.degree
        self.const = model.const
        
        self.k = k
        self.lambda_knn = lambda_knn
        if hasattr(model, 'Z_train_') and model.Z_train_ is not None:
            self.Xtr = torch.tensor(model.Z_train_[:, :-1], dtype=torch.float32, device=self.device) # (n, p-1)
            self.ytr = torch.tensor(model.Z_train_[:, -1], dtype=torch.float32, device=self.device) # (n,)

            self.mean_y = float(model.Z_train_[:, -1].mean())
        else:
            print("Warning: MomentPolynomialPCA model has no Z_train_ data. kNN disabled.")
            self.k = 0
            self.lambda_knn = 0.0
            self.Xtr = torch.empty(0, 0, device=self.device)
            self.ytr = torch.empty(0, 0, device=self.device)
            self.mean_y = 0.0

        self.exponents = torch.tensor(np.array(model.exponents_), dtype=torch.int64, device=self.device)
        self.bin_sqrts = torch.tensor(model.bin_sqrts_, dtype=torch.float32, device=self.device).view(-1, 1)
        self.d_p = self.exponents.shape[0]

        self.V_m = torch.tensor(model.eigvecs_, dtype=torch.float32, device=self.device)
        self.m_feat = torch.tensor(model.m_feat_, dtype=torch.float32, device=self.device).view(-1, 1)
        self.has_const = (self.const is not None) and (self.const != 0.0)
        self.sqrt_u = float(self.const) ** 0.5 if self.has_const else 0.0
    
    def _torch_feature_map(self, Z_batch: torch.Tensor) -> torch.Tensor:
        Z_trans = Z_batch.T
        all_powers = Z_trans.unsqueeze(0) ** self.exponents.unsqueeze(2)
        Phi = torch.prod(all_powers, dim=1)
        return self.bin_sqrts * Phi

    def E_phi_batch(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        b = X.shape[0]
        if self.has_const:
            sqrt_u_tensor = torch.full((b, 1), self.sqrt_u, device=self.device)
            Z_batch = torch.cat([X, Y, sqrt_u_tensor], dim=1)
        else:
            Z_batch = torch.cat([X, Y], dim=1)

        Phi = self._torch_feature_map(Z_batch)
        Phi_c = Phi - self.m_feat
        total_sq_norm = torch.sum(Phi_c * Phi_c, dim=0)
        
        if self.m == 0:
            return total_sq_norm
            
        T = self.V_m.T @ Phi_c
        proj_sq_norm = torch.sum(T * T, dim=0)
        error = total_sq_norm - proj_sq_norm
        return torch.clamp(error, min=0.0)
    
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

def fit_models(Z_train: Array,
               method: str = 'kernel',
               evr_target: Optional[float] = 0.95,
               **kwargs) -> List[Union[KPCAResult, MomentPolynomialPCA]]:
    """
    Master function to fit models
    """
    if method == 'kernel':
        print("--- Fitting Kernel PCA model ---")
        kernel_choice = kwargs.get('kernel_choice', 'all')
        poly_degrees = kwargs.get('poly_degrees', 'all')
        rbf_sigmas = kwargs.get('rbf_sigmas', 'all')
        
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
            model = poly_pca.kpca_fit(Z_train, cfg, evr_target=evr_target)
            models.append(model)
        print("Model fitting complete.")
        return models
    
    elif method == 'moment':
        print("--- Fitting Moment PCA Pipeline ---")
        degrees = kwargs.get('degrees', [1, 2, 3])
        consts = kwargs.get('consts', [0.0, 1.0])

        models = []
        print(f"Fitting {len(degrees) * len(consts)} moment-based models...")
        for d in degrees:
            for c in consts:
                print(f"  Fitting degree={d}, const={c}")
                model = poly_pca.MomentPolynomialPCA(degree=d, const=c, evr_target=evr_target)
                model.fit(Z_train)
                models.append(model)
        print("Model fitting complete.")
        return models
    else:
        raise ValueError(f"Unknown method: {method}")

def predict(models: List[Union[KPCAResult, MomentPolynomialPCA]],
            X_test: Array,
            y_test: Optional[Array] = None,
            k: int = 0,
            lambda_knn: float = 0.0,
            batch_size: Optional[int] = None,
            lr: float = 0.05,
            steps: int = 300) -> pd.DataFrame:

    rows = []
    results = []

    for model in models:
        y_pred: Array
        projector: Union[TorchProjector_Kernel, TorchProjector_Moment]
        kernel_name: str
        param_name: str
        if isinstance(model, KPCAResult):
            projector = TorchProjector_Kernel(model, k=k, lambda_knn=lambda_knn)
            kernel_name = model.kernel_cfg.kind
            param_name = str(model.kernel_cfg.params)
            print(f"Predicting with: {kernel_name}, params: {param_name}, k={k}, lambda={lambda_knn}")
        elif isinstance(model, MomentPolynomialPCA):
            projector = TorchProjector_Moment(model, k=k, lambda_knn=lambda_knn)
            kernel_name = "poly-moment"
            param_name = f"degree={model.degree}, const={model.const}"
            print(f"Predicting with: moment, params: {param_name}, k={k}, lambda={lambda_knn}")
        else:
            raise ValueError(f"Unknown model type for prediction. Model types are: {type(model)}")
        
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
            root_mean_residual = float(np.sqrt(mean_residual))
        rows.append({
            "kernel": kernel_name,
            "params": param_name,
            "m": model.m,
            "RMSE_yhat_vs_y": rmse_y,
            "root_mean_feature": root_mean_residual,
        })
        results.append({
            "kernel": kernel_name,
            "params": param_name,
            "pred": y_pred
        })

    return pd.DataFrame(rows), results
 

if __name__ == "__main__":

    PCA_METHOD = 'moment' # 'kernel' or 'moment'
    EVR_TARGET = 0.9
    kernel_params = {
        'kernel_choice': 'all',
        'poly_degrees': [1, 2, 3],
        'rbf_sigmas': 'all'
    }
    moment_params = {
        'degrees': [1, 2, 3],
        'consts': [0.0]
    }
    knn_params = {
        'k': 6,
        'lambda_knn': 0.5
    }
    fit_args = kernel_params if PCA_METHOD == 'kernel' else moment_params

    Z_train, Z_test = make_data_combined(n_train=450, n_test=50, n_features=1 ,seed=0, noise_level=0.15)
    Z_train_clean, Z_test_clean = make_data_combined(n_train=450, n_test=50, n_features=1 ,seed=0, noise_level=0.0)

    X_train = Z_train[:, :-1]
    y_train = Z_train[:, -1]

    Xq = np.linspace(X_train.min(), X_train.max(), 400)[:, None]

    X_test = Z_test[:, :-1]
    y_test = Z_test[:, -1]

    fitted_models = fit_models(
        Z_train,
        method=PCA_METHOD,
        evr_target=EVR_TARGET,
        **fit_args
    )

    df_argmin, results = predict(
        fitted_models,
        Xq,
        y_test=None,
        batch_size=None,
        lr=0.05,
        steps=300,
        **knn_params
    )
    
    print(f"\n--- {PCA_METHOD.upper()} PCA Prediction on Dense Grid (Xq) ---")
    print(df_argmin)

    yq_preds = { (row['kernel'], str(row['params'])): row['pred'] for row in results }

    for (kernel, params), yq_pred in yq_preds.items():
        plt.figure(figsize=(8, 5))
        plt.plot(Xq[:, 0], yq_pred, label=f"Prediction: {kernel}, {params}")
        plt.scatter(X_train[:, 0], y_train, s=25, color="red", alpha=0.8, label="train")
        plt.scatter(X_test[:, 0], y_test, s=25, color="blue", alpha=0.7, label="test")
        plt.xlabel("x")
        plt.ylabel("y / prediction")
        plt.title(f"NPCA Regression ({PCA_METHOD} method)")
        plt.legend()

    plt.show()