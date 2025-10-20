import numpy as np
import pandas as pd
from typing import List, Dict, Any

from make_dataset import make_data_separated, make_data_combined
from NPCA_reg_op import evaluate_kernels_with_torch_argmin
from spline_smoothing import TPSRegressor
from kernel_reg import KernelRegression, zscore, cdist, pairwise_sq_dists_numpy

def run_npca_regression(Z_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, **kwargs) -> Dict[str, Any]:
    """
    Runs the NPCA regression model and returns the best result.
    """
    print("Running NPCA Regression...")
    df_results, _ = evaluate_kernels_with_torch_argmin(Z_train, X_test, y_test, **kwargs)
    if df_results.empty:
        return {"mse": np.inf, "params": {}}
    best_result = df_results.loc[df_results['MSE_yhat_vs_y'].idxmin()]
    return {
        "mse": best_result['MSE_yhat_vs_y'],
        "params": f"kernel={best_result['kernel']}, params={best_result['params']}"
    }

def run_spline_smoothing(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, **kwargs) -> Dict[str, Any]:
    """
    Runs the Thin Plate Spline regression model.
    """
    print("Running Spline Smoothing...")
    model = TPSRegressor(**kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred)**2)
    return {"mse": mse, "params": kwargs}

def run_kernel_regression(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, **kwargs) -> Dict[str, Any]:
    """
    Runs the Kernel regression model with bandwidth selection.
    """
    print("Running Kernel Regression...")
    model = KernelRegression(**kwargs)
    model.fit(X_train, y_train)

    Xz, _, _ = zscore(X_train)
    if cdist:
        dists = cdist(Xz, Xz)
    else:
        dists = np.sqrt(pairwise_sq_dists_numpy(Xz, Xz))
    
    med = np.median(dists[np.triu_indices(len(X_train),1)])
    grid = np.logspace(np.log10(med * 0.1), np.log10(med * 10), 20)
    
    best_h = model.select_bandwidth(grid)
    model.h = best_h
    
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred)**2)
    
    params = kwargs.copy()
    params['selected_h'] = best_h
    return {"mse": mse, "params": params}

def evaluate_models(datasets: int, models_to_run: List[str], model_params: Dict[str, Dict[str, Any]]):
    """
    Main evaluation loop.
    """
    results = []
    
    for i in range(datasets):
        print(f"\n--- Starting Dataset {i+1}/{datasets} ---")
        
        X_train, y_train, X_test, y_test = make_data_separated(
            n_train=400, n_test=100, n_features=1, seed=i
        )
        Z_train, Z_test = make_data_combined(
            n_train=400, n_test=100, n_features=1, seed=i
        )
        # The Z_test from make_data_combined includes y, so we need to use the X and y from separated
        # to ensure consistency for the NPCA model's test set.
        
        for model_name in models_to_run:
            if model_name == 'npca':
                params = model_params.get('npca', {})
                result = run_npca_regression(Z_train, Z_test[:, :-1], Z_test[:, -1], **params)
            elif model_name == 'spline':
                params = model_params.get('spline', {})
                result = run_spline_smoothing(X_train, y_train, X_test, y_test, **params)
            elif model_name == 'kernel':
                params = model_params.get('kernel', {})
                result = run_kernel_regression(X_train, y_train, X_test, y_test, **params)
            else:
                print(f"Warning: Model '{model_name}' not recognized.")
                continue
                
            results.append({
                "dataset": i + 1,
                "model": model_name,
                "mse": result['mse'],
                "params": str(result['params'])
            })
            print(f"  {model_name.upper()} MSE: {result['mse']:.4f}")

    return pd.DataFrame(results)

if __name__ == '__main__':
    NUM_DATASETS = 5
    
    # Specify which models to run: 'npca', 'spline', 'kernel' or a list
    MODELS = ['npca', 'spline', 'kernel'] # Use 'all' or a list e.g., ['spline', 'kernel']

    # --- Model Parameters ---
    # Define parameters for each model.
    # For NPCA, you can specify kernel types and their parameters.
    # For Spline and Kernel, you specify their respective constructor arguments.
    MODEL_CONFIGS = {
        'npca': {
            'kernel_choice': "all", # 'rbf', 'poly', or 'all'
            'poly_degrees': [2, 3],
            'rbf_sigmas': [0.5, 1.0, 2.0],
            'evr_target': 0.99,
            'lr': 0.05,
            'steps': 200
        },
        'spline': {
            'lambda_': 0.01,
            'm': 2,
            'standardize': True
        },
        'kernel': {
            'kernel': 'rbf',
            'standardize': True
        }
    }

    if 'all' in MODELS:
        models_to_run = ['npca', 'spline', 'kernel']
    else:
        models_to_run = MODELS

    final_results = evaluate_models(NUM_DATASETS, models_to_run, MODEL_CONFIGS)
    
    print("\n--- Final Model Rankings ---")

    ranking = final_results.groupby('model')['mse'].mean().sort_values().reset_index()
    ranking['rank'] = ranking.index + 1
    
    print("Average MSE across all datasets:")
    print(ranking)
    
    print("\nDetailed results per dataset:")
    print(final_results.sort_values(by=['dataset', 'mse']))
