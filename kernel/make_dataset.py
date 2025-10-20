import numpy as np
from typing import Tuple, Optional, Dict, Any
from numpy.typing import NDArray as Array

def _prepare_data(
    n_train: int,
    n_test: int,
    n_features: int,
    seed: int,
    noise_level: float = 0.05,
    X: Optional[Array] = None,
    y: Optional[Array] = None,
    distribution: str = 'uniform',
    dist_params: Optional[Dict[str, Any]] = None
) -> Tuple[Array, Array]:
    """
    Internal helper to prepare data by either using an existing dataset or generating a new one.

    Args:
        n_train: Number of training samples.
        n_test: Number of testing samples.
        n_features: Number of features (used only for data generation).
        seed: Random seed for reproducibility.
        X: Optional pre-existing feature matrix.
        y: Optional pre-existing target vector.
        distribution: Name of the distribution for data generation ('uniform', 'normal', 'multivariate_normal').
        dist_params: Dictionary of parameters for the chosen distribution.

    Returns:
        A tuple of (X_shuffled, y_shuffled).
    """
    rng = np.random.default_rng(seed)
    
    if X is not None and y is not None:
        if X.ndim == 1: X = X.reshape(-1, 1)
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        n_total = len(X)
        if n_train + n_test > n_total:
            raise ValueError(f"n_train ({n_train}) + n_test ({n_test}) cannot exceed total samples ({n_total}).")
        
        X_gen, y_gen = X, y

    else:
        n_total = n_train + n_test
        dist_params = dist_params or {}

        if distribution == 'uniform':
            low = dist_params.get('low', -np.pi)
            high = dist_params.get('high', np.pi)
            X_gen = rng.uniform(low=low, high=high, size=(n_total, n_features))
        
        elif distribution == 'normal':
            loc = dist_params.get('mean', 0.0)
            scale = dist_params.get('scale', 1.0)
            X_gen = rng.normal(loc=loc, scale=scale, size=(n_total, n_features))
        
        elif distribution == 'multivariate_normal':
            mean = dist_params.get('mean', np.zeros(n_features))
            cov = dist_params.get('cov', np.eye(n_features))
            if len(mean) != n_features or cov.shape != (n_features, n_features):
                raise ValueError("Shape of 'mean' or 'cov' is inconsistent with n_features.")
            X_gen = rng.multivariate_normal(mean=mean, cov=cov, size=n_total)

        else:
            raise ValueError(f"Unknown distribution '{distribution}'. Choose from 'uniform', 'normal', 'multivariate_normal'.")
            
        # Generate target y based on the features
        x_sum = np.sum(X_gen, axis=1)
        y_gen = np.sin(x_sum) + noise_level * rng.standard_normal(n_total)

    # Shuffle the data (either provided or generated)
    idx = rng.permutation(len(X_gen))
    X_shuffled = X_gen[idx]
    y_shuffled = y_gen[idx]
    
    return X_shuffled, y_shuffled

def make_data_separated(
    n_train: int = 80, 
    n_test: int = 80, 
    n_features: int = 1,
    seed: int = 0,
    noise_level: float = 0.05,
    X: Optional[Array] = None,
    y: Optional[Array] = None,
    distribution: str = 'uniform',
    dist_params: Optional[Dict[str, Any]] = None
) -> Tuple[Array, Array, Array, Array]:
    """
    Prepares a dataset, returning separate train/test sets for X and y.

    Can operate in two modes:
    1. Data Generation: If X and y are None, generates new data based on the specified distribution.
    2. Data Splitting: If X and y are provided, it shuffles and splits the existing data.

    Args:
        n_train: Number of training samples.
        n_test: Number of testing samples.
        n_features: Dimensionality of the dataset (for data generation).
        seed: Random seed for reproducibility.
        X: (Optional) Pre-existing feature matrix.
        y: (Optional) Pre-existing target vector.
        distribution: Distribution for data generation ('uniform', 'normal', 'multivariate_normal').
        dist_params: Parameters for the distribution (e.g., {'mean': 0, 'scale': 2}).

    Returns:
        A tuple of four numpy arrays in the order: (X_test, y_test, X_train, y_train).
    """
    X_shuffled, y_shuffled = _prepare_data(
        n_train, n_test, n_features, seed, noise_level, X, y, distribution, dist_params
    )

    # Define the split point. Use n_train for splitting.
    split_idx = n_train
    
    X_train = X_shuffled[:split_idx]
    y_train = y_shuffled[:split_idx]
    
    X_test = X_shuffled[split_idx : split_idx + n_test]
    y_test = y_shuffled[split_idx : split_idx + n_test]
    
    return X_train, y_train, X_test, y_test

def make_data_combined(
    n_train: int = 80, 
    n_test: int = 80, 
    n_features: int = 1,
    seed: int = 0,
    noise_level: float = 0.05,
    X: Optional[Array] = None,
    y: Optional[Array] = None,
    distribution: str = 'uniform',
    dist_params: Optional[Dict[str, Any]] = None
) -> Tuple[Array, Array]:
    """
    Prepares a dataset, returning combined train/test sets (Z).

    Can operate in two modes:
    1. Data Generation: If X and y are None, generates new data.
    2. Data Splitting: If X and y are provided, it shuffles and splits them.

    In the returned arrays, the target y is the last column.

    Returns:
        A tuple of two numpy arrays in the order: (Z_test, Z_train).
    """
    X_shuffled, y_shuffled = _prepare_data(
        n_train, n_test, n_features, seed, noise_level, X, y, distribution, dist_params
    )
    Z_shuffled = np.column_stack((X_shuffled, y_shuffled))

    # Define the split point
    split_idx = n_train
    
    Z_train = Z_shuffled[:split_idx]
    Z_test = Z_shuffled[split_idx : split_idx + n_test]
    
    return Z_train, Z_test