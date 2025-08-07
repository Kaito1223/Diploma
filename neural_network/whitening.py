import numpy as np

def whitening(data, name = 'pca'):
    """
    Apply whitening transformation to the given data.

    Parameters:
    name (str): The name of the whitening method.
    data (array-like): The input data to be whitened. Size should be (n_features, n_samples).
    n_feature: dimension of each data point (size of the signal vector at a given time step)
    n_samples: number of data points (number of time steps)

    Returns:
    array-like: The whitened data.
    """
    if name == 'zca':
        return zca_whitening(data)
    elif name == 'pca':
        return pca_whitening(data)
    else:
        raise ValueError(f"Unknown whitening method: {name}")

import numpy as np

def zca_whitening(X, n_components=None, epsilon=1e-5):
    """
    Perform ZCA whitening with optional dimensionality reduction.

    Parameters:
    -----------
    X : np.ndarray (m, T)
        Input data with m features and T samples.
    n_components : int or None
        Number of components to keep (n < m). If None, keep all m.
    epsilon : float
        Small constant for numerical stability.

    Returns:
    --------
    X_zca : np.ndarray (m, T) or (n, T)
        ZCA whitened data in original space but reduced to n components if specified.
    U_zca : np.ndarray (n, m) or (m, m)
        Whitening matrix.
    """
    m, T = X.shape
    if n_components is None:
        n_components = m  # Keep all components

    # Step 1: Center data
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_centered = X - X_mean

    # Step 2: Covariance matrix
    cov = np.cov(X_centered, rowvar=True, bias=True)  # Shape (m, m)

    # Step 3: Eigendecomposition (ascending order)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]  # Sort descending
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Step 4: Keep top n_components
    eigvals_n = eigvals[:n_components]  # (n,)
    eigvecs_n = eigvecs[:, :n_components]  # (m, n)

    # Step 5: Whitening matrix (n, m)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals_n + epsilon))  # (n, n)
    U_zca = eigvecs_n @ D_inv_sqrt @ eigvecs_n.T  # (m, m) still

    # If reduced, only keep top n directions
    if n_components < m:
        U_zca_reduced = D_inv_sqrt @ eigvecs_n.T  # (n, m)
        X_zca = U_zca_reduced @ X_centered  # (n, T)
        return X_zca, U_zca_reduced

    # Step 6: Whiten data (full m dimensions)
    X_zca = U_zca @ X_centered  # (m, T)
    return X_zca, U_zca



def pca_whitening(data, n_components=None, epsilon=1e-5):
    """
    Apply PCA whitening to reduce to n_components dimensions.

    Parameters:
    -----------
    data : array-like (m, T)
        Input data with m features and T samples.
    n_components : int or None
        Number of components to keep. If None, keep all m.
    epsilon : float
        Small constant to avoid division by zero.

    Returns:
    --------
    X_pca : np.ndarray (n, T)
        PCA whitened and reduced data.
    U_pca : np.ndarray (n, m)
        Whitening matrix.
    """
    m, T = data.shape
    if n_components is None:
        n_components = m  # Keep full dimension

    # Step 1: Center the data
    data_mean = np.mean(data, axis=1, keepdims=True)
    data_centered = data - data_mean

    # Step 2: Covariance matrix
    cov_matrix = np.cov(data_centered, rowvar=True)

    # Step 3: Eigen decomposition (descending)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Step 4: Keep top n_components
    eigvals_n = eigvals[:n_components]
    eigvecs_n = eigvecs[:, :n_components]  # Shape (m, n)

    # Step 5: Whitening matrix (n, m)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals_n + epsilon))
    U_pca = D_inv_sqrt @ eigvecs_n.T  # Shape (n, m)

    # Step 6: Whiten data (n, T)
    X_pca = U_pca @ data_centered

    return X_pca, U_pca

    

# Example usage:
#data = np.random.rand(5, 100)  # Example data with 5 features and 100 samples
#print(data)

#whitened_data, whitening_matrix = whitening(data, name='pca')
#print("Whitened Data:\n", whitened_data)
#print("Whitening Matrix:\n", whitening_matrix)

#print("Mean of Whitened Data:\n", np.mean(whitened_data, axis=1))
#print("Covariance of Whitened Data:\n", np.cov(whitened_data, rowvar=True))