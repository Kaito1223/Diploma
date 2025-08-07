import itertools
from math import lgamma, comb
import numpy as np

def degree_d_multi_indices(d, p):
    """
    Returns a lexicographically sorted list of all exponent vectors
    (j1,...,jp) with non-negative integers summing to d.
    """
    #star and bar method
    result = []
    for dividers in itertools.combinations(range(d+p-1), p-1):
        last = -1
        index = []
        for bar in dividers + (d+p-1,):
            index.append(bar - last - 1)
            last = bar
        result.append(tuple(index))
    return list(reversed(result))

def sample_data(p, n, seed=None):
    """
    Generate a sample dataset of shape (p, n) with standard normal distribution.
    Parameters
    ----------
    p : int
        Number of features
    n : int
        Number of samples
    seed : int or None
        Random seed for reproducibility
    Returns
    -------
    np.ndarray, shape (p, n)
        Sample dataset. 
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((p, n)) 


def poly_kernel(x, y, degree, const = 0):
    """
    Homogeneous polynomial kernel  K(x,y) = (xᵀy)^degree.

    Parameters
    ----------
    x, y   : ndarray, shape (p,1) Column vectors in R^p
    degree : int  (d ≥ 0)

    Returns
    -------
    float
        Kernel value.
    """
    return float((x.T @ y + const) ** degree)

import numpy as np

def gram_matrix(X, degree):
    """
    X   : data sample, ndarray, shape (p, n)
    degree : int  (polynomial degree d)

    Returns the n*n Gram matrix with entries (xᵢᵀ xⱼ)^d.
    """
    return (X.T @ X) ** degree

def eigen(K):
    """
    K : Gram matrix, ndarray, shape (n, n)

    Returns the eigenvalues and eigenvectors of the Gram matrix.
    """
    eigen_vals, eigen_vecs = np.linalg.eigh(K)
    idx = np.argsort(eigen_vals)[::-1]
    return eigen_vals[idx], eigen_vecs[:, idx]

def multinomial_coefficients_sqrt(d, j_vec):
    """
    Returns the multinomial coefficients for the polynomial degree d and number of variables p.
    Using log-gamma to avoid overflow issues.
    lgamma(n+1) = log(n!)
    Parameters
    ----------
    d :     int
        Polynomial degree.
    j_vec : list
        List of indices (j1, ..., jp) with non-negative integers summing to d.
    
    Returns
    -------
    np.ndarray, shape ((d+p-1) choose (p-1),)
        Array of multinomial coefficients.
    """
    log_d = lgamma(d + 1)
    log_j_vec = sum(lgamma(j_k + 1) for j_k in j_vec)
    return np.sqrt(np.exp(log_d - log_j_vec))

def emprircal_moment(X, moments_vec):
    return float(np.mean(np.prod(X ** moments_vec.reshape(-1, 1), axis=0)))

def moment_matrix(d, p, X, exponents):
    d_p = len(exponents)
    M = np.zeros((d_p, d_p))
    for r, i_vec in enumerate(exponents):
        bin_i = multinomial_coefficients_sqrt(d, i_vec)
        for s, j_vec in enumerate(exponents):
            bin_j = multinomial_coefficients_sqrt(d, j_vec)
            moments_vec = np.array([i_vec[k] + j_vec[k] for k in range(p)])
            M[r, s] = bin_i * bin_j * emprircal_moment(X, moments_vec)

    return M


def feature_matrix(X, d, exponents, bin_sqrts):
    p, n = X.shape
    d_p = len(exponents)
    Z = np.empty((d_p, n))
    for r, (j, bin_sqrt) in enumerate(zip(exponents, bin_sqrts)):
        Z[r] = bin_sqrt * np.prod(X ** np.array(j)[:, None], axis=0)
    return Z


def polynomial_pca(X, d, components=2):
    p, n = X.shape

    exponents = degree_d_multi_indices(d, p)
    bin_sqrts = [multinomial_coefficients_sqrt(d, j) for j in exponents]
    d_p = len(exponents)

    M = moment_matrix(d, p, X, exponents)
    eigen_vals, eigen_vecs = eigen(M)

    m = min(components, eigen_vals.size)
    eigen_vals, eigen_vecs = eigen_vals[:m], eigen_vecs[:, :m]
    
    phi = feature_matrix(X, d, exponents, bin_sqrts)
    Z = (eigen_vecs.T @ phi) / eigen_vals[: , None]
    return Z

def kpca_from_gram(Kc, eig_vals, eig_vecs, m=None):
    n = Kc.shape[0]

    nonzero = eig_vals > 1e-12
    eig_vals = eig_vals[nonzero]
    eig_vecs = eig_vecs[:, nonzero]

    if m is not None:
        m = min(m, eig_vals.size)
        eig_vals = eig_vals[:m]
        eig_vecs = eig_vecs[:, :m]
    else:
        m = eig_vals.size

    Z = (eig_vecs.T @ Kc) / eig_vals[:, None]

    return Z



# p, n, d = 2, 800, 2
# exponents = degree_d_multi_indices(d, p)
# X = sample_data(p, n, seed=42)
# K = gram_matrix(X, d)

# eigen_vals_original, eigen_vecs_original = eigen(X.T@X)

# print("Eigenvalues of the original data matrix:", eigen_vals_original)

# eigen_vals_gram, eigen_vecs_gram = eigen(K)
# eigen_vals_gram = eigen_vals_gram/n

# M = moment_matrix(d, p, X, exponents)
# eigen_vals_M2, eigen_vecs_M2 = eigen(M)

# print("Eigenvalues of the Gram matrix:", eigen_vals_gram)
# print("Eigenvalues of the moment matrix M2:", eigen_vals_M2)

# # print("Eigenvectors of the Gram matrix:\n", eigen_vecs_gram)
# print("Eigenvectors of the moment matrix M2:\n", eigen_vecs_M2)

# polynomial_pca_result = polynomial_pca(X, d, components=1)
# print("KPCA from moment matrix result:\n", polynomial_pca_result)

# pca_from_gram = kpca_from_gram(K, eigen_vals_gram, eigen_vecs_gram, m=1)
# print("KPCA from Gram matrix result:\n", pca_from_gram)

#print(polynomial_pca_result / pca_from_gram)