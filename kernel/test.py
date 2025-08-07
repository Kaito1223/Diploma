import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from kernel_polunomial_pca_general import *

def generate_xor_data(n_per_quad=50, noise=0.3, seed=None):
    """
    Produce an XOR toy dataset in ℝ².

    Parameters
    ----------
    n_per_quad : int   # points in each quadrant (total = 4× this)
    noise       : float   # Gaussian noise σ around each corner
    seed        : int or None

    Returns
    -------
    X : ndarray (2, n)   # columns are samples
    y : ndarray (n,)     # labels +1 / -1
    """
    rng = np.random.default_rng(seed)

    q1 = rng.normal([ 1,  1], noise, size=(n_per_quad, 2))
    q2 = rng.normal([-1,  1], noise, size=(n_per_quad, 2))
    q3 = rng.normal([-1, -1], noise, size=(n_per_quad, 2))
    q4 = rng.normal([ 1, -1], noise, size=(n_per_quad, 2))

    X = np.vstack([q1, q3, q2, q4]).T
    y = np.hstack([ np.ones(2*n_per_quad),
                   -np.ones(2*n_per_quad)])
    return X, y

def generate_circles(n_per_ring=300, r_inner=1.0, r_outer=2.0,
                     noise=0.05, seed=None):
    rng   = np.random.default_rng(seed)
    theta = rng.uniform(0, 2*np.pi, size=n_per_ring)

    inner = np.vstack([r_inner*np.cos(theta),
                       r_inner*np.sin(theta)]).T
    outer = np.vstack([r_outer*np.cos(theta),
                       r_outer*np.sin(theta)]).T

    inner += rng.normal(scale=noise, size=inner.shape)
    outer += rng.normal(scale=noise, size=outer.shape)

    X = np.vstack([inner, outer]).T
    y = np.hstack([np.zeros(n_per_ring),
                   np.ones (n_per_ring)])
    return X, y

X, y = generate_circles(n_per_ring=250, noise=0.06, seed=0)
#X, y = generate_xor_data(n_per_quad=20, noise=0.25, seed=0)

plt.scatter(X[0], X[1], c=y, cmap='bwr', alpha=0.7, s=15)
plt.axhline(0, color='grey', lw=0.6); plt.axvline(0, color='grey', lw=0.6)
plt.gca().set_aspect('equal', 'box')
plt.title("Circle dataset")
plt.show()


pca = PCA(n_components=1)
X_PCA = pca.fit_transform(X.T)
plt.scatter(X_PCA, np.zeros_like(X_PCA), c=y, cmap='bwr', alpha=0.7, s=15)
plt.axhline(0, color='grey', lw=0.6); plt.axvline(0, color='grey', lw=0.6)
plt.title("Linear PCA projection of XOR dataset")
plt.show()

degree = 2
X_NPCA = polynomial_pca(X, degree, components=1)
plt.scatter(X_NPCA, np.zeros_like(X_NPCA), c=y, cmap='bwr', alpha=0.7, s=15)
plt.axhline(0, color='grey', lw=0.6); plt.axvline(0, color='grey', lw=0.6)
plt.title("Polynomial PCA projection of XOR dataset")
plt.show()

X_NPCA_gram = kpca_from_gram(gram_matrix(X, degree), *eigen(gram_matrix(X, degree)), m=1)
plt.scatter(X_NPCA_gram, np.zeros_like(X_NPCA_gram), c=y, cmap='bwr', alpha=0.7, s=15)
plt.axhline(0, color='grey', lw=0.6); plt.axvline(0, color='grey', lw=0.6)
plt.title("KPCA projection from Gram matrix of XOR dataset")
plt.show()
