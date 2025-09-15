import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.datasets import make_moons, make_circles, load_digits
import time


cv = StratifiedKFold(5, shuffle=True, random_state=0)
clf = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                    LogisticRegression(max_iter=1000))

from poly_pca import (
    MomentPolynomialPCA,
    PolynomialKernelPCA,
    T_statistic,
    Q_statistic,
)

def generate_xor3d_data(n_per_corner=50, noise=0.3, seed=None):
    """
    3D XOR (parity) dataset in R^3.

    Returns
    -------
    X : ndarray, shape (3, n)   # columns are samples
    y : ndarray, shape (n,)     # labels in {+1, -1}
    """
    rng = np.random.default_rng(seed)

    # 8 corners of the cube
    centers = np.array([
        [ 1,  1,  1],
        [-1,  1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [ 1, -1, -1],
        [-1, -1, -1],
    ], dtype=float)

    # Label by parity: product of signs > 0 (even #neg) -> +1, else -1
    labels = np.sign(np.prod(centers, axis=1)).astype(int)

    X_chunks = []
    y_chunks = []
    for c, lab in zip(centers, labels):
        samples = rng.normal(loc=c, scale=noise, size=(n_per_corner, 3))
        X_chunks.append(samples)
        y_chunks.append(np.full(n_per_corner, lab, dtype=int))

    X = np.vstack(X_chunks).T   # (3, 8*n_per_corner)
    y = np.concatenate(y_chunks)
    return X, y


def generate_circles(n_per_ring=300, r_inner=1.0, r_outer=2.0,
                     noise=0.05, seed=None):
    """
    Two noisy concentric circles in ℝ².

    Returns
    -------
    X : ndarray (2, n)
    y : ndarray (n,)
    """
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

#X, y = generate_circles(n_per_ring=20, noise=0.06, seed=0)

X, y = generate_xor3d_data(n_per_corner=60, noise=0.25, seed=0)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X[0], X[1], X[2],
    c=y, cmap="coolwarm", alpha=0.7, s=40
)

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
ax.set_title("3D XOR Dataset")
plt.show()

# X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1, random_state=0)
# X = X.T

start_pca = time.time()
pca = PCA(n_components=3)
X_PCA = pca.fit_transform(X.T)  # shape (n, 1)
evr_pca = pca.explained_variance_ratio_.sum()
acc_pca = cross_val_score(clf, X_PCA, y, cv=cv).mean()
end_pca = time.time()
print(f"PCA time: {end_pca - start_pca:.3f} sec")


plt.scatter(X_PCA[:, 0], X_PCA[:, 1], c=y, cmap='bwr', alpha=0.7, s=15)
plt.axhline(0, color='grey', lw=0.6); plt.axvline(0, color='grey', lw=0.6)
plt.title("Dataset")
plt.show()

degree = 3
start_kpca = time.time()
kpca_sklearn = KernelPCA(
    kernel='poly',
    n_components=4,
    degree=degree,
    coef0=0.0,
)
Z_kpca_sklearn = kpca_sklearn.fit_transform(X.T)
end_kpca = time.time()
print(f"sklearn KPCA time: {end_kpca - start_kpca:.3f} sec")
# plt.figure()
# plt.scatter(Z_kpca_sklearn[:, 0], Z_kpca_sklearn[:, 1], c=y, cmap='bwr', alpha=0.7, s=15)
# plt.axhline(0, color='grey', lw=0.6); plt.axvline(0, color='grey', lw=0.6)
# plt.title("Kernel PCA (sklearn, polynomial, 2D projection)")
# plt.gca().set_aspect('equal', 'box')

start_mpca = time.time()
mpca = MomentPolynomialPCA(
    degree=degree,
    const= 0.0,
    n_components=4,
    tol=1e-12,
    score_norm="none"
)
Z_moment = mpca.fit_transform(X)
enfd_mpca = time.time()
print(f"Moment-Poly PCA time: {enfd_mpca - start_mpca:.3f} sec")

acc_kpca = cross_val_score(clf, Z_moment.T, y, cv=cv).mean()
# plt.figure()
# plt.scatter(Z_moment[0], Z_moment[1], c=y, cmap='bwr', alpha=0.7, s=15)
# plt.axhline(0, color='grey', lw=0.6); plt.axvline(0, color='grey', lw=0.6)
# plt.title("Moment Polynomial PCA (2D projection)")
# plt.gca().set_aspect('equal', 'box')
# plt.show()

print("Moment-Poly PCA eigenvalues:", mpca.eigvals_)
print("Moment-Poly PCA eigenvectors:\n", mpca.eigvecs_)

kpca = PolynomialKernelPCA(
    degree=degree,
    const= 0.0,
    n_components=4,
    center_kernel=True,
    tol=1e-12,
    score_norm="eig"
)
Z_kernel = kpca.fit_transform(X)
# plt.figure()
# plt.scatter(Z_kernel[0], Z_kernel[1], c=y, cmap='bwr', alpha=0.7, s=15)
# plt.axhline(0, color='grey', lw=0.6); plt.axvline(0, color='grey', lw=0.6)
# plt.title("Kernel PCA (polynomial, 2D projection)")
# plt.gca().set_aspect('equal', 'box')
# plt.show()

# print("Kernel PCA (Gram) eigenvalues:", kpca.eigvals_)

print(f"PCA-CV-Acc: {acc_pca:.3f}")
print(f"NPCA-CV-Acc: {acc_kpca:.3f}")
