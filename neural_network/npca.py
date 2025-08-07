import numpy as np
from scipy.linalg import sqrtm
from whitening import zca_whitening, pca_whitening
from sklearn.decomposition import PCA
import numpy as np
import cv2
import matplotlib.pyplot as plt

def orth(W):
    """
    Perform symmetric orthogonalization of the matrix W.
    """
    return np.linalg.inv(sqrtm(W @ W.T)) @ W

def phi(y):
    """
    Apply the non-linear function phi to the input y.
    Simplified activation function φ(y) from Eq. (14)
    Uses kurtosis to detect sub/super-Gaussianity.
    """
    E_y4 = np.mean(y ** 4, axis=1, keepdims=True)  # shape (n, 1)
    E_y2 = np.mean(y ** 2, axis=1, keepdims=True)  # shape (n, 1)
    return np.sign(E_y4 - 3 * E_y2 ** 2) * (y ** 3 - 3 * y * E_y2)

def npca(X, n_components=None, block_size=10, mu=0.1, max_iter=1000, tol=1e-6):
    """
    Perform NPCA on dataset X (Eq. 16).

    Parameters
    ----------
    X : np.ndarray (m, T)
        Input data (m features, T samples)
    n_components : int or None
        Number of components to extract (if None, use m)
    block_size : int
        Number of samples per block for averaging (L)
    mu : float
        Step size parameter
    max_iter : int
        Max number of iterations
    tol : float
        Convergence tolerance on W

    Returns
    -------
    S : np.ndarray (n, T)
        Estimated source signals (separated signals)
    W : np.ndarray (n, m)
        Separating matrix
    """
    m, T = X.shape
    if n_components is None:
        n_components = m

    # Step 1: Whitening
    X_whitened, U_whiten = pca_whitening(X, n_components)
    #print("Whitening matrix shape:", U_whiten.shape)

    # Step 2: Initialize W randomly (n x n orthonormal matrix)
    W = np.linalg.qr(np.random.randn(n_components, n_components))[0]

    for iteration in range(max_iter):
        W_prev = W.copy()

        # Step 3: Process in blocks
        num_blocks = T // block_size
        grad = np.zeros_like(W)

        for b in range(num_blocks):
            idx_start = b * block_size
            idx_end = (b + 1) * block_size
            V_b = X_whitened[:, idx_start:idx_end]  # (n, block_size)

            # y_t = W_{t-1} * v_t(l)
            Y_b = W @ V_b  # (n, block_size)
            Phi_b = phi(Y_b)  # (n, block_size)

            # Update gradient
            grad += Phi_b @ V_b.T / block_size  # (n, n)

        grad *= (mu / num_blocks)

        # Update W (Eq. 16)
        W = orth(W + grad)

        # Check convergence
        delta = np.max(np.abs(W - W_prev))
        if delta < tol:
            print(f"Converged at iteration {iteration}, Δ={delta:.2e}")
            break

    # Step 4: Recover original scale sources
    S = W @ X_whitened  # Estimated sources (n, T)
    separating_matrix = W @ U_whiten  # Final separating matrix (n, m)
    return S, separating_matrix

def normalize(img):
    img = img - np.min(img)
    return img / np.max(img)

def mix_sources(sources, A):
    return A @ sources

def load_and_prepare_image(path, size=(128, 128)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    return img

# Simulate mixed signals
np.random.seed(0)
img1 = load_and_prepare_image('lena.png')       # Provide paths to your images
img2 = load_and_prepare_image('camera_man.png')  # Replace with actual file paths

# Flatten images into signals
s1 = img1.flatten()
s2 = img2.flatten()

# Stack as sources: shape (2, N_pixels)
S = np.vstack([s1, s2]).astype(np.float64)

# Normalize sources
S = normalize(S)
A = np.random.rand(1000, 2)  # Random mixing matrix
X = mix_sources(S, A)     # Mixed signals

print("Mixed signals shape:", X.shape)

S_est, B = npca(X, n_components=2, block_size=500, mu=0.5, max_iter=500)
print("Estimated sources shape:", S_est.shape)
img1_est = S_est[0, :].reshape(img1.shape)
img2_est = S_est[1, :].reshape(img2.shape)

fig, axes = plt.subplots(3, 2, figsize=(10, 12))
axes = axes.ravel()

# Original Sources
axes[0].imshow(img1, cmap='gray')
axes[0].set_title('Original Image 1')
axes[1].imshow(img2, cmap='gray')
axes[1].set_title('Original Image 2')

# Mixed Signals
axes[2].imshow(X[0, :].reshape(img1.shape), cmap='gray')
axes[2].set_title('Mixed Image 1')
axes[3].imshow(X[1, :].reshape(img1.shape), cmap='gray')
axes[3].set_title('Mixed Image 2')

# Recovered Sources
axes[4].imshow(normalize(img1_est), cmap='gray')
axes[4].set_title('Recovered Image 1 (NPCA)')
axes[5].imshow(normalize(img2_est), cmap='gray')
axes[5].set_title('Recovered Image 2 (NPCA)')

for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()