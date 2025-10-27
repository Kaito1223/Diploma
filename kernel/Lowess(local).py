import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess

from make_dataset import make_data_separated

def new_window(title):
    fig = plt.figure(figsize=(7,5))
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass
    return fig

def lowess_predict(x_train, y_train, x_eval, frac=0.18, it=0):
    """
    Non-robust LOWESS: local weighted linear smoothing.
    frac in (0,1): neighborhood proportion. it=0 disables robust reweighting.
    """
    lo = lowess(y_train, x_train.ravel(), frac=frac, it=it, return_sorted=True)
    return np.interp(x_eval.ravel(), lo[:,0], lo[:,1])

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = make_data_separated(
        n_train=400, n_test=120, n_features=1, seed=0
    )

    spans = [0.01, 0.12, 0.3, 0.45, 0.7, 0.9]

    x_line = np.linspace(X_train.min(), X_train.max(), 600).reshape(-1, 1)

    for frac in spans:
        title = f"lowess (frac={frac})"
        y_pred = lowess_predict(X_train, y_train, X_test, frac=frac, it=0)
        mse = mean_squared_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)
        print(f"{title}: mse={mse:.4f}, r2={r2:.4f}")

        new_window(title)
        plt.scatter(X_train[:,0], y_train, s=12, alpha=0.5, label="train")
        plt.scatter(X_test[:,0],  y_test,  s=12, alpha=0.5, label="test")
        y_line = lowess_predict(X_train, y_train, x_line, frac=frac, it=0)
        plt.plot(x_line[:,0], y_line, lw=2, label=title)
        plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.tight_layout()

    plt.show()