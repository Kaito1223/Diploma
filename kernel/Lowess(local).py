import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess

from make_dataset import make_data_separated

def new_window(title):
    fig = plt.figure(figsize=(7,5))
    fig.canvas.manager.set_window_title(title)
    return fig

def lowess_predict(x_train, y_train, x_eval, frac=0.18, it=0):
    """
    Non-robust LOWESS: local weighted linear smoothing.
    frac in (0,1): neighborhood proportion. it=0 disables robust reweighting.
    """
    lo = lowess(y_train, x_train.ravel(), frac=frac, it=it, return_sorted=True)
    return np.interp(x_eval.ravel(), lo[:,0], lo[:,1])

def contaminate_y(y, rho = 0.1, k_sigma = 6.0, rng = np.random.default_rng(0)):
    y = y.copy()
    n = len(y)
    m = max(1, int(round(rho * n)))
    idx = rng.choice(n, size = m, replace = False)
    s = np.std(y, ddof = 1)
    jumps = k_sigma * s * np.sign(rng.standard_normal(m))
    y[idx] += jumps
    return y, idx

def evaluate_clean_and_plot(X_train, y_train, X_test, y_test, spans, it, x_line):
    """
    for each span(frac), fitting LOWESS, printing MSE/R2
    """
    rows = []
    for frac in spans:
        title = f"LOWESS (span = {frac})"
        y_pred = lowess_predict(X_train, y_train, X_test, frac = frac, it = it)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{title}: mse = {mse:.4f}, r2 = {r2:.4f}")
        rows.append({"spans": frac, "it": it, "mse_clean": mse, "R2_score": r2})

        new_window(title)
        plt.scatter(X_train[:, 0], y_train, s=12, alpha = 0.5, label = "train")
        plt.scatter(X_test[:, 0], y_test, s=12, alpha = 0.5, label = "test")
        y_line = lowess_predict(X_train, y_train, x_line, frac = frac, it = it)
        plt.plot(x_line[:, 0], y_line, lw=2, label="prediction")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        plt.legend()
        plt.tight_layout()

    
    return pd.DataFrame(rows)

def robustness_eval_lowess(Xtr, ytr, Xte, yte, spans, rhos, it=0, k_sigma=6.0, seed=0):
    rows = []
    for frac in spans:
        yhat0 = lowess_predict(Xtr, ytr, Xte, frac=frac, it=it)
        mse0  = mean_squared_error(yte, yhat0)
        r20   = r2_score(yte, yhat0)

        rdrs = []
        for rho in rhos:
            ytr_bad, _ = contaminate_y(
                ytr, rho=rho, k_sigma=k_sigma,
                rng=np.random.default_rng(seed + int(1000 * rho))
            )
            yhat = lowess_predict(Xtr, ytr_bad, Xte, frac=frac, it=it)
            mse  = mean_squared_error(yte, yhat)
            rdr  = mse / mse0
            rdrs.append(rdr)
            rows.append({
                "span": frac, "it": it, "rho": rho,
                "MSE_clean": mse0, "MSE_rho": mse, "RDR": rdr, "R2_clean": r20
            })

        rs = 100.0 / np.mean(rdrs)  # Robustness Score
        rows.append({
            "span": frac, "it": it, "rho": "RS",
            "MSE_clean": mse0, "MSE_rho": None, "RDR": rs, "R2_clean": r20
        })

        sub = [r for r in rows if r["span"] == frac and r["it"] == it and r["rho"] != "RS"]
        sub = pd.DataFrame(sub).sort_values("rho")
        title = f"LOWESS, span={frac}"
        new_window(title)
        plt.plot(sub["rho"], sub["RDR"], marker="o", lw=2)
        plt.axhline(1.0, ls="--", alpha=0.6, label="no change")
        plt.xlabel("ρ")
        plt.ylabel("RDR = MSEρ / MSE0")
        plt.title(title)
        plt.legend()
        plt.tight_layout()

    return pd.DataFrame(rows)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = make_data_separated(
        n_train=400, n_test=120, n_features=1, seed=0
    )
    x_line = np.linspace(X_train.min(), X_train.max(), 600).reshape(-1, 1)

    spans = [0.01, 0.12, 0.30, 0.45, 0.70, 0.90]
    rhos  = [0.05, 0.10, 0.20, 0.30]

    for it in [0]:
        print(f"\nlowess clean performance")
        df_clean = evaluate_clean_and_plot(
            X_train, y_train, X_test, y_test, spans, it, x_line
        )
        print(f"\nlowess robustness")
        df_rob = robustness_eval_lowess(
            X_train, y_train, X_test, y_test, spans, rhos, it=it, k_sigma=6.0, seed=0
        )
        summary = (df_rob[df_rob["rho"].eq("RS")]
                   .loc[:, ["span", "it", "R2_clean", "MSE_clean", "RDR"]]
                   .rename(columns={"RDR": "RobustnessScore_RS"}))
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    plt.show()

