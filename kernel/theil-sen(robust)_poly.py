import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import mean_squared_error, r2_score
from make_dataset import make_data_separated

# X_train, y_train, X_test, y_test = make_data_separated(
#     n_train=400, n_test=120, n_features=1, seed=0
# )

def new_window(title):
    fig = plt.figure(figsize=(7,5))
    fig.canvas.manager.set_window_title(title)
    return fig

def build_ts_pipeline(deg):
    return Pipeline([
        ("poly",   PolynomialFeatures(degree=deg, include_bias=True)),
        ("scale",  StandardScaler(with_mean=True, with_std=True)),
        ("theil",  TheilSenRegressor(random_state=0, n_jobs=-1)),
    ])

def contaminate_y(y, rho = 0.1, k_sigma = 6.0, rng=np.random.default_rng(0)):
    """
    add large label outliers to rho-fration fo training points.
    """
    y = y.copy()
    n = len(y)
    m = max(1, int(np.round(rho * n)))
    idx = rng.choice(n, size = m, replace = False)
    s = np.std(y, ddof = 1)
    jumps = k_sigma * s * np.sign(rng.standard_normal(m))
    y[idx] += jumps
    return y, idx

def evaluate_clean_and_plot(X_train, y_train, X_test, y_test, degrees, x_line):
    rows = []
    for deg in degrees:
        title = f"Theil-Sen(degree = {deg})"
        model = build_ts_pipeline(deg)
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_line = model.predict(x_line)
        mse = mean_squared_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        print(f"{title}: MSE = {mse:.4f}, R2 = {r2:.4f}")
        rows.append({"degree:": deg, "MSE": mse, "R2": r2})

        new_window(title)
        plt.scatter(X_train[:, 0], y_train, s = 12, alpha = 0.5, label = "train")
        plt.scatter(X_test[:, 0], y_test, s = 12, alpha = 0.5, label = "test")
        plt.plot(x_line[:, 0], y_pred_line, lw = 2, label = "prediction")
        plt.xlabel("x"); plt.ylabel("y")
        plt.title(title)
        plt.legend()
        plt.tight_layout()

    return pd.DataFrame(rows)

def robustness_eval(Xtr, ytr, Xte, yte, degrees, rhos, k_sigma=6.0, seed=0):
    rows = []
    for deg in degrees:
        model0 = build_ts_pipeline(deg)
        model0.fit(Xtr, ytr)
        yhat0 = model0.predict(Xte)
        mse0  = mean_squared_error(yte, yhat0)
        r20   = r2_score(yte, yhat0)

        rdrs = []
        for rho in rhos:
            ytr_bad, _ = contaminate_y(
                ytr, rho=rho, k_sigma=k_sigma,
                rng=np.random.default_rng(seed + int(1000 * rho))
            )
            model_c = build_ts_pipeline(deg)
            model_c.fit(Xtr, ytr_bad)
            yhat = model_c.predict(Xte)
            mse  = mean_squared_error(yte, yhat)
            rdr  = mse / mse0
            rdrs.append(rdr)
            rows.append({
                "degree": deg, "rho": rho,
                "MSE_clean": mse0, "MSE_rho": mse, "RDR": rdr, "R2_clean": r20
            })

        rs = 100.0 / np.mean(rdrs) # robustness score
        rows.append({
            "degree": deg, "rho": "RS",
            "MSE_clean": mse0, "MSE_rho": None, "RDR": rs, "R2_clean": r20
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = make_data_separated(
        n_train=400, n_test=120, n_features=1, seed=0
    )
    degrees = [2, 3, 4, 5, 6]
    rhos    = [0.05, 0.10, 0.20, 0.30]
    x_line  = np.linspace(X_train.min(), X_train.max(), 400).reshape(-1, 1)

    print("\n Clean performance (per degree)")
    df_clean = evaluate_clean_and_plot(X_train, y_train, X_test, y_test, degrees, x_line)

    df_rob = robustness_eval(X_train, y_train, X_test, y_test, degrees, rhos, k_sigma=6.0, seed=0)

    print("\n clean performance and robustness Score")
    summary = (df_rob[df_rob["rho"].eq("RS")]
               .loc[:, ["degree", "R2_clean", "MSE_clean", "RDR"]]
               .rename(columns={"RDR": "RobustnessScore_RS"}))
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    plt.show()
