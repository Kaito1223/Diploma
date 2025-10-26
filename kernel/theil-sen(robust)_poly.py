import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import mean_squared_error, r2_score
from make_dataset import make_data_separated

X_train, y_train, X_test, y_test = make_data_separated(
    n_train=400, n_test=120, n_features=1, seed=0
)

def new_window(title):
    fig = plt.figure(figsize=(7,5))
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass
    return fig

x_line = np.linspace(X_train.min(), X_train.max(), 400).reshape(-1, 1)

degrees = [2, 3, 4, 5, 6] #integers only

for deg in degrees:
    title = f"Theil–Sen (degree={deg})"
    model = Pipeline([
        ("poly",   PolynomialFeatures(degree=deg, include_bias=True)),
        ("scale",  StandardScaler(with_mean=True, with_std=True)),
        ("theil",  TheilSenRegressor(random_state=0, n_jobs=-1))
    ])
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_line = model.predict(x_line)

    mse = mean_squared_error(y_test, y_pred_test)
    r2  = r2_score(y_test, y_pred_test)
    print(f"{title}: MSE={mse:.4f}, R2={r2:.4f}")

    new_window(title)
    plt.scatter(X_train[:,0], y_train, s=12, alpha=0.5, label="train")
    plt.scatter(X_test[:,0],  y_test,  s=12, alpha=0.5, label="test")
    plt.plot(x_line[:,0], y_pred_line, lw=2, label=f"degree={deg}")
    plt.xlabel("x"); plt.ylabel("y"); plt.title(title); plt.legend()
    plt.tight_layout()

plt.show()
