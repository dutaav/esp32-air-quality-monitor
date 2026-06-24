import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from train_model import (FEATURES, TARGET, load_data, least_squares_svd,
                         metrics, split_series)

DIR_HERE = os.path.dirname(os.path.abspath(__file__))
PATH_PNG = os.path.join(DIR_HERE, "model_analysis.png")


def main() -> None:
    df, path = load_data()
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = split_series(X, y, 0.8)
    bias, weights = least_squares_svd(X_train, y_train)
    split_idx = len(X_train)

    y_pred_all = X @ weights + bias
    residual = y - y_pred_all
    mae, rmse, r2 = metrics(y_test, y_pred_all[split_idx:])

    fig, axs = plt.subplots(3, 1, figsize=(11, 9))
    fig.suptitle(f"Linear Regression AQI (test): MAE={mae:.1f}  RMSE={rmse:.1f}  R^2={r2:.3f}",
                 fontsize=12, fontweight="bold")

    idx = np.arange(len(y))

    # 1. Actual vs predicted over time, mark train/test split
    axs[0].plot(idx, y, label="Actual", color="steelblue", lw=1.0)
    axs[0].plot(idx, y_pred_all, label="Predicted", color="orange", lw=1.0, alpha=0.85)
    axs[0].axvline(split_idx, color="red", ls="--", lw=1.0, label="Train/test split")
    axs[0].set_title("Actual vs Predicted AQI")
    axs[0].set_xlabel("Sample"); axs[0].set_ylabel("AQI")
    axs[0].legend(); axs[0].grid(alpha=0.3)

    # 2. Raw residuals + MA(20) (shows short-term bias)
    rm = pd.Series(residual).rolling(20, min_periods=1).mean()
    axs[1].plot(idx, residual, color="lightgray", lw=0.5, label="Residual")
    axs[1].plot(idx, rm, color="crimson", lw=1.3, label="MA(20)")
    axs[1].axhline(0, color="black", lw=0.7)
    axs[1].set_title("Residuals & moving average")
    axs[1].set_xlabel("Sample"); axs[1].set_ylabel("Actual - Predicted")
    axs[1].legend(); axs[1].grid(alpha=0.3)

    # 3. Feature weights sorted by |w|
    order = np.argsort(np.abs(weights))
    names = [FEATURES[i] for i in order]
    values = weights[order]
    colors = ["tomato" if v < 0 else "steelblue" for v in values]
    axs[2].barh(names, values, color=colors)
    axs[2].axvline(0, color="black", lw=0.7)
    axs[2].set_title("Feature weights (sorted by |w|)")
    axs[2].set_xlabel("Coefficient value")
    axs[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PATH_PNG, dpi=140, bbox_inches="tight")
    print(f"Data source : {path}")
    print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  R^2={r2:.4f}")
    print(f"Plot saved  : {PATH_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
