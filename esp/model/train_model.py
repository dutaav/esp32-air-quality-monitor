import os
import numpy as np
import pandas as pd

DIR_HERE  = os.path.dirname(os.path.abspath(__file__))
PATH_DATA = os.path.join(DIR_HERE, "..", "dataset", "cleaned.csv")

FEATURES = ["AQI", "AQI_prev1", "AQI_prev2",
            "delta_AQI", "delta2_AQI", "Temp", "Humidity"]
TARGET = "PredictedAQI"

# mapping to const names in esp.ino
C_NAMES = ["W_AQI", "W_PREV1", "W_PREV2",
           "W_DELTA", "W_ACCEL", "W_TEMP", "W_HUM"]


def least_squares_svd(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Minimum-norm solution to Xw = y via SVD.

    theta = pinv(X') @ y  where X' = [1 | X]. lstsq uses SVD so it
    stays valid when X is singular / collinear (our delta & delta2 case).
    """
    n = X.shape[0]
    X1 = np.hstack([np.ones((n, 1)), X])
    theta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return float(theta[0]), theta[1:]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    err = y_pred - y_true
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mae, rmse, r2


def split_series(X: np.ndarray, y: np.ndarray, ratio: float = 0.8):
    split = int(len(X) * ratio)
    return X[:split], X[split:], y[:split], y[split:]


def load_data() -> tuple[pd.DataFrame, str]:
    if not os.path.exists(PATH_DATA):
        raise FileNotFoundError(
            "Dataset not found. Run record_serial.py first (records + cleans in one step)."
        )
    df = pd.read_csv(PATH_DATA)
    return df, PATH_DATA


def print_coefficients_c(bias: float, weights: np.ndarray) -> None:
    print("\n--- Weights (paste into esp.ino) ---")
    print(f"const float W_BIAS  = {bias:+.5f}f;")
    for name, w in zip(C_NAMES, weights):
        print(f"const float {name:<7} = {w:+.5f}f;")


def main() -> None:
    df, path = load_data()
    print(f"Data source : {path}")
    print(f"Total rows  : {len(df)}")

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = split_series(X, y, 0.8)
    bias, weights = least_squares_svd(X_train, y_train)

    y_pred = X_test @ weights + bias
    mae, rmse, r2 = metrics(y_test, y_pred)

    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
    print("--- Test set performance ---")
    print(f"MAE  = {mae:.2f} AQI units")
    print(f"RMSE = {rmse:.2f}")
    print(f"R^2  = {r2:.4f}")

    print_coefficients_c(bias, weights)


if __name__ == "__main__":
    main()
