import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def _adjust_window(series_length: int, window: int, order: int) -> int:
    if window > series_length:
        window = series_length
    if window <= order:
        window = order + 1
    if window % 2 == 0:
        window -= 1
    if window < 3:
        window = 3
    return window

def smooth_price(price_series: pd.Series, window: int = 21, order: int = 2) -> pd.Series:
    arr = price_series.values
    w = _adjust_window(len(arr), window, order)
    try:
        sm = savgol_filter(arr, window_length=w, polyorder=order, mode='mirror')
        return pd.Series(sm, index=price_series.index)
    except Exception:
        return price_series.copy()

def get_slope(price_series: pd.Series, window: int = 21, order: int = 2) -> pd.Series:
    arr = price_series.values
    w = _adjust_window(len(arr), window, order)

    try:
        sl = savgol_filter(arr, window_length=w, polyorder=order, deriv=1, mode='mirror')
    except Exception:
        sl = np.gradient(arr) if len(arr) >= 2 else np.zeros_like(arr)

    return pd.Series(sl, index=price_series.index)

def get_acceleration(price_series: pd.Series, window: int = 21, order: int = 2) -> pd.Series:
    arr = price_series.values
    w = _adjust_window(len(arr), window, order)

    try:
        ac = savgol_filter(arr, window_length=w, polyorder=order, deriv=2, mode='mirror')
    except Exception:
        ac = np.gradient(np.gradient(arr)) if len(arr) >= 3 else np.zeros_like(arr)

    return pd.Series(ac, index=price_series.index)

def fit_polynomial(x_vals: np.ndarray, y_vals: np.ndarray, degree: int = 2) -> np.ndarray:
    return np.polyfit(x_vals, y_vals, degree)

def eval_polynomial(coeffs: np.ndarray, x_vals: np.ndarray) -> np.ndarray:
    poly = np.poly1d(coeffs)
    return poly(x_vals)

def get_polynomial_features(price_series: pd.Series, window: int = 21, degree: int = 2):
    length = len(price_series)
    if length < window:
        raise ValueError("Series length must exceed the window size")
    x = np.arange(window)
    dates, a_vals, b_vals, c_vals = [], [], [], []
    for i in range(window, length + 1):
        y = price_series.iloc[i-window:i].values
        coeffs = np.polyfit(x, y, degree)
        a_vals.append(coeffs[0])
        b_vals.append(coeffs[1])
        c_vals.append(coeffs[2] if degree == 2 else 0)
        dates.append(price_series.index[i-1])
    return (
        pd.Series(a_vals, index=dates),
        pd.Series(b_vals, index=dates),
        pd.Series(c_vals, index=dates),
    )
