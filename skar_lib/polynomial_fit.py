from scipy.signal import savgol_filter
import numpy as np
import pandas as pd


def _adjust_window(series_length: int, window: int, order: int) -> int:
    """
    Ensure Savitzky-Golay window length is valid:
      - window_length <= series_length
      - window_length > polyorder
      - window_length is odd
    """
    # Cap window at series length (make odd)
    if window > series_length:
        window = series_length if series_length % 2 == 1 else series_length - 1
    # Minimum window must exceed polynomial order
    min_win = order + 2
    if window <= order:
        window = min_win if min_win % 2 == 1 else min_win + 1
    # Ensure odd
    if window % 2 == 0:
        window -= 1
    return max(3, window)


def smooth_price(price_series: pd.Series, window: int = 21, order: int = 2) -> pd.Series:
    """
    Smooth price series using Savitzky–Golay filter (mirror padding).
    Fallback to raw series if filter fails.
    Returns a pandas Series matching the original index.
    """
    values = price_series.values
    try:
        length = len(values)
        w = _adjust_window(length, window, order)
        smoothed = savgol_filter(
            values,
            window_length=w,
            polyorder=order,
            mode='mirror'
        )
    except Exception:
        smoothed = values
    return pd.Series(smoothed, index=price_series.index)


def get_slope(price_series: pd.Series, window: int = 21, order: int = 2) -> pd.Series:
    """
    Compute the first derivative (slope) of the price series.
    Uses Savitzky–Golay filter with mirror padding. Falls back to numpy gradient if filter fails.
    Returns a pandas Series.
    """
    values = price_series.values
    try:
        length = len(values)
        w = _adjust_window(length, window, order)
        slope = savgol_filter(
            values,
            window_length=w,
            polyorder=order,
            deriv=1,
            mode='mirror'
        )
    except Exception:
        slope = np.gradient(values)
    return pd.Series(slope, index=price_series.index)


def get_acceleration(price_series: pd.Series, window: int = 21, order: int = 2) -> pd.Series:
    """
    Compute the second derivative (acceleration) of the price series.
    Uses Savitzky–Golay filter with mirror padding. Falls back to \
numpy second-order gradient if filter fails.
    Returns a pandas Series.
    """
    values = price_series.values
    try:
        length = len(values)
        w = _adjust_window(length, window, order)
        accel = savgol_filter(
            values,
            window_length=w,
            polyorder=order,
            deriv=2,
            mode='mirror'
        )
    except Exception:
        # fallback: second derivative via numpy gradient twice
        accel = np.gradient(np.gradient(values))
    return pd.Series(accel, index=price_series.index)


def fit_polynomial(x_vals, y_vals, degree: int = 2):
    """
    Fit a polynomial of specified degree to x_vals and y_vals.
    Returns numpy array of coefficients.
    """
    return np.polyfit(x_vals, y_vals, degree)


def eval_polynomial(coeffs, x_vals):
    """
    Evaluate polynomial with given coeffs over x_vals.
    """
    poly = np.poly1d(coeffs)
    return poly(x_vals)


def get_polynomial_features(price_series: pd.Series, window: int = 21, degree: int = 2):
    """
    Compute rolling polynomial coefficients for the price series.
    Returns three pandas Series for coefficients (a, b, c).
    """
    length = len(price_series)
    if length < window:
        raise ValueError("Series length must exceed window for polynomial features")

    x = np.arange(window)
    a_vals, b_vals, c_vals = [], [], []
    dates = []

    for i in range(window, length + 1):
        y = price_series.values[i - window:i]
        coeffs = np.polyfit(x, y, degree)
        a_vals.append(coeffs[0])
        b_vals.append(coeffs[1])
        c_vals.append(coeffs[2] if degree == 2 else 0)
        dates.append(price_series.index[i - 1])

    return (
        pd.Series(a_vals, index=dates),
        pd.Series(b_vals, index=dates),
        pd.Series(c_vals, index=dates),
    )
