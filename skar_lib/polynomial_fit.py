from scipy.signal import savgol_filter
import numpy as np


def smooth_price(price_series, window=21, order=2):
    """
    Apply Savitzky–Golay filter to smooth price series.
    """
    return savgol_filter(price_series, window_length=window, polyorder=order)


def get_slope(price_series, window=21, order=2):
    """
    Compute the first derivative (slope) of the smoothed price series.
    """
    return savgol_filter(price_series, window_length=window, polyorder=order, deriv=1)


def get_acceleration(price_series, window=21, order=2):
    """
    Compute the second derivative (acceleration) of the smoothed price series.
    """
    return savgol_filter(price_series, window_length=window, polyorder=order, deriv=2)


def fit_polynomial(x_vals, y_vals, degree=2):
    """
    Fit a polynomial of specified degree to the data and return coefficients.
    """
    return np.polyfit(x_vals, y_vals, degree)


def eval_polynomial(coeffs, x_vals):
    """
    Evaluate a polynomial at the given x values.
    """
    poly = np.poly1d(coeffs)
    return poly(x_vals)


def get_polynomial_features(price_series, window=21, degree=2):
    """
    Fit polynomial over a rolling window and return time series of coefficients.
    Returns three arrays: a_coeffs, b_coeffs, c_coeffs for quadratic fits.
    """
    a_coeffs, b_coeffs, c_coeffs = [], [], []
    x = np.arange(window)

    for i in range(window, len(price_series)):
        y_window = price_series[i - window : i]
        coeffs = np.polyfit(x, y_window, degree)
        a_coeffs.append(coeffs[0])
        b_coeffs.append(coeffs[1])
        c_coeffs.append(coeffs[2] if degree == 2 else 0)

    return np.array(a_coeffs), np.array(b_coeffs), np.array(c_coeffs)

