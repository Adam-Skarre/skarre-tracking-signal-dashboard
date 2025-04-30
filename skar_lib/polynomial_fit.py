import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def get_slope(
    price_series: pd.Series,
    window: int = 21,
    polyorder: int = 3
) -> pd.Series:
    """
    Smooth the price_series with a Savitzky–Golay filter and compute the first derivative (slope).

    Parameters:
    - price_series: pd.Series of price values indexed by datetime.
    - window: odd integer window length for smoothing/filtering.
    - polyorder: polynomial order for the filter.

    Returns:
    - pd.Series of slope values aligned with the input index.
    """
    # Validate window length
    if window % 2 == 0 or window < 3:
        raise ValueError("window must be an odd integer >= 3")

    # Apply Savitzky–Golay: deriv=1 for first derivative
    # Mode 'interp' to avoid edge distortions
    slope_values = savgol_filter(
        price_series.values,
        window_length=window,
        polyorder=polyorder,
        deriv=1,
        mode='interp'
    )

    slope_series = pd.Series(data=slope_values, index=price_series.index)
    return slope_series


def get_acceleration(
    price_series: pd.Series,
    window: int = 21,
    polyorder: int = 3
) -> pd.Series:
    """
    Compute the second derivative (acceleration) of price_series using Savitzky–Golay.

    Parameters:
    - price_series: pd.Series of price values indexed by datetime.
    - window: odd integer window length for smoothing/filtering.
    - polyorder: polynomial order for the filter.

    Returns:
    - pd.Series of acceleration values aligned with the input index.
    """
    if window % 2 == 0 or window < 3:
        raise ValueError("window must be an odd integer >= 3")

    accel_values = savgol_filter(
        price_series.values,
        window_length=window,
        polyorder=polyorder,
        deriv=2,
        mode='interp'
    )

    accel_series = pd.Series(data=accel_values, index=price_series.index)
    return accel_series
