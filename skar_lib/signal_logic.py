import pandas as pd
import numpy as np

def get_slope(prices: pd.Series, window: int = 5) -> pd.Series:
    """
    1st derivative (momentum) via rolling linear fit.
    """
    return prices.diff().rolling(window).mean().fillna(0)

def get_acceleration(prices: pd.Series, window: int = 5) -> pd.Series:
    """
    2nd derivative: change of slope.
    """
    slope = get_slope(prices, window)
    return slope.diff().fillna(0)

def generate_skarre_signal(
    price_series: pd.Series,
    entry_slope_threshold: float,
    exit_slope_threshold: float,
    entry_sst_threshold: float,
    exit_sst_threshold: float,
    slope_window: int = 5,
    ma_window: int = 20,
    vol_window: int = 20,
    min_holding_days: int = 1
) -> pd.Series:
    """
    Combined slope + SST signal.
    SST = (price - rolling MA) / rolling volatility.
    """
    # Derivatives
    slope = get_slope(price_series, slope_window)
    accel = get_acceleration(price_series, slope_window)

    # SST
    ma  = price_series.rolling(ma_window).mean()
    vol = price_series.pct_change().rolling(vol_window).std().fillna(price_series.pct_change().std())
    sst = ((price_series - ma) / vol).fillna(0)

    # Generate positions
    positions = pd.Series(0, index=price_series.index)
    pos = 0
    last_change = None

    for idx in price_series.index:
        if last_change is None:
            held = min_holding_days
        else:
            held = (idx - last_change).days

        cslope = slope.loc[idx]
        csst   = sst.loc[idx]

        # Entry
        if pos == 0 and cslope > entry_slope_threshold and csst > entry_sst_threshold and held >= min_holding_days:
            pos = 1
            last_change = idx
        # Exit
        elif pos == 1 and (cslope < exit_slope_threshold or csst < exit_sst_threshold) and held >= min_holding_days:
            pos = 0
            last_change = idx

        positions.loc[idx] = pos

    return positions
