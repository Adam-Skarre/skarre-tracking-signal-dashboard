import pandas as pd
import numpy as np


def generate_signals(
    slope: pd.Series,
    accel: pd.Series,
    entry_slope: float,
    exit_slope: float,
    use_sst: bool = False,
    price: pd.Series = None,
    ma_window: int = 50,
    vol_window: int = 20,
    min_holding_days: int = 3
) -> pd.Series:
    """
    Generate long/flat trading signals based on slope, acceleration, and optional SST.

    Parameters:
    - slope: pd.Series of first derivatives
    - accel: pd.Series of second derivatives (unused if use_sst)
    - entry_slope: threshold to enter a long position
    - exit_slope: threshold to exit a long position
    - use_sst: whether to use SST = (price - MA)/volatility instead of raw slope
    - price: pd.Series of prices (required if use_sst=True)
    - ma_window: lookback window for moving average in SST
    - vol_window: lookback window for volatility in SST
    - min_holding_days: minimum bars to hold a position

    Returns:
    - pd.Series of 0 (flat) or 1 (long) signals
    """
    if use_sst and price is None:
        raise ValueError("price series must be provided when use_sst=True")

    # Precompute SST if needed
    if use_sst:
        ma = price.rolling(window=ma_window).mean()
        vol = price.pct_change().rolling(window=vol_window).std().replace(0, np.nan)
        sst = (price - ma) / vol
        entry_signal = sst > entry_slope
        exit_signal  = sst < exit_slope
    else:
        entry_signal = slope > entry_slope
        exit_signal  = slope < exit_slope

    # Initialize signal series
    signal = pd.Series(0, index=slope.index, dtype=int)
    position = 0
    hold_count = 0

    for t in slope.index:
        if position == 0:
            if entry_signal.loc[t]:
                position = 1
                hold_count = 1
        else:
            hold_count += 1
            if exit_signal.loc[t] and hold_count >= min_holding_days:
                position = 0
                hold_count = 0
        signal.loc[t] = position

    return signal
