import numpy as np
import pandas as pd


def generate_signals(
    slope_series: pd.Series,
    accel_series: pd.Series = None,
    entry_slope: float = 0.0,
    exit_slope: float = 0.0,
    use_acceleration: bool = False
) -> pd.Series:
    """
    Generate long/flat position signals based on slope and optional acceleration.

    Parameters:
    - slope_series: pd.Series of first derivative values (slope)
    - accel_series: pd.Series of second derivative values (acceleration)
    - entry_slope: threshold above which to enter a long position
    - exit_slope: threshold below which to exit the long position
    - use_acceleration: if True, require accel_series > 0 for entry and < 0 for exit

    Returns:
    - pd.Series of integer positions: 1 for long, 0 for flat
    """
    n = len(slope_series)
    positions = np.zeros(n, dtype=int)
    current_pos = 0

    for i in range(1, n):
        slope = slope_series.iloc[i]
        accel = accel_series.iloc[i] if (use_acceleration and accel_series is not None) else None

        # Entry rule
        if current_pos <= 0 and slope > entry_slope and (not use_acceleration or accel > 0):
            current_pos = 1
        # Exit rule
        elif current_pos >= 1 and slope < exit_slope and (not use_acceleration or accel < 0):
            current_pos = 0

        positions[i] = current_pos

    return pd.Series(positions, index=slope_series.index)
