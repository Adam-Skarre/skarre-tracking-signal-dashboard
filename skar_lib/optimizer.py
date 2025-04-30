import numpy as np
import pandas as pd
from .signal_logic import generate_signals
from .backtester import backtest


def optimize_thresholds(
    price_series: pd.Series,
    slope_series: pd.Series,
    accel_series: pd.Series = None,
    entry_min: float = 0.0,
    entry_max: float = 1.0,
    exit_min: float = -1.0,
    exit_max: float = 0.0,
    step: float = 0.1,
    use_acceleration: bool = False,
    metric: str = 'Sharpe'
) -> pd.DataFrame:
    """
    Perform a grid search over entry and exit thresholds to optimize a performance metric.

    Returns a DataFrame with entry thresholds as index, exit thresholds as columns, and metric values.

    Parameters:
    - price_series: pd.Series of prices
    - slope_series: pd.Series of slope values
    - accel_series: pd.Series of acceleration values (optional)
    - entry_min/entry_max: range for entry slope threshold
    - exit_min/exit_max: range for exit slope threshold
    - step: threshold increment
    - use_acceleration: if True, include acceleration filter
    - metric: performance metric to optimize (e.g. 'Sharpe', 'Max Drawdown')
    """
    entry_vals = np.arange(entry_min, entry_max + step, step)
    exit_vals = np.arange(exit_min, exit_max + step, step)

    results = pd.DataFrame(index=np.round(entry_vals, 4), columns=np.round(exit_vals, 4))

    for entry in entry_vals:
        for exit_th in exit_vals:
            signals = generate_signals(
                slope_series, accel_series, entry, exit_th, use_acceleration
            )
            back = backtest(price_series, signals)
            results.loc[np.round(entry, 4), np.round(exit_th, 4)] = back['performance'].get(metric, np.nan)

    return results.astype(float)


def get_optimal_thresholds(results_df: pd.DataFrame) -> tuple:
    """
    Identify the entry, exit pair that maximizes the metric in the results DataFrame.

    Returns a tuple (entry_threshold, exit_threshold).
    """
    # Flatten and find max location
    idx = results_df.stack()  # Series with MultiIndex
    best = idx.idxmax()
    return best  # (entry_val, exit_val)
