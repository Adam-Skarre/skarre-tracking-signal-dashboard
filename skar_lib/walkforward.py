import pandas as pd
from backtester import run_backtest  # your existing backtester entry-point
from typing import Callable, List, Dict

def run_walkforward(
    price: pd.Series,
    signal_fn: Callable[[pd.Series], pd.Series],
    train_window: int,
    test_window: int,
    step: int
) -> pd.DataFrame:
    """
    Splits `price` into rolling train/test windows, applies `signal_fn` to each train,
    backtests on each test set, and aggregates fold metrics.

    Returns a DataFrame with one row per fold and columns for:
      - train_start, train_end, test_start, test_end
      - return, sharpe, max_drawdown, etc.
    """
    results: List[Dict] = []
    n = len(price)
    idx = price.index

    # walk-forward
    for start in range(0, n - train_window - test_window + 1, step):
        t0, t1 = start, start + train_window
        u0, u1 = t1, t1 + test_window

        train = price.iloc[t0:t1]
        test  = price.iloc[u0:u1]

        # fit signal on train
        sig_train = signal_fn(train)

        # backtest on test using the same params / signal logic
        metrics = run_backtest(test, lambda series: signal_fn(series))
        
        # record fold
        results.append({
            "train_start": idx[t0].strftime("%Y-%m-%d"),
            "train_end":   idx[t1-1].strftime("%Y-%m-%d"),
            "test_start":  idx[u0].strftime("%Y-%m-%d"),
            "test_end":    idx[u1-1].strftime("%Y-%m-%d"),
            **metrics  # unpack return, sharpe, drawdown, etc.
        })

    return pd.DataFrame(results)
