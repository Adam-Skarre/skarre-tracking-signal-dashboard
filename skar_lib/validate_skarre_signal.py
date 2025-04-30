import numpy as np
import pandas as pd
from backtester import backtest


def bootstrap_sharpe(
    price: pd.Series,
    signal: pd.Series,
    n_boot: int = 1000,
    block_size: int = 20,
    cost: pd.Series = None,
    pos_size: pd.Series = None
) -> (float, float):
    """
    Compute observed Sharpe and p-value under null by block bootstrap of returns.

    Returns:
    - observed_sharpe, p_value
    """
    # Full backtest to get observed Sharpe
    df, perf = backtest(price, signal, cost, pos_size)
    obs_sharpe = perf['Sharpe']

    # Calculate strategy returns
    strat_ret = df['Return']

    # Block bootstrap p-value
    n = len(strat_ret)
    boot_sharpes = []
    for _ in range(n_boot):
        # sample start index for block
        starts = np.random.randint(0, n-block_size, size=int(np.ceil(n/block_size)))
        resampled = []
        for s in starts:
            resampled.extend(strat_ret.values[s:s+block_size])
            if len(resampled) >= n:
                break
        resampled = np.array(resampled[:n])
        # compute Sharpe
        sr = np.nan
        if np.std(resampled, ddof=1) > 0:
            sr = np.mean(resampled)/np.std(resampled, ddof=1)*np.sqrt(252)
        boot_sharpes.append(sr)
    boot_sharpes = np.array(boot_sharpes)

    # p-value: fraction of boot Sharpe >= obs if obs>0, else <= obs
    if obs_sharpe >= 0:
        p_value = np.mean(boot_sharpes >= obs_sharpe)
    else:
        p_value = np.mean(boot_sharpes <= obs_sharpe)

    return obs_sharpe, p_value


def regime_performance(
    price: pd.Series,
    signal: pd.Series,
    slope: pd.Series
) -> pd.DataFrame:
    """
    Split performance into bull/bear regimes based on slope >0.

    Returns a DataFrame with index ['bull','bear'] and columns ['Sharpe','Total Return','Max Drawdown'].
    """
    # Define regimes
    bull = slope > 0
    bear = ~bull

    results = {}
    for name, mask in [('bull', bull), ('bear', bear)]:
        # filter price+signal
        pr = price[mask]
        sg = signal[mask]
        if len(pr) < 2:
            # insufficient data
            results[name] = {'Sharpe': np.nan, 'Total Return': np.nan, 'Max Drawdown': np.nan}
            continue
        df, perf = backtest(pr, sg)
        results[name] = {
            'Sharpe': perf['Sharpe'],
            'Total Return': perf['Total Return'],
            'Max Drawdown': perf['Max Drawdown']
        }
    return pd.DataFrame(results).T
