# validate_skarre_signal.py

import numpy as np
import pandas as pd
from signal_logic import generate_skarre_signal
from backtester import evaluate_strategy
import matplotlib.pyplot as plt

def bootstrap_sharpe(returns, n_bootstrap=10000):
    """
    Returns observed Sharpe ratio and p-value from bootstrap sampling.
    """
    returns = returns.dropna()
    obs_sharpe = (returns.mean() / returns.std(ddof=1)) * np.sqrt(252)

    bootstraps = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        sh = (np.mean(sample) / np.std(sample, ddof=1)) * np.sqrt(252)
        bootstraps.append(sh)

    p_value = np.mean([s >= obs_sharpe for s in bootstraps])
    return obs_sharpe, p_value


def regime_labels(price_series, ma_period=200):
    """
    Label each day as 'bull', 'sideways', or 'bear' based on 200-day trend.
    """
    trend = price_series.pct_change(ma_period)
    regime = pd.cut(trend, bins=[-np.inf, -0.01, 0.01, np.inf], labels=["bear", "sideways", "bull"])
    return regime.reindex(price_series.index)


def regime_performance(price_series, signal_func, **kwargs):
    """
    Evaluate signal strategy performance by regime.
    """
    regime = regime_labels(price_series)
    results = []

    for r in ["bull", "sideways", "bear"]:
        sub_idx = regime[regime == r].index
        sub_prices = price_series.loc[sub_idx]

        if len(sub_prices) < 252:
            continue

        sig = signal_func(sub_prices, **kwargs)
        sh, ret, dd, trades, _ = evaluate_strategy(sub_prices, sig)

        results.append({
            "Regime": r,
            "Sharpe": sh,
            "Return": ret,
            "Drawdown": dd,
            "Trades": trades
        })

    return pd.DataFrame(results)


def plot_comparison(price_series, signal_series, title="Strategy vs Buy & Hold"):
    """
    Plot strategy and benchmark cumulative returns.
    """
    returns = price_series.pct_change().fillna(0)
    strat_returns = signal_series.shift(1) * returns
    strat_returns = strat_returns.fillna(0)
    cost_adj = strat_returns - 0.001  # Approximate penalty

    cumulative_strat = (1 + cost_adj).cumprod()
    cumulative_bh = (1 + returns).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_strat, label='Skarre Strategy')
    plt.plot(cumulative_bh, label='Buy & Hold', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
