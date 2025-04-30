# skar_lib/backtester.py

import pandas as pd
import numpy as np

def evaluate_strategy(
    price_series: pd.Series,
    signal_series: pd.Series,
    cost_params: dict = None
):
    """
    Returns: sharpe, total_return, max_drawdown, trade_count, cumulative_returns
    """
    # 1) Align returns with signals
    returns = price_series.pct_change().fillna(0)
    sig     = signal_series.shift(1).fillna(0)
    strat_r = returns * sig

    # 2) Dynamic transaction cost: base + slippage * volatility
    vol  = returns.rolling(20).std().fillna(returns.std())
    base = cost_params.get("base_cost", 0.001) if cost_params else 0.001
    slip = cost_params.get("slippage_factor", 0.5) if cost_params else 0.5
    cost = base + slip * vol

    # 3) Compute trade penalties
    trades        = sig.diff().abs()
    cost_penalty  = cost * trades

    # 4) Net returns and equity curve
    net_r = strat_r - cost_penalty
    cum   = (1 + net_r).cumprod()

    # 5) If no trades (empty equity), bail out gracefully
    if cum.empty:
        # Return NaN for Sharpe, zeros for others, and an empty series
        return np.nan, 0.0, 0.0, 0, pd.Series(dtype=float)

    # 6) Performance metrics
    mean    = net_r.mean()
    std     = net_r.std(ddof=1)
    sharpe  = (mean / std) * np.sqrt(252) if std > 0 else np.nan
    tot_ret = cum.iloc[-1] - 1
    max_dd  = (cum / cum.cummax() - 1).min()
    n_trades = int(trades.sum())

    return sharpe, tot_ret, max_dd, n_trades, cum


def backtest(
    price_series: pd.Series,
    signal_series: pd.Series,
    cost_params: dict = None
) -> dict:
    """
    Executes signals and returns:
      - performance dict
      - equity_curve (pd.Series)
      - trade_log (pd.DataFrame)
    """
    # Run evaluation
    sharpe, tot_ret, max_dd, n_trades, cum = evaluate_strategy(
        price_series, signal_series, cost_params
    )

    # Build a simple trade log
    log = []
    prev = signal_series.iloc[0]
    for idx, curr in signal_series.items():
        if curr != prev:
            log.append({"Date": idx, "Position": int(curr)})
            prev = curr
    trade_df = pd.DataFrame(log)

    # Assemble performance dict
    years = (price_series.index[-1] - price_series.index[0]).days / 365
    perf = {
        "Sharpe": sharpe,
        "Total Return": tot_ret,
        "Max Drawdown": max_dd,
        "Trade Frequency": (n_trades / years) if years > 0 else n_trades
    }

    return {
        "performance": perf,
        "equity_curve": cum,
        "trade_log": trade_df
    }
