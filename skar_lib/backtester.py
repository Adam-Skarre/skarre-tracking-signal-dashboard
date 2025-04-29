# backtester.py

import pandas as pd
import numpy as np
from signal_logic import generate_skarre_signal
from optimizer import grid_search_optimizer

def evaluate_strategy(price_series, signal_series, cost_params=None):
    """
    Evaluate performance of a trading signal.
    Returns: Sharpe, total return, max drawdown, trades, cumulative returns
    """
    returns = price_series.pct_change().fillna(0)
    signal = signal_series.shift(1).reindex(returns.index).fillna(0)
    strat_returns = returns * signal

    # Transaction cost model
    vol = returns.rolling(20).std().fillna(returns.std())
    base_cost = cost_params.get("base_cost", 0.001) if cost_params else 0.001
    slip_factor = cost_params.get("slippage_factor", 0.5) if cost_params else 0.5
    cost_per_trade = base_cost + slip_factor * vol
    trades = signal.diff().abs()
    cost_penalty = cost_per_trade * trades

    net_returns = strat_returns - cost_penalty
    cumulative = (1 + net_returns).cumprod()

    mean = net_returns.mean()
    std = net_returns.std(ddof=1)
    sharpe = (mean / std) * np.sqrt(252) if std > 0 else np.nan
    total_return = cumulative.iloc[-1] - 1
    max_dd = (cumulative / cumulative.cummax() - 1).min()
    trade_count = trades.sum()

    return sharpe, total_return, max_dd, trade_count, cumulative


def walk_forward_backtest(
    price_series,
    entry_slope_grid,
    exit_slope_grid,
    entry_sst_grid,
    exit_sst_grid,
    cost_params=None,
    train_years=5,
    test_years=1,
    min_holding_days=5,
    slope_window=5,
    ma_window=20,
    vol_window=20
):
    """
    Performs walk-forward optimization and backtesting.
    Returns DataFrame of performance by window.
    """
    result_rows = []
    start_dates = pd.date_range(price_series.index[0], price_series.index[-1], freq='12M')

    for start in start_dates:
        train_end = start + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        test_end = train_end + pd.DateOffset(years=test_years)

        train_data = price_series[start:train_end]
        test_data = price_series[train_end + pd.Timedelta(days=1):test_end]

        if len(train_data) < 252 * train_years or len(test_data) < 252 * test_years:
            continue

        # Optimize on train
        best_params, _, _ = grid_search_optimizer(
            train_data,
            entry_slope_grid,
            exit_slope_grid,
            entry_sst_grid,
            exit_sst_grid,
            slope_window,
            ma_window,
            vol_window,
            min_holding_days,
            cost_params
        )

        # Apply on test
        es, xs, est, xst = best_params
        signal_test = generate_skarre_signal(
            test_data,
            entry_slope_threshold=es,
            exit_slope_threshold=xs,
            entry_sst_threshold=est,
            exit_sst_threshold=xst,
            slope_window=slope_window,
            ma_window=ma_window,
            vol_window=vol_window,
            min_holding_days=min_holding_days
        )

        sharpe, ret, dd, trades, _ = evaluate_strategy(test_data, signal_test, cost_params)

        result_rows.append({
            "train_start": start.date(),
            "train_end": train_end.date(),
            "test_end": test_end.date(),
            "entry_slope": es,
            "exit_slope": xs,
            "entry_sst": est,
            "exit_sst": xst,
            "Sharpe": sharpe,
            "Return": ret,
            "Drawdown": dd,
            "Trades": trades
        })

    return pd.DataFrame(result_rows)
