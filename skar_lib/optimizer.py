# optimizer.py

import numpy as np
import pandas as pd
from signal_logic import generate_skarre_signal
from backtester import evaluate_strategy

def grid_search_optimizer(
    price_series,
    entry_slope_grid,
    exit_slope_grid,
    entry_sst_grid,
    exit_sst_grid,
    slope_window=5,
    ma_window=20,
    vol_window=20,
    min_holding_days=5,
    cost_params=None,
):
    """
    Search best parameters across slope and SST thresholds.
    Returns best Sharpe and associated parameter set.
    """
    best_score = -np.inf
    best_params = (0, 0, 0, 0)
    results = []

    for es in entry_slope_grid:
        for xs in exit_slope_grid:
            for est in entry_sst_grid:
                for xst in exit_sst_grid:
                    sig = generate_skarre_signal(
                        price_series,
                        entry_slope_threshold=es,
                        exit_slope_threshold=xs,
                        entry_sst_threshold=est,
                        exit_sst_threshold=xst,
                        slope_window=slope_window,
                        ma_window=ma_window,
                        vol_window=vol_window,
                        min_holding_days=min_holding_days
                    )
                    sharpe, ret, dd, trades, _ = evaluate_strategy(price_series, sig, cost_params)

                    results.append({
                        "entry_slope": es,
                        "exit_slope": xs,
                        "entry_sst": est,
                        "exit_sst": xst,
                        "Sharpe": sharpe,
                        "Return": ret,
                        "Drawdown": dd,
                        "Trades": trades
                    })

                    if sharpe > best_score:
                        best_score = sharpe
                        best_params = (es, xs, est, xst)

    result_df = pd.DataFrame(results).sort_values(by="Sharpe", ascending=False)
    return best_params, best_score, result_df
