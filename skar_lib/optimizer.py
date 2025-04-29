import numpy as np
import pandas as pd
from signal_logic import generate_skarre_signal
from backtester    import evaluate_strategy

def grid_search_optimizer(
    price_series: pd.Series,
    entry_slope_grid: list,
    exit_slope_grid: list,
    entry_sst_grid: list,
    exit_sst_grid: list,
    slope_window: int = 5,
    ma_window: int = 20,
    vol_window: int = 20,
    min_holding_days: int = 5,
    cost_params: dict = None
) -> tuple:
    """
    Performs a 4D grid search over entry/exit slope and entry/exit SST.
    Returns: (best_params, best_sharpe, results_df)
    """
    best_sharpe = -np.inf
    best_params = None
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
                    sharpe, r, dd, tr, _ = evaluate_strategy(
                        price_series, sig, cost_params
                    )

                    results.append({
                        "entry_slope": es,
                        "exit_slope": xs,
                        "entry_sst": est,
                        "exit_sst": xst,
                        "Sharpe": sharpe,
                        "Return": r,
                        "Drawdown": dd,
                        "Trades": tr
                    })

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = (es, xs, est, xst)

    df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)
    return best_params, best_sharpe, df
