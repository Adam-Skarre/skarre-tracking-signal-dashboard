import pandas as pd
from backtester import backtest
from signal_logic import generate_signals


def grid_search_optimizer(
    price: pd.Series,
    slope: pd.Series,
    accel: pd.Series,
    entry_range: list,
    exit_range: list,
    use_sst: bool = False,
    ma_window: int = 50,
    vol_window: int = 20,
    min_holding_days: int = 3,
    cost_series: pd.Series = None,
    pos_size_series: pd.Series = None
) -> (tuple, pd.DataFrame):
    """
    Perform grid search over entry and exit thresholds to maximize Sharpe ratio.

    Parameters:
    - price: pd.Series of prices
    - slope: pd.Series of first derivative
    - accel: pd.Series of second derivative
    - entry_range: sequence of entry threshold values
    - exit_range: sequence of exit threshold values
    - use_sst: flag to use SST logic instead of raw slope
    - ma_window, vol_window: parameters for SST
    - min_holding_days: minimum holding period for signals
    - cost_series: pd.Series of transaction costs (optional)
    - pos_size_series: pd.Series of position sizes (optional)

    Returns:
    - best_params: (entry_threshold, exit_threshold, Sharpe)
    - results_df: pd.DataFrame with columns ['entry', 'exit', 'Sharpe', 'Total Return', 'Max Drawdown']
    """
    records = []
    for entry in entry_range:
        for exit in exit_range:
            # generate signals
            signals = generate_signals(
                slope=slope,
                accel=accel,
                entry_slope=entry,
                exit_slope=exit,
                use_sst=use_sst,
                price=price,
                ma_window=ma_window,
                vol_window=vol_window,
                min_holding_days=min_holding_days
            )
            # backtest
            df, perf = backtest(
                price=price,
                signal=signals,
                cost=cost_series,
                pos_size=pos_size_series
            )
            records.append({
                'entry': entry,
                'exit': exit,
                'Sharpe': perf['Sharpe'],
                'Total Return': perf['Total Return'],
                'Max Drawdown': perf['Max Drawdown']
            })
    results_df = pd.DataFrame.from_records(records)
    # drop nan Sharpe
    results_df = results_df.dropna(subset=['Sharpe'])
    # find best Sharpe
    best = results_df.loc[results_df['Sharpe'].idxmax()]
    best_params = (best['entry'], best['exit'], best['Sharpe'])
    return best_params, results_df
