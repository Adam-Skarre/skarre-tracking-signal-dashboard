import pandas as pd
import numpy as np
from .metrics import sharpe_ratio, max_drawdown, win_rate, trade_frequency

def generate_trade_log(price_series: pd.Series, positions: pd.Series) -> pd.DataFrame:
    """
    Generate a log of trades from position signals.

    Parameters:
    - price_series: pd.Series of prices indexed by date
    - positions: pd.Series of position signals (1 for long, 0 for flat)

    Returns:
    - DataFrame with Entry Date, Exit Date, Entry Price, Exit Price, and Return columns
    """
    trades = []
    entry_date = None
    prev_positions = positions.shift(1).fillna(0)

    for date in positions.index:
        prev = prev_positions.loc[date]
        pos = positions.loc[date]

        # Entry: flat -> long
        if prev == 0 and pos == 1:
            entry_date = date

        # Exit: long -> flat
        elif prev == 1 and pos == 0 and entry_date is not None:
            exit_date = date
            entry_price = price_series.loc[entry_date]
            exit_price = price_series.loc[exit_date]
            ret = exit_price / entry_price - 1
            trades.append({
                'Entry Date': entry_date,
                'Exit Date': exit_date,
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Return': ret
            })
            entry_date = None

    return pd.DataFrame(trades)


def backtest(
    price_series: pd.Series,
    positions: pd.Series,
    initial_capital: float = 100.0
) -> dict:
    """
    Run a backtest given price series and position signals.

    Returns a dict with:
      - equity_curve: pd.Series of portfolio value over time
      - strategy_returns: pd.Series of periodic returns
      - trade_log: DataFrame of individual trades
      - performance: dict of performance metrics
    """
    # Compute daily returns
    returns = price_series.pct_change().fillna(0)
    # Apply positions (shifted by 1 to avoid lookahead)
    strategy_returns = positions.shift(1).fillna(0) * returns

    # Build equity curve
    equity_curve = initial_capital * (1 + strategy_returns).cumprod()

    # Extract trade log
    trade_log = generate_trade_log(price_series, positions)

    # Calculate metrics
    perf = {
        'Sharpe': sharpe_ratio(strategy_returns),
        'Max Drawdown': max_drawdown(equity_curve),
        'Win Rate': win_rate(trade_log['Return'] if not trade_log.empty else []),
        'Trade Frequency': trade_frequency(trade_log['Entry Date'],
                                           (price_series.index[-1] - price_series.index[0]).days)
    }

    return {
        'equity_curve': equity_curve,
        'strategy_returns': strategy_returns,
        'trade_log': trade_log,
        'performance': perf
    }
