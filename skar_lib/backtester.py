import pandas as pd
import numpy as np


def backtest(
    price: pd.Series,
    signal: pd.Series,
    cost: pd.Series = None,
    pos_size: pd.Series = None
) -> (pd.DataFrame, dict):
    """
    Run a backtest of the given signal on price data.

    Parameters:
    - price: pd.Series of asset prices indexed by datetime
    - signal: pd.Series of 0/1 position flags (1 = long)
    - cost: pd.Series of transaction cost per trade (optional)
    - pos_size: pd.Series of position sizing multipliers (optional)

    Returns:
    - df: pd.DataFrame with columns ['Return', 'Equity', 'Position']
    - perf: dict of performance metrics (Total Return, CAGR, Sharpe, Max Drawdown)
    """
    # Daily returns
    returns = price.pct_change().fillna(0)

    # Position sizing
    if pos_size is None:
        position = signal.shift(1).fillna(0)
    else:
        position = (signal.shift(1).fillna(0) * pos_size.shift(1).fillna(0)).fillna(0)

    # Strategy returns
    strat_ret = returns * position

    # Transaction costs
    if cost is not None:
        trades = signal.diff().abs().fillna(0)
        strat_ret = strat_ret - (cost * trades)

    # Equity curve
    equity = (1 + strat_ret).cumprod()

    # Performance metrics
    total_ret = equity.iloc[-1] - 1
    # Approximate CAGR
    days = (equity.index[-1] - equity.index[0]).days
    cagr = (equity.iloc[-1]) ** (365.0 / days) - 1 if days > 0 else np.nan
    sharpe = (strat_ret.mean() / strat_ret.std(ddof=1)) * np.sqrt(252) if strat_ret.std(ddof=1) > 0 else np.nan
    max_dd = (equity / equity.cummax() - 1).min()

    perf = {
        'Total Return': total_ret,
        'CAGR': cagr,
        'Sharpe': sharpe,
        'Max Drawdown': max_dd
    }

    # Build output DataFrame
    df = pd.DataFrame({
        'Return': strat_ret,
        'Equity': equity,
        'Position': position
    })

    return df, perf
