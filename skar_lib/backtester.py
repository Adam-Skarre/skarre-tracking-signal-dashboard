import pandas as pd
import numpy as np


def backtest(price: pd.Series, signal: pd.Series) -> dict:
    """
    Simple backtest engine:
    - Buys when signal == 1
    - Sells when signal == -1
    - Holds otherwise
    Returns a dict with 'metrics' and 'trade_log'.
    """
    # Align signal to next bar (trade on open next day)
    position = signal.shift(1).fillna(0)

    # Calculate daily returns
    returns = price.pct_change().fillna(0)
    strat_returns = returns * position

    # Equity curve
    equity = (1 + strat_returns).cumprod()

    # Build trade log DataFrame using Series directly for alignment
    trades = pd.DataFrame({
        "Price": price,
        "Signal": signal,
        "Position": position,
        "Return": strat_returns,
        "Equity": equity
    })

    # Compute metrics
    total_return = equity.iloc[-1] - 1
    sharpe = (strat_returns.mean() / strat_returns.std(ddof=1)) * np.sqrt(252) if strat_returns.std(ddof=1) != 0 else np.nan
    max_drawdown = (equity / equity.cummax() - 1).min()

    # Safely calculate trade frequency using the price index
    if len(price.index) > 1:
        trade_frequency = len(trades) / ((price.index[-1] - price.index[0]).days / 365)
    else:
        trade_frequency = 0

    metrics = {
        "total_return": round(total_return, 6),
        "sharpe": round(sharpe, 6),
        "max_drawdown": round(max_drawdown, 6),
        "trade_frequency": round(trade_frequency, 2)
    }

    return {"metrics": metrics, "trade_log": trades}
