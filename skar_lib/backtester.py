import pandas as pd
import numpy as np

def backtest(price: pd.Series, signal: pd.Series) -> dict:
    """
    Simple backtest engine:
      - Buys when signal == 1
      - Sells when signal == -1
      - Holds otherwise
    Returns a dict with 'metrics' and 'trade_log' (both pandas objects).
    """
    # 1. Align signal to next bar
    position = signal.shift(1).fillna(0)

    # 2. Daily returns and strategy returns
    returns = price.pct_change().fillna(0)
    strat_returns = returns * position

    # 3. Equity curve
    equity = (1 + strat_returns).cumprod()

    # 4. Build trade_log DataFrame
    trades = pd.DataFrame({
        "Price": price,
        "Signal": signal,
        "Position": position,
        "Return": strat_returns,
        "Equity": equity
    })

    # 5. Handle empty data case
    if equity.empty:
        metrics = {
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "trade_frequency": 0.0
        }
        return {"metrics": metrics, "trade_log": trades}

    # 6. Compute summary metrics
    total_return = equity.iloc[-1] - 1
    sharpe = (strat_returns.mean() / strat_returns.std(ddof=1)) * np.sqrt(252) \
             if strat_returns.std(ddof=1) != 0 else np.nan
    max_drawdown = (equity / equity.cummax() - 1).min()

    # 7. Compute trade frequency
    if len(price.index) > 1:
        trade_frequency = len(trades) / ((price.index[-1] - price.index[0]).days / 365)
    else:
        trade_frequency = 0.0

    metrics = {
        "total_return": round(total_return, 6),
        "sharpe": round(sharpe, 6),
        "max_drawdown": round(max_drawdown, 6),
        "trade_frequency": round(trade_frequency, 2)
    }

    return {"metrics": metrics, "trade_log": trades}
