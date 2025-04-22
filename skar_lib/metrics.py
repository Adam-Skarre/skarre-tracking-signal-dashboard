import numpy as np


def sharpe_ratio(returns, period=252, risk_free_rate=0.0):
    """
    Calculate annualized Sharpe ratio of a return series.

    Parameters:
    - returns: array-like of periodic returns (e.g., daily returns)
    - period: number of periods per year (default 252 for trading days)
    - risk_free_rate: annual risk-free rate (decimal)

    Returns:
    - Annualized Sharpe ratio
    """
    excess = np.array(returns) - risk_free_rate / period
    mean_excess = np.mean(excess)
    std_excess = np.std(excess)
    if std_excess == 0:
        return np.nan
    return (mean_excess / std_excess) * np.sqrt(period)


def max_drawdown(equity_curve):
    """
    Calculate the maximum drawdown of an equity curve.

    Parameters:
    - equity_curve: array-like of portfolio values over time

    Returns:
    - Maximum drawdown as a negative decimal
    """
    equity = np.array(equity_curve)
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / peaks
    return np.min(drawdowns)


def win_rate(trade_returns):
    """
    Calculate the win rate of trades.

    Parameters:
    - trade_returns: array-like of individual trade returns (decimal)

    Returns:
    - Win rate as a decimal
    """
    returns = np.array(trade_returns)
    if len(returns) == 0:
        return np.nan
    return np.mean(returns > 0)


def trade_frequency(trade_dates, period_days=365):
    """
    Estimate annualized trade frequency.

    Parameters:
    - trade_dates: list of datetime dates for each trade
    - period_days: length of backtest in days (default 365)

    Returns:
    - Trades per year
    """
    n_trades = len(trade_dates)
    if period_days == 0:
        return np.nan
    return n_trades * (365.0 / period_days)
