import pandas as pd
import numpy as np

def backtest(price_series, positions):
    """
    Basic backtesting engine:
    - Calculates equity curve from positions
    - Tracks trades and returns a trade log
    """

    df = pd.DataFrame({'Price': price_series, 'Position': positions}).copy()
    df['Return'] = df['Price'].pct_change().fillna(0)
    df['Strategy Return'] = df['Return'] * df['Position'].shift().fillna(0)

    # Calculate equity curve
    df['Equity'] = (1 + df['Strategy Return']).cumprod()

    # Build trade log
    trade_log = []
    in_position = False
    entry_date = None
    entry_price = None

    for date, row in df.iterrows():
        pos = row['Position']
        price = row['Price']

        if pos != 0 and not in_position:
            in_position = True
            entry_date = date
            entry_price = price
        elif pos == 0 and in_position:
            exit_date = date
            exit_price = price
            trade_log.append({
                'Entry Date': entry_date,
                'Exit Date': exit_date,
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Return': (exit_price - entry_price) / entry_price
            })
            in_position = False

    trade_df = pd.DataFrame(trade_log)

    # Performance metrics
    perf = {
        'Sharpe': df['Strategy Return'].mean() / df['Strategy Return'].std() * np.sqrt(252) if df['Strategy Return'].std() else 0,
        'Max Drawdown': (df['Equity'] / df['Equity'].cummax() - 1).min(),
        'Win Rate': (trade_df['Return'] > 0).mean() if not trade_df.empty else 0,
        'Trade Frequency': len(trade_df) / ((df.index[-1] - df.index[0]).days / 365)
    }

    return {
        'performance': perf,
        'equity_curve': df['Equity'],
        'trade_log': trade_df
    }

