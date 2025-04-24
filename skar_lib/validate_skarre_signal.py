import numpy as np
import pandas as pd
from skar_lib.data_loader import download_price_data
from skar_lib.polynomial_fit import get_slope, get_acceleration
from skar_lib.signal_logic import generate_signals
from skar_lib.backtester import backtest

tickers = ["SPY", "QQQ", "DIA", "AAPL", "MSFT", "XLF"]
start_date = "2015-01-01"
end_date = "2024-12-31"
entry_threshold = 0.5
exit_threshold = -0.5

results = []

for ticker in tickers:
    print(f"Running validation on {ticker}...")
    try:
        df = download_price_data([ticker], start_date, end_date)
        price = df[ticker]
        slope = get_slope(price)
        accel = get_acceleration(price)
        signals = generate_signals(slope, accel, entry_threshold, exit_threshold, use_acceleration=True)
        result = backtest(price, signals)
        perf = result["performance"]
        results.append({
            "Ticker": ticker,
            "Sharpe Ratio": round(perf["Sharpe"], 2),
            "Max Drawdown": f"{perf['Max Drawdown']:.0%}",
            "Win Rate": f"{perf['Win Rate']:.0%}",
            "Trades/Year": round(perf["Trade Frequency"], 1)
        })
    except Exception as e:
        results.append({
            "Ticker": ticker,
            "Sharpe Ratio": "ERROR",
            "Max Drawdown": "N/A",
            "Win Rate": "N/A",
            "Trades/Year": "N/A",
            "Error": str(e)
        })

df_results = pd.DataFrame(results)
print("\n=== SKARRE SIGNAL STRATEGY VALIDATION ===")
print(df_results.to_string(index=False))
