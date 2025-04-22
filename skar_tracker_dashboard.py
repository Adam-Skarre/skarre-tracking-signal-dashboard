import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime

from skar_lib.polynomial_fit import get_slope, get_acceleration
from skar_lib.signal_logic import generate_signals
from skar_lib.backtester import backtest

st.set_page_config(page_title="Skarre Tracker Quantitative Dashboard", layout="wide")

# Sidebar controls
st.sidebar.title("Skarre Tracker Signal")
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="SPY").upper()
start_date = st.sidebar.date_input("Start Date", datetime(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2024, 12, 31))
entry_th = st.sidebar.slider("Entry Threshold", 0.0, 2.0, 0.5, 0.1)
exit_th = st.sidebar.slider("Exit Threshold", -2.0, 0.0, -0.5, 0.1)
show_signals = st.sidebar.checkbox("Show Skarre Buy/Sell Points", value=True)

# Download price data
@st.cache_data(show_spinner=False)
def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[['Close']].dropna()
    df.columns = ['Price']
    return df

price_df = get_data(ticker, start_date, end_date)
if price_df.empty:
    st.warning("No data found for this ticker and date range.")
    st.stop()

price_series = price_df["Price"]

# Derivatives and signals
slope = get_slope(price_series)
accel = get_acceleration(price_series)
signals = generate_signals(slope, accel, entry_th, exit_th, use_acceleration=True)

# Skarre Signal Chart
st.title(f"{ticker} | Skarre Signal Dashboard")

fig = go.Figure()
fig.add_trace(go.Scatter(x=price_series.index, y=price_series, mode='lines', name='Price'))
if show_signals:
    buy_points = price_series[signals == 1]
    sell_points = price_series[signals == -1]
    fig.add_trace(go.Scatter(x=buy_points.index, y=buy_points, mode='markers', name='Buy Signal',
                             marker=dict(color='green', size=8, symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=sell_points.index, y=sell_points, mode='markers', name='Sell Signal',
                             marker=dict(color='red', size=8, symbol='triangle-down')))
fig.update_layout(title=f"{ticker} Price with Skarre Signals", height=500)
st.plotly_chart(fig, use_container_width=True)

# Derivative Diagnostics
st.subheader("Derivative Diagnostics")
st.line_chart(pd.DataFrame({'Slope': slope}))
st.line_chart(pd.DataFrame({'Acceleration': accel}))

# Strategy Performance
st.subheader("Strategy vs Buy & Hold Performance")
result = backtest(price_series, signals)

perf = result['performance']
equity_curve = result['equity_curve']
trade_log = result['trade_log']

if not trade_log.empty:
    st.metric("Sharpe Ratio", f"{perf['Sharpe']:.2f}")
    st.metric("Max Drawdown", f"{perf['Max Drawdown']:.0%}")
    st.metric("Win Rate", f"{perf['Win Rate']:.0%}")
    st.metric("Trades/Year", f"{perf['Trade Frequency']:.1f}")

    st.line_chart(pd.DataFrame({
        "Skarre Equity": equity_curve,
        "Buy & Hold": (1 + price_series.pct_change().fillna(0)).cumprod()
    }))
else:
    st.warning("No trades generated with current thresholds. Try adjusting entry/exit levels.")

# Trade Log Table
st.subheader("Trade Log")
if trade_log.empty:
    st.write("No trades to display.")
else:
    st.dataframe(trade_log)

# Download Option
if not trade_log.empty:
    csv = trade_log.to_csv(index=False)
    st.download_button("Download Trade Log CSV", csv, file_name=f"{ticker}_trade_log.csv")
