import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime

from skar_lib.polynomial_fit import get_slope, get_acceleration
from skar_lib.signal_logic import generate_signals
from skar_lib.backtester import backtest

st.set_page_config(page_title="Skarre Tracker Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select View", [
    "About",
    "Live Signal Tracker",
    "Derivative Diagnostics",
    "Strategy Performance",
    "Trade Log"
])

# Shared config and cache
@st.cache_data(show_spinner=False)
def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[['Close']].dropna()
    df.columns = ['Price']
    return df

# Live Signal Tracker Page
if page == "Live Signal Tracker":
    st.title("Live Signal Tracker")

    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="SPY").upper()
    start_date = st.sidebar.date_input("Start Date", datetime(2022, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime(2024, 12, 31))
    entry_th = st.sidebar.slider("Entry Threshold", 0.0, 2.0, 0.5, 0.1)
    exit_th = st.sidebar.slider("Exit Threshold", -2.0, 0.0, -0.5, 0.1)
    show_signals = st.sidebar.checkbox("Show Skarre Buy/Sell Points", value=True)

    price_df = get_data(ticker, start_date, end_date)
    if price_df.empty:
        st.warning("No data found for this ticker and date range.")
        st.stop()

    price_series = price_df["Price"]
    slope = get_slope(price_series)
    accel = get_acceleration(price_series)
    signals = generate_signals(slope, accel, entry_th, exit_th, use_acceleration=True)

    # Price with Signal Chart
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

    # Derivatives preview
    st.subheader("Derivative Preview")
    st.line_chart(pd.DataFrame({'Slope': slope}))
    st.line_chart(pd.DataFrame({'Acceleration': accel}))

    # Equity comparison
    result = backtest(price_series, signals)
    equity_curve = result["equity_curve"]
    st.subheader("Strategy vs Buy & Hold")
    st.line_chart(pd.DataFrame({
        "Skarre Equity": equity_curve,
        "Buy & Hold": (1 + price_series.pct_change().fillna(0)).cumprod()
    }))

# About Page
elif page == "About":
    st.title("About the Skarre Tracker Dashboard")
    st.markdown("""
The Skarre Signal Dashboard is a quantitative research and trading platform driven by engineering logic and financial precision.
It leverages polynomial regression and calculus-based analysis to anticipate market turns and evaluate investment strategies.
Features include live signal analysis, equity comparison, and fully exportable trade logs.
    """)

# Derivative Diagnostics Page
elif page == "Derivative Diagnostics":
    st.title("Derivative Diagnostics")
    ticker = "SPY"
    price_series = get_data(ticker, "2022-01-01", "2024-12-31")["Price"]
    slope = get_slope(price_series)
    accel = get_acceleration(price_series)

    st.line_chart(price_series.rename("SPY Price"))
    st.line_chart(slope.rename("Slope (1st Derivative)"))
    st.line_chart(accel.rename("Acceleration (2nd Derivative)"))

# Strategy Performance Page
elif page == "Strategy Performance":
    st.title("Strategy Performance (SPY Example)")
    price_series = get_data("SPY", "2022-01-01", "2024-12-31")["Price"]
    slope = get_slope(price_series)
    accel = get_acceleration(price_series)
    signals = generate_signals(slope, accel, 0.5, -0.5, use_acceleration=True)
    result = backtest(price_series, signals)

    if result['trade_log'].empty:
        st.warning("No trades triggered for SPY under default thresholds.")
    else:
        perf = result['performance']
        st.metric("Sharpe Ratio", f"{perf['Sharpe']:.2f}")
        st.metric("Max Drawdown", f"{perf['Max Drawdown']:.0%}")
        st.metric("Win Rate", f"{perf['Win Rate']:.0%}")
        st.metric("Trades/Year", f"{perf['Trade Frequency']:.1f}")
        st.line_chart(result["equity_curve"])

# Trade Log Page
elif page == "Trade Log":
    st.title("Trade Log (SPY Example)")
    price_series = get_data("SPY", "2022-01-01", "2024-12-31")["Price"]
    slope = get_slope(price_series)
    accel = get_acceleration(price_series)
    signals = generate_signals(slope, accel, 0.5, -0.5, use_acceleration=True)
    result = backtest(price_series, signals)
    trade_df = result.get("trade_log", pd.DataFrame())

    if trade_df.empty:
        st.warning("No trades to display.")
    else:
        st.dataframe(trade_df)
        csv = trade_df.to_csv(index=False)
        st.download_button("Download Trade Log CSV", csv, file_name="SPY_trade_log.csv")
