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
    "Polynomial Fit Curve",
    "Derivative Histograms",
    "Threshold Optimization",
    "Strategy Performance",
    "Trade Log"
])

@st.cache_data(show_spinner=False)
def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[['Close']].dropna()
    df.columns = ['Price']
    return df

# LIVE SIGNAL TRACKER
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

    st.subheader("Derivative Preview")
    st.line_chart(pd.DataFrame({'Slope': slope}))
    st.line_chart(pd.DataFrame({'Acceleration': accel}))

    result = backtest(price_series, signals)
    equity_curve = result["equity_curve"]
    st.subheader("Strategy vs Buy & Hold")
    st.line_chart(pd.DataFrame({
        "Skarre Equity": equity_curve,
        "Buy & Hold": (1 + price_series.pct_change().fillna(0)).cumprod()
    }))

# ABOUT
elif page == "About":
    st.title("About the Skarre Tracker Dashboard")
    st.markdown("""
The Skarre Signal Dashboard is a quantitative trading platform rooted in engineering logic and calculus-based analytics. 
It applies polynomial fitting, slope curvature, and dynamic threshold optimization to deliver robust entry/exit signals, 
targeting superior performance over traditional investing benchmarks.
    """)

# DERIVATIVE DIAGNOSTICS
elif page == "Derivative Diagnostics":
    st.title("Derivative Diagnostics")
    price_series = get_data("SPY", "2022-01-01", "2024-12-31")["Price"]
    slope = get_slope(price_series)
    accel = get_acceleration(price_series)

    st.line_chart(price_series.rename("SPY Price"))
    st.line_chart(slope.rename("Slope (1st Derivative)"))
    st.line_chart(accel.rename("Acceleration (2nd Derivative)"))

# POLYNOMIAL FIT CURVE
elif page == "Polynomial Fit Curve":
    st.title("Polynomial Fit Curve Analysis")
    ticker_poly = st.sidebar.text_input("Ticker for Polynomial Fit", value="SPY").upper()
    price_poly = get_data(ticker_poly, "2022-01-01", "2024-12-31")["Price"]
    window = st.sidebar.slider("Polynomial Fit Window Size", 10, 50, 21, 2)

    x_vals = np.arange(window)
    fit_values = []
    dates = []

    for i in range(window, len(price_poly)):
        y_vals = price_poly.iloc[i - window:i].values
        coeffs = np.polyfit(x_vals, y_vals, 2)
        poly = np.poly1d(coeffs)
        fit_curve = poly(x_vals)
        fit_values.append(fit_curve[-1])
        dates.append(price_poly.index[i - 1])

    fit_series = pd.Series(fit_values, index=dates)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_poly.index, y=price_poly, name="Price", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=fit_series.index, y=fit_series, name="Poly Fit (Rolling)", line=dict(color="blue")))
    st.plotly_chart(fig, use_container_width=True)

# DERIVATIVE HISTOGRAMS
elif page == "Derivative Histograms":
    st.title("Slope and Acceleration Histograms")
    price_hist = get_data("SPY", "2022-01-01", "2024-12-31")["Price"]
    slope_hist = get_slope(price_hist)
    accel_hist = get_acceleration(price_hist)

    st.subheader("Slope Distribution")
    st.bar_chart(slope_hist.value_counts(bins=30).sort_index())

    st.subheader("Acceleration Distribution")
    st.bar_chart(accel_hist.value_counts(bins=30).sort_index())

# THRESHOLD OPTIMIZATION
elif page == "Threshold Optimization":
    st.title("Threshold Optimization Heatmap")
    price_opt = get_data("SPY", "2022-01-01", "2024-12-31")["Price"]
    slope_opt = get_slope(price_opt)
    accel_opt = get_acceleration(price_opt)

    entry_range = np.arange(0.0, 1.1, 0.1)
    exit_range = np.arange(-1.0, 0.1, 0.1)
    heatmap = []

    for entry in entry_range:
        row = []
        for exit in exit_range:
            signals = generate_signals(slope_opt, accel_opt, entry, exit, use_acceleration=True)
            result = backtest(price_opt, signals)
            row.append(result["performance"]["Sharpe"])
        heatmap.append(row)

    heatmap_df = pd.DataFrame(heatmap, index=[f"{e:.1f}" for e in entry_range], columns=[f"{x:.1f}" for x in exit_range])
    st.subheader("Sharpe Ratio Heatmap (Entry vs Exit Threshold)")
    st.dataframe(heatmap_df.style.background_gradient(cmap="RdYlGn", axis=None))

# STRATEGY PERFORMANCE
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
        equity_curve = result["equity_curve"]
        buy_hold = (1 + price_series.pct_change().fillna(0)).cumprod()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sharpe Ratio", f"{perf['Sharpe']:.2f}")
        col2.metric("Max Drawdown", f"{perf['Max Drawdown']:.0%}")
        col3.metric("Win Rate", f"{perf['Win Rate']:.0%}")
        col4.metric("Trades/Year", f"{perf['Trade Frequency']:.1f}")

        st.line_chart(pd.DataFrame({
            "Skarre Equity": equity_curve,
            "Buy & Hold": buy_hold
        }))

        st.subheader("Final Value Comparison")
        comparison_df = pd.DataFrame({
            "Total Return (%)": [
                (equity_curve.iloc[-1] - 1) * 100,
                (buy_hold.iloc[-1] - 1) * 100
            ],
            "CAGR (%)": [
                ((equity_curve.iloc[-1]) ** (1 / 2.8) - 1) * 100,
                ((buy_hold.iloc[-1]) ** (1 / 2.8) - 1) * 100
            ]
        }, index=["Skarre Strategy", "Buy & Hold"])
        st.dataframe(comparison_df.style.format("{:.2f}"))

# TRADE LOG
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
