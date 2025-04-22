import streamlit as st
import pandas as pd
import numpy as np

from skar_lib.polynomial_fit import get_slope, get_acceleration
from skar_lib.signal_logic import generate_signals
from skar_lib.optimizer import optimize_thresholds, get_optimal_thresholds
from skar_lib.backtester import backtest
from skar_lib.plots import (
    plot_equity_curve,
    plot_drawdown,
    plot_signals,
    plot_heatmap,
)

st.set_page_config(page_title="Skarre Signal Dashboard", layout="wide")
st.markdown("<style>h1, h2, h3 { margin-top: 0.5rem !important; }</style>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About", "Performance", "Optimization", "Diagnostics", "Trade Log"], index=0)

# Load Dummy SPY Data with Trend (More realistic)
@st.cache_data
def load_price():
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=300)
    noise = np.random.normal(0, 1, size=len(dates))
    price = 400 + np.cumsum(0.3 + noise)  # Upward trend
    return pd.Series(price, index=dates)

price_spy = load_price()

# Compute Derivatives
slope_spy = get_slope(price_spy)
accel_spy = get_acceleration(price_spy)

# About
if page == "About":
    st.title("About the Skarre Signal Dashboard")
    st.markdown("""
The Skarre Signal Dashboard is a quantitative research platform using polynomial regression and calculus-based indicators (slope, acceleration) to optimize financial strategies.

Inspired by Renaissance Technologies, it applies engineering precision to financial signal processing.
""")

# Performance
elif page == "Performance":
    st.title("Strategy Performance vs. SPY")
    entry_th, exit_th = 0.5, -0.5
    positions = generate_signals(slope_spy, accel_spy, entry_th, exit_th, use_acceleration=True)
    result = backtest(price_spy, positions)

    if result["trade_log"].empty:
        st.warning("⚠️ No trades triggered. Adjust thresholds or confirm slope direction.")
    else:
        st.subheader("Equity Curve")
        st.pyplot(plot_equity_curve(result["equity_curve"]))

        st.subheader("Drawdown")
        st.pyplot(plot_drawdown(result["equity_curve"]))

        st.subheader("Performance Metrics")
        perf = result["performance"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sharpe Ratio", f"{perf['Sharpe']:.2f}")
        col2.metric("Max Drawdown", f"{perf['Max Drawdown']:.0%}")
        col3.metric("Win Rate", f"{perf['Win Rate']:.0%}")
        col4.metric("Trades/Year", f"{perf['Trade Frequency']:.1f}")

# Optimization
elif page == "Optimization":
    st.title("Threshold Optimization")
    st.markdown("Grid search over slope thresholds to maximize Sharpe Ratio.")
    results = optimize_thresholds(
        price_series=price_spy,
        slope_series=slope_spy,
        accel_series=accel_spy,
        entry_min=0.0, entry_max=1.0,
        exit_min=-1.0, exit_max=0.0,
        step=0.1,
        use_acceleration=True,
        metric='Sharpe'
    )
    best_entry, best_exit = get_optimal_thresholds(results)

    st.write(f"**Optimal Entry:** {best_entry:.2f}  **Optimal Exit:** {best_exit:.2f}")
    st.pyplot(plot_heatmap(results, xlabel='Exit Slope', ylabel='Entry Slope', title='Sharpe Heatmap'))

# Diagnostics
elif page == "Diagnostics":
    st.title("Signal Diagnostics")
    st.line_chart(price_spy.rename("SPY Price"))

    st.subheader("Slope (1st Derivative)")
    st.line_chart(slope_spy.rename("Slope"))

    st.subheader("Acceleration (2nd Derivative)")
    st.line_chart(accel_spy.rename("Acceleration"))

    st.subheader("Signal Sample")
    signal_sample = generate_signals(slope_spy, accel_spy, 0.5, -0.5, use_acceleration=True)
    st.line_chart(signal_sample.rename("Signal"))

# Trade Log
elif page == "Trade Log":
    st.title("Trade Log")
    signals = generate_signals(slope_spy, accel_spy, 0.5, -0.5, use_acceleration=True)
    result = backtest(price_spy, signals)
    trade_df = result.get("trade_log", pd.DataFrame())

    if trade_df.empty:
        st.warning("⚠️ No trades found. Adjust your thresholds or verify signal generation.")
    else:
        st.dataframe(trade_df)
