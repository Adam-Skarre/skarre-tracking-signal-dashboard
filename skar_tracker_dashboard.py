
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

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["About", "Performance", "Optimization", "Diagnostics", "Trade Log"],
    index=0
)

# Load Dummy Price Data (Replace with real data loader if needed)
@st.cache_data
def load_price():
    dates = pd.date_range(start="2022-01-01", periods=300)
    price_spy = pd.Series(np.cumsum(np.random.randn(300)) + 420, index=dates)
    price_qqq = pd.Series(np.cumsum(np.random.randn(300)) + 330, index=dates)
    return price_spy, price_qqq

price_spy, price_qqq = load_price()

# Compute derivatives
slope_spy = get_slope(price_spy)
accel_spy = get_acceleration(price_spy)

# About Page
if page == "About":
    st.title("About the Skarre Signal Dashboard")
    st.markdown("""
The Skarre Signal Dashboard is an engineering‑driven, data‑science approach to algorithmic trading,
inspired by the research rigor of Renaissance Technologies. We use **polynomial regression** and
**calculus‑based derivatives** to anticipate market turns, optimize entry/exit thresholds, and
benchmark against passive indices.

**Key Features**:
- Slope/acceleration analysis using Savitzky–Golay filters
- Optimization of entry/exit thresholds
- Modular backtesting engine with performance metrics
    """)

# Performance Page
elif page == "Performance":
    st.title("Strategy Performance vs. SPY")
    entry_th, exit_th = 0.5, -0.5
    positions = generate_signals(slope_spy, accel_spy, entry_th, exit_th, use_acceleration=True)
    result = backtest(price_spy, positions)

    st.subheader("Equity Curve")
    st.pyplot(plot_equity_curve(result['equity_curve']))

    st.subheader("Drawdown")
    st.pyplot(plot_drawdown(result['equity_curve']))

    st.subheader("Performance Metrics")
    perf = result['performance']
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sharpe Ratio", f"{perf['Sharpe']:.2f}")
    col2.metric("Max Drawdown", f"{perf['Max Drawdown']:.0%}")
    col3.metric("Win Rate", f"{perf['Win Rate']:.0%}")
    col4.metric("Trades/Year", f"{perf['Trade Frequency']:.1f}")

# Optimization Page
elif page == "Optimization":
    st.title("Threshold Optimization")
    st.markdown("Grid‑search over entry/exit slope thresholds to maximize Sharpe Ratio.")
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

    st.write(f"**Optimal Entry:** {best_entry:.2f} **Optimal Exit:** {best_exit:.2f}")
    st.pyplot(plot_heatmap(results, xlabel='Exit Slope', ylabel='Entry Slope', title='Sharpe Heatmap'))

# Diagnostics Page
elif page == "Diagnostics":
    st.title("Derivative Diagnostics")
    st.markdown("Visualize price and slope/acceleration signals.")
    st.pyplot(plot_signals(price_spy, generate_signals(slope_spy, accel_spy, 0.0, 0.0)))

    st.subheader("Slope (1st Derivative)")
    st.line_chart(pd.DataFrame({'Slope': slope_spy}))

    st.subheader("Acceleration (2nd Derivative)")
    st.line_chart(pd.DataFrame({'Acceleration': accel_spy}))

# Trade Log Page
elif page == "Trade Log":
    st.title("Trade Log")
    signals = generate_signals(slope_spy, accel_spy, 0.5, -0.5, use_acceleration=True)
    trade_df = backtest(price_spy, signals)['trade_log']
    st.dataframe(trade_df)
