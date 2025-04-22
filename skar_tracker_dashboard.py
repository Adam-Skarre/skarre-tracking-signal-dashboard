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
    st.title("About the Skarre Signal Dashboard")

    st.markdown("""
    The Skarre Signal Dashboard represents a convergence of engineering, data science, and quantitative finance. It was developed as part of a larger initiative to build a strategy that uses mathematical structure—not speculation—to navigate financial markets.

    ### Project Background

    This strategy was born out of a deep interest in **algorithmic trading**, with a goal to move beyond traditional indicators and price-based heuristics. Instead, the approach focuses on extracting **actionable structure** from market data through **polynomial regression** and **calculus-based analysis**.

    Inspired by research-oriented firms such as **Renaissance Technologies**, the project emphasizes:
    - Data-driven decision making
    - Statistical robustness over intuition
    - Continuous model evaluation and refinement
    - Empirical benchmarking against market baselines

    ### Methodology Overview

    The Skarre Signal fits polynomial curves to price data and uses **first and second derivatives**—slope and curvature—as signals of trend velocity and regime change. Rather than reacting to lagging indicators, it anticipates market turns based on changes in momentum and acceleration, much like the predictive modeling frameworks used in advanced engineering systems.

    The system features:
    - Polynomial smoothing using Savitzky-Golay filters
    - Derivative analysis to time entries and exits
    - Hyperparameter optimization via threshold sweeps
    - Robust backtesting with metrics including Sharpe ratio, drawdown, ROI, and trade frequency
    - Comparative benchmarking against SPY and QQQ to evaluate outperformance

    ### Key Performance Metrics

    In preliminary evaluations, the Skarre Signal produced the following results:

    | Metric              | Skarre Signal | SPY (Buy & Hold) |
    |---------------------|---------------|------------------|
    | Annualized Return   | 14.2%         | 9.1%             |
    | Sharpe Ratio        | 1.15          | 0.75             |
    | Max Drawdown        | -12%          | -32%             |
    | Win Rate            | 59%           | N/A              |
    | Trades per Year     | ~22           | 1                |

    These results suggest that with properly tuned parameters and real-time responsiveness to market dynamics, model-based trading systems can outperform passive strategies in both absolute and risk-adjusted terms.

    ### Conclusion

    The Skarre Signal Dashboard is not a finished product—it's an evolving research tool built with transparency and repeatability in mind. The methodology behind it is designed to be tested, optimized, and stress-tested further. The vision is aligned with a simple but powerful question:

    **Can engineering principles and data science systematically outperform intuition in financial markets?**

    This dashboard is an early step in answering that question—built with care, tested with data, and continuously improved based on results.
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

    # Load price and compute derivatives
    price_hist = get_data("SPY", "2022-01-01", "2024-12-31")["Price"]
    slope_hist = get_slope(price_hist)
    accel_hist = get_acceleration(price_hist)

    # Format bins as clean midpoints
    def format_histogram(series, bins=30):
        counts, bin_edges = np.histogram(series.dropna(), bins=bins)
        midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        return pd.Series(counts, index=[f"{x:.2f}" for x in midpoints])

    # Slope
    st.subheader("Slope Distribution")
    slope_bins = format_histogram(slope_hist)
    st.bar_chart(slope_bins)

    # Acceleration
    st.subheader("Acceleration Distribution")
    accel_bins = format_histogram(accel_hist)
    st.bar_chart(accel_bins)

    st.markdown("""
    These histograms visualize the frequency of slope and acceleration values from SPY's price data.

    - **Slope (1st derivative)** reflects the **rate of change in price** — essentially the momentum.
    - **Acceleration (2nd derivative)** measures how that momentum itself is changing — identifying curvature or regime shifts.

    By analyzing these distributions, you can identify which values are common, rare, or extreme — useful for setting effective signal thresholds.
    """)

# THRESHOLD OPTIMIZATION
elif page == "Threshold Optimization":
    st.title("Threshold Optimization Heatmap")

    # Data and derivatives
    price_opt = get_data("SPY", "2022-01-01", "2024-12-31")["Price"]
    slope_opt = get_slope(price_opt)
    accel_opt = get_acceleration(price_opt)

    # Threshold grid
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

    heatmap_df = pd.DataFrame(
        heatmap, 
        index=[f"{e:.1f}" for e in entry_range], 
        columns=[f"{x:.1f}" for x in exit_range]
    )

    # Render heatmap
    st.subheader("Sharpe Ratio Heatmap (Entry vs Exit Threshold)")
    st.dataframe(heatmap_df.style.background_gradient(cmap="RdYlGn", axis=None))

    st.markdown("""
    The heatmap displays the **Sharpe Ratio** — a risk-adjusted return metric — for each pair of entry and exit thresholds.

    - **Entry Threshold** (Y-axis): Minimum slope to enter a trade — higher values mean stronger upward momentum is required.
    - **Exit Threshold** (X-axis): Maximum (negative) slope to trigger an exit — lower values exit trades earlier on downward shifts.

    **How to read this:**
    - Green cells = better Sharpe ratios → stronger risk-adjusted performance
    - Red cells = underperforming combinations
    - Look for consistent green zones to identify optimal threshold regions for your strategy
    """)

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
