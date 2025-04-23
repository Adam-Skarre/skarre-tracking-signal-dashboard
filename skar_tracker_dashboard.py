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
    st.title("Engineering, Optimization, and Comparison of Algorithmic Trading")

    st.markdown("""
    This dashboard is the core deliverable of an independent research project titled “Engineering, Optimization, and Comparison of Algorithmic Trading.”  
    The project investigates whether structured, engineering-based techniques can be used to design, analyze, and optimize algorithmic trading strategies that outperform passive benchmarks.

    ### Research Objectives

    The primary goals of the study are to:
    - Apply mathematical modeling and calculus to extract actionable structure from market data
    - Develop a signal generation system using first and second derivatives of price curves
    - Optimize entry/exit thresholds for improved performance across various regimes
    - Compare strategy behavior to passive investing (e.g. SPY, QQQ) through backtesting
    - Present findings in a transparent, interactive format

    ### Methodology Overview

    The system:
    - Uses Savitzky–Golay filters to smooth price data while preserving critical turning points
    - Computes **first and second derivatives** (slope and curvature) to detect acceleration-based signals
    - Applies threshold logic to identify long/exit conditions based on derivative magnitudes and direction
    - Performs backtests with full metrics including Sharpe ratio, drawdown, win rate, and ROI
    - Benchmarks strategy output against SPY buy-and-hold to evaluate relative performance

    ### System Features

    This dashboard allows users to:
    - Select tickers and explore real-time signal overlays
    - Visualize moving slopes and curvature-based inflection zones
    - Run parameter sweeps and generate heatmaps for optimization
    - Compare strategy returns against traditional market exposure
    - Download trade logs and inspect signal performance over time

    ### Live Performance Results (SPY: 2020–2024)

    The results below are dynamically generated using the strategy's current thresholds:
    """)

    # Live backtest result for SPY (2020–2024)
    price = get_data("SPY", "2020-01-01", "2024-12-31")["Price"]
    slope = get_slope(price)
    accel = get_acceleration(price)
    signals = generate_signals(slope, accel, 0.5, -0.5, use_acceleration=True)
    result = backtest(price, signals)

    buy_hold = (1 + price.pct_change().fillna(0)).cumprod()
    buy_hold_drawdown = (buy_hold / buy_hold.cummax() - 1).min()

    comparison_df = pd.DataFrame({
        "Metric": ["Annualized Return", "Sharpe Ratio", "Max Drawdown", "Win Rate", "Trades per Year"],
        "Skarre Signal": [
            f"{(result['equity_curve'].iloc[-1] - 1) * 100:.1f}%",
            f"{result['performance']['Sharpe']:.2f}",
            f"{result['performance']['Max Drawdown'] * 100:.0f}%",
            f"{result['performance']['Win Rate'] * 100:.0f}%",
            f"{result['performance']['Trade Frequency']:.1f}"
        ],
        "SPY (Buy & Hold)": [
            f"{(buy_hold.iloc[-1] - 1) * 100:.1f}%",
            "N/A",
            f"{buy_hold_drawdown * 100:.0f}%",
            "N/A",
            "1"
        ]
    }).set_index("Metric")

    with st.expander("View Live Performance Table"):
        st.dataframe(comparison_df)

    st.markdown("""
    *Note: These values are computed using fixed thresholds (entry = 0.5, exit = –0.5) over the 2020–2024 period, and are subject to change based on market conditions.*

    ### Conclusion

    This dashboard reflects a structured, engineering-first approach to market modeling.  
    It was developed as part of the academic study "Engineering, Optimization, and Comparison of Algorithmic Trading,” which blends principles from mechanical systems analysis, data science, and quantitative finance. 

    The Skarre Signal is not intended as financial advice or a final product. Instead, it is an evolving model and presentation tool — built to explore whether mathematical rigor and optimization can consistently outperform intuition in dynamic financial systems.
    """)
# DERIVATIVE DIAGNOSTICS
elif page == "Strategy Overview":
    st.title("DEBUG MODE: Derivative Diagnostics")

    st.markdown("""
    This debug mode shows what's working or failing in the derivative signal generation pipeline.  
    Useful if the screen is blank or if you're unsure whether the app is loading valid data.
    """)

    # Sidebar input
    ticker = st.sidebar.text_input("Ticker", value="SPY").upper()
    start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.today())
    entry_th = st.sidebar.slider("Entry Threshold", 0.0, 2.0, 0.5, 0.1)
    exit_th = st.sidebar.slider("Exit Threshold", -2.0, 0.0, -0.5, 0.1)

    st.write("Ticker:", ticker)
    st.write("Start Date:", start_date)
    st.write("End Date:", end_date)

    try:
        # Load data
        price_series = get_data(ticker, start_date, end_date)["Price"]
        st.write("✅ Loaded price data:", price_series.head())

        # Derivatives
        slope = get_slope(price_series)
        accel = get_acceleration(price_series)

        st.write("✅ Slope (first 5):", slope.head())
        st.write("✅ Acceleration (first 5):", accel.head())

        # Signals
        signals = generate_signals(slope, accel, entry_th, exit_th, use_acceleration=True)
        st.write("✅ Signals (first 5):", signals.head())

        # Chart 1: Price + Buy/Sell Signals
        st.subheader(f"{ticker} Price with Skarre Signals")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_series.index, y=price_series, name="Price", line=dict(color="black")))

        buy_points = price_series[signals == 1]
        sell_points = price_series[signals == -1]
        if not buy_points.empty:
            fig.add_trace(go.Scatter(x=buy_points.index, y=buy_points, name="Buy", mode="markers",
                                     marker=dict(color="green", symbol="triangle-up", size=8)))
        if not sell_points.empty:
            fig.add_trace(go.Scatter(x=sell_points.index, y=sell_points, name="Sell", mode="markers",
                                     marker=dict(color="red", symbol="triangle-down", size=8)))

        st.plotly_chart(fig, use_container_width=True)

        # Chart 2: Slope & Acceleration
        st.subheader("Slope and Acceleration")
        st.line_chart(pd.DataFrame({"Slope": slope, "Acceleration": accel}))

        # Chart 3: Backtest vs Buy & Hold
        st.subheader("Backtest Performance")
        result = backtest(price_series, signals)
        equity_curve = result["equity_curve"]
        buy_hold = (1 + price_series.pct_change().fillna(0)).cumprod()
        st.line_chart(pd.DataFrame({
            "Skarre Strategy": equity_curve,
            "Buy & Hold": buy_hold
        }))

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
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
