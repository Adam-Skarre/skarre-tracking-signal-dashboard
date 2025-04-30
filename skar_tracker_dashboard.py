# skar_tracker_dashboard.py

import os, sys
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf

# ─── Ensure skar_lib is on path ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
LIB_DIR  = os.path.join(BASE_DIR, "skar_lib")
if os.path.isdir(LIB_DIR) and LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

# ─── Core strategy modules ────────────────────────────────────────────────────
from polynomial_fit            import get_slope, get_acceleration
from signal_logic              import generate_signals
from backtester                import backtest, evaluate_strategy
from optimizer                 import grid_search_optimizer
from validate_skarre_signal    import bootstrap_sharpe, regime_performance

# ─── Streamlit config ────────────────────────────────────────────────────────
st.set_page_config(page_title="Skarre Tracker Dashboard", layout="wide")

# ─── Data fetching (inclusive end date) ───────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data(ticker: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """
    Download daily Close prices for ticker over [start, end], inclusive.
    """
    # Yahoo Finance end date is exclusive, so add one day:
    df = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False)
    if df.empty or "Close" not in df:
        return pd.DataFrame(columns=["Price"])
    df = df[["Close"]].rename(columns={"Close": "Price"})
    df.index = pd.to_datetime(df.index)
    # Filter to exactly start/end
    mask = (df.index.date >= start) & (df.index.date <= end)
    return df.loc[mask]

# ─── Cached, coarse grid search for demo speed ─────────────────────────────────
@st.cache_data(show_spinner=False)
def run_grid_search(price, slope, accel):
    entry_range = np.arange(0.0, 2.1, 0.5)   # coarse steps for speed
    exit_range  = np.arange(-2.0, 0.1, 0.5)
    best, df = grid_search_optimizer(
        price=price, slope=slope, accel=accel,
        entry_range=entry_range, exit_range=exit_range,
        use_sst=True, ma_window=50, vol_window=20, min_holding_days=3
    )
    return best, df

# ─── Cached bootstrap for demo speed ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_bootstrap(price, signal):
    # 500 resamples only
    return bootstrap_sharpe(price, signal, n_boot=500, block_size=20)

# ─── Sidebar navigation ──────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "About",
    "Live Signal Tracker",
    "Derivative Diagnostics",
    "Polynomial Fit Curve",
    "Derivative Histograms",
    "Threshold Optimization",
    "Strategy Performance",
    "Trade Log"
])

# ─── Page: About ──────────────────────────────────────────────────────────────
if page == "About":
    st.title("Skarre Tracker Dashboard")
    st.markdown("""
**Version 3** of the Skarre Signal system:
- Regime-aware, SST-adjusted slope signals  
- Realistic costs & dynamic position sizing  
- Multi-asset, live-ready Streamlit UI  
- Rigorous walk-forward & bootstrap validation  
    """)
    st.markdown("Built as part of ME 3296 Independent Study — aiming for robust, real-world outperformance.")

# ─── Page: Live Signal Tracker ────────────────────────────────────────────────
elif page == "Live Signal Tracker":
    st.title("Live Signal Tracker")
    ticker     = st.sidebar.text_input("Ticker", "SPY").upper()
    start_date = st.sidebar.date_input("Start Date", datetime(2022, 1, 1).date())
    end_date   = st.sidebar.date_input("End Date",   datetime.today().date())
    entry_th   = st.sidebar.slider("Entry Threshold (slope)",  0.0, 5.0, 0.5, 0.1)
    exit_th    = st.sidebar.slider("Exit Threshold (slope)",  -5.0, 0.0, -0.5, 0.1)
    use_accel  = st.sidebar.checkbox("Use Acceleration", True)
    show_marks = st.sidebar.checkbox("Show Buy/Sell Markers", True)

    df = get_data(ticker, start_date, end_date)
    if df.empty:
        st.warning(f"No data found for {ticker} from {start_date} to {end_date}.")
        st.stop()

    price = df["Price"]
    slope = get_slope(price)
    accel = get_acceleration(price)
    sig   = generate_signals(slope, accel, entry_th, exit_th, use_acceleration=use_accel)

    fig = go.Figure([go.Scatter(x=price.index, y=price, name="Price")])
    if show_marks:
        buys  = price[sig ==  1]
        sells = price[sig == -1]
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys, mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=8, color="green")
        ))
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells, mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", size=8, color="red")
        ))
    fig.update_layout(height=500, title=f"{ticker} with Skarre Signals")
    st.plotly_chart(fig, use_container_width=True)

# ─── Page: Derivative Diagnostics ────────────────────────────────────────────
elif page == "Derivative Diagnostics":
    st.title("Derivative Diagnostics")
    ticker  = st.sidebar.text_input("Ticker", "SPY").upper()
    sd      = st.sidebar.date_input("Start Date", datetime(2010, 1, 1).date())
    ed      = st.sidebar.date_input("End Date",   datetime.today().date())
    entryS  = st.sidebar.slider("Entry Slope", 0.0, 10.0, 0.5, 0.1)
    exitS   = st.sidebar.slider("Exit Slope",  -10.0, 0.0, -0.5, 0.1)
    useA    = st.sidebar.checkbox("Use Acceleration", False)

    df = get_data(ticker, sd, ed)
    if df.empty:
        st.warning("No data for that range."); st.stop()

    price = df["Price"]
    slope = get_slope(price)
    accel = get_acceleration(price)

    st.subheader("Price, Slope & Acceleration")
    diag_df = pd.DataFrame({
        "Price": price,
        "Slope": slope,
        "Accel": accel
    })
    st.line_chart(diag_df)

# ─── Page: Polynomial Fit Curve ──────────────────────────────────────────────
elif page == "Polynomial Fit Curve":
    st.title("Polynomial Fit Curve")
    ticker = st.sidebar.text_input("Ticker", "SPY").upper()
    window = st.sidebar.slider("Window Size", 10, 50, 21, 1)

    df = get_data(ticker, datetime(2022,1,1).date(), datetime.today().date())
    if df.empty:
        st.warning("No data"); st.stop()

    price = df["Price"]
    xs = np.arange(window)
    fit_vals, dates = [], []
    for i in range(window, len(price)):
        ys = price.iloc[i-window:i].values
        coeffs = np.polyfit(xs, ys, 2)
        fit_vals.append(np.poly1d(coeffs)(xs)[-1])
        dates.append(price.index[i-1])

    fit = pd.Series(fit_vals, index=dates)
    fig = go.Figure([
        go.Scatter(x=price.index, y=price, name="Price"),
        go.Scatter(x=fit.index,   y=fit,   name="Poly Fit")
    ])
    st.plotly_chart(fig, use_container_width=True)

# ─── Page: Derivative Histograms ───────────────────────────────────────────
elif page == "Derivative Histograms":
    st.title("Slope & Acceleration Histograms")
    df    = get_data("SPY", datetime(2022,1,1).date(), datetime.today().date())
    price = df["Price"]
    slope = get_slope(price).dropna()
    accel = get_acceleration(price).dropna()

    st.subheader("Slope Distribution")
    st.bar_chart(slope.value_counts(bins=30).sort_index())
    st.subheader("Acceleration Distribution")
    st.bar_chart(accel.value_counts(bins=30).sort_index())

# ─── Page: Threshold Optimization ───────────────────────────────────────────
elif page == "Threshold Optimization":
    st.title("Threshold Optimization (Coarse Grid)")
    df    = get_data("SPY", datetime(2022,1,1).date(), datetime.today().date())
    price = df["Price"]
    slope = get_slope(price)
    accel = get_acceleration(price)

    with st.spinner("Running grid search…"):
        (best_e, best_x, best_sh), results_df = run_grid_search(price, slope, accel)

    st.success(f"Best entry={best_e:.2f}, exit={best_x:.2f} → Sharpe={best_sh:.2f}")
    st.dataframe(results_df.style.background_gradient("RdYlGn"))

# ─── Page: Strategy Performance ─────────────────────────────────────────────
elif page == "Strategy Performance":
    st.title("Strategy vs Buy-and-Hold")
    df    = get_data("SPY", datetime(2022,1,1).date(), datetime.today().date())
    price = df["Price"]
    slope = get_slope(price)
    accel = get_acceleration(price)
    sig   = generate_signals(slope, accel, 0.5, -0.5, use_acceleration=True)

    res = backtest(price, sig)
    eq  = res["equity_curve"]
    bh  = (1 + price.pct_change().fillna(0)).cumprod()

    st.line_chart(pd.DataFrame({"Strategy":eq, "BuyHold":bh}))
    perf = res["performance"]
    st.metric("Sharpe Ratio",    f"{perf['Sharpe']:.2f}")
    st.metric("Total Return",    f"{perf['Total Return']*100:.1f}%")
    st.metric("Max Drawdown",    f"{perf['Max Drawdown']*100:.1f}%")
    st.metric("Trades per Year", f"{perf['Trade Frequency']:.1f}")

# ─── Page: Trade Log ─────────────────────────────────────────────────────────
elif page == "Trade Log":
    st.title("Trade Log")
    df    = get_data("SPY", datetime(2022,1,1).date(), datetime.today().date())
    price = df["Price"]
    slope = get_slope(price)
    accel = get_acceleration(price)
    sig   = generate_signals(slope, accel, 0.5, -0.5, use_acceleration=True)

    res = backtest(price, sig)
    trades = res.get("trade_log", pd.DataFrame())
    if trades.empty:
        st.info("No trades triggered.")
    else:
        st.dataframe(trades)
        csv = trades.to_csv(index=False)
        st.download_button("Download CSV", csv, "trade_log.csv")
