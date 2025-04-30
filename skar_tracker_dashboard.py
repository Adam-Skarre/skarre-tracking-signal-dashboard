# skar_tracker_dashboard.py
import os, sys

# ─── ENSURE skar_lib/ IS ON PYTHON PATH ─────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
SKAR_LIB_DIR = os.path.join(BASE_DIR, "skar_lib")
if os.path.isdir(SKAR_LIB_DIR) and SKAR_LIB_DIR not in sys.path:
    sys.path.insert(0, SKAR_LIB_DIR)
# ────────────────────────────────────────────────────────────────────────────────
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Core modules (must reside alongside this file)
from polynomial_fit          import get_slope, get_acceleration
from signal_logic            import generate_skarre_signal
from backtester              import backtest, evaluate_strategy
from optimizer               import grid_search_optimizer
from validate_skarre_signal  import bootstrap_sharpe, regime_performance

st.set_page_config(page_title="Skarre Tracker Dashboard", layout="wide")

@st.cache_data(show_spinner=False)
def get_data(ticker, start, end):
    # 1) Download full history
    raw = yf.download(ticker, progress=False)
    if raw.empty:
        return pd.DataFrame(columns=["Price"])
    raw = raw[["Close"]].dropna()
    raw.columns = ["Price"]
    # 2) Ensure index is datetime.date
    raw.index = pd.to_datetime(raw.index).date
    # 3) Filter by start/end (inclusive)
    mask = (raw.index >= start) & (raw.index <= end)
    df = raw.loc[mask]
    return df

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
], key="nav_page")

# === About ===
if page == "About":
    st.title("About Skarre Tracker")
    st.markdown("""
    **Skarre Tracker v2.0** is a research-grade trading dashboard:
    - **Slope (1st derivative)** and **Acceleration (2nd derivative)** signals
    - **Standardized Skarre Score (SST)**: deviation from moving average / volatility
    - **Walk-forward validation**, **dynamic transaction costs**, **bootstrap Sharpe**, **regime analysis**
    - **4-parameter optimization**: entry/exit slope & entry/exit SST
    """)
    st.caption("For educational/research purposes only.")

# === Live Signal Tracker ===
elif page == "Live Signal Tracker":
    st.title("Live Signal Tracker")
    st.markdown("Overlay real-time Skarre signals on price data with adjustable thresholds.")

    # Sidebar controls
    ticker       = st.sidebar.text_input("Ticker", "SPY", key="live_ticker").upper()
    start_date   = st.sidebar.date_input("Start Date", datetime(2022,1,1), key="live_start")
    end_date     = st.sidebar.date_input("End Date",   datetime(2024,12,31), key="live_end")
    entry_slope  = st.sidebar.slider("Entry Slope Threshold",   0.0, 0.5, 0.1, step=0.01, key="live_entry_slope")
    exit_slope   = st.sidebar.slider("Exit Slope Threshold",   -0.5, 0.0, -0.1, step=0.01, key="live_exit_slope")
    entry_sst    = st.sidebar.slider("Entry SST Threshold",     0.0, 2.0, 1.0, step=0.1, key="live_entry_sst")
    exit_sst     = st.sidebar.slider("Exit SST Threshold",    -2.0, 0.0, -1.0, step=0.1, key="live_exit_sst")
    holding_days = st.sidebar.slider("Min. Holding Days",       1,   20,   5, key="live_holding_days")
    show_pts     = st.sidebar.checkbox("Show Buy/Sell Markers", True, key="live_show_markers")

      # LIVE SIGNAL TRACKER
    if page == "Live Signal Tracker":
        st.title("Live Signal Tracker")

        ticker     = st.sidebar.text_input("Enter Ticker Symbol", value="SPY").upper()
        start_date = st.sidebar.date_input("Start Date", datetime(2022, 1, 1))
        end_date   = st.sidebar.date_input("End Date",   datetime(2024, 12, 31))

        # <-- DEBUG BLOCK (must align with the 4-space indent under the if) -->
        st.write("🔍 Debug — ticker, start, end:", ticker, start_date, end_date)
        price_df = get_data(ticker, start_date, end_date)
        st.write("📊 Debug — rows fetched:", price_df.shape[0])
        st.write(price_df.head(3))
        st.write(price_df.tail(3))
        if price_df.empty:
            st.warning(f"No data found for {ticker} from {start_date} to {end_date}.")
            st.stop()

        # … rest of your Live Signal Tracker code …

    price_series = df['Price']
    slope  = get_slope(price_series)
    accel  = get_acceleration(price_series)
    signals = generate_skarre_signal(
        price_series,
        entry_slope_threshold=entry_slope,
        exit_slope_threshold= exit_slope,
        entry_sst_threshold=  entry_sst,
        exit_sst_threshold=   exit_sst,
        min_holding_days=     holding_days
    )

    # Price & signals chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_series.index, y=price_series, mode='lines', name='Price'))
    if show_pts:
        buys  = price_series[signals==1]
        sells = price_series[signals==-1]
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys, mode='markers', name='Buy',
            marker=dict(color='green', symbol='triangle-up', size=8)))
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells, mode='markers', name='Sell',
            marker=dict(color='red', symbol='triangle-down', size=8)))
    fig.update_layout(title=f"{ticker} Price & Skarre Signals", height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green▲ = Buy, Red▼ = Sell")

    # Derivative preview
    st.subheader("Derivative Preview: Slope & Acceleration")
    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(pd.DataFrame({'Slope': slope}))
        st.caption("Slope = rate of change (momentum)")
    with c2:
        st.line_chart(pd.DataFrame({'Acceleration': accel}))
        st.caption("Acceleration = change of momentum")

    # Equity & metrics
    res = backtest(price_series, signals)
    eq = res['equity_curve']
    bh = (1 + price_series.pct_change().fillna(0)).cumprod()
    st.subheader("Equity Curve vs Buy & Hold")
    st.line_chart(pd.DataFrame({'Strategy': eq, 'Buy & Hold': bh}))
    st.caption("Cumulative returns comparison")

    perf = res['performance']
    st.subheader("Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sharpe Ratio",    f"{perf['Sharpe']:.2f}")
    m2.metric("Total Return",    f"{perf['Total Return']*100:.2f}%")
    m3.metric("Max Drawdown",    f"{perf['Max Drawdown']*100:.2f}%")
    m4.metric("Trades/Year",     f"{perf['Trade Frequency']:.1f}")
    st.caption("Costs & slippage applied dynamically")

# === Derivative Diagnostics ===
elif page == "Derivative Diagnostics":
    st.title("Derivative Diagnostics")
    st.markdown("Inspect slope & acceleration distributions before trading.")

    ticker2    = st.sidebar.text_input("Ticker", "SPY", key="diag_ticker").upper()
    sd2        = st.sidebar.date_input("Start Date", datetime(2010,1,1), key="diag_start")
    ed2        = st.sidebar.date_input("End Date",   datetime.today(), key="diag_end")
    show_marks = st.sidebar.checkbox("Show Buy/Sell Markers", True, key="diag_show_markers")

    df2 = get_data(ticker2, sd2, ed2)
    if df2.empty:
        st.warning("No data found.")
        st.stop()

    ps2 = df2['Price']
    sl2 = get_slope(ps2)
    ac2 = get_acceleration(ps2)

    st.line_chart(pd.DataFrame({'Slope': sl2, 'Acceleration': ac2}))
    st.caption("Slope = momentum; Acceleration = change of momentum")

# === Polynomial Fit Curve ===
elif page == "Polynomial Fit Curve":
    st.title("Polynomial Fit Curve")
    st.markdown("Fit a rolling 2nd-degree polynomial to price data.")

    ticker_pf = st.sidebar.text_input("Ticker", "SPY", key="pf_ticker").upper()
    window    = st.sidebar.slider("Window Size", 10, 50, 21, step=2, key="pf_window")

    df_pf = get_data(ticker_pf, "2022-01-01", "2024-12-31")['Price']
    xv    = np.arange(window)
    fits, dates = [], []
    for i in range(window, len(df_pf)):
        yvals  = df_pf.iloc[i-window:i].values
        coeffs = np.polyfit(xv, yvals, 2)
        fits.append(np.poly1d(coeffs)(xv)[-1])
        dates.append(df_pf.index[i-1])
    fit_s = pd.Series(fits, index=dates)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_pf.index, y=df_pf, name='Price'))
    fig3.add_trace(go.Scatter(x=fit_s.index, y=fit_s, name='Poly Fit'))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Rolling polynomial fit captures curvature")

# === Derivative Histograms ===
elif page == "Derivative Histograms":
    st.title("📊 Derivative Histograms")
    st.markdown("Distribution of slope & acceleration values for SPY (2022–2024).")

    df_h = get_data("SPY", "2022-01-01", "2024-12-31")['Price']
    s_h = get_slope(df_h)
    a_h = get_acceleration(df_h)

    def make_hist(x):
        counts, bins = np.histogram(x.dropna(), bins=30)
        return pd.Series(counts, index=bins[:-1])

    st.subheader("Slope Distribution")
    st.bar_chart(make_hist(s_h))
    st.caption("Frequency of momentum values")

    st.subheader("Acceleration Distribution")
    st.bar_chart(make_hist(a_h))
    st.caption("Frequency of change-of-momentum values")

# === Threshold Optimization ===
elif page == "Threshold Optimization":
    st.title("Threshold Optimization")
    st.markdown("Heatmap of Sharpe ratio over slope & SST thresholds for SPY (2022–2024).")

    price_opt = get_data("SPY", "2022-01-01", "2024-12-31")['Price']
    entry_slope_grid = np.linspace(0,0.5,6)
    exit_slope_grid  = np.linspace(-0.5,0,6)
    entry_sst_grid   = np.linspace(0,2,5)
    exit_sst_grid    = np.linspace(-2,0,5)

    best, best_sh, df_opt = grid_search_optimizer(
        price_opt,
        entry_slope_grid, exit_slope_grid,
        entry_sst_grid, exit_sst_grid
    )

    st.write(f"**Best Params:** slope_in={best[0]}, slope_out={best[1]}, sst_in={best[2]}, sst_out={best[3]}  → Sharpe {best_sh:.2f}")
    heatmap = df_opt.pivot_table(
    index="entry_slope",
    columns="exit_sst",
    values="Sharpe",
    aggfunc="mean"
)
    st.dataframe(heatmap.style.background_gradient(axis=None))
    st.caption("Green = better risk-adjusted performance")

# === Strategy Performance ===
elif page == "Strategy Performance":
    st.title("Strategy Performance")
    st.markdown("Equity curve comparison & regime analysis for SPY (2022–2024).")

    df_sp = get_data("SPY", "2022-01-01", "2024-12-31")['Price']
    # Use default or last optimized params
    signals_sp = generate_skarre_signal(df_sp, entry_slope_threshold=0.1, exit_slope_threshold=-0.1,
                                        entry_sst_threshold=1.0, exit_sst_threshold=-1.0,
                                        min_holding_days=5)
    res_sp = backtest(df_sp, signals_sp)
    eq_sp  = res_sp['equity_curve']
    bh_sp  = (1 + df_sp.pct_change().fillna(0)).cumprod()

    st.subheader("Equity Curve")
    st.line_chart(pd.DataFrame({'Strategy': eq_sp, 'Buy & Hold': bh_sp}))
    st.caption("Strategy vs passive benchmark returns")

    st.subheader("Performance Metrics")
    df_perf = pd.DataFrame(res_sp['performance'], index=["Value"]).T
    st.dataframe(df_perf.style.format({
        "Sharpe": "{:.2f}",
        "Total Return": "{:.1%}",
        "Max Drawdown": "{:.1%}",
        "Trade Frequency": "{:.1f}"
    }))

    st.subheader("Regime Analysis")
    df_reg = regime_performance(
        df_sp, generate_skarre_signal,
        entry_slope_threshold=0.1, exit_slope_threshold=-0.1,
        entry_sst_threshold=1.0, exit_sst_threshold=-1.0,
        min_holding_days=5
    )
    st.dataframe(df_reg)
    st.caption("Performance segmented by market regimes")

# === Trade Log ===
elif page == "Trade Log":
    st.title("Trade Log")
    st.markdown("Detailed list of executed trades for SPY.")

    df_tl = get_data("SPY", "2022-01-01", "2024-12-31")['Price']
    signals_tl = generate_skarre_signal(df_tl, entry_slope_threshold=0.1, exit_slope_threshold=-0.1,
                                        entry_sst_threshold=1.0, exit_sst_threshold=-1.0,
                                        min_holding_days=5)
    res_tl = backtest(df_tl, signals_tl)
    trade_log = res_tl.get('trade_log', pd.DataFrame())

    if trade_log.empty:
        st.warning("No trades executed under current thresholds.")
    else:
        st.dataframe(trade_log)
        csv = trade_log.to_csv(index=False)
        st.download_button("Download Trade Log CSV", csv, "trade_log.csv")
    st.caption("Timestamped buy/sell entries")
