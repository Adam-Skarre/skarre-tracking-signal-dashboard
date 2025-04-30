# skar_tracker_dashboard.py
import os, sys, importlib.util
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf
from datetime import datetime, timedelta

# ─── Ensure skar_lib/ is on PYTHONPATH ─────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
LIB_DIR  = os.path.join(BASE_DIR, "skar_lib")
if os.path.isdir(LIB_DIR) and LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

# ─── Dynamically load a module by filename ────────────────────────────────────
def _load_module(name):
    path = os.path.join(LIB_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ─── Load core modules ────────────────────────────────────────────────────────
_pf  = _load_module("polynomial_fit")
_sl  = _load_module("signal_logic")
_bt  = _load_module("backtester")
_opt = _load_module("optimizer")
_val = _load_module("validate_skarre_signal")
_dl  = _load_module("data_loader")

# ─── Bind functions to consistent names ───────────────────────────────────────
get_slope        = _pf.get_slope
get_acceleration = _pf.get_acceleration

generate_signals      = _sl.generate_signals
backtest              = _bt.backtest
grid_search_optimizer = _opt.grid_search_optimizer
bootstrap_sharpe      = _val.bootstrap_sharpe
regime_performance    = _val.regime_performance

download_price_data = _dl.load_data  # adjust if named differently

# ─── Streamlit config ────────────────────────────────────────────────────────
st.set_page_config(page_title="Skarre Tracker V3", layout="wide")

# ─── Data fetching with caching ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data(ticker: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    # include end date by adding one day (yf.download is exclusive)
    df = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False)
    if df.empty:
        return pd.DataFrame(columns=["Price"])
    df = df[["Close"]].rename(columns={"Close": "Price"})
    df.index = pd.to_datetime(df.index)
    mask = (df.index.date >= start) & (df.index.date <= end)
    return df.loc[mask]

# ─── Cached, coarse grid search for demo ──────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_grid_search(price, slope, accel):
    # coarse steps for speed demo
    entry_range = np.arange(0.0, 2.1, 0.5)
    exit_range  = np.arange(-2.0, 0.1, 0.5)
    best, df = grid_search_optimizer(
        price=price,
        slope=slope,
        accel=accel,
        entry_range=entry_range,
        exit_range=exit_range,
        use_sst=True,
        ma_window=50,
        vol_window=20,
        min_holding_days=3
    )
    return best, df

# ─── Cached bootstrap for demo ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_bootstrap(price, signal):
    return bootstrap_sharpe(price, signal, n_boot=500, block_size=20)

# ─── Sidebar navigation ─────────────────────────────────────────────────────
st.sidebar.title("Navigation")
pages = [
    "About",
    "Live Signal Tracker",
    "Derivative Diagnostics",
    "Polynomial Fit Curve",
    "Derivative Histograms",
    "Threshold Optimization",
    "Strategy Performance",
    "Trade Log"
]
page = st.sidebar.radio("Select View", pages)

# ─── Page: About ─────────────────────────────────────────────────────────────
if page == "About":
    st.title("Skarre Tracker V3: Engineering & Validation")
    st.markdown("""
**V3 Highlights**  
- Regime-aware, SST-adjusted signals  
- Realistic costs, position sizing  
- Multi-asset & live-ready Streamlit UI  
- Rigorous walk-forward & bootstrap validation  
    """)
    st.markdown("Developed as part of ME 3296 Independent Study — aiming for robust, real-world outperformance.")

# ─── Page: Live Signal Tracker ───────────────────────────────────────────────
elif page == "Live Signal Tracker":
    st.title("Live Signal Tracker (SPY example)")
    ticker     = st.sidebar.text_input("Ticker", "SPY").upper()
    start_date = st.sidebar.date_input("Start Date",  datetime(2022,1,1))
    end_date   = st.sidebar.date_input("End Date",    datetime.today().date())
    entry_th   = st.sidebar.slider("Entry Th (slope)", 0.0, 5.0, 0.5, 0.1)
    exit_th    = st.sidebar.slider("Exit Th (slope)", -5.0, 0.0, -0.5, 0.1)
    use_accel  = st.sidebar.checkbox("Use Acceleration", True)
    show_mark  = st.sidebar.checkbox("Show Buy/Sell",    True)

    df = get_data(ticker, start_date, end_date)
    if df.empty:
        st.warning(f"No data for {ticker} from {start_date} to {end_date}.")
        st.stop()

    price = df["Price"]
    slope = get_slope(price)
    accel = get_acceleration(price)
    sig   = generate_signals(slope, accel, entry_th, exit_th, use_acceleration=use_accel)

    fig = go.Figure([go.Scatter(x=price.index, y=price, name="Price")])
    if show_mark:
        buys  = price[sig ==  1]
        sells = price[sig == -1]
        fig.add_trace(go.Scatter(x=buys.index,  y=buys,  mode="markers", name="Buy",  marker_symbol="triangle-up"))
        fig.add_trace(go.Scatter(x=sells.index, y=sells, mode="markers", name="Sell", marker_symbol="triangle-down"))
    st.plotly_chart(fig, use_container_width=True)

# ─── Page: Derivative Diagnostics ───────────────────────────────────────────
elif page == "Derivative Diagnostics":
    st.title("Derivative Diagnostics")
    ticker = st.sidebar.text_input("Ticker", "SPY").upper()
    sd     = st.sidebar.date_input("Start", datetime(2010,1,1))
    ed     = st.sidebar.date_input("End",   datetime.today().date())
    entryS = st.sidebar.slider("Entry Slope",  0.0, 10.0, 0.5, 0.1)
    exitS  = st.sidebar.slider("Exit Slope",  -10.0, 0.0, -0.5, 0.1)
    useA   = st.sidebar.checkbox("Use Acceleration", False)

    df = get_data(ticker, sd, ed)
    if df.empty: st.warning("No data"); st.stop()
    price = df["Price"]
    slope = get_slope(price)
    accel = get_acceleration(price)
    st.line_chart(pd.DataFrame({"Price":price,"Slope":slope,"Accel":accel}))

# ─── Page: Polynomial Fit Curve ─────────────────────────────────────────────
elif page == "Polynomial Fit Curve":
    st.title("Polynomial Fit Curve")
    ticker = st.sidebar.text_input("Ticker", "SPY").upper()
    window = st.sidebar.slider("Window Size", 10, 100, 21, 1)

    df = get_data(ticker, datetime(2022,1,1), datetime.today().date())
    if df.empty: st.warning("No data"); st.stop()
    price = df["Price"]

    xs, fit_vals = np.arange(window), []
    dates = []
    for i in range(window, len(price)):
        ys = price.iloc[i-window:i].values
        coeffs = np.polyfit(xs, ys, 2)
        fit_vals.append(np.poly1d(coeffs)(xs)[-1])
        dates.append(price.index[i-1])
    fit = pd.Series(fit_vals, index=dates)

    st.line_chart(pd.DataFrame({"Price":price, "PolyFit":fit}))

# ─── Page: Derivative Histograms ───────────────────────────────────────────
elif page == "Derivative Histograms":
    st.title("Derivative Value Distributions")
    df = get_data("SPY", datetime(2022,1,1), datetime.today().date())
    price = df["Price"]
    slope = get_slope(price).dropna()
    accel = get_acceleration(price).dropna()
    st.bar_chart(slope.value_counts(bins=30).sort_index())
    st.bar_chart(accel.value_counts(bins=30).sort_index())

# ─── Page: Threshold Optimization ───────────────────────────────────────────
elif page == "Threshold Optimization":
    st.title("Threshold Optimization")
    df = get_data("SPY", datetime(2022,1,1), datetime.today().date())
    price = df["Price"]
    slope = get_slope(price)
    accel = get_acceleration(price)

    with st.spinner("Grid search…"):
        (best_e, best_x, best_sh), results_df = run_grid_search(price, slope, accel)

    st.success(f"Best entry={best_e}, exit={best_x} → Sharpe={best_sh:.2f}")
    st.dataframe(results_df.style.background_gradient(cmap="RdYlGn"))

# ─── Page: Strategy Performance ─────────────────────────────────────────────
elif page == "Strategy Performance":
    st.title("Strategy vs Buy & Hold")
    df = get_data("SPY", datetime(2022,1,1), datetime.today().date())
    price = df["Price"]
    slope = get_slope(price)
    accel = get_acceleration(price)
    sig   = generate_signals(slope, accel, 0.5, -0.5, use_acceleration=True)

    res = backtest(price, sig)
    eq  = res["equity_curve"] if "equity_curve" in res else res["Equity"]
    bh  = (1 + price.pct_change().fillna(0)).cumprod()

    st.line_chart(pd.DataFrame({"Strategy":eq, "BuyHold":bh}))
    perf = res["performance"]
    st.metric("Sharpe",           f"{perf['Sharpe']:.2f}")
    st.metric("Total Return",     f"{perf['Total Return']*100:.2f}%")
    st.metric("Max Drawdown",     f"{perf['Max Drawdown']*100:.2f}%")
    st.metric("Trades per Year",  f"{perf.get('Trade Frequency', np.nan):.1f}")

# ─── Page: Trade Log ─────────────────────────────────────────────────────────
elif page == "Trade Log":
    st.title("Trade Log")
    df = get_data("SPY", datetime(2022,1,1), datetime.today().date())
    price = df["Price"]
    slope = get_slope(price)
    accel = get_acceleration(price)
    sig   = generate_signals(slope, accel, 0.5, -0.5, use_acceleration=True)

    res = backtest(price, sig)
    log = res.get("trade_log", pd.DataFrame())
    if log.empty:
        st.info("No trades triggered.")
    else:
        st.dataframe(log)
        csv = log.to_csv(index=False)
        st.download_button("Download CSV", csv, "trade_log.csv")
