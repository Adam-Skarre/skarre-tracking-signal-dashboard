import os
import sys
import importlib.util
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta

# ─── Configure project and library paths ─────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
LIB_DIR = os.path.join(BASE_DIR, "skar_lib")
if os.path.isdir(LIB_DIR) and LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

# ─── Dynamic loader for your modules ───────────────────────────────────────────
def _load_module(name: str):
    """Load a module from skar_lib by filename without .py"""
    path = os.path.join(LIB_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# ─── Load core modules and bind functions ─────────────────────────────────────
_pf  = _load_module("polynomial_fit")
_sl  = _load_module("signal_logic")
_bt  = _load_module("backtester")
_opt = _load_module("optimizer")
_val = _load_module("validate_skarre_signal")
_dl  = _load_module("data_loader")

get_slope            = _pf.get_slope
get_acceleration     = _pf.get_acceleration
generate_skarre_signal = _sl.generate_skarre_signal
backtest             = _bt.backtest
evaluate_strategy    = _bt.evaluate_strategy
grid_search_optimizer = _opt.grid_search_optimizer
bootstrap_sharpe     = _val.bootstrap_sharpe
regime_performance   = _val.regime_performance
load_data            = _dl.load_data  # function to fetch price data

# ─── Streamlit app configuration ─────────────────────────────────────────────
st.set_page_config(page_title="Skarre Tracker Dashboard", layout="wide")

# ─── Sidebar navigation ──────────────────────────────────────────────────────
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

# ─── Utility: fetch data with caching ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Wrapper around your data_loader.load_data to ensure date filtering
    """
    df = load_data(ticker, start, end)
    # Expect df.index as datetime (or convert)
    if isinstance(df.index, pd.DatetimeIndex) is False:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df

# ─── Page: About ─────────────────────────────────────────────────────────────
if page == "About":
    st.title("Engineering, Optimization, and Comparison of Algorithmic Trading")
    st.markdown(
        """
        This dashboard is part of the independent study "Engineering Models, Optimization, and Comparison of Algorithmic Trading".
        Explore multi-version signal development, backtests, and live signal demo.
        """
    )

# ─── Page: Live Signal Tracker ────────────────────────────────────────────────
elif page == "Live Signal Tracker":
    st.title("Live Signal Tracker")
    ticker = st.sidebar.text_input("Ticker Symbol", "SPY").upper()
    start_date = st.sidebar.date_input("Start Date", datetime(2022, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.today())
    entry_th = st.sidebar.slider("Entry Threshold", 0.0, 5.0, 0.5, 0.1)
    exit_th  = st.sidebar.slider("Exit Threshold", -5.0, 0.0, -0.5, 0.1)
    use_accel = st.sidebar.checkbox("Use Acceleration", True)
    show_markers = st.sidebar.checkbox("Show Buy/Sell Markers", True)

    df = get_data(ticker, start_date, end_date)
    if df.empty:
        st.warning(f"No data found for {ticker} from {start_date} to {end_date}.")
        st.stop()

    price = df['Close'] if 'Close' in df else df.iloc[:, 0]
    slope = get_slope(price)
    accel = get_acceleration(price)
    signals = generate_skarre_signal(slope, accel, entry_th, exit_th, use_acceleration=use_accel)

    fig = go.Figure([go.Scatter(x=price.index, y=price, name='Price')])
    if show_markers:
        buys  = price[signals ==  1]
        sells = price[signals == -1]
        fig.add_trace(go.Scatter(x=buys.index,  y=buys,  mode='markers', name='Buy',  marker_symbol='triangle-up'))
        fig.add_trace(go.Scatter(x=sells.index, y=sells, mode='markers', name='Sell', marker_symbol='triangle-down'))
    st.plotly_chart(fig, use_container_width=True)

# ─── Page: Derivative Diagnostics ───────────────────────────────────────────
elif page == "Derivative Diagnostics":
    st.title("Derivative Diagnostics")
    ticker = st.sidebar.text_input("Ticker Symbol", "SPY").upper()
    sd = st.sidebar.date_input("Start Date", datetime(2010, 1, 1))
    ed = st.sidebar.date_input("End Date", datetime.today())
    entry_s = st.sidebar.slider("Entry Slope Th", 0.0, 10.0, 0.5, 0.1)
    exit_s  = st.sidebar.slider("Exit Slope Th", -10.0, 0.0, -0.5, 0.1)
    use_a   = st.sidebar.checkbox("Use Acceleration", False)

    df = get_data(ticker, sd, ed)
    if df.empty: st.warning("No data"); st.stop()
    price = df.iloc[:, -1]
    slope = get_slope(price)
    accel = get_acceleration(price)
    sig   = generate_skarre_signal(slope, accel, entry_s, exit_s, use_acceleration=use_a)

    st.line_chart(pd.DataFrame({'Price': price, 'Slope': slope, 'Accel': accel}))

# ─── Page: Polynomial Fit Curve ─────────────────────────────────────────────
elif page == "Polynomial Fit Curve":
    st.title("Polynomial Fit Curve Analysis")
    ticker = st.sidebar.text_input("Ticker", "SPY").upper()
    window = st.sidebar.slider("Window Size", 10, 100, 21, 1)
    df = get_data(ticker, datetime(2022,1,1), datetime.today())
    if df.empty: st.warning("No data"); st.stop()
    price = df.iloc[:, -1]
    fit_vals = []
    dates = []
    x = np.arange(window)
    for i in range(window, len(price)):
        y = price.iloc[i-window:i].values
        coeffs = np.polyfit(x, y, 2)
        poly = np.poly1d(coeffs)
        fit_vals.append(poly(x)[-1])
        dates.append(price.index[i-1])
    fit = pd.Series(fit_vals, index=dates)
    st.line_chart(pd.DataFrame({'Price': price, 'PolyFit': fit}))

# ─── Page: Derivative Histograms ───────────────────────────────────────────
elif page == "Derivative Histograms":
    st.title("Slope & Acceleration Distribution")
    df = get_data("SPY", datetime(2022,1,1), datetime.today())
    price = df.iloc[:, -1]
    slope = get_slope(price)
    accel = get_acceleration(price)
    st.bar_chart(slope.dropna().value_counts(bins=30))
    st.bar_chart(accel.dropna().value_counts(bins=30))

# ─── Page: Threshold Optimization ───────────────────────────────────────────
elif page == "Threshold Optimization":
    st.title("Threshold Optimization Heatmap")
    df = get_data("SPY", datetime(2022,1,1), datetime.today())
    price = df.iloc[:, -1]
    slope = get_slope(price)
    accel = get_acceleration(price)
    params, best_sh, heat_df = grid_search_optimizer(price, slope, accel)
    st.write("Best Params:", params, "Sharpe:", best_sh)
    st.dataframe(heat_df)

# ─── Page: Strategy Performance ─────────────────────────────────────────────
elif page == "Strategy Performance":
    st.title("Strategy vs Buy & Hold")
    df = get_data("SPY", datetime(2022,1,1), datetime.today())
    price = df.iloc[:, -1]
    slope = get_slope(price)
    accel = get_acceleration(price)
    sig = generate_skarre_signal(slope, accel, 0.5, -0.5, use_acceleration=True)
    res = backtest(price, sig)
    eq = res['equity_curve']
    bh = (1 + price.pct_change().fillna(0)).cumprod()
    st.line_chart(pd.DataFrame({'Strategy': eq, 'BuyHold': bh}))
    st.metric("Sharpe", res['performance']['Sharpe'])
    st.metric("Return", f"{res['performance']['Total Return']*100:.2f}%")
    st.metric("Drawdown", f"{res['performance']['Max Drawdown']*100:.2f}%")
    st.metric("Trades/Year", f"{res['performance']['Trade Frequency']:.1f}")

# ─── Page: Trade Log ─────────────────────────────────────────────────────────
elif page == "Trade Log":
    st.title("Trade Log")
    df = get_data("SPY", datetime(2022,1,1), datetime.today())
    price = df.iloc[:, -1]
    slope = get_slope(price)
    accel = get_acceleration(price)
    sig = generate_skarre_signal(slope, accel, 0.5, -0.5, use_acceleration=True)
    res = backtest(price, sig)
    log = res.get('trade_log', pd.DataFrame())
    if log.empty:
        st.write("No trades.")
    else:
        st.dataframe(log)
        st.download_button("Download CSV", log.to_csv(index=False), "trade_log.csv")
