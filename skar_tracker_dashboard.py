import os
import sys
# Add skar_lib folder to PYTHONPATH for direct module imports
base_dir = os.path.abspath(os.path.dirname(__file__))
lib_path = os.path.join(base_dir, "skar_lib")
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime

# Direct imports from skar_lib modules
from polynomial_fit import get_slope, get_acceleration
from signal_logic import generate_signals
from backtester import backtest
from data_loader import get_data
from walkforward import run_walkforward
from skar_lib.polynomial_fit import get_slope, get_acceleration
from skar_lib.signal_logic import generate_signals
from skar_lib.backtester import backtest
from skar_lib.data_loader import get_data
from skar_lib.walkforward import run_walkforward

st.set_page_config(page_title="Skarre Tracker Dashboard V3", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select View", [
    "About",
    "Live Signal",
    "Backtest V1",
    "Plot Signals",
    "Metrics Summary",
    "Optimization",
    "Validation",
    "Plots",
    "Walk-Forward",
    "Dashboard",
    "Trade Log (SPY Example)"
])

# "About" page
if page == "About":
    st.title("🚀 Skarre Tracker Dashboard V3")
    st.markdown(
        "This dashboard provides signal generation, backtesting, and now walk-forward testing for robust validation."
    )

# "Live Signal" page
elif page == "Live Signal":
    st.header("Live Signal Generation")
    tick = st.sidebar.text_input("Ticker", "SPY")
    start = st.sidebar.date_input("Start Date", datetime(2022,1,1))
    end   = st.sidebar.date_input("End Date", datetime.today())
    price_df = get_data(tick, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    price_series = price_df["Price"]
    slope = get_slope(price_series)
    accel = get_acceleration(price_series)
    entry = st.sidebar.number_input("Entry Threshold", value=0.5)
    exit_ = st.sidebar.number_input("Exit Threshold", value=-0.5)
    use_acc = st.sidebar.checkbox("Use Acceleration", True)
    signals = generate_signals(
        slope, accel, entry, exit_, use_acceleration=use_acc
    )
    st.line_chart(signals)

# "Backtest V1" page
elif page == "Backtest V1":
    st.header("Backtest V1: Skarre Signal")
    tick = st.sidebar.text_input("Ticker", "SPY")
    start = st.sidebar.date_input("Start Date", datetime(2022,1,1))
    end   = st.sidebar.date_input("End Date", datetime.today())
    price_series = get_data(tick, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))["Price"]
    slope = get_slope(price_series)
    accel = get_acceleration(price_series)
    entry = st.sidebar.number_input("Entry Threshold", value=0.5)
    exit_ = st.sidebar.number_input("Exit Threshold", value=-0.5)
    use_acc = st.sidebar.checkbox("Use Acceleration", True)
    signals = generate_signals(slope, accel, entry, exit_, use_acceleration=use_acc)
    result = backtest(price_series, signals)
    st.write("**Performance Metrics**")
    st.json(result.get("metrics", {}))

# "Plot Signals" page
elif page == "Plot Signals":
    st.header("Signal vs Price Plot")
    st.info("Plot Signals module coming soon.")

# "Metrics Summary" page
elif page == "Metrics Summary":
    st.header("Metrics Summary")
    st.info("Metrics Summary module coming soon.")

# "Optimization" page
elif page == "Optimization":
    st.header("Parameter Optimization")
    st.info("Optimization module coming soon.")

# "Validation" page
elif page == "Validation":
    st.header("Signal Validation")
    st.info("Validation module coming soon.")

# "Plots" page
elif page == "Plots":
    st.header("Additional Plots")
    st.info("Additional Plots module coming soon.")

# "Walk-Forward" page (V3)
elif page == "Walk-Forward":
    st.header("Walk-Forward Testing")
    tick = st.sidebar.text_input("Ticker", "SPY")
    entry = st.sidebar.number_input("Entry Threshold", value=0.5)
    exit_ = st.sidebar.number_input("Exit Threshold", value=-0.5)
    train_window = st.sidebar.number_input("Train Window (days)", min_value=30, value=252)
    test_window  = st.sidebar.number_input("Test Window (days)",  min_value=30, value=63)
    step         = st.sidebar.number_input("Step Size (days)",    min_value=1,  value=63)
    price_df = get_data(tick, "2000-01-01", datetime.today().strftime("%Y-%m-%d"))
    series = price_df["Price"]
    def signal_fn(series_in):
        s = get_slope(series_in)
        a = get_acceleration(series_in)
        return generate_signals(s, a, entry, exit_, use_acceleration=True)
    df_fw = run_walkforward(series, signal_fn, train_window, test_window, step)
    st.subheader("Fold-by-Fold Results")
    st.dataframe(df_fw)
    if not df_fw.empty:
        st.line_chart(df_fw[["return","sharpe","max_drawdown"]])

# "Dashboard" page
elif page == "Dashboard":
    st.header("Comprehensive Dashboard")
    st.info("Dashboard module coming soon.")

# "Trade Log (SPY Example)" page
elif page == "Trade Log (SPY Example)":
    st.header("Trade Log (SPY Example)")
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
