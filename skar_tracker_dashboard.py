import os
import sys
import importlib.util
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime

# --- Helper to dynamically load modules from skar_lib ---
def load_mod(name: str, rel_path: str):
    module_path = os.path.join(os.path.dirname(__file__), rel_path)
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load data_loader
dl_mod = load_mod("data_loader", "skar_lib/data_loader.py")
get_data = dl_mod.get_data

# Load backtester
bt_mod = load_mod("backtester", "skar_lib/backtester.py")
backtest = bt_mod.backtest

# Load polynomial_fit
pf_mod = load_mod("polynomial_fit", "skar_lib/polynomial_fit.py")
get_slope = pf_mod.get_slope
get_acceleration = pf_mod.get_acceleration

# Load signal_logic
sl_mod = load_mod("signal_logic", "skar_lib/signal_logic.py")
generate_signals = sl_mod.generate_signals

# Load walkforward
wf_mod = load_mod("walkforward", "skar_lib/walkforward.py")
run_walkforward = wf_mod.run_walkforward

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Skarre Tracker Dashboard V3", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select View",
    [
        "About",
        "Live Signal",
        "Backtest V1",
        "Walk-Forward",
        "Trade Log",
    ],
)

# --- About Page ---
if page == "About":
    st.title("🚀 Skarre Tracker Dashboard V3")
    st.markdown(
        "This dashboard now supports robust walk-forward testing alongside your original V1 signal backtest."
    )

# --- Live Signal Page ---
elif page == "Live Signal":
    st.header("Live Signal Generation")
    ticker = st.sidebar.text_input("Ticker", "SPY")
    start = st.sidebar.date_input("Start Date", datetime(2022,1,1))
    end   = st.sidebar.date_input("End Date", datetime.today())
    df_price = get_data(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    price = df_price["Price"]

    slope = get_slope(price)
    accel = get_acceleration(price)
    entry = st.sidebar.number_input("Entry Threshold", value=0.5)
    exit_ = st.sidebar.number_input("Exit Threshold", value=-0.5)
    use_acc = st.sidebar.checkbox("Use Acceleration", True)

    signals = generate_signals(
        slope, accel, entry, exit_, use_acceleration=use_acc
    )
    st.line_chart(signals)

# --- Backtest V1 Page ---
elif page == "Backtest V1":
    st.header("Backtest V1: Skarre Signal")
    ticker = st.sidebar.text_input("Ticker", "SPY")
    start = st.sidebar.date_input("Start Date", datetime(2022,1,1))
    end   = st.sidebar.date_input("End Date", datetime.today())
    price = get_data(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))["Price"]

    slope = get_slope(price)
    accel = get_acceleration(price)
    entry = st.sidebar.number_input("Entry Threshold", value=0.5)
    exit_ = st.sidebar.number_input("Exit Threshold", value=-0.5)
    use_acc = st.sidebar.checkbox("Use Acceleration", True)

    signals = generate_signals(slope, accel, entry, exit_, use_acceleration=use_acc)
    result = backtest(price, signals)
    st.json(result.get("metrics", {}))

# --- Walk-Forward Page ---
elif page == "Walk-Forward":
    st.header("Walk-Forward Testing")
    ticker = st.sidebar.text_input("Ticker", "SPY")
    entry = st.sidebar.number_input("Entry Threshold", value=0.5)
    exit_ = st.sidebar.number_input("Exit Threshold", value=-0.5)
    train_window = st.sidebar.number_input("Train Window (days)", min_value=30, value=252)
    test_window  = st.sidebar.number_input("Test Window (days)",  min_value=30, value=63)
    step         = st.sidebar.number_input("Step Size (days)",    min_value=1,  value=63)

    price = get_data(ticker, "2000-01-01", datetime.today().strftime("%Y-%m-%d"))["Price"]
    def signal_fn(series_in):
        s = get_slope(series_in)
        a = get_acceleration(series_in)
        return generate_signals(s, a, entry, exit_, use_acceleration=True)

    df_fw = run_walkforward(price, signal_fn, train_window, test_window, step)
    st.dataframe(df_fw)
    if not df_fw.empty:
        st.line_chart(df_fw[["return","sharpe","max_drawdown"]])

# --- Trade Log Page ---
elif page == "Trade Log":
    st.header("Trade Log (SPY Example)")
    price = get_data("SPY", "2022-01-01", "2024-12-31")["Price"]
    slope = get_slope(price)
    accel = get_acceleration(price)
    signals = generate_signals(slope, accel, 0.5, -0.5, use_acceleration=True)
    result = backtest(price, signals)
    trade_df = result.get("trade_log", pd.DataFrame())
    if trade_df.empty:
        st.warning("No trades to display.")
    else:
        st.dataframe(trade_df)
        csv = trade_df.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="trade_log.csv")
