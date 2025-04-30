import streamlit as st
import pandas as pd
from datetime import datetime

from skar_lib.data_loader import get_data
from skar_lib.backtester import backtest
from skar_lib.signal_logic import generate_signals
from skar_lib.polynomial_fit import get_slope, get_acceleration
from skar_lib.walkforward import run_walkforward

st.set_page_config(page_title="Skarre Tracker Dashboard V3", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select View", ["About", "Live Signal", "Backtest V1", "Walk-Forward"])

if page == "About":
    st.title("📊 Skarre Tracker Dashboard — V3")
    st.markdown(
        """
        This V3 dashboard features:
        - Signal generation using slope + acceleration
        - V1 backtest logic with entry/exit tuning
        - V3 walk-forward validation with fold-by-fold metrics
        """
    )

elif page == "Live Signal":
    st.header("📈 Live Signal Viewer")
    ticker = st.sidebar.text_input("Ticker", value="SPY")
    start = st.sidebar.date_input("Start Date", datetime(2022, 1, 1))
    end = st.sidebar.date_input("End Date", datetime.today())

    df = get_data(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    price = df["Price"]
    slope = get_slope(price)
    accel = get_acceleration(price)

    entry = st.sidebar.number_input("Entry Threshold", value=0.5)
    exit_ = st.sidebar.number_input("Exit Threshold", value=-0.5)
    use_acc = st.sidebar.checkbox("Use Acceleration", True)

    signals = generate_signals(slope, accel, entry, exit_, use_acc)
    st.line_chart(signals)

elif page == "Backtest V1":
    st.header("🔁 Backtest V1: Original Skarre Signal")
    ticker = st.sidebar.text_input("Ticker", value="SPY")
    start = st.sidebar.date_input("Start Date", datetime(2022, 1, 1))
    end = st.sidebar.date_input("End Date", datetime.today())

    df = get_data(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    price = df["Price"]
    slope = get_slope(price)
    accel = get_acceleration(price)

    entry = st.sidebar.number_input("Entry Threshold", value=0.5)
    exit_ = st.sidebar.number_input("Exit Threshold", value=-0.5)
    use_acc = st.sidebar.checkbox("Use Acceleration", True)

    signals = generate_signals(slope, accel, entry, exit_, use_acc)
    result = backtest(price, signals)
    st.subheader("Performance Metrics")
    st.json(result.get("metrics", {}))

elif page == "Walk-Forward":
    st.header("🧪 Walk-Forward Validation (V3)")
    ticker = st.sidebar.text_input("Ticker", value="SPY")
    entry = st.sidebar.number_input("Entry Threshold", value=0.5)
    exit_ = st.sidebar.number_input("Exit Threshold", value=-0.5)
    train_window = st.sidebar.number_input("Train Window (days)", min_value=30, value=252)
    test_window = st.sidebar.number_input("Test Window (days)", min_value=30, value=63)
    step = st.sidebar.number_input("Step Size (days)", min_value=1, value=63)

    df = get_data(ticker, "2000-01-01", datetime.today().strftime("%Y-%m-%d"))
    price = df["Price"]

    def signal_fn(p):
        s = get_slope(p)
        a = get_acceleration(p)
        return generate_signals(s, a, entry, exit_, True)

    df_folds = run_walkforward(price, signal_fn, train_window, test_window, step)
    st.subheader("Fold-by-Fold Results")
    st.dataframe(df_folds)
    if not df_folds.empty:
        st.line_chart(df_folds[["return", "sharpe", "max_drawdown"]])
