# skar_tracker_dashboard.py


import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime

# Now these will load from the same folder
from polynomial_fit             import get_slope, get_acceleration
from signal_logic               import generate_skarre_signal
from backtester                 import backtest, evaluate_strategy
from optimizer                  import grid_search_optimizer
from validate_skarre_signal     import bootstrap_sharpe, regime_performance

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
    """Fetch Close prices for given ticker/date range."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[['Close']].dropna()
    df.columns = ['Price']
    return df

# === ABOUT ===
if page == "About":
    st.title("About Skarre Tracker")
    st.markdown("""
    **Skarre Tracker v2.0** is a research-grade trading dashboard that:
      - Extracts **slope** (1st derivative) and **acceleration** (2nd derivative) from price data.
      - Computes a **Standardized Skarre Score (SST)**: deviation from moving average divided by volatility.
      - Generates buy/sell signals only when slope > entry_threshold, acceleration > 0, and SST > entry_SST, and exits when slope < exit_threshold or SST < exit_SST.
      - Validates via **walk-forward**, **dynamic transaction costs**, **bootstrap Sharpe**, and **regime analysis**.
    """)
    st.caption("This is for educational/research purposes only — not financial advice.")

# === LIVE SIGNAL TRACKER ===
elif page == "Live Signal Tracker":
    st.title("🔴 Live Signal Tracker")
    st.markdown("Overlay real-time Skarre signals on price data.")

    # Sidebar controls
    ticker       = st.sidebar.text_input("Ticker", "SPY").upper()
    start_date   = st.sidebar.date_input("Start Date",  datetime(2022,1,1))
    end_date     = st.sidebar.date_input("End Date",    datetime(2024,12,31))
    entry_slope  = st.sidebar.slider("Entry Slope Threshold",  0.0, 0.5, 0.1, 0.01)
    exit_slope   = st.sidebar.slider("Exit Slope Threshold",  -0.5, 0.0, -0.1, 0.01)
    entry_sst    = st.sidebar.slider("Entry SST Threshold",    0.0, 2.0, 1.0, 0.1)
    exit_sst     = st.sidebar.slider("Exit SST Threshold",    -2.0, 0.0, -1.0, 0.1)
    holding_days = st.sidebar.slider("Minimum Holding Days",   1, 20, 5)
    show_points  = st.sidebar.checkbox("Show Buy/Sell Markers", True)

    df = get_data(ticker, start_date, end_date)
    if df.empty:
        st.warning("No data for this ticker/date.")
        st.stop()

    price = df['Price']
    slope = get_slope(price)
    accel = get_acceleration(price)

    # Generate signals using new V2 logic
    signals = generate_skarre_signal(
        price,
        entry_slope_threshold=entry_slope,
        exit_slope_threshold=exit_slope,
        entry_sst_threshold=entry_sst,
        exit_sst_threshold=exit_sst,
        min_holding_days=holding_days
    )

    # Price + signals plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price.index, y=price, mode='lines', name='Price'))
    if show_points:
        buys  = price[signals==1]
        sells = price[signals==-1]
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys, mode='markers', name='Buy',
            marker=dict(color='green',symbol='triangle-up',size=8)
        ))
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells, mode='markers', name='Sell',
            marker=dict(color='red',symbol='triangle-down',size=8)
        ))
    fig.update_layout(title=f"{ticker} Price & Skarre Signals", height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green triangles = buy; red triangles = sell.")

    # Derivative preview
    st.subheader("Derivatives: Slope & Acceleration")
    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(pd.DataFrame({'Slope': slope}))
        st.caption("Slope = 1st derivative of price (momentum).")
    with c2:
        st.line_chart(pd.DataFrame({'Acceleration': accel}))
        st.caption("Acceleration = 2nd derivative (change of momentum).")

    # Equity curve vs Buy & Hold
    res = backtest(price, signals)
    eq  = res['equity_curve']
    bh  = (1 + price.pct_change().fillna(0)).cumprod()
    st.subheader("Equity Curve Comparison")
    st.line_chart(pd.DataFrame({'Strategy': eq, 'Buy & Hold': bh}))
    st.caption("Cumulative returns over selected period.")

    # Performance metrics
    perf = res['performance']
    st.subheader("Performance Metrics")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Sharpe Ratio",    f"{perf['Sharpe']:.2f}")
    m2.metric("Total Return",    f"{perf['Total Return']*100:.2f}%")
    m3.metric("Max Drawdown",    f"{perf['Max Drawdown']*100:.2f}%")
    m4.metric("Trades per Year", f"{perf['Trade Frequency']:.1f}")
    st.caption("Sharpe and drawdown account for dynamic trading costs.")

# === DERIVATIVE DIAGNOSTICS ===
elif page == "Derivative Diagnostics":
    st.title("🔍 Derivative Diagnostics")
    st.markdown("Visualize momentum and acceleration without trading logic.")
    ticker = st.sidebar.text_input("Ticker", "SPY").upper()
    sd     = st.sidebar.date_input("Start Date", datetime(2010,1,1))
    ed     = st.sidebar.date_input("End Date",   datetime.today())
    df2    = get_data(ticker, sd, ed)
    if df2.empty:
        st.warning("No data.")
        st.stop()
    s2 = get_slope(df2['Price']); a2 = get_acceleration(df2['Price'])
    st.line_chart(pd.DataFrame({'Slope': s2, 'Acceleration': a2}))
    st.caption("Use these plots to choose threshold values.")

# === POLYNOMIAL FIT CURVE ===
elif page == "Polynomial Fit Curve":
    st.title("🔧 Polynomial Fit Curve")
    st.markdown("Fit a rolling 2nd-degree polynomial to reveal trend shape.")
    ticker_pf = st.sidebar.text_input("Ticker", "SPY").upper()
    df_pf     = get_data(ticker_pf, "2022-01-01", "2024-12-31")['Price']
    window    = st.sidebar.slider("Window Size", 10, 50, 21, 2)
    xv        = np.arange(window)
    fits, dates= [], []
    for i in range(window, len(df_pf)):
        vals = df_pf.iloc[i-window:i].values
        coeff= np.polyfit(xv, vals, 2)
        fits.append(np.poly1d(coeff)(xv)[-1])
        dates.append(df_pf.index[i-1])
    fit_s = pd.Series(fits, index=dates)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_pf.index, y=df_pf, name='Price'))
    fig2.add_trace(go.Scatter(x=fit_s.index, y=fit_s, name='Poly Fit'))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Rolling polynomial fit to detect curvature.")

# === DERIVATIVE HISTOGRAMS ===
elif page == "Derivative Histograms":
    st.title("📊 Derivative Histograms")
    st.markdown("Distribution of slope and acceleration values.")
    df_h = get_data("SPY","2022-01-01","2024-12-31")['Price']
    s_h = get_slope(df_h); a_h = get_acceleration(df_h)
    def hist(ser):
        counts, bins = np.histogram(ser.dropna(), bins=30)
        return pd.Series(counts, index=bins[:-1])
    st.bar_chart(hist(s_h)); st.caption("Slope distribution.")
    st.bar_chart(hist(a_h)); st.caption("Acceleration distribution.")

# === THRESHOLD OPTIMIZATION ===
elif page == "Threshold Optimization":
    st.title("🌡️ Threshold Optimization")
    st.markdown("Heatmap of Sharpe for various slope+SST thresholds.")
    df_opt = get_data("SPY","2022-01-01","2024-12-31")['Price']
    entry_slope_grid = np.linspace(0,0.5,6)
    exit_slope_grid  = np.linspace(-0.5,0,6)
    entry_sst_grid   = np.linspace(0,2,5)
    exit_sst_grid    = np.linspace(-2,0,5)
    best, best_sh, res = grid_search_optimizer(
        df_opt,
        entry_slope_grid, exit_slope_grid,
        entry_sst_grid, exit_sst_grid
    )
    st.write(f"**Best Params**: {best} → Sharpe: {best_sh:.2f}")
    heat = res.pivot('entry_slope','exit_sst','Sharpe')
    st.dataframe(heat.style.background_gradient(axis=None))
    st.caption("Find thresholds that maximize Sharpe.")

# === STRATEGY PERFORMANCE ===
elif page == "Strategy Performance":
    st.title("📈 Strategy Performance")
    st.markdown("Aggregate and regime-based performance.")
    df_sp = get_data("SPY","2022-01-01","2024-12-31")['Price']
    # Use best params from above
    signals_sp = generate_skarre_signal(df_sp, *best, min_holding_days=holding_days)
    res_sp     = backtest(df_sp, signals_sp)
    eq_sp      = res_sp['equity_curve']
    bh_sp      = (1+df_sp.pct_change().fillna(0)).cumprod()
    st.line_chart(pd.DataFrame({'Strategy':eq_sp,'B&H':bh_sp}))
    st.caption("Equity curve comparison over sample period.")
    # Regime analysis
    reg_df = regime_performance(df_sp, generate_skarre_signal,
                                entry_slope_threshold=best[0],
                                exit_slope_threshold=best[1],
                                entry_sst_threshold=best[2],
                                exit_sst_threshold=best[3],
                                min_holding_days=holding_days)
    st.subheader("Regime Performance")
    st.dataframe(reg_df)
    st.caption("Performance segmented by bull/sideways/bear markets.")

# === TRADE LOG ===
elif page == "Trade Log":
    st.title("🗒️ Trade Log")
    st.markdown("Detailed list of executed trades.")
    df_tl = get_data("SPY","2022-01-01","2024-12-31")['Price']
    signals_tl = generate_skarre_signal(df_tl, *best, min_holding_days=holding_days)
    res_tl = backtest(df_tl, signals_tl)
    trade_log = res_tl.get('trade_log', pd.DataFrame())
    if trade_log.empty:
        st.warning("No trades executed under current thresholds.")
    else:
        st.dataframe(trade_log)
        csv = trade_log.to_csv(index=False)
        st.download_button("Download Trade Log CSV", csv, "trade_log.csv")
    st.caption("Timestamped buy/sell executions.")
