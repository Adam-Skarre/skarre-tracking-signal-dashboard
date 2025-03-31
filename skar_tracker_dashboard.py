import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import linregress
import plotly.graph_objs as go

# -----------------------
# Data Download & Processing
# -----------------------

@st.cache_data(show_spinner=False)
def get_data(ticker, start, end):
    """
    Download historical data using yfinance.
    - Flatten multi-index columns if needed.
    - Normalize column names to Title Case.
    - If all 5 columns are identically named, rename them to Open, High, Low, Close, Volume.
    - Calculate daily returns using 'Adj Close' if available, else 'Close'.
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    
    if df.empty:
        st.error("No data returned. Please check the ticker and date range.")
        st.stop()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)
    
    df.columns = [col.title() for col in df.columns]
    
    if len(set(df.columns)) == 1 and df.shape[1] == 5:
        st.warning("All columns have the same name. Reassigning to Open, High, Low, Close, Volume.")
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
    
    df.dropna(inplace=True)
    
    if 'Adj Close' in df.columns:
        df['Return'] = df['Adj Close'].pct_change()
    elif 'Close' in df.columns:
        df['Return'] = df['Close'].pct_change()
    else:
        st.error("Downloaded data does not contain 'Adj Close' or 'Close'.")
        st.stop()
    
    return df

def compute_skarre_signal(df, ma_window=150, vol_window=14):
    """
    Compute the Skarre Signal (Z-score) as:
      Z = (Price - Moving Average) / (Rolling Std of (Price - MA))
    Uses 'Adj Close' if available, else 'Close'.
    """
    df = df.copy()
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    
    df['MA'] = df[price_col].rolling(window=ma_window, min_periods=1).mean()
    df['Deviation'] = df[price_col] - df['MA']
    df['Vol'] = df['Deviation'].rolling(window=vol_window, min_periods=1).std()
    df['Skarre_Signal'] = df.apply(lambda row: (row['Deviation'] / row['Vol']) if row['Vol'] != 0 else 0, axis=1)
    return df

# -----------------------
# Backtesting & Performance
# -----------------------

def backtest_strategy(df, 
                      strategy="Contrarian", 
                      entry_threshold=-2.0, 
                      exit_threshold=0.5, 
                      trailing_stop=0.08, 
                      profit_target=0.1, 
                      initial_capital=100000):
    """
    Backtest the strategy based on the Skarre Signal.
    - For "Contrarian": buy when signal <= entry_threshold, exit when signal >= exit_threshold.
    - For "Momentum": buy when signal >= entry_threshold, exit when signal <= exit_threshold.
    Also applies a trailing stop and profit target.
    
    Returns:
      - trades: List of trade dictionaries.
      - equity_df: DataFrame tracking portfolio equity over time.
      - buy_hold: Series representing buy-and-hold performance.
    """
    df = df.copy().reset_index()
    position = 0
    entry_price = 0
    max_price = 0
    capital = initial_capital
    equity_curve = []
    trades = []
    
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    initial_price = df[price_col].iloc[0]
    buy_hold = initial_capital * (df[price_col] / initial_price)
    
    for i, row in df.iterrows():
        date = row['Date']
        price = row[price_col]
        signal = row['Skarre_Signal']
        
        current_equity = capital * (price / entry_price) if position == 1 else capital
        equity_curve.append((date, current_equity))
        
        if position == 0:
            if strategy == "Contrarian" and signal <= entry_threshold:
                position = 1
                entry_price = price
                max_price = price
                trades.append({"Entry Date": date, "Entry Price": price, "Exit Date": None, "Exit Price": None, "Return": None})
            elif strategy == "Momentum" and signal >= abs(entry_threshold):
                position = 1
                entry_price = price
                max_price = price
                trades.append({"Entry Date": date, "Entry Price": price, "Exit Date": None, "Exit Price": None, "Return": None})
        else:
            if price > max_price:
                max_price = price
            exit_trade = False
            if strategy == "Contrarian" and signal >= exit_threshold:
                exit_trade = True
            elif strategy == "Momentum" and signal <= -abs(exit_threshold):
                exit_trade = True
            if price < max_price * (1 - trailing_stop):
                exit_trade = True
            if price >= entry_price * (1 + profit_target):
                exit_trade = True
            
            if exit_trade:
                exit_price = price
                trade_return = (exit_price / entry_price) - 1
                trades[-1]["Exit Date"] = date
                trades[-1]["Exit Price"] = exit_price
                trades[-1]["Return"] = trade_return
                capital *= (exit_price / entry_price)
                position = 0
                entry_price = 0
                max_price = 0

    equity_df = pd.DataFrame(equity_curve, columns=["Date", "Equity"]).set_index("Date")
    return trades, equity_df, buy_hold

def compute_performance_metrics(equity_df, initial_capital):
    """
    Compute performance metrics: Total Return, CAGR, Sharpe Ratio, Sortino Ratio, and Maximum Drawdown.
    """
    final_equity = equity_df['Equity'].iloc[-1]
    total_return = final_equity / initial_capital - 1
    
    dates = equity_df.index
    duration_years = (dates[-1] - dates[0]).days / 365.25
    CAGR = (final_equity / initial_capital) ** (1/duration_years) - 1
    
    equity_df['Daily Return'] = equity_df['Equity'].pct_change().fillna(0)
    avg_ret = equity_df['Daily Return'].mean()
    std_ret = equity_df['Daily Return'].std()
    sharpe = (avg_ret / std_ret * np.sqrt(252)) if std_ret != 0 else np.nan
    
    downside = equity_df['Daily Return'][equity_df['Daily Return'] < 0]
    downside_std = downside.std() if not downside.empty else np.nan
    sortino = (avg_ret / downside_std * np.sqrt(252)) if downside_std and downside_std != 0 else np.nan
    
    equity_df['Cumulative Max'] = equity_df['Equity'].cummax()
    equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Cumulative Max']) / equity_df['Cumulative Max']
    max_drawdown = equity_df['Drawdown'].min()
    
    return {
        "Total Return (%)": round(total_return * 100, 2),
        "CAGR (%)": round(CAGR * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Max Drawdown (%)": round(max_drawdown * 100, 2)
    }

# -----------------------
# Polynomial Analysis Functions
# -----------------------

def polynomial_analysis(df, window=30):
    """
    For a rolling window, fit a quadratic polynomial (degree=2) to the price data.
    Returns a DataFrame of coefficients with columns: Quadratic, Linear, Intercept.
    """
    price_col = 'Close' if 'Close' in df.columns else ('Adj Close' if 'Adj Close' in df.columns else None)
    if price_col is None:
        st.error("No valid price column for polynomial analysis.")
        st.stop()
    
    prices = df[price_col]
    coeffs = []
    dates = []
    for i in range(window, len(prices)):
        x = np.arange(window)
        y = prices.iloc[i-window:i].values
        p = np.polyfit(x, y, 2)
        coeffs.append(p)
        dates.append(prices.index[i])
    coeff_df = pd.DataFrame(coeffs, index=dates, columns=['Quadratic', 'Linear', 'Intercept'])
    return coeff_df

def plot_polynomial_sample(df, window=30):
    """
    Plot a sample quadratic polynomial fit over the most recent window.
    """
    price_col = 'Close' if 'Close' in df.columns else ('Adj Close' if 'Adj Close' in df.columns else None)
    if price_col is None:
        st.error("No valid price column for polynomial plot.")
        st.stop()
    prices = df[price_col].iloc[-window:]
    x = np.arange(window)
    p = np.polyfit(x, prices.values, 2)
    poly_fit = np.polyval(p, x)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(prices.index, prices, 'bo-', label='Actual Price')
    ax.plot(prices.index, poly_fit, 'r--', label=f'Quadratic Fit: a={p[0]:.4f}, b={p[1]:.4f}, c={p[2]:.4f}')
    ax.set_title("Sample Quadratic Polynomial Fit")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

# -----------------------
# Plotting Function for Trade Signals
# -----------------------

def plot_price_and_signals(df, trades):
    """
    Plot the price series, moving average, and mark trade signals.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    ax.plot(df.index, df[price_col], label='Price', color='blue')
    ax.plot(df.index, df['MA'], label='Moving Average', color='orange', linestyle='--')
    
    for trade in trades:
        entry_date = trade["Entry Date"]
        exit_date = trade["Exit Date"]
        entry_price = trade["Entry Price"]
        ax.scatter(entry_date, entry_price, marker="^", color="green", s=100, label="Buy")
        if exit_date:
            exit_price = trade["Exit Price"]
            ax.scatter(exit_date, exit_price, marker="v", color="red", s=100, label="Sell")
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_title("Price Chart with Trade Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

# -----------------------
# App Layout
# -----------------------

st.set_page_config(page_title="Skarre Tracker Quantitative Portfolio Dashboard", layout="wide")
st.title("Skarre Tracker Quantitative Portfolio Dashboard")

# Sidebar Inputs
st.sidebar.header("Inputs & Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY")
start_date = st.sidebar.date_input("Start Date", value=datetime(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())
ma_window = st.sidebar.number_input("Moving Average Window (days)", min_value=10, max_value=300, value=150)
vol_window = st.sidebar.number_input("Volatility Window (days)", min_value=5, max_value=60, value=14)
strategy = st.sidebar.selectbox("Strategy Type", options=["Contrarian", "Momentum"])
if strategy == "Contrarian":
    entry_threshold = st.sidebar.number_input("Entry Threshold (Z-score)", value=-2.0, step=0.1)
    exit_threshold = st.sidebar.number_input("Exit Threshold (Z-score)", value=0.5, step=0.1)
else:
    entry_threshold = st.sidebar.number_input("Entry Threshold (Z-score)", value=2.0, step=0.1)
    exit_threshold = st.sidebar.number_input("Exit Threshold (Z-score)", value=-0.5, step=0.1)
trailing_stop = st.sidebar.number_input("Trailing Stop Loss (%)", value=8.0, step=0.5) / 100.0
profit_target = st.sidebar.number_input("Profit Target (%)", value=10.0, step=0.5) / 100.0
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000, step=1000)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (annual %)", value=2.0, step=0.1) / 100.0
refresh_interval = st.sidebar.number_input("Live Graph Refresh (sec)", value=30, step=5)

with st.spinner("Downloading data..."):
    df_raw = get_data(ticker, start_date, end_date)
    df = compute_skarre_signal(df_raw, ma_window=ma_window, vol_window=vol_window)
    df.index = pd.to_datetime(df.index)
price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

# -----------------------
# Tabs
# -----------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Live Graph", "Data & Signals", "Backtest & Comparison", "Performance Metrics",
    "Polynomial Analysis", "About"
])

# Tab 1: Live Graph using Plotly
with tab1:
    st.header("Live Price Graph")
    fig_live = go.Figure()
    fig_live.add_trace(go.Scatter(
        x=df.index,
        y=df[price_col],
        mode='lines',
        name='Price',
        line=dict(color='blue')
    ))
    fig_live.add_trace(go.Scatter(
        x=df.index,
        y=df['MA'],
        mode='lines',
        name=f'{ma_window}-day MA',
        line=dict(color='orange')
    ))
    # Use "Vol" column (computed in compute_skarre_signal) for volatility
    entry_line = df['MA'].iloc[-1] + entry_threshold * df['Vol'].iloc[-1]
    fig_live.add_hline(
        y=entry_line,
        line_dash="dash",
        line_color="green",
        annotation_text="Entry Threshold",
        annotation_position="top right"
    )
    fig_live.update_layout(
        title="Live Price Data",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified'
    )
    st.plotly_chart(fig_live, use_container_width=True)
    
    latest_price = df[price_col].iloc[-1]
    latest_signal = df['Skarre_Signal'].iloc[-1]
    signal_status = "✅ Buy" if latest_signal >= entry_threshold else "⏸️ Hold/Cash"
    st.metric(label=f"Current {ticker} Price", value=f"${latest_price:.2f}")
    st.metric(label="Current Skarre Signal", value=f"{latest_signal:.2f}", delta=signal_status)
    st.write("This live graph updates on page reload (approximately every", refresh_interval, "seconds).")

# Tab 2: Data & Signals
with tab2:
    st.header("Historical Data & Skarre Signal")
    st.dataframe(df.tail(10))
    st.subheader("Price & Central Moving Average")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df[price_col], label='Price', color='blue')
    ax.plot(df.index, df['MA'], label='Moving Average', color='orange', linestyle='--')
    ax.set_title("Price & Central Moving Average")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
    st.write("The Skarre Signal is computed as the Z-score of the deviation from the moving average.")

# Tab 3: Backtest & Comparison
with tab3:
    st.header("Strategy Backtest & Buy-Hold Comparison")
    trades, equity_df, buy_hold = backtest_strategy(
        df, strategy=strategy,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        trailing_stop=trailing_stop,
        profit_target=profit_target,
        initial_capital=initial_capital
    )
    if trades is None:
        trades = []
    
    st.subheader("Trade Log")
    if trades and len(trades) > 0:
        st.dataframe(pd.DataFrame(trades))
    else:
        st.write("No trades executed with these parameters.")
    
    st.subheader("Equity Curve Comparison")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_df.index, equity_df['Equity'], label="Signal Strategy", color='magenta')
    ax.plot(buy_hold.index, buy_hold, label="Buy & Hold", color='gray', linestyle='--')
    ax.set_title("Equity Curve: Signal Strategy vs. Buy & Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("Price Chart with Trade Signals")
    if trades and len(trades) > 0:
        plot_price_and_signals(df, trades)
    else:
        st.write("No trade signals to plot.")

# Tab 4: Performance Metrics
with tab4:
    st.header("Performance Metrics")
    metrics = compute_performance_metrics(equity_df, initial_capital)
    for k, v in metrics.items():
        st.write(f"**{k}:** {v}")

# Tab 5: Polynomial Analysis
with tab5:
    st.header("Polynomial Analysis: Parabolic Trends")
    coeff_df = polynomial_analysis(df, window=30)
    st.subheader("Time Series of Quadratic Coefficient (Curvature)")
    st.line_chart(coeff_df['Quadratic'])
    st.subheader("Histogram of Coefficients")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(coeff_df['Quadratic'], bins=20, color='skyblue')
    ax[0].set_title("Quadratic Coefficient")
    ax[1].hist(coeff_df['Linear'], bins=20, color='salmon')
    ax[1].set_title("Linear Coefficient")
    ax[2].hist(coeff_df['Intercept'], bins=20, color='lightgreen')
    ax[2].set_title("Intercept")
    st.pyplot(fig)
    st.subheader("Sample Quadratic Fit")
    plot_polynomial_sample(df, window=30)

# Tab 6: About
with tab6:
    st.header("About Skarre Tracking Signal")
    st.markdown("""
    **Mission & Vision**  
    The Skarre Tracking Signal Dashboard is a comprehensive quantitative tool designed to analyze market dynamics 
    and identify potential trading opportunities using a proprietary signal methodology. Our mission is to combine 
    advanced statistical analysis, robust backtesting, and innovative polynomial trend analysis to empower users 
    with actionable insights for informed decision-making.
    
    **Overview of the Approach**  
    - **Data & Signals:** Historical price data is processed to compute a central moving average and a signal (as a Z-score) 
      that highlights deviations from typical price behavior.  
    - **Backtesting:** The strategy simulates trades based on the signal, comparing its performance with a traditional 
      buy-and-hold approach to evaluate risk and reward.  
    - **Polynomial Analysis:** Rolling quadratic fits capture parabolic trends in the market, providing a unique view 
      of market curvature and momentum.
    - **Live Graph:** An interactive Plotly graph displays real-time price data with dynamic annotations.
    
    **Our Goal**  
    Our aim is to deliver a professional-grade quantitative analysis platform that is both robust and intuitive. 
    While certain aspects of our proprietary signal remain confidential, the dashboard offers rich analytics and visualization 
    to foster further research and refinement of algorithmic trading strategies.
    
    **Explore & Experiment**  
    Adjust the parameters in the sidebar to explore different market scenarios and discover how the signal performs 
    under various conditions. Your feedback is vital as we continue to enhance this tool.
    """)
    st.write("Thank you for exploring the Skarre Tracking Signal Dashboard. We welcome your feedback!")
