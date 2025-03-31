import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.stats import linregress

# -----------------------
# Helper Functions
# -----------------------

@st.cache_data(show_spinner=False)
def get_data(ticker, start, end):
    """
    Download historical data using yfinance.
    Flattens multi-index columns if necessary.
    """
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        st.error("No data returned. Please check the ticker and date range.")
        return df
    # Flatten multi-index columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)
    df.dropna(inplace=True)
    if 'Adj Close' in df.columns:
        df['Return'] = df['Adj Close'].pct_change()
    elif 'Close' in df.columns:
        df['Return'] = df['Close'].pct_change()
    else:
        st.error("Downloaded data does not contain 'Adj Close' or 'Close' column.")
    return df

def compute_skarre_signal(df, ma_window=150, vol_window=14):
    """
    Compute the Skarre Signal (Z-score) as:
       (Price - MA) / (Rolling Std of (Price - MA))
    Uses 'Adj Close' if available, else 'Close'.
    """
    df = df.copy()
    if 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    elif 'Close' in df.columns:
        price_col = 'Close'
    else:
        st.error("Data does not contain necessary price column.")
        return df

    df['MA'] = df[price_col].rolling(window=ma_window, min_periods=1).mean()
    df['Deviation'] = df[price_col] - df['MA']
    df['Vol'] = df['Deviation'].rolling(window=vol_window, min_periods=1).std()
    # Compute the signal; avoid division by zero
    df['Skarre_Signal'] = df.apply(lambda row: (row['Deviation'] / row['Vol']) if row['Vol'] != 0 else 0, axis=1)
    return df

def backtest_strategy(df, 
                      strategy="Contrarian", 
                      entry_threshold=-2.0, 
                      exit_threshold=0.5, 
                      trailing_stop=0.08, 
                      profit_target=0.1, 
                      initial_capital=100000,
                      risk_free_rate=0.02):
    """
    Backtest a strategy using the Skarre Signal.
    
    For "Contrarian" strategy, enter when signal <= entry_threshold and exit when signal >= exit_threshold.
    For "Momentum" strategy, logic is reversed.
    
    Also includes trailing stop and profit target.
    
    Returns a list of trades and a DataFrame of the equity curve.
    """
    df = df.copy().reset_index()
    position = 0      # 0: no position, 1: long position
    entry_price = 0
    max_price = 0     # for trailing stop
    capital = initial_capital
    equity_curve = [] # list of tuples (date, equity value)
    trades = []       # list of trade dictionaries
    
    for i, row in df.iterrows():
        date = row['Date']
        # Use available price column
        if 'Adj Close' in df.columns:
            price = row['Adj Close']
        elif 'Close' in df.columns:
            price = row['Close']
        else:
            st.error("Price column not found in data.")
            return trades, pd.DataFrame()
            
        signal = row['Skarre_Signal']
        
        # Record current equity (mark-to-market if in a trade)
        if position == 1:
            current_equity = capital * (price / entry_price)
        else:
            current_equity = capital
        equity_curve.append((date, current_equity))
        
        # Entry logic
        if position == 0:
            if strategy == "Contrarian" and signal <= entry_threshold:
                position = 1
                entry_price = price
                max_price = price
                trade = {"Entry Date": date, "Entry Price": price, "Exit Date": None, "Exit Price": None, "Return": None}
                trades.append(trade)
            elif strategy == "Momentum" and signal >= abs(entry_threshold):
                position = 1
                entry_price = price
                max_price = price
                trade = {"Entry Date": date, "Entry Price": price, "Exit Date": None, "Exit Price": None, "Return": None}
                trades.append(trade)
        # Exit logic
        else:
            if price > max_price:
                max_price = price
            exit_trade = False
            if strategy == "Contrarian":
                if signal >= exit_threshold:
                    exit_trade = True
            else:  # Momentum strategy
                if signal <= -abs(exit_threshold):
                    exit_trade = True
            # Check trailing stop and profit target
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
                capital = capital * (exit_price / entry_price)
                position = 0
                entry_price = 0
                max_price = 0

    equity_df = pd.DataFrame(equity_curve, columns=["Date", "Equity"]).set_index("Date")
    return trades, equity_df

def compute_performance_metrics(equity_df, initial_capital, risk_free_rate=0.02):
    """
    Compute performance metrics including Total Return, CAGR, Sharpe Ratio, Sortino Ratio, and Max Drawdown.
    """
    final_equity = equity_df['Equity'].iloc[-1]
    total_return = final_equity / initial_capital - 1

    dates = equity_df.index
    duration_days = (dates[-1] - dates[0]).days
    duration_years = duration_days / 365.25

    CAGR = (final_equity / initial_capital) ** (1 / duration_years) - 1

    equity_df['Daily Return'] = equity_df['Equity'].pct_change().fillna(0)
    avg_daily_return = equity_df['Daily Return'].mean()
    std_daily_return = equity_df['Daily Return'].std()
    sharpe = (avg_daily_return - risk_free_rate/252) / std_daily_return * np.sqrt(252) if std_daily_return != 0 else np.nan

    downside_returns = equity_df['Daily Return'][equity_df['Daily Return'] < 0]
    downside_std = downside_returns.std() if not downside_returns.empty else np.nan
    sortino = (avg_daily_return - risk_free_rate/252) / downside_std * np.sqrt(252) if downside_std and downside_std != 0 else np.nan

    equity_df['Cumulative Max'] = equity_df['Equity'].cummax()
    equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Cumulative Max']) / equity_df['Cumulative Max']
    max_drawdown = equity_df['Drawdown'].min()

    metrics = {
        "Total Return (%)": round(total_return * 100, 2),
        "CAGR (%)": round(CAGR * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Max Drawdown (%)": round(max_drawdown * 100, 2)
    }
    return metrics

def plot_price_and_signals(df, trades):
    """
    Plot the price series along with the moving average and mark buy/sell signals.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    if 'Adj Close' in df.columns:
        price_series = df['Adj Close']
    elif 'Close' in df.columns:
        price_series = df['Close']
    else:
        st.error("Price column not found in data.")
        return
    ax.plot(df.index, price_series, label='Price', color='blue')
    ax.plot(df.index, df['MA'], label='Moving Average', color='orange', linestyle='--')
    
    for trade in trades:
        entry_date = trade["Entry Date"]
        exit_date = trade["Exit Date"]
        entry_price = trade["Entry Price"]
        ax.scatter(entry_date, entry_price, marker="^", color="green", s=100, label="Buy")
        if exit_date:
            exit_price = trade["Exit Price"]
            ax.scatter(exit_date, exit_price, marker="v", color="red", s=100, label="Sell")
    
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_title("Price Chart with Buy/Sell Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

def plot_skarre_signal(df, entry_threshold, exit_threshold, strategy):
    """
    Plot the Skarre Signal (Z-score) over time with threshold lines.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, df['Skarre_Signal'], label='Skarre Signal', color='purple')
    if strategy == "Contrarian":
        ax.axhline(entry_threshold, color='green', linestyle='--', label=f'Entry Threshold ({entry_threshold})')
        ax.axhline(exit_threshold, color='red', linestyle='--', label=f'Exit Threshold ({exit_threshold})')
    else:
        ax.axhline(entry_threshold, color='green', linestyle='--', label=f'Entry Threshold ({entry_threshold})')
        ax.axhline(-abs(exit_threshold), color='red', linestyle='--', label=f'Exit Threshold ({-abs(exit_threshold)})')
    ax.set_title("Skarre Signal Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Signal (Z-score)")
    ax.legend()
    st.pyplot(fig)

def plot_equity_curve(equity_df):
    """
    Plot the portfolio equity curve over time.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_df.index, equity_df['Equity'], color='magenta', label="Strategy Equity")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    st.pyplot(fig)

def rolling_sharpe(equity_df, window=252, risk_free_rate=0.02):
    """
    Calculate a rolling Sharpe ratio over a given window (default 1 year ~252 trading days).
    """
    returns = equity_df['Equity'].pct_change().fillna(0)
    rolling_sharpe = returns.rolling(window=window).apply(
        lambda x: ((np.mean(x) - risk_free_rate/252) / np.std(x) * np.sqrt(252)) if np.std(x) != 0 else np.nan
    )
    return rolling_sharpe

def grid_search_optimization(df, strategy, initial_capital, risk_free_rate):
    """
    Perform a grid search over entry and exit thresholds to optimize the Sharpe Ratio.
    Returns the best parameters, best Sharpe, and a DataFrame of results.
    """
    best_sharpe = -np.inf
    best_params = {"entry_threshold": None, "exit_threshold": None}
    results = []
    
    entry_range = np.arange(-3.0, -1.0, 0.5) if strategy == "Contrarian" else np.arange(1.0, 3.0, 0.5)
    exit_range = np.arange(0.0, 2.0, 0.5) if strategy == "Contrarian" else np.arange(-2.0, 0.0, 0.5)
    
    for entry_thr in entry_range:
        for exit_thr in exit_range:
            trades, equity_df_test = backtest_strategy(df, strategy=strategy, 
                                                         entry_threshold=entry_thr, 
                                                         exit_threshold=exit_thr, 
                                                         initial_capital=initial_capital)
            equity_df_test['Daily Return'] = equity_df_test['Equity'].pct_change().fillna(0)
            avg_ret = equity_df_test['Daily Return'].mean()
            std_ret = equity_df_test['Daily Return'].std()
            sharpe = (avg_ret - risk_free_rate/252) / std_ret * np.sqrt(252) if std_ret != 0 else np.nan
            results.append((entry_thr, exit_thr, sharpe))
            if sharpe is not np.nan and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params["entry_threshold"] = entry_thr
                best_params["exit_threshold"] = exit_thr
                
    results_df = pd.DataFrame(results, columns=["Entry Threshold", "Exit Threshold", "Sharpe Ratio"])
    return best_params, best_sharpe, results_df

# -----------------------
# Streamlit App Layout
# -----------------------

st.title("Skarre Tracker Quantitative Portfolio Dashboard")

# Sidebar: User Inputs
st.sidebar.header("User Inputs")
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

with st.spinner("Downloading data..."):
    df_raw = get_data(ticker, start_date, end_date)
    if df_raw.empty:
        st.stop()
    df = compute_skarre_signal(df_raw, ma_window=ma_window, vol_window=vol_window)
    df.index = pd.to_datetime(df.index)

# Create Tabs for Organization
tabs = st.tabs(["Data & Signals", "Backtest", "Performance Metrics", "Rolling Analysis", "Optimization"])

with tabs[0]:
    st.header("Historical Data & Skarre Signal")
    st.write(df.tail())
    st.subheader("Skarre Signal Chart")
    plot_skarre_signal(df, entry_threshold, exit_threshold, strategy)

with tabs[1]:
    st.header("Strategy Backtest")
    trades, equity_df = backtest_strategy(df, strategy=strategy, 
                                            entry_threshold=entry_threshold, 
                                            exit_threshold=exit_threshold, 
                                            trailing_stop=trailing_stop, 
                                            profit_target=profit_target, 
                                            initial_capital=initial_capital,
                                            risk_free_rate=risk_free_rate)
    st.write("### Trade Log")
    if trades:
        trades_df = pd.DataFrame(trades)
        st.dataframe(trades_df)
    else:
        st.write("No trades executed with these parameters.")
    
    st.write("### Equity Curve")
    plot_equity_curve(equity_df)
    
    st.write("### Price Chart with Signals")
    plot_price_and_signals(df, trades)

with tabs[2]:
    st.header("Performance Metrics")
    metrics = compute_performance_metrics(equity_df, initial_capital, risk_free_rate)
    for key, value in metrics.items():
        st.write(f"**{key}:** {value}")

with tabs[3]:
    st.header("Rolling Sharpe Ratio (1-Year Window)")
    equity_df_copy = equity_df.copy()
    equity_df_copy['Rolling Sharpe'] = rolling_sharpe(equity_df_copy, window=252, risk_free_rate=risk_free_rate)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_df_copy.index, equity_df_copy['Rolling Sharpe'], color='teal')
    ax.set_title("Rolling Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    st.pyplot(fig)

with tabs[4]:
    st.header("Grid Search Optimization")
    if st.button("Run Grid Search"):
        best_params, best_sharpe, results_df = grid_search_optimization(df, strategy, initial_capital, risk_free_rate)
        st.write(f"**Best Entry Threshold:** {best_params['entry_threshold']}")
        st.write(f"**Best Exit Threshold:** {best_params['exit_threshold']}")
        st.write(f"**Best Sharpe Ratio:** {round(best_sharpe, 2)}")
        st.dataframe(results_df.sort_values(by="Sharpe Ratio", ascending=False))
    else:
        st.write("Press the button to run grid search optimization over thresholds.")
