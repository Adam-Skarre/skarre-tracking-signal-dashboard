import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
import scipy

# -----------------------
# Data Download & Processing
# -----------------------

@st.cache_data(show_spinner=False)
def get_data(ticker, start, end):
    """
    Download historical data using yfinance.
    - Uses group_by='ticker' to obtain properly labeled columns.
    - Flattens multi-index columns and normalizes them to Title Case.
    - If all 5 columns are identically named, renames them to Open, High, Low, Close, Volume.
    - Computes daily returns using 'Adj Close' if available, else 'Close'.
    """
    df = yf.download(ticker, start=start, end=end, progress=False, group_by='ticker')
    
    if df.empty:
        st.error("No data returned. Please check the ticker and date range.")
        st.stop()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)
    else:
        df.columns = [col.title() for col in df.columns]
    
    df.columns = [col.title() for col in df.columns]
    
    # If all columns are the same and there are 5 columns, reassign them.
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
      Z = (Price - MA) / (Rolling Std of (Price - MA))
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

def backtest_strategy(df, strategy="Contrarian", entry_threshold=-2.0, exit_threshold=0.5,
                      trailing_stop=0.08, profit_target=0.1, initial_capital=100000):
    """
    Backtest the strategy based on the Skarre Signal.
    - For "Contrarian": Buy when signal <= entry_threshold; sell when signal >= exit_threshold.
    - For "Momentum":  Buy when signal >= entry_threshold; sell when signal <= exit_threshold.
    Also applies a trailing stop and profit target.
    
    Returns:
      - trades: List of trade dictionaries.
      - equity_df: DataFrame tracking portfolio equity over time.
      - buy_hold: Series of buy-and-hold performance.
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
# Polynomial Analysis
# -----------------------

def polynomial_analysis(df, window=30):
    """
    Fit a rolling quadratic polynomial (degree=2) to the price data.
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

def plot_polynomial_sample_plotly(df, window=30):
    """
    Plot a sample quadratic polynomial fit over the most recent window using Plotly.
    """
    price_col = 'Close' if 'Close' in df.columns else ('Adj Close' if 'Adj Close' in df.columns else None)
    if price_col is None:
        st.error("No valid price column for polynomial sample.")
        st.stop()
    
    prices = df[price_col].iloc[-window:]
    x = np.arange(window)
    p = np.polyfit(x, prices.values, 2)
    poly_fit = np.polyval(p, x)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices.index, 
        y=prices, 
        mode='lines+markers', 
        name='Actual Price',
        line=dict(color='blue')
    ))
    fit_label = f"Quadratic Fit: a={p[0]:.4f}, b={p[1]:.4f}, c={p[2]:.4f}"
    fig.add_trace(go.Scatter(
        x=prices.index, 
        y=poly_fit, 
        mode='lines', 
        name=fit_label, 
        line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title="Sample Quadratic Polynomial Fit",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified',
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_price_and_signals_plotly(df, trades):
    """
    Plot the price, moving average, and buy/sell signals interactively using Plotly.
    """
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    for trade in trades:
        buy_x.append(trade["Entry Date"])
        buy_y.append(trade["Entry Price"])
        if trade["Exit Date"]:
            sell_x.append(trade["Exit Date"])
            sell_y.append(trade["Exit Price"])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df[price_col],
        mode='lines',
        name='Price',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['MA'],
        mode='lines',
        name='Moving Average',
        line=dict(color='orange', dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=buy_x, 
        y=buy_y,
        mode='markers',
        name='Buy',
        marker=dict(symbol='triangle-up', color='green', size=12)
    ))
    fig.add_trace(go.Scatter(
        x=sell_x, 
        y=sell_y,
        mode='markers',
        name='Sell',
        marker=dict(symbol='triangle-down', color='red', size=12)
    ))
    fig.update_layout(
        title="Price Chart with Trade Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified',
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Main App Layout
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

# Create Tabs
tabs = st.tabs(["Live Graph", "Data & Signals", "Backtest & Comparison", "Performance Metrics", "Polynomial Analysis", "About"])

# Tab 1: Live Graph
with tabs[0]:
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
        line=dict(color='orange', dash='dot')
    ))
    # Dynamic entry threshold line using Vol (volatility)
    entry_line = df['MA'].iloc[-1] + entry_threshold * df['Vol'].iloc[-1]
    fig_live.add_hline(
        y=entry_line,
        line_dash="dash",
        line_color="green",
        annotation_text="Entry Threshold",
        annotation_position="top right"
    )
    # Add range slider for better navigation
    fig_live.update_layout(
        title="Live Price Data",
        xaxis=dict(title="Date", rangeslider=dict(visible=True)),
        yaxis_title="Price",
        hovermode='x unified',
        template="plotly_white"
    )
    st.plotly_chart(fig_live, use_container_width=True)
    
    latest_price = df[price_col].iloc[-1]
    latest_signal = df['Skarre_Signal'].iloc[-1]
    signal_status = "✅ Buy" if latest_signal >= entry_threshold else "⏸️ Hold/Cash"
    st.metric(label=f"Current {ticker} Price", value=f"${latest_price:.2f}")
    st.metric(label="Current Skarre Signal", value=f"{latest_signal:.2f}", delta=signal_status)
    st.write("This live graph refreshes on page reload (approximately every", refresh_interval, "seconds).")

# Tab 2: Data & Signals
with tabs[1]:
    st.header("Historical Data & Skarre Signal")
    st.dataframe(df.tail(10))
    fig_data = go.Figure()
    fig_data.add_trace(go.Scatter(
        x=df.index,
        y=df[price_col],
        mode='lines',
        name='Price',
        line=dict(color='blue')
    ))
    fig_data.add_trace(go.Scatter(
        x=df.index,
        y=df['MA'],
        mode='lines',
        name='Moving Average',
        line=dict(color='orange', dash='dot')
    ))
    fig_data.update_layout(
        title="Price & Central Moving Average",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified',
        template="plotly_white"
    )
    st.plotly_chart(fig_data, use_container_width=True)
    st.write("The Skarre Signal is computed as the Z-score of the deviation from the moving average.")

# Tab 3: Backtest & Comparison
with tabs[2]:
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
        st.write("No trades executed with these parameters. (Check if your date range includes recent data such as 2024.)")
    
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_df.index,
        y=equity_df['Equity'],
        mode='lines',
        name='Signal Strategy',
        line=dict(color='magenta')
    ))
    fig_equity.add_trace(go.Scatter(
        x=buy_hold.index,
        y=buy_hold,
        mode='lines',
        name='Buy & Hold',
        line=dict(color='gray', dash='dot')
    ))
    fig_equity.update_layout(
        title="Equity Curve: Signal Strategy vs. Buy & Hold",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        hovermode='x unified',
        template="plotly_white",
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig_equity, use_container_width=True)
    
    st.subheader("Price Chart with Trade Signals")
    if trades and len(trades) > 0:
        plot_price_and_signals_plotly(df, trades)
    else:
        st.write("No trade signals to plot.")

# Tab 4: Performance Metrics
with tabs[3]:
    st.header("Performance Metrics")
    metrics = compute_performance_metrics(equity_df, initial_capital)
    for k, v in metrics.items():
        st.write(f"**{k}:** {v}")

# Tab 5: Polynomial Analysis
with tabs[4]:
    st.header("Polynomial Analysis: Parabolic Trends")
    coeff_df = polynomial_analysis(df, window=30)
    fig_quad = go.Figure()
    fig_quad.add_trace(go.Scatter(
        x=coeff_df.index,
        y=coeff_df['Quadratic'],
        mode='lines',
        name='Quadratic Coefficient',
        line=dict(color='blue')
    ))
    fig_quad.update_layout(
        title="Quadratic Coefficient Over Time",
        xaxis_title="Date",
        yaxis_title="Coefficient (a)",
        hovermode='x unified',
        template="plotly_white"
    )
    st.plotly_chart(fig_quad, use_container_width=True)
    
    fig_hist = make_subplots(rows=1, cols=3, subplot_titles=["Quadratic", "Linear", "Intercept"])
    fig_hist.add_trace(go.Histogram(
        x=coeff_df['Quadratic'], nbinsx=20, marker_color='skyblue', name="Quadratic"
    ), row=1, col=1)
    fig_hist.add_trace(go.Histogram(
        x=coeff_df['Linear'], nbinsx=20, marker_color='salmon', name="Linear"
    ), row=1, col=2)
    fig_hist.add_trace(go.Histogram(
        x=coeff_df['Intercept'], nbinsx=20, marker_color='lightgreen', name="Intercept"
    ), row=1, col=3)
    fig_hist.update_layout(
        title="Histogram of Polynomial Coefficients",
        template="plotly_white",
        hovermode='x unified'
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.subheader("Sample Quadratic Fit (Most Recent Window)")
    plot_polynomial_sample_plotly(df, window=30)

# Tab 6: About
with tabs[5]:
    st.header("About Skarre Tracking Signal")
    st.markdown("""
    **Mission & Vision**  
    I developed the Skarre Tracking Signal Dashboard as a passion project during my college studies. 
    This tool is designed to analyze market dynamics and highlight potential trading opportunities using a unique, proprietary signal. 
    My goal was to combine rigorous statistical analysis, robust backtesting, and innovative trend analysis into one comprehensive platform.
    
    **What This Dashboard Does**  
    - **Data & Signals:** It processes historical price data to calculate a central moving average and computes a signal (as a Z-score) that captures deviations from normal price behavior.  
    - **Backtesting:** It simulates trades based on this signal and compares the performance against a simple buy-and-hold strategy to help evaluate risk and reward.
    - **Polynomial Analysis:** By fitting rolling quadratic models to the price data, it captures parabolic trends that provide insight into market curvature and momentum.
    - **Live Graph:** The live graph uses interactive Plotly charts with range sliders, clear annotations, and dynamic tooltips so you can explore the most up-to-date price information.
    
    **My Mission**  
    I built this dashboard to push the boundaries of quantitative analysis as part of my college projects. 
    I want to empower fellow students and traders with a tool that is both sophisticated and easy to use—providing clear insights and supporting informed decision-making.
    
    **Explore & Experiment**  
    Feel free to adjust the parameters in the sidebar to see how the signal performs under various conditions. 
    Your feedback is crucial as I continue to refine and enhance this project.
    """)
    st.write("Thank you for exploring my Skarre Tracking Signal Dashboard!")
