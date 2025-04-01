import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime

# -----------------------
# Data Download & Processing
# -----------------------

@st.cache_data(show_spinner=False)
def get_data(ticker, start, end):
    """
    Download historical data from Yahoo Finance.
    Uses group_by='ticker' to ensure standard OHLCV columns.
    Flattens multi-index columns, normalizes them to Title Case, and computes daily returns.
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
    
    # If 5 columns are all identical, reassign to OHLCV.
    if len(set(df.columns)) == 1 and df.shape[1] == 5:
        st.warning("All columns had the same name. Reassigning to Open, High, Low, Close, Volume.")
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
      (Price - Moving Average) / (Rolling Std of (Price - MA))
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

def backtest_strategy(df, strategy="Momentum", entry_threshold=2.0, exit_threshold=-0.5,
                      trailing_stop=0.08, profit_target=0.1, initial_capital=100000, transaction_cost=0.0):
    """
    Backtest the Skarre Signal strategy.
    - For Contrarian: Buy when signal <= entry_threshold; Sell when signal >= exit_threshold.
    - For Momentum:  Buy when signal >= entry_threshold; Sell when signal <= exit_threshold.
    Applies trailing stop, profit target, transaction costs, and forces any open position to close at the end.
    
    Returns:
      trades: List of trade dictionaries.
      equity_df: DataFrame of portfolio equity over time.
      buy_hold: Series representing Buy & Hold performance.
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
        
        current_equity = capital * (price / entry_price) if position else capital
        equity_curve.append((date, current_equity))
        
        if position == 0:
            if (strategy == "Contrarian" and signal <= entry_threshold) or \
               (strategy == "Momentum" and signal >= entry_threshold):
                # Buy signal
                shares = capital / price
                capital_after_cost = capital * (1 - transaction_cost)
                position = shares
                entry_price = price
                max_price = price
                trades.append({
                    "Entry Date": date,
                    "Entry Price": price,
                    "Exit Date": None,
                    "Exit Price": None,
                    "Return": None
                })
                capital = 0  # fully invested
        else:
            if price > max_price:
                max_price = price
            exit_trade = False
            if strategy == "Contrarian" and signal >= exit_threshold:
                exit_trade = True
            elif strategy == "Momentum" and signal <= exit_threshold:
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
                proceeds = position * exit_price
                proceeds_after_cost = proceeds * (1 - transaction_cost)
                capital = proceeds_after_cost
                position = 0
                entry_price = 0
                max_price = 0
    
    # Force close any open position at the end
    if position:
        final_date = df.iloc[-1]['Date']
        final_price = df.iloc[-1][price_col]
        trade_return = (final_price / entry_price) - 1
        trades[-1]["Exit Date"] = final_date
        trades[-1]["Exit Price"] = final_price
        trades[-1]["Return"] = trade_return
        capital = position * final_price
        equity_curve[-1] = (final_date, capital)
        position = 0
        entry_price = 0
        max_price = 0
    
    equity_df = pd.DataFrame(equity_curve, columns=["Date", "Equity"]).set_index("Date")
    # Reindex to cover the full date range
    full_index = pd.date_range(start=df['Date'].iloc[0], end=df['Date'].iloc[-1])
    equity_df.index = pd.to_datetime(equity_df.index)
    buy_hold.index = pd.to_datetime(buy_hold.index)
    equity_df = equity_df.reindex(full_index, method='ffill')
    buy_hold = buy_hold.reindex(full_index, method='ffill')
    
    return trades, equity_df, buy_hold

def compute_performance_metrics(equity_df, initial_capital):
    """
    Compute performance metrics: Total Return, CAGR, Sharpe Ratio, Sortino Ratio, and Maximum Drawdown.
    """
    final_equity = equity_df['Equity'].iloc[-1]
    total_return = final_equity / initial_capital - 1
    
    dates = equity_df.index
    duration_days = (dates[-1] - dates[0]).days
    years = duration_days / 365.25 if duration_days > 0 else 1
    CAGR = (final_equity / initial_capital)**(1/years) - 1
    
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
    If insufficient data, returns an empty DataFrame.
    """
    price_col = 'Close' if 'Close' in df.columns else ('Adj Close' if 'Adj Close' in df.columns else None)
    if price_col is None:
        st.error("No valid price column for polynomial analysis.")
        st.stop()
    prices = df[price_col]
    if len(prices) < window:
        st.error("Not enough data for polynomial analysis.")
        return pd.DataFrame(columns=['Quadratic', 'Linear', 'Intercept'])
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

def plot_signals_backtest(df, trades):
    """
    Plot a price chart with trade signals using Plotly.
    """
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    for t in trades:
        buy_x.append(t["Entry Date"])
        buy_y.append(t["Entry Price"])
        if t["Exit Date"]:
            sell_x.append(t["Exit Date"])
            sell_y.append(t["Exit Price"])
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
    return fig

# -----------------------
# Main App Layout
# -----------------------

st.set_page_config(page_title="Skarre Tracker Quantitative Portfolio Dashboard", layout="wide")
st.title("Skarre Tracker Quantitative Portfolio Dashboard")

# Sidebar Inputs
st.sidebar.header("Inputs & Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY", help="Enter a valid ticker (e.g., SPY).")
benchmark_ticker = st.sidebar.text_input("Benchmark (optional)", value="^GSPC", help="Optional benchmark ticker (e.g., ^GSPC).")
start_date = st.sidebar.date_input("Start Date", value=datetime(2010, 1, 1), help="Start date for historical data.")
end_date = st.sidebar.date_input("End Date", value=datetime.today(), help="End date for historical data.")
ma_window = st.sidebar.number_input("Moving Average Window (days)", 10, 300, 150, help="Window for the moving average calculation.")
vol_window = st.sidebar.number_input("Volatility Window (days)", 5, 60, 14, help="Window for the volatility calculation.")
# Default strategy set to Momentum
strategy = st.sidebar.selectbox("Strategy Type", ["Momentum", "Contrarian"], help="Momentum: buy trending; Contrarian: buy oversold.")
if strategy == "Contrarian":
    entry_threshold = st.sidebar.number_input("Entry Threshold (Z-score)", value=-2.0, step=0.1, help="Buy when signal is <= this value.")
    exit_threshold = st.sidebar.number_input("Exit Threshold (Z-score)", value=0.5, step=0.1, help="Sell when signal is >= this value.")
else:
    entry_threshold = st.sidebar.number_input("Entry Threshold (Z-score)", value=2.0, step=0.1, help="Buy when signal is >= this value.")
    exit_threshold = st.sidebar.number_input("Exit Threshold (Z-score)", value=-0.5, step=0.1, help="Sell when signal is <= this value.")
trailing_stop = st.sidebar.number_input("Trailing Stop Loss (%)", 1.0, 50.0, 8.0, 0.5, help="Stop if price drops this percentage from its peak.")
profit_target = st.sidebar.number_input("Profit Target (%)", 1.0, 100.0, 10.0, 0.5, help="Take profit if price rises this percentage from entry.")
initial_capital = st.sidebar.number_input("Initial Capital ($)", 1000, 10000000, 100000, 1000, help="Starting portfolio value.")
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (annual %)", 0.0, 10.0, 2.0, 0.1, help="Risk-free rate for Sharpe calculation.") / 100.0
transaction_cost = st.sidebar.number_input("Transaction Cost (decimal)", 0.0, 0.01, 0.0, 0.001, help="E.g., 0.001 = 0.1% per trade.")
refresh_interval = st.sidebar.number_input("Live Graph Refresh (sec)", 5, 120, 30, 5, help="Graph refresh interval (page reload).")

with st.spinner("Downloading data..."):
    df_raw = get_data(ticker, start_date, end_date)
    df = compute_skarre_signal(df_raw, ma_window, vol_window)
    df.index = pd.to_datetime(df.index)

price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

# -----------------------
# Tabs
# -----------------------

tabs = st.tabs(["Live Graph", "Data & Signals", "Backtest & Comparison", "Performance Metrics", "Polynomial Analysis", "About"])

# Tab 1: Live Graph
with tabs[0]:
    st.header("Live Graph")
    st.subheader("Candlestick Chart with Volume")
    fig_candle = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               row_heights=[0.8, 0.2], vertical_spacing=0.02)
    fig_candle.update_layout(template="plotly_white", hovermode='x unified')
    fig_candle.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candlestick",
        increasing_line_color='green',
        decreasing_line_color='red'
    ), row=1, col=1)
    if 'Volume' in df.columns:
        fig_candle.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name="Volume",
            marker_color='rgba(60,60,150,0.5)'
        ), row=2, col=1)
    fig_candle.update_yaxes(title_text="Price", row=1, col=1)
    fig_candle.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig_candle, use_container_width=True)
    
    st.subheader("Skarre Signal Overlay")
    fig_skarre = go.Figure()
    fig_skarre.add_trace(go.Scatter(
        x=df.index,
        y=df[price_col],
        mode='lines',
        name='Price',
        line=dict(color='blue')
    ))
    fig_skarre.add_trace(go.Scatter(
        x=df.index,
        y=df['MA'],
        mode='lines',
        name=f'{ma_window}-day MA',
        line=dict(color='orange', dash='dot')
    ))
    dynamic_entry = df['MA'].iloc[-1] + entry_threshold * df['Vol'].iloc[-1]
    fig_skarre.add_hline(
        y=dynamic_entry,
        line_dash="dash",
        line_color="green",
        annotation_text="Entry Threshold",
        annotation_position="top right"
    )
    fig_skarre.update_layout(
        title="Live Price & Signal Overlay",
        xaxis=dict(title="Date", rangeslider=dict(visible=True)),
        yaxis_title="Price",
        hovermode='x unified',
        template="plotly_white"
    )
    st.plotly_chart(fig_skarre, use_container_width=True)
    
    latest_price = df[price_col].iloc[-1]
    latest_signal = df['Skarre_Signal'].iloc[-1]
    st.metric(label=f"Current {ticker} Price", value=f"${latest_price:.2f}")
    st.metric(label="Current Skarre Signal", value=f"{latest_signal:.2f}")
    st.write("Note: Data may be delayed ~15 minutes. The live graph refreshes on page reload (~", refresh_interval, "sec).")

# Tab 2: Data & Signals
with tabs[1]:
    st.header("Historical Data & Signals")
    st.write("Latest 15 rows:")
    st.dataframe(df.tail(15))
    st.write("Dataset shape:", df.shape)
    st.write("Columns:", list(df.columns))

# Tab 3: Backtest & Comparison
with tabs[2]:
    st.header("Backtest & Buy-Hold Comparison")
    trades, equity_df, buy_hold = backtest_strategy(
        df, strategy=strategy,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        trailing_stop=trailing_stop,
        profit_target=profit_target,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost
    )
    st.subheader("Trade Log")
    if trades:
        st.dataframe(pd.DataFrame(trades))
    else:
        st.write("No trades executed. Consider adjusting parameters or extending the date range (ensure 2024 data is included).")
    
    # Reindex equity_df and buy_hold for full date range
    full_index = pd.date_range(start=df.index[0], end=df.index[-1])
    equity_df.index = pd.to_datetime(equity_df.index)
    buy_hold.index = pd.to_datetime(buy_hold.index)
    equity_df = equity_df.reindex(full_index, method='ffill')
    buy_hold = buy_hold.reindex(full_index, method='ffill')
    
    st.subheader("Equity Curve Comparison")
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=equity_df.index,
        y=equity_df['Equity'],
        mode='lines',
        name='Signal Strategy',
        line=dict(color='magenta')
    ))
    fig_eq.add_trace(go.Scatter(
        x=buy_hold.index,
        y=buy_hold,
        mode='lines',
        name='Buy & Hold',
        line=dict(color='gray', dash='dot')
    ))
    fig_eq.update_layout(
        title="Equity Curve: Signal Strategy vs. Buy & Hold",
        xaxis=dict(title="Date", rangeslider=dict(visible=True)),
        yaxis_title="Portfolio Value",
        hovermode='x unified',
        template="plotly_white"
    )
    st.plotly_chart(fig_eq, use_container_width=True)
    
    st.subheader("Price Chart with Trade Signals")
    if trades:
        fig_signals = plot_signals_backtest(df, trades)
        st.plotly_chart(fig_signals, use_container_width=True)
    else:
        st.write("No trade signals to display.")

# Tab 4: Performance Metrics
with tabs[3]:
    st.header("Performance Metrics")
    metrics = compute_performance_metrics(equity_df, initial_capital)
    help_texts = {
        "Total Return (%)": "Overall percentage gain/loss over the period.",
        "CAGR (%)": "Annualized growth rate of the portfolio.",
        "Sharpe Ratio": "Risk-adjusted return (higher is better, typically >1).",
        "Sortino Ratio": "Risk-adjusted return focused on downside risk.",
        "Max Drawdown (%)": "Largest peak-to-trough decline."
    }
    for k, v in metrics.items():
        st.write(f"**{k}:** {v}  \n*{help_texts.get(k, '')}*")

# Tab 5: Polynomial Analysis
with tabs[4]:
    st.header("Polynomial Analysis: Parabolic Trends")
    st.write("Rolling quadratic fits (degree=2) capture parabolic behavior in price data.")
    coeff_df = polynomial_analysis(df, window=30)
    st.subheader("Quadratic Coefficient Over Time")
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
    
    st.subheader("Histogram of Polynomial Coefficients")
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
    
    st.subheader("Sample Quadratic Fit (Most Recent 30-day Window)")
    plot_polynomial_sample_plotly(df, window=30)

# Tab 6: About
with tabs[5]:
    st.header("About Skarre Tracking Signal")
    st.markdown("""
    **My Inspiration**  
    I developed the Skarre Tracking Signal Dashboard during my college years as a personal project fueled by my passion for quantitative finance and engineering. This platform is a practical application of financial and quantitative engineering, integrating rigorous data analysis, algorithmic strategy design, and interactive visualization.

    **What This Dashboard Does**  
    - **Data & Signals:** Downloads historical price data from Yahoo Finance, calculates a central moving average, and computes a proprietary Z-score signal to flag significant market conditions.  
    - **Backtesting:** Simulates trading strategies (Momentum or Contrarian) with trailing stops, profit targets, and transaction cost adjustments, comparing the results to a Buy & Hold benchmark.  
    - **Polynomial Analysis:** Uses rolling quadratic regression to uncover parabolic trends, offering insights into market curvature and potential turning points.  
    - **Live Graph:** Presents real-time market data through interactive charts, including candlestick and volume displays.

    **My Mission**  
    My mission is to build a robust, professional-grade quantitative analysis platform that empowers traders and financial engineers to explore and understand market dynamics. I strive to create tools that are both rigorous in their analysis and practical in their application.

    **Future Enhancements**  
    Planned improvements include incorporating advanced risk metrics (such as the Calmar and Omega Ratios), real-time alerts, multi-asset analysis, and enhanced export options to further refine this platform.

    **Disclaimer**  
    This dashboard is a rigorously engineered research tool based on historical data and quantitative analysis. While no strategy can guarantee future performance, the methodologies implemented are grounded in robust engineering principles and provide valuable insights into market behavior.

    **Thank You**  
    Thank you for exploring the Skarre Tracking Signal Dashboard. Your feedback is essential as I continue to refine and enhance this project.
    """)
    st.write("Thank you for exploring my dashboard!")
