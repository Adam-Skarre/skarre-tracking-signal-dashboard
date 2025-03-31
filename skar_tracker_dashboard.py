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
    Download historical data from Yahoo Finance with group_by='ticker' to ensure standard columns.
    Flatten multi-index columns if needed, normalize them to Title Case, and compute daily returns.
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

    # If 5 columns all the same name, rename them to standard OHLCV
    if len(set(df.columns)) == 1 and df.shape[1] == 5:
        st.warning("All columns had the same name. Reassigning to Open, High, Low, Close, Volume.")
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
    
    df.dropna(inplace=True)
    
    # Add a 'Return' column
    if 'Adj Close' in df.columns:
        df['Return'] = df['Adj Close'].pct_change()
    elif 'Close' in df.columns:
        df['Return'] = df['Close'].pct_change()
    else:
        st.error("Downloaded data lacks 'Adj Close' or 'Close'.")
        st.stop()
    
    return df

def compute_skarre_signal(df, ma_window=150, vol_window=14):
    """
    Compute the Skarre Signal (Z-score):
       (Price - MA) / Rolling Std(Price - MA).
    """
    df = df.copy()
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    
    df['MA'] = df[price_col].rolling(ma_window, min_periods=1).mean()
    df['Deviation'] = df[price_col] - df['MA']
    df['Vol'] = df['Deviation'].rolling(vol_window, min_periods=1).std()
    df['Skarre_Signal'] = df.apply(
        lambda row: row['Deviation']/row['Vol'] if row['Vol'] != 0 else 0, axis=1
    )
    return df

# -----------------------
# Backtesting & Performance
# -----------------------

def backtest_strategy(
    df, 
    strategy="Contrarian", 
    entry_threshold=-2.0, 
    exit_threshold=0.5,
    trailing_stop=0.08, 
    profit_target=0.1, 
    initial_capital=100000,
    transaction_cost=0.0
):
    """
    Backtest the Skarre Signal strategy:
    - Contrarian: Buy if signal <= entry_threshold, sell if signal >= exit_threshold.
    - Momentum:  Buy if signal >= entry_threshold, sell if signal <= exit_threshold.
    Includes trailing stop, profit target, and optional transaction_cost in decimal (e.g. 0.001 for 0.1%).
    
    Returns:
      trades: List of trade dictionaries.
      equity_df: DataFrame of portfolio equity over time.
      buy_hold: Series for buy-and-hold performance.
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
        
        # Mark-to-market
        current_equity = capital * (price / entry_price) if position == 1 else capital
        equity_curve.append((date, current_equity))
        
        # Entry Logic
        if position == 0:
            if (strategy == "Contrarian" and signal <= entry_threshold) or \
               (strategy == "Momentum"  and signal >= abs(entry_threshold)):
                # Buy
                shares = capital / price
                # Subtract transaction cost (percentage of capital)
                capital_after_cost = capital * (1 - transaction_cost)
                # Adjust position
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
                # Capital is now in the stock (0 cash)
                capital = 0 if position > 0 else capital
        else:
            # If in position, track max price for trailing stop
            if price > max_price:
                max_price = price
            exit_trade = False

            # Contrarian exit
            if strategy == "Contrarian" and signal >= exit_threshold:
                exit_trade = True
            # Momentum exit
            if strategy == "Momentum" and signal <= -abs(exit_threshold):
                exit_trade = True
            # Trailing stop
            if price < max_price * (1 - trailing_stop):
                exit_trade = True
            # Profit target
            if price >= entry_price * (1 + profit_target):
                exit_trade = True

            if exit_trade:
                # Sell
                exit_price = price
                trade_return = (exit_price / entry_price) - 1
                trades[-1]["Exit Date"] = date
                trades[-1]["Exit Price"] = exit_price
                trades[-1]["Return"] = trade_return
                # Reconvert shares to capital
                proceeds = position * exit_price
                # Subtract transaction cost again
                proceeds_after_cost = proceeds * (1 - transaction_cost)
                capital = proceeds_after_cost
                position = 0
                entry_price = 0
                max_price = 0

    equity_df = pd.DataFrame(equity_curve, columns=["Date", "Equity"]).set_index("Date")
    return trades, equity_df, buy_hold

def compute_performance_metrics(equity_df, initial_capital):
    """
    Compute essential performance metrics:
      - Total Return, CAGR, Sharpe, Sortino, Max Drawdown.
    """
    final_equity = equity_df['Equity'].iloc[-1]
    total_return = final_equity / initial_capital - 1
    
    dates = equity_df.index
    duration_days = (dates[-1] - dates[0]).days
    years = duration_days / 365.25 if duration_days > 0 else 1
    CAGR = (final_equity / initial_capital)**(1/years) - 1 if years > 0 else np.nan
    
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
        "Total Return (%)": round(total_return*100, 2),
        "CAGR (%)": round(CAGR*100, 2) if not np.isnan(CAGR) else None,
        "Sharpe Ratio": round(sharpe, 2) if not np.isnan(sharpe) else None,
        "Sortino Ratio": round(sortino, 2) if not np.isnan(sortino) else None,
        "Max Drawdown (%)": round(max_drawdown*100, 2) if not np.isnan(max_drawdown) else None
    }

# -----------------------
# Plotly Graphs
# -----------------------

def plot_candlestick_volume(df):
    """
    Candlestick chart + volume sub-plot for the 'Live Graph' tab.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.8, 0.2],
                        vertical_spacing=0.02)
    fig.update_layout(template="plotly_white", hovermode='x unified')
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Candlestick",
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # Volume
    if 'Volume' in df.columns:
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name="Volume",
                marker_color='rgba(60,60,150,0.5)'
            ),
            row=2, col=1
        )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(title_text="Date", rangeslider_visible=False, row=2, col=1)
    fig.update_layout(title="Candlestick + Volume", showlegend=False)
    return fig

def plot_skarre_signal_live(df, ma_window, entry_threshold):
    """
    Overlays the moving average + dynamic threshold line for the live graph tab.
    """
    # Using the last row for threshold
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    entry_line = df['MA'].iloc[-1] + entry_threshold * df['Vol'].iloc[-1]
    
    fig_line = go.Figure()
    fig_line.update_layout(template="plotly_white", hovermode='x unified')
    
    fig_line.add_trace(go.Scatter(
        x=df.index,
        y=df[price_col],
        mode='lines',
        name='Price',
        line=dict(color='blue')
    ))
    fig_line.add_trace(go.Scatter(
        x=df.index,
        y=df['MA'],
        mode='lines',
        name=f'{ma_window}-day MA',
        line=dict(color='orange', dash='dot')
    ))
    # Entry threshold line (horizontal) at the last computed value
    fig_line.add_hline(
        y=entry_line,
        line_dash="dash",
        line_color="green",
        annotation_text="Entry Threshold",
        annotation_position="top right"
    )
    fig_line.update_layout(
        title="Live Skarre Signal Overlay",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    return fig_line

def plot_signals_backtest(df, trades):
    """
    Price + signals for backtest results.
    """
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    
    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []
    
    for t in trades:
        buy_dates.append(t["Entry Date"])
        buy_prices.append(t["Entry Price"])
        if t["Exit Date"]:
            sell_dates.append(t["Exit Date"])
            sell_prices.append(t["Exit Price"])
    
    fig = go.Figure()
    fig.update_layout(template="plotly_white", hovermode='x unified')
    
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
        x=buy_dates,
        y=buy_prices,
        mode='markers',
        name='Buy',
        marker=dict(symbol='triangle-up', color='green', size=12)
    ))
    fig.add_trace(go.Scatter(
        x=sell_dates,
        y=sell_prices,
        mode='markers',
        name='Sell',
        marker=dict(symbol='triangle-down', color='red', size=12)
    ))
    fig.update_layout(
        title="Price Chart with Trade Signals",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    return fig

# -----------------------
# Main App Layout
# -----------------------

st.set_page_config(page_title="Skarre Tracker Quantitative Portfolio Dashboard", layout="wide")
st.title("Skarre Tracker Quantitative Portfolio Dashboard")

# Sidebar
st.sidebar.header("Inputs & Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY", help="Enter a valid Yahoo Finance ticker symbol.")
benchmark_ticker = st.sidebar.text_input("Benchmark (optional)", value="^GSPC", help="Compare performance to a benchmark (e.g., ^GSPC). Leave blank to skip.")
start_date = st.sidebar.date_input("Start Date", value=datetime(2010,1,1), help="Initial date for historical data.")
end_date = st.sidebar.date_input("End Date", value=datetime.today(), help="Final date for historical data.")
ma_window = st.sidebar.number_input("Moving Average Window (days)", 10, 300, 150, help="Smoothing period for the Skarre Signal.")
vol_window = st.sidebar.number_input("Volatility Window (days)", 5, 60, 14, help="Rolling std window for the signal's denominator.")
strategy = st.sidebar.selectbox("Strategy Type", ["Contrarian", "Momentum"], help="Contrarian: buy oversold, sell overbought. Momentum: buy uptrend, sell downtrend.")
if strategy == "Contrarian":
    entry_threshold = st.sidebar.number_input("Entry Threshold (Z-score)", value=-2.0, step=0.1, help="Buy when signal <= this.")
    exit_threshold = st.sidebar.number_input("Exit Threshold (Z-score)", value=0.5, step=0.1, help="Sell when signal >= this.")
else:
    entry_threshold = st.sidebar.number_input("Entry Threshold (Z-score)", value=2.0, step=0.1, help="Buy when signal >= this.")
    exit_threshold = st.sidebar.number_input("Exit Threshold (Z-score)", value=-0.5, step=0.1, help="Sell when signal <= this.")
trailing_stop = st.sidebar.number_input("Trailing Stop Loss (%)", 1.0, 50.0, 8.0, 0.5, help="Stop triggered if price drops from peak by this percent.")
profit_target = st.sidebar.number_input("Profit Target (%)", 1.0, 100.0, 10.0, 0.5, help="Take profit if price rises by this percent from entry.")
initial_capital = st.sidebar.number_input("Initial Capital ($)", 1000, 10000000, 100000, 1000, help="Starting portfolio size.")
transaction_cost = st.sidebar.number_input("Transaction Cost (decimal)", 0.0, 0.01, 0.0, 0.001, help="E.g. 0.001 = 0.1% cost per trade.")
refresh_interval = st.sidebar.number_input("Live Graph Refresh (sec)", 5, 120, 30, 5, help="Approximate reload interval for the Live Graph tab.")

# Data
with st.spinner("Downloading main ticker data..."):
    df_raw = get_data(ticker, start_date, end_date)
    df = compute_skarre_signal(df_raw, ma_window, vol_window)
    df.index = pd.to_datetime(df.index)

# Optional benchmark
if benchmark_ticker.strip():
    with st.spinner("Downloading benchmark data..."):
        bench_df_raw = get_data(benchmark_ticker, start_date, end_date)
        # We'll just store it for later comparison if needed
else:
    bench_df_raw = None

tabs = st.tabs(["Live Graph", "Data & Signals", "Backtest & Comparison", "Performance Metrics", "Polynomial Analysis", "About"])

# -----------------------
# Tab 1: Live Graph
# -----------------------
with tabs[0]:
    st.header("Live Graph: Candlestick & Volume")
    # 1) Candlestick + Volume
    fig_candle = plot_candlestick_volume(df)
    st.plotly_chart(fig_candle, use_container_width=True)
    
    # 2) Skarre Signal Overlay
    st.subheader("Skarre Signal Overlay")
    fig_skarre_line = plot_skarre_signal_live(df, ma_window, entry_threshold)
    st.plotly_chart(fig_skarre_line, use_container_width=True)
    
    # Live metrics
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    latest_price = df[price_col].iloc[-1]
    latest_signal = df['Skarre_Signal'].iloc[-1]
    st.metric(label=f"Current {ticker} Price", value=f"${latest_price:.2f}")
    st.metric(label="Current Skarre Signal", value=f"{latest_signal:.2f}")
    st.write("Note: Data is from Yahoo Finance and may be delayed by ~15 minutes. This graph refreshes on page reload (~",
             refresh_interval, "seconds).")

# -----------------------
# Tab 2: Data & Signals
# -----------------------
with tabs[1]:
    st.header("Historical Data & Skarre Signal")
    st.write("Latest 15 rows for quick reference:")
    st.dataframe(df.tail(15))
    st.write("Full dataset shape:", df.shape)
    st.write("Columns:", list(df.columns))

# -----------------------
# Tab 3: Backtest & Comparison
# -----------------------
with tabs[2]:
    st.header("Strategy Backtest & Buy-Hold Comparison")
    trades, equity_df, buy_hold = backtest_strategy(
        df,
        strategy=strategy,
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
        st.write("No trades were executed with these parameters. Try adjusting thresholds or date range.")
    
    st.subheader("Equity Curve Comparison")
    fig_eq = go.Figure()
    fig_eq.update_layout(template="plotly_white", hovermode='x unified')
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
        yaxis_title="Portfolio Value"
    )
    st.plotly_chart(fig_eq, use_container_width=True)
    
    st.subheader("Price Chart with Trade Signals")
    if trades:
        fig_signals = plot_signals_backtest(df, trades)
        st.plotly_chart(fig_signals, use_container_width=True)
    else:
        st.write("No trade signals to display.")

# -----------------------
# Tab 4: Performance Metrics
# -----------------------
with tabs[3]:
    st.header("Performance Metrics")
    metrics = compute_performance_metrics(equity_df, initial_capital)
    # Tooltips / help text for each metric
    help_texts = {
        "Total Return (%)": "Overall percentage gain/loss from start to end.",
        "CAGR (%)": "Compound Annual Growth Rate – annualized rate of return.",
        "Sharpe Ratio": "Risk-adjusted return (excess return / volatility). >1 is often considered good.",
        "Sortino Ratio": "Focuses on downside volatility only (better for skewed distributions).",
        "Max Drawdown (%)": "Worst peak-to-trough drop during the period."
    }
    for k, v in metrics.items():
        tip = help_texts.get(k, "")
        st.write(f"**{k}:** {v}  \n{tip}")

# -----------------------
# Tab 5: Polynomial Analysis
# -----------------------
with tabs[4]:
    st.header("Polynomial Analysis: Parabolic Trends")
    st.write("Fitting a rolling quadratic polynomial (degree=2) to capture parabolic behavior in the price data.")
    
    coeff_df = polynomial_analysis(df, window=30)
    st.subheader("Quadratic Coefficient Over Time")
    fig_quad = go.Figure()
    fig_quad.update_layout(template="plotly_white", hovermode='x unified')
    fig_quad.add_trace(go.Scatter(
        x=coeff_df.index,
        y=coeff_df['Quadratic'],
        mode='lines',
        name='Quadratic (a)',
        line=dict(color='blue')
    ))
    fig_quad.update_layout(
        title="Quadratic Coefficient (a) Over Time",
        xaxis_title="Date",
        yaxis_title="Coefficient (a)"
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
        title="Distribution of Polynomial Coefficients",
        template="plotly_white"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.subheader("Most Recent Quadratic Fit (30-day window)")
    plot_polynomial_sample_plotly(df, window=30)

# -----------------------
# Tab 6: About
# -----------------------
with tabs[5]:
    st.header("About Skarre Tracking Signal")
    st.markdown("""
    **My Inspiration**  
    The Skarre Tracking Signal Dashboard was born out of my passion for quantitative finance and 
    the desire to create a cohesive platform for exploring market data, backtesting strategies, 
    and analyzing advanced trends like parabolic (quadratic) fits.

    **How It Works**  
    - **Data & Signals**: We fetch historical price data from Yahoo Finance (which may be delayed ~15 minutes). 
      A rolling average and volatility measure yield the Skarre Signal, effectively a Z-score that helps 
      detect overbought/oversold conditions.
    - **Backtesting**: Using the selected parameters (Contrarian or Momentum approach, trailing stop, 
      profit target, transaction cost, etc.), the app simulates trades over the historical period 
      and compares performance to a simple Buy & Hold approach.
    - **Polynomial Analysis**: We fit quadratic polynomials on rolling windows of price data to uncover 
      parabolic trends, which can sometimes indicate accelerating momentum or potential turning points.
    
    **Future Plans**  
    - Additional metrics like the Calmar Ratio and custom alerts (email/SMS) for threshold crossings.
    - Multi-asset portfolio backtesting and correlation analysis.
    - Transaction cost refinements and advanced order execution logic.

    **Disclaimer**  
    This dashboard is for educational and demonstration purposes. The Skarre Signal strategy is not 
    guaranteed to yield profits. Always conduct your own due diligence and consider consulting a 
    professional advisor before making real trades.

    **Thank You**  
    I appreciate your interest in the Skarre Tracking Signal Dashboard. Your feedback is invaluable 
    as I continue improving this project and exploring new frontiers in quant research.
    """)
    st.write("Enjoy exploring, and happy trading!")
