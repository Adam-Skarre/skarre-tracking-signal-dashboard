import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.title('🚀 Comprehensive Skar Tracking Signal Dashboard with Live Data & Backtest')

# Sidebar Inputs
ticker = st.sidebar.text_input('Enter Stock Ticker', value='TSLA').upper()
MA_period = st.sidebar.number_input('Moving Average Period (days)', min_value=50, max_value=300, value=150)
vol_period = st.sidebar.number_input('Volatility Period (days)', min_value=5, max_value=60, value=14)
entry_threshold = st.sidebar.slider('Entry Threshold', min_value=0.0, max_value=3.0, value=0.7, step=0.1)
stop_loss_pct = st.sidebar.slider('Trailing Stop Loss (%)', min_value=1, max_value=20, value=8)
profit_take_pct = st.sidebar.slider('Profit Target (%)', min_value=1, max_value=50, value=15)

@st.cache_data(ttl=300)
def get_data(ticker_symbol):
    df = yf.download(ticker_symbol, start='2010-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'), progress=False)
    df = df[['Close']]
    df.columns = ['Price']
    df.dropna(inplace=True)
    return df

data = get_data(ticker)

if data.empty:
    st.error(f"Ticker '{ticker}' is invalid or has no data. Please enter a valid ticker.")
    st.stop()

data['Return'] = data['Price'].pct_change()
data['Volatility'] = data['Price'].rolling(window=vol_period).std()
data['MA'] = data['Price'].rolling(window=MA_period).mean()
data['Skar Signal'] = (data['Price'] - data['MA']) / data['Volatility']

# Live metrics
latest_signal = data['Skar Signal'].iloc[-1]
latest_price = data['Price'].iloc[-1]

signal_status = "✅ Buy" if latest_signal >= entry_threshold else "⏸️ Hold/Cash"
st.metric(label=f"Current {ticker} Price", value=f"${latest_price:.2f}")
st.metric(label="Current Skar Signal", value=f"{latest_signal:.2f}", delta=signal_status)

# Enhanced Backtesting logic
initial_capital = 100000
position = 0
entry_price = 0
cash = initial_capital
portfolio_values = []
peak_price = 0

for i in range(len(data)):
    signal = data['Skar Signal'].iloc[i]
    price = data['Price'].iloc[i]

    if position == 0 and signal >= entry_threshold:
        position = cash / price
        entry_price = price
        cash = 0
        peak_price = price

    if position > 0:
        peak_price = max(peak_price, price)
        if (signal < entry_threshold or 
            price <= peak_price * (1 - stop_loss_pct / 100) or 
            price >= entry_price * (1 + profit_take_pct / 100)):
            cash = position * price
            position = 0

    total_portfolio_value = cash + (position * price)
    portfolio_values.append(total_portfolio_value)

data['Portfolio'] = portfolio_values

# Performance metrics
portfolio_series = pd.Series(portfolio_values, index=data.index)
total_return = (portfolio_series.iloc[-1] / initial_capital - 1)
years = (portfolio_series.index[-1] - portfolio_series.index[0]).days / 365.25
CAGR = (portfolio_series.iloc[-1] / initial_capital) ** (1 / years) - 1
annual_vol = portfolio_series.pct_change().std() * np.sqrt(252)
sharpe_ratio = (portfolio_series.pct_change().mean() * 252) / annual_vol
max_drawdown = ((portfolio_series / portfolio_series.cummax()) - 1).min()

# Display metrics
st.subheader('Enhanced Backtest Performance Metrics')
st.write(f"**Total Return:** {total_return:.2%}")
st.write(f"**CAGR:** {CAGR:.2%}")
st.write(f"**Annualized Volatility:** {annual_vol:.2%}")
st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
st.write(f"**Max Drawdown:** {max_drawdown:.2%}")

# Plot portfolio value
fig_portfolio = go.Figure()
fig_portfolio.add_trace(go.Scatter(x=data.index, y=data['Portfolio'], mode='lines', name='Portfolio Value', line=dict(color='green')))
fig_portfolio.update_layout(title='Optimized Portfolio Value Over Time', xaxis_title='Date', yaxis_title='Value ($)', hovermode='x unified')
st.plotly_chart(fig_portfolio, use_container_width=True)

# Price & Signal Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines', name=f'{ticker} Price', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=data.index, y=data['MA'], mode='lines', name=f'{MA_period}-day MA', line=dict(color='orange')))
entry_line = data['MA'].iloc[-1] + entry_threshold * data['Volatility'].iloc[-1]
fig.add_hline(y=entry_line, line_dash="dash", line_color="green", annotation_text="Entry Threshold", annotation_position="top right")
fig.update_layout(title=f'{ticker} Price & Skar Signal', xaxis_title='Date', yaxis_title='Price ($)', hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)

# Skar Signal Plot
fig_signal = go.Figure()
fig_signal.add_trace(go.Scatter(x=data.index, y=data['Skar Signal'], mode='lines', name='Skar Signal', line=dict(color='purple')))
fig_signal.add_hline(y=entry_threshold, line_dash="dash", line_color="green", annotation_text="Entry Threshold", annotation_position="top right")
fig_signal.update_layout(title=f'Skar Signal Over Time for {ticker}', xaxis_title='Date', yaxis_title='Skar Signal (Z-Score)', hovermode='x unified')
st.plotly_chart(fig_signal, use_container_width=True)
