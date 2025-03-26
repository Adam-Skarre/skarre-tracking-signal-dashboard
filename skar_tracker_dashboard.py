import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.title('📈 Universal Skar Tracking Signal Dashboard')

ticker = st.sidebar.text_input('Enter Stock Ticker', value='TSLA').upper()
MA_period = st.sidebar.number_input('Moving Average Period (days)', min_value=50, max_value=300, value=200)
vol_period = st.sidebar.number_input('Volatility Period (days)', min_value=5, max_value=60, value=21)
entry_threshold = st.sidebar.slider('Entry Threshold', min_value=0.0, max_value=3.0, value=0.5, step=0.1)

@st.cache_data(ttl=300)
def get_data(ticker_symbol):
    df = yf.download(ticker_symbol, period='1y', progress=False)
    df = df[['Close']]
    df.columns = ['Price']
    df.dropna(inplace=True)
    return df

data = get_data(ticker)

if data.empty:
    st.error(f"Ticker '{ticker}' is invalid or has no data. Please enter a valid ticker.")
    st.stop()

data['Return'] = data['Price'].pct_change()
data['Volatility'] = data['Return'].rolling(window=vol_period).std()
data['MA'] = data['Price'].rolling(window=MA_period).mean()
data['Skar Signal'] = (data['Price'] - data['MA']) / data['Volatility']

latest_signal = data['Skar Signal'].iloc[-1]
latest_price = data['Price'].iloc[-1]

st.metric(label=f"Current {ticker} Price", value=f"${latest_price:.2f}")
st.metric(label="Current Skar Signal", value=f"{latest_signal:.2f}", delta="✅ Buy" if latest_signal >= entry_threshold else "⏸️ Hold/Cash")

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines', name=f'{ticker} Price', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=data.index, y=data['MA'], mode='lines', name=f'{MA_period}-day MA', line=dict(color='orange')))

fig.add_hline(y=data['MA'].iloc[-1] + entry_threshold * data['Volatility'].iloc[-1], 
              line_dash="dash", line_color="green", 
              annotation_text="Entry Threshold", annotation_position="top right")

fig.update_layout(title=f'{ticker} Price & Skar Signal', xaxis_title='Date', yaxis_title='Price ($)', hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)

fig_signal = go.Figure()
fig_signal.add_trace(go.Scatter(x=data.index, y=data['Skar Signal'], mode='lines', name='Skar Signal', line=dict(color='purple')))
fig_signal.add_hline(y=entry_threshold, line_dash="dash", line_color="green", annotation_text="Entry Threshold", annotation_position="top right")

fig_signal.update_layout(title=f'Skar Signal Over Time for {ticker}', xaxis_title='Date', yaxis_title='Skar Signal (Z-Score)', hovermode='x unified')
st.plotly_chart(fig_signal, use_container_width=True)
