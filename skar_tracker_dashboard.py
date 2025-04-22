import streamlit as st
import pandas as pd
import numpy as np
from skar_lib.polynomial_fit import get_slope, get_acceleration

st.set_page_config(page_title="Skarre Signal Dashboard", layout="wide")
st.title("📈 Skarre Signal Dashboard")

# --- Dummy SPY Price Series for Now ---
dates = pd.date_range(start="2022-01-01", periods=200)
price_spy = pd.Series(np.cumsum(np.random.randn(200)) + 400, index=dates)

# --- Compute Slope and Acceleration ---
slope = get_slope(price_spy)
accel = get_acceleration(price_spy)

# --- Display ---
st.subheader("SPY Price")
st.line_chart(price_spy)

st.subheader("Slope (1st Derivative)")
st.line_chart(slope)

st.subheader("Acceleration (2nd Derivative)")
st.line_chart(accel)
