import streamlit as st
import pandas as pd
import yfinance as yf
import altair as alt

# --- Title ---
st.title("📈 Nifty Market Breadth Dashboard")

# --- Load live Nifty data ---
def load_nifty():
    ticker = yf.Ticker("^NSEI")  # Nifty 50 index
    hist = ticker.history(period="6mo")
    hist = hist.reset_index()
    return hist

nifty_df = load_nifty()

# --- Display latest values ---
latest = nifty_df.iloc[-1]
st.metric(label="Nifty Last Price", value=f"{latest['Close']:.2f}", delta=f"{latest['Close']-latest['Open']:.2f}")

# --- Market Breadth Placeholder Computation ---
# In a real system, you would compute Adv/Decl from NSE API or your data source.
# Here we simulate breadth values for demonstration.

def compute_breadth(df):
    df = df.copy()
    df['Advances'] = (df['Close'] > df['Open']).astype(int) * 800  # placeholder
    df['Declines'] = (df['Close'] < df['Open']).astype(int) * 700
    df['% Above 10 DMA'] = (df['Close'] > df['Close'].rolling(10).mean()).astype(int) * 50
    df['% Above 20 DMA'] = (df['Close'] > df['Close'].rolling(20).mean()).astype(int) * 50
    df['% Above 40 DMA'] = (df['Close'] > df['Close'].rolling(40).mean()).astype(int) * 50
    return df

breadth_df = compute_breadth(nifty_df)

# --- Show table similar to screenshot ---
st.subheader("📊 Market Breadth Table (Simplified)")
show_cols = [
    'Date', 'Close', 'Advances', 'Declines', '% Above 10 DMA', '% Above 20 DMA', '% Above 40 DMA'
]
st.dataframe(breadth_df[show_cols].tail(60), use_container_width=True)

# --- Chart: Nifty Price ---
st.subheader("📈 Nifty Price Chart")
chart = (
    alt.Chart(nifty_df)
    .mark_line()
    .encode(
        x='Date:T',
        y='Close:Q'
    )
    .properties(height=300)
)
st.altair_chart(chart, use_container_width=True)

# --- Chart: Breadth Indicator (% Above 20 DMA) ---
st.subheader("📉 Breadth Trend: % Above 20 DMA")
breadth_chart = (
    alt.Chart(breadth_df)
    .mark_area(opacity=0.4)
    .encode(
        x='Date:T',
        y='% Above 20 DMA:Q'
    )
    .properties(height=300)
)
st.altair_chart(breadth_chart, use_container_width=True)

st.info("This is a demo version. For full replication of your spreadsheet (Adv/Decl, % Above DMA buckets, Up/Down by %, etc.), plug in your actual NSE breadth dataset.")
