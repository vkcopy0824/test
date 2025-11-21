import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import altair as alt
from datetime import datetime, timedelta
from typing import List

st.set_page_config(layout="wide", page_title="NIFTY50 Market Breadth (20d)")

# --------------- Config ----------------
# Public proxy used only for adv/dec fetch (can be overridden in Secrets or UI)
DEFAULT_PROXY_BASE = "https://nse-api-proxy.vercel.app"

# Local image path (user-uploaded) — developer provided this path and we use it as-is.
SAMPLE_IMAGE_PATH = "/mnt/data/IMG_1890.JPG"

# History window in days (as requested)
HIST_DAYS = 20
YF_CHUNK = 25

# --------------- Static NIFTY50 list (plain symbols) ---------------
# These are plain NSE symbols — the code will append '.NS' for yfinance lookups.
NIFTY50_SYMBOLS = [
    "RELIANCE","TCS","HDFCBANK","HDFC","ICICIBANK","INFY","ITC","KOTAKBANK","LT","SBIN",
    "BHARTIARTL","AXISBANK","HINDUNILVR","BAJAJFINANCE","MARUTI","ASIANPAINT","NTPC","ONGC","POWERGRID","SUNPHARMA",
    "NESTLEIND","TITAN","UPL","M&M","WIPRO","GRASIM","DIVISLAB","BAJAJ-AUTO","TECHM","HCLTECH",
    "EICHERMOT","INDUSINDBK","BRITANNIA","ADANIENT","ADANIPORTS","COALINDIA","JSWSTEEL","IOC","LTIM","SHREECEM",
    "SBILIFE","HINDALCO","ULTRACEMCO","VEDL","CIPLA","ICICIGI","TATASTEEL","POWERHOUSE","BANDHANBNK","PEL"
]

# Note: above list aims to be a representative, up-to-date list — replace any symbols you prefer.

# --------------- Helpers ----------------
def get_proxy_base():
    pb = None
    try:
        pb = st.secrets.get("PROXY_BASE", None)
    except Exception:
        pb = None
    if st.session_state.get("PROXY_BASE_OVERRIDE"):
        return st.session_state["PROXY_BASE_OVERRIDE"].rstrip("/")
    return (pb or DEFAULT_PROXY_BASE).rstrip("/")

def proxy_get(path: str, timeout: int = 10):
    """
    GET to proxy. Example path: '/adv-dec?index=NIFTY%2050'
    """
    base = get_proxy_base()
    url = base + path
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/"
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Proxy request failed: {url} — {e}")
        return None

@st.cache_data(ttl=30)
def fetch_breadth_proxy():
    # public proxy adv/dec endpoint (works with the public proxy we discussed)
    return proxy_get("/adv-dec?index=NIFTY%2050")

@st.cache_data(ttl=300)
def fetch_histories_yf(tickers: List[str], days: int = HIST_DAYS):
    """
    Download history for tickers using yfinance.
    Returns dict ticker -> DataFrame
    """
    end = datetime.now().date()
    start = end - timedelta(days=days + 10)  # buffer for non-trading days
    results = {}
    if not tickers:
        return results
    for i in range(0, len(tickers), YF_CHUNK):
        chunk = tickers[i:i+YF_CHUNK]
        try:
            df = yf.download(chunk,
                             start=start.strftime("%Y-%m-%d"),
                             end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                             progress=False,
                             group_by='ticker',
                             threads=True,
                             auto_adjust=False)
        except Exception as e:
            st.warning(f"yfinance download failed for chunk {i//YF_CHUNK}: {e}")
            df = pd.DataFrame()
        if df.empty:
            for tk in chunk:
                results[tk] = pd.DataFrame()
            continue
        # parse into per-ticker DataFrames
        if len(chunk) == 1:
            results[chunk[0]] = df
        else:
            if isinstance(df.columns, pd.MultiIndex):
                for tk in chunk:
                    try:
                        results[tk] = df[tk].dropna(how='all')
                    except Exception:
                        results[tk] = pd.DataFrame()
            else:
                # fallback: assign full df to each (best-effort)
                for tk in chunk:
                    results[tk] = df.copy()
    return results

def analyze_history_df(h: pd.DataFrame):
    """
    Compute last_close, prev_close, DMAs (close-based), flags, daily % and % over window
    """
    if h is None or h.empty or 'Close' not in h.columns:
        return None
    close = h['Close'].dropna()
    if close.empty:
        return None
    res = {}
    res['last_close'] = float(close.iloc[-1])
    res['prev_close'] = float(close.iloc[-2]) if len(close) >= 2 else res['last_close']
    res['dma10'] = float(close.rolling(10).mean().iloc[-1]) if len(close) >= 10 else np.nan
    res['dma20'] = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else np.nan
    res['dma40'] = float(close.rolling(40).mean().iloc[-1]) if len(close) >= 40 else np.nan
    res['above_10'] = int(not np.isnan(res['dma10']) and res['last_close'] > res['dma10'])
    res['above_20'] = int(not np.isnan(res['dma20']) and res['last_close'] > res['dma20'])
    res['above_40'] = int(not np.isnan(res['dma40']) and res['last_close'] > res['dma40'])
    res['last_day_pct'] = ((res['last_close'] - res['prev_close']) / res['prev_close'] * 100) if res['prev_close'] != 0 else 0.0
    # pct over HIST_DAYS window (earliest available within window)
    if len(close) >= HIST_DAYS:
        earliest = float(close.iloc[-HIST_DAYS])
    else:
        earliest = float(close.iloc[0])
    res['pct_over_window'] = ((res['last_close'] - earliest) / earliest * 100) if earliest != 0 else np.nan
    return res

# --------------- UI ----------------
st.title("📈 NIFTY50 Market Breadth Dashboard — 20-day window (static constituents)")

# Top row: proxy input, live breadth, sample image
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.subheader("Proxy")
    current = get_proxy_base()
    proxy_input = st.text_input("Proxy Base (override)", value=current, key="proxy_input")
    if st.button("Apply Proxy Override"):
        st.session_state["PROXY_BASE_OVERRIDE"] = proxy_input.strip()
        st.experimental_rerun()
    st.caption("Set PROXY_BASE in Streamlit Secrets to avoid exposing it in UI.")

with col2:
    st.subheader("NSE Live Breadth (proxy)")
    breadth = fetch_breadth_proxy()
    if breadth:
        adv = breadth.get("advances") or breadth.get("adv") or breadth.get("advancesCount") or breadth.get("advancer")
        dec = breadth.get("declines") or breadth.get("dec") or breadth.get("declinesCount") or breadth.get("decliner")
        unc = breadth.get("unchanged") or breadth.get("unch")
        st.metric("Advances (NSE)", adv if adv is not None else "—")
        st.metric("Declines (NSE)", dec if dec is not None else "—")
        st.metric("Unchanged (NSE)", unc if unc is not None else "—")
    else:
        st.info("Could not fetch adv/dec from proxy. Check proxy or try again.")

with col3:
    st.subheader("Reference Image")
    try:
        st.image(SAMPLE_IMAGE_PATH, caption="Reference screenshot", use_column_width=True)
    except Exception as e:
        st.text(f"Could not load sample image at {SAMPLE_IMAGE_PATH}: {e}")

st.markdown("---")

# Build tickers for yfinance
symbols = NIFTY50_SYMBOLS
tickers_yf = [s + ".NS" for s in symbols]

st.write(f"Using static list with {len(tickers_yf)} tickers. Fetching {HIST_DAYS} days history for each via yfinance (may take ~30–90s).")

histories = fetch_histories_yf(tickers_yf, days=HIST_DAYS)

rows = []
for sym, yf_sym in zip(symbols, tickers_yf):
    h = histories.get(yf_sym, pd.DataFrame())
    info = analyze_history_df(h)
    if info is None:
        rows.append({
            "Symbol": sym,
            "Ticker": yf_sym,
            "Close": np.nan,
            "%Above10": np.nan,
            "%Above20": np.nan,
            "%Above40": np.nan,
            "Up4": 0,
            "Down4": 0,
            "Up25": 0,
            "Down25": 0,
            "Up50": 0,
            "Down50": 0
        })
    else:
        last = info['last_close']
        day_pct = info['last_day_pct']
        window_pct = info['pct_over_window']
        rows.append({
            "Symbol": sym,
            "Ticker": yf_sym,
            "Close": last,
            "%Above10": info['above_10'],
            "%Above20": info['above_20'],
            "%Above40": info['above_40'],
            "Up4": 1 if day_pct > 4 else 0,
            "Down4": 1 if day_pct < -4 else 0,
            "Up25": 1 if (not np.isnan(window_pct) and window_pct > 25) else 0,
            "Down25": 1 if (not np.isnan(window_pct) and window_pct < -25) else 0,
            "Up50": 1 if (not np.isnan(window_pct) and window_pct > 50) else 0,
            "Down50": 1 if (not np.isnan(window_pct) and window_pct < -50) else 0
        })

df = pd.DataFrame(rows)

# Aggregates
total = len(df)
up4_total = int(df["Up4"].sum())
down4_total = int(df["Down4"].sum())
pct_above_10 = df["%Above10"].sum() / total * 100 if total else np.nan
pct_above_20 = df["%Above20"].sum() / total * 100 if total else np.nan
pct_above_40 = df["%Above40"].sum() / total * 100 if total else np.nan
up25_total = int(df["Up25"].sum())
down25_total = int(df["Down25"].sum())
up50_total = int(df["Up50"].sum())
down50_total = int(df["Down50"].sum())

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Stocks > +4% (today)", up4_total)
c2.metric("Stocks < -4% (today)", down4_total)
c3.metric("% Above 10 DMA", f"{pct_above_10:.1f}%" if not np.isnan(pct_above_10) else "—")
c4.metric("% Above 20 DMA", f"{pct_above_20:.1f}%" if not np.isnan(pct_above_20) else "—")
c5.metric("Up 25% (20d)", up25_total)
c6.metric("Down 25% (20d)", down25_total)

st.markdown("---")

# Display table with heatmap coloring for flags
display_cols = ["Ticker", "Close", "Up4", "Down4", "Up25", "Down25", "Up50", "Down50", "%Above10", "%Above20", "%Above40"]
display_df = df.set_index("Symbol")[display_cols]

def color_flag(v):
    if pd.isna(v):
        return ""
    if v == 1:
        return "background-color: #b6d7a8"
    if v == 0:
        return "background-color: #f4cccc"
    return ""

styled = display_df.style.format({"Close": "{:.2f}"}).applymap(color_flag, subset=["Up4","Down4","Up25","Down25","Up50","Down50","%Above10","%Above20","%Above40"])

st.subheader("Per-stock Breadth Table (static constituents)")
st.dataframe(styled, use_container_width=True)

st.markdown("---")
st.subheader("Aggregate: % Above 20 DMA over recent days (sample)")
days_back = st.slider("Days to aggregate (max 20)", min_value=5, max_value=min(20, HIST_DAYS), value=min(10, HIST_DAYS))
dates = pd.date_range(end=datetime.now(), periods=days_back)
agg_rows = []
for d in dates:
    cnt = 0
    valid = 0
    for yf_sym in histories:
        h = histories.get(yf_sym)
        if h is None or h.empty or 'Close' not in h.columns:
            continue
        close = h['Close'].dropna()
        c = close[close.index <= d]
        if len(c) < 20:
            continue
        dma20 = c.rolling(20).mean().iloc[-1]
        if np.isnan(dma20):
            continue
        valid += 1
        if c.iloc[-1] > dma20:
            cnt += 1
    pct = (cnt / valid * 100) if valid else np.nan
    agg_rows.append({"date": d, "pct_above20": pct})

agg_df = pd.DataFrame(agg_rows).dropna()
if not agg_df.empty:
    chart = alt.Chart(agg_df).mark_area(opacity=0.4).encode(x='date:T', y='pct_above20:Q').properties(height=300)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Not enough data to compute aggregate trend for the requested window.")

st.markdown("---")
st.caption("Notes: DMAs are computed on Close prices. Up/Down 4% counts stocks with today % change >4% / <-4%. 25%/50% are computed over the 20-day window (increase HIST_DAYS if you want longer-term % changes).")
