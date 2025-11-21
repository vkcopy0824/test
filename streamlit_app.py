import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import altair as alt
from datetime import datetime, timedelta
from typing import List

st.set_page_config(layout="wide", page_title="NIFTY Market Breadth Dashboard (20d)")

# ---------------- Configuration ----------------
# Default public proxy that forwards NSE endpoints (replace if you have another)
DEFAULT_PROXY_BASE = "https://nse-api-proxy.vercel.app"

# Local image path (user-uploaded image). Developer requested this exact path be used.
SAMPLE_IMAGE_PATH = "/mnt/data/IMG_1890.JPG"

# History window in days (user requested 20 days)
HIST_DAYS = 20

# yfinance download chunk size
YF_CHUNK = 25

# ---------------- Helpers ----------------
def get_proxy_base():
    # Prefer secrets, then session override, then default
    pb = st.secrets.get("PROXY_BASE", None) if hasattr(st, "secrets") else None
    if "PROXY_BASE_OVERRIDE" in st.session_state:
        return st.session_state["PROXY_BASE_OVERRIDE"]
    return pb or DEFAULT_PROXY_BASE

def proxy_get(full_path: str, timeout: int = 10):
    """
    Perform GET to proxy. `full_path` should start with '/' and include any query string if needed.
    Example: '/api/marketData-pre-open?key=NIFTY'
    """
    base = get_proxy_base().rstrip("/")
    url = base + full_path
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

@st.cache_data(ttl=60)
def fetch_breadth_via_proxy():
    # Common NSE endpoint for breadth (proxied)
    return proxy_get("/api/marketData-pre-open?key=NIFTY")

@st.cache_data(ttl=600)
def fetch_nifty50_constituents_via_proxy():
    # Endpoint returning stock list for NIFTY 50 (proxied)
    # Some proxies may return different shapes; handle common variants.
    j = proxy_get("/api/equity-stockIndices?index=NIFTY%2050")
    if not j:
        return None
    # try common structures
    items = j.get("data") if isinstance(j, dict) and "data" in j else j
    symbols = []
    try:
        for it in items:
            if isinstance(it, dict):
                sym = it.get("symbol") or it.get("tradingSymbol") or it.get("symbolName")
            else:
                sym = None
            if sym:
                # ensure .NS suffix for yfinance
                yf_sym = sym if sym.endswith(".NS") else f"{sym}.NS"
                symbols.append((sym, yf_sym))
    except Exception:
        return None
    return symbols

@st.cache_data(ttl=300)
def fetch_histories_yf(tickers: List[str], days: int = HIST_DAYS):
    """
    Fetch history via yfinance for multiple tickers.
    Returns dict ticker -> DataFrame (with Date index and 'Open','Close' etc.)
    """
    end = datetime.now().date()
    start = end - timedelta(days=days + 10)  # extra buffer for non-trading days
    results = {}
    if not tickers:
        return results
    # chunked downloads
    for i in range(0, len(tickers), YF_CHUNK):
        chunk = tickers[i:i + YF_CHUNK]
        try:
            df = yf.download(chunk, start=start.strftime("%Y-%m-%d"), end=(end + timedelta(days=1)).strftime("%Y-%m-%d"), progress=False, group_by='ticker', threads=True, auto_adjust=False)
        except Exception as e:
            st.warning(f"yfinance download failed for chunk {i//YF_CHUNK}: {e}")
            df = pd.DataFrame()
        # parse results
        if df.empty:
            for tk in chunk:
                results[tk] = pd.DataFrame()
            continue
        # If only one ticker, df has normal columns
        if len(chunk) == 1:
            results[chunk[0]] = df
        else:
            # Multi-index columns with ticker as top-level
            if isinstance(df.columns, pd.MultiIndex):
                for tk in chunk:
                    if tk in df.columns.levels[0]:
                        try:
                            results[tk] = df[tk].dropna(how='all')
                        except Exception:
                            results[tk] = pd.DataFrame()
                    else:
                        # fallback: try to build from columns containing the ticker
                        results[tk] = pd.DataFrame()
            else:
                # flat columns - ambiguous; assign entire df (best-effort)
                for tk in chunk:
                    results[tk] = df.copy()
    return results

def analyze_history_df(h: pd.DataFrame):
    """
    Given a history DataFrame with 'Close' column, compute:
    - last_close, prev_close
    - 10/20/40 DMA (if possible)
    - above DMAs flags
    - last_day_pct (today vs prev)
    - pct_change_over_period (period = HIST_DAYS-1 between oldest and last)
    """
    if h is None or h.empty or 'Close' not in h.columns:
        return None
    close = h['Close'].dropna()
    if close.empty:
        return None
    res = {}
    res['last_close'] = float(close.iloc[-1])
    res['prev_close'] = float(close.iloc[-2]) if len(close) >= 2 else res['last_close']
    # DMAs
    res['dma10'] = float(close.rolling(10).mean().iloc[-1]) if len(close) >= 10 else np.nan
    res['dma20'] = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else np.nan
    res['dma40'] = float(close.rolling(40).mean().iloc[-1]) if len(close) >= 40 else np.nan
    res['above_10'] = int(not np.isnan(res['dma10']) and res['last_close'] > res['dma10'])
    res['above_20'] = int(not np.isnan(res['dma20']) and res['last_close'] > res['dma20'])
    res['above_40'] = int(not np.isnan(res['dma40']) and res['last_close'] > res['dma40'])
    # daily pct change (last vs prev)
    res['last_day_pct'] = ((res['last_close'] - res['prev_close']) / res['prev_close'] * 100) if res['prev_close'] != 0 else 0.0
    # percent change over the requested HIST_DAYS window: from earliest available within window to last
    if len(close) >= HIST_DAYS:
        earliest = float(close.iloc[-HIST_DAYS])
        res['pct_over_window'] = ((res['last_close'] - earliest) / earliest * 100) if earliest != 0 else np.nan
    else:
        # use earliest available
        earliest = float(close.iloc[0])
        res['pct_over_window'] = ((res['last_close'] - earliest) / earliest * 100) if earliest != 0 else np.nan
    return res

# ---------------- UI ----------------
st.title("📈 NIFTY50 Market Breadth Dashboard — 20-day window (stocks-based breadth)")

# Top: proxy config and live breadth
col_p1, col_p2, col_p3 = st.columns([2, 2, 1])
with col_p1:
    st.subheader("Proxy configuration")
    proxy_val = st.text_input("Proxy Base (override)", value=get_proxy_base(), key="proxy_override_input")
    if st.button("Apply proxy override"):
        st.session_state["PROXY_BASE_OVERRIDE"] = proxy_val.strip()
        st.experimental_rerun()
    st.caption("Set PROXY_BASE in Streamlit Secrets to avoid exposing it in UI. Default uses a public proxy.")

with col_p2:
    st.subheader("Live breadth (from proxy)")
    breadth = fetch_breadth_via_proxy()
    if breadth:
        adv = breadth.get("advances") or breadth.get("advancer") or breadth.get("adv") or breadth.get("advancesCount")
        dec = breadth.get("declines") or breadth.get("decliner") or breadth.get("dec") or breadth.get("declinesCount")
        unc = breadth.get("unchanged") or breadth.get("unch")
        st.metric("Advances (NSE)", adv if adv is not None else "—")
        st.metric("Declines (NSE)", dec if dec is not None else "—")
        st.metric("Unchanged (NSE)", unc if unc is not None else "—")
    else:
        st.info("Could not fetch breadth from proxy. Check proxy or try again.")

with col_p3:
    st.subheader("Reference Image")
    try:
        st.image(SAMPLE_IMAGE_PATH, caption="Reference screenshot", use_column_width=True)
    except Exception as e:
        st.text(f"Could not load sample image: {e}")

st.markdown("---")

# Fetch constituents from proxy (dynamic list)
st.subheader("Fetching NIFTY50 constituents (via proxy)")
symbols = fetch_nifty50_constituents_via_proxy()
if not symbols:
    st.warning("Could not fetch constituents from proxy; using a default fallback list.")
    fallback = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    symbols = [(s, f"{s}.NS") for s in fallback]

st.write(f"Found {len(symbols)} symbols. Fetching {HIST_DAYS} days history for each (yfinance). This can take ~30-90s depending on network.")

tickers_yf = [yf for (_, yf) in symbols]

# Fetch histories
histories = fetch_histories_yf(tickers_yf, days=HIST_DAYS)

# Analyze each symbol
rows = []
for orig_sym, yf_sym in symbols:
    h = histories.get(yf_sym, pd.DataFrame())
    info = analyze_history_df(h)
    if info is None:
        rows.append({
            "Symbol": orig_sym,
            "Ticker": yf_sym,
            "Last": np.nan,
            "Chg% (day)": np.nan,
            "Pct (20d)": np.nan,
            "Above10DMA": np.nan,
            "Above20DMA": np.nan,
            "Above40DMA": np.nan,
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
        above10 = info['above_10']
        above20 = info['above_20']
        above40 = info['above_40']  # likely NaN because HIST_DAYS=20
        # thresholds:
        up4 = 1 if day_pct > 4 else 0
        down4 = 1 if day_pct < -4 else 0
        up25 = 1 if (not np.isnan(window_pct) and window_pct > 25) else 0
        down25 = 1 if (not np.isnan(window_pct) and window_pct < -25) else 0
        up50 = 1 if (not np.isnan(window_pct) and window_pct > 50) else 0
        down50 = 1 if (not np.isnan(window_pct) and window_pct < -50) else 0

        rows.append({
            "Symbol": orig_sym,
            "Ticker": yf_sym,
            "Last": last,
            "Chg% (day)": round(day_pct, 2),
            f"Pct ({HIST_DAYS}d)": round(window_pct, 2) if not np.isnan(window_pct) else np.nan,
            "Above10DMA": above10,
            "Above20DMA": above20,
            "Above40DMA": above40,
            "Up4": up4,
            "Down4": down4,
            "Up25": up25,
            "Down25": down25,
            "Up50": up50,
            "Down50": down50
        })

df = pd.DataFrame(rows)

# Compute summary breadth counts
total = len(df)
adv_calc = int((df["Chg% (day)"] > 0).sum())
dec_calc = int((df["Chg% (day)"] < 0).sum())
pct_above_10 = df["Above10DMA"].sum() / total * 100 if total else np.nan
pct_above_20 = df["Above20DMA"].sum() / total * 100 if total else np.nan
pct_above_40 = df["Above40DMA"].sum() / total * 100 if total else np.nan
up4_total = int(df["Up4"].sum())
down4_total = int(df["Down4"].sum())
up25_total = int(df["Up25"].sum())
down25_total = int(df["Down25"].sum())
up50_total = int(df["Up50"].sum())
down50_total = int(df["Down50"].sum())

# Top metrics
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Advances (calc)", adv_calc)
m2.metric("Declines (calc)", dec_calc)
m3.metric("% Above 10 DMA", f"{pct_above_10:.1f}%" if not np.isnan(pct_above_10) else "—")
m4.metric("% Above 20 DMA", f"{pct_above_20:.1f}%" if not np.isnan(pct_above_20) else "—")
m5.metric("Up 4% (count)", up4_total)
m6.metric("Down 4% (count)", down4_total)

m7, m8, m9 = st.columns(3)
m7.metric(f"Up 25% (20d)", up25_total)
m8.metric(f"Down 25% (20d)", down25_total)
m9.metric(f"Up 50% (20d)", up50_total)

st.markdown("---")

# Styling for table: green for flags=1, red for 0
def color_flag(v):
    if pd.isna(v):
        return ""
    return "background-color: #b6d7a8" if v == 1 else "background-color: #f4cccc"

display_df = df.set_index("Symbol")[["Ticker", "Last", "Chg% (day)", f"Pct ({HIST_DAYS}d)", "Above10DMA", "Above20DMA", "Above40DMA", "Up4", "Down4", "Up25", "Down25", "Up50", "Down50"]]

st.subheader("Breadth Table — per-stock computed (heatmap flags)")
st.dataframe(display_df.style.format({"Last": "{:.2f}", "Chg% (day)": "{:.2f}", f"Pct ({HIST_DAYS}d)": "{:.2f}"}).applymap(color_flag, subset=["Above10DMA","Above20DMA","Above40DMA","Up4","Down4","Up25","Down25","Up50","Down50"]), use_container_width=True)

st.markdown("---")
st.subheader("Aggregate Trend: % Above 20 DMA over last few days (sample)")
# Build simple aggregate over recent days for % above 20 DMA (requires histories)
days_back = st.slider("Days to aggregate (max 20)", min_value=5, max_value=min(60, HIST_DAYS), value=min(10, HIST_DAYS))
dates = pd.date_range(end=datetime.now(), periods=days_back)
agg = []
for d in dates:
    cnt = 0
    total_valid = 0
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
        total_valid += 1
        if c.iloc[-1] > dma20:
            cnt += 1
    pct = (cnt / total_valid * 100) if total_valid else np.nan
    agg.append({"date": d, "pct_above20": pct})

agg_df = pd.DataFrame(agg).dropna()
if not agg_df.empty:
    chart = alt.Chart(agg_df).mark_area(opacity=0.4).encode(x='date:T', y='pct_above20:Q').properties(height=300)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Not enough data to compute aggregate trend for the requested window.")

st.markdown("---")
st.caption("Notes: 'Up/Down 4%' uses today's percent change. 'Up/Down 25/50%' use change over the configured window (20 days). Increase the history window if you want reliable 40 DMA and longer-term % changes.")
