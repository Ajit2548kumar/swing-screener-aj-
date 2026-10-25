import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="NSE Swing Screener", layout="wide")
st.title("📈 NSE Swing Screener (8-Point Check)")

import io, os, time
import requests

LISTS = {
    "Nifty 50": "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "Nifty 100": "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "Nifty 500": "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "All NSE (2000+, slow)": "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "Custom (khud likho)": None,
}
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}


@st.cache_data(ttl=86400, show_spinner=False)
def load_list(name):
    """NSE se list lao; na mile to repo ki local file lists/<naam>.csv use karo."""
    url = LISTS[name]
    local = os.path.join("lists", name.split(" (")[0].lower().replace(" ", "") + ".csv")
    try:
        txt = requests.get(url, headers=HEADERS, timeout=15).text
        df = pd.read_csv(io.StringIO(txt))
    except Exception:
        if not os.path.exists(local):
            return []
        df = pd.read_csv(local)
    df.columns = [c.strip().upper() for c in df.columns]
    if "SERIES" in df.columns:
        df = df[df["SERIES"].str.strip() == "EQ"]
    return sorted(df["SYMBOL"].astype(str).str.strip().unique().tolist())


universe = st.selectbox("Kaunsa universe screen karna hai?", list(LISTS.keys()), index=2)
if LISTS[universe] is None:
    symbols = st.text_area("Stock symbols (comma se alag, bina .NS ke)",
                           "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, ITC, LT")
else:
    symbols = ""
capital = st.number_input("Total capital (₹)", value=500000, step=10000)
risk_pct = st.number_input("Per trade risk (%)", value=1.0, step=0.25)
min_score = st.slider("Sirf wahi stocks dikhao jinka score kam-se-kam ...", 0, 8, 6)
swing_n = 5       # swing high/low window
lookback = 20     # resistance = pichle 20 din ka high
vol_mult = 1.5    # breakout volume >= 1.5x 20-din average


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def swing_points(s, kind):
    pts = []
    for i in range(swing_n, len(s) - swing_n):
        w = s.iloc[i - swing_n:i + swing_n + 1]
        if (kind == "high" and s.iloc[i] == w.max()) or (kind == "low" and s.iloc[i] == w.min()):
            pts.append(s.iloc[i])
    return pts


@st.cache_data(ttl=3600, show_spinner=False)
def load_chunk(syms):
    """Ek chunk (<=100 stocks) EK call mein; 1 ghante tak cache."""
    tickers = [x + ".NS" for x in syms]
    return yf.download(tickers, period="1y", interval="1d", group_by="ticker",
                       progress=False, auto_adjust=True, threads=True)


def analyse(sym, data):
    try:
        df = data[sym + ".NS"].dropna(how="all")
    except KeyError:
        return None
    if df.empty or len(df) < 80:
        return None
    o, c, h, l, v = df["Open"], df["Close"], df["High"], df["Low"], df["Volume"]
    e20, e50 = (c.ewm(span=n, adjust=False).mean() for n in (20, 50))
    price = c.iloc[-1]
    r = rsi(c).iloc[-1]

    # 1. Trend (जरूरी): price EMA20 aur EMA50 dono ke upar; EMA20>EMA50 ho to behtar
    trend_ok = price > e20.iloc[-1] and price > e50.iloc[-1]
    ema_better = e20.iloc[-1] > e50.iloc[-1]

    # 2. Higher High + Higher Low
    sh, sl = swing_points(h, "high")[-2:], swing_points(l, "low")[-2:]
    hhhl = len(sh) == 2 and len(sl) == 2 and sh[1] > sh[0] and sl[1] > sl[0]

    # 3. Resistance breakout: close resistance ke upar + majboot (green, upper-range) candle
    res = h.shift(1).rolling(lookback).max()
    rng = (h - l).replace(0, np.nan)
    strong = (c > o) & ((c - l) / rng >= 0.7)
    brk = (c > res) & strong
    recent = brk.iloc[-10:]
    breakout = bool(recent.any())

    # 4. Volume: breakout din par volume >= 1.5x pichle 20 din ka average
    vol_ratio = (v / v.shift(1).rolling(20).mean()).iloc[-10:]
    vol_ok = bool(breakout and (vol_ratio[recent] >= vol_mult).any())
    brk_level = res.iloc[-10:][recent].iloc[-1] if breakout else np.nan

    # 5. RSI 55-70 (70 se upar = jaldbaazi mat karo)
    rsi_ok = 55 <= r <= 70

    # 6. Pullback / Retest: breakout level ke ±3% paas, support hold, aaj green
    retest = bool(breakout and abs(price - brk_level) / brk_level <= 0.03
                  and l.iloc[-3:].min() >= brk_level * 0.97 and c.iloc[-1] > c.iloc[-2])

    # 7. Risk-Reward >= 1:2
    stop = min(sl) * 0.995 if sl else l.iloc[-10:].min() * 0.995
    risk = price - stop
    prior_high = h.iloc[-60:].max()
    target = prior_high if prior_high > price + 2 * risk else price + 2 * risk
    rr = (target - price) / risk if risk > 0 else 0
    rr_ok = rr >= 2

    # 8. Stop loss + position size + gap risk
    gap = (o / c.shift(1) - 1).abs().iloc[-20:].max() * 100
    qty = int((capital * risk_pct / 100) // risk) if risk > 0 else 0
    sl_ok = 0 < risk / price <= 0.08 and gap < 4 and qty > 0

    checks = [trend_ok, hhhl, breakout, vol_ok, rsi_ok, retest, rr_ok, sl_ok]
    return {
        "Stock": sym, "Price": round(price, 2), "RSI": round(r, 1),
        "1 Trend (जरूरी)": trend_ok, "EMA20>50": bool(ema_better), "2 HH/HL": hhhl,
        "3 Breakout": breakout, "4 Volume 1.5x": vol_ok, "5 RSI 55-70": rsi_ok,
        "6 Retest": retest, "7 R:R 1:2": rr_ok, "8 SL/Size": sl_ok,
        "Stop Loss": round(stop, 2), "Target": round(target, 2),
        "R:R": round(rr, 2), "Qty": qty, "Gap% (20d max)": round(gap, 1),
        "Score /8": sum(checks),
    }


if st.button("Screen karo"):
    if LISTS[universe] is None:
        syms = [x.strip().upper() for x in symbols.split(",") if x.strip()]
    else:
        syms = load_list(universe)
        if not syms:
            st.error("NSE list nahi mili. Repo mein lists/ folder mein CSV daalo (README dekho).")
            st.stop()
    st.write(f"{len(syms)} stocks screen ho rahe hain...")
    CH = 100
    chunks = [tuple(syms[i:i + CH]) for i in range(0, len(syms), CH)]
    rows, failed, bar = [], 0, st.progress(0.0)
    for i, ch in enumerate(chunks):
        try:
            data = load_chunk(ch)
            for sym in ch:
                try:
                    res = analyse(sym, data)
                    if res:
                        rows.append(res)
                    else:
                        failed += 1
                except Exception:
                    failed += 1
        except Exception:
            failed += len(ch)
        bar.progress((i + 1) / len(chunks))
        time.sleep(1)  # Yahoo rate-limit se bachne ke liye
    if rows:
        out = pd.DataFrame(rows).sort_values("Score /8", ascending=False)
        out = out[out["Score /8"] >= min_score]
        st.caption(f"{len(rows)} stocks ka data mila, {failed} ka nahi mila. {len(out)} score filter pass kar gaye.")
        show = out.copy()
        for col in show.columns[3:12]:
            show[col] = show[col].map({True: "✅", False: "❌"})
        st.dataframe(show, use_container_width=True, hide_index=True)
        top = out[(out["Score /8"] >= 7) & out["1 Trend (जरूरी)"]]["Stock"]
        st.success("Top picks (trend ✅ + 7/8 ya zyada): " + ", ".join(top)) if len(top) \
            else st.info("Aaj koi stock 7/8 score aur trend ✅ pe nahi hai.")
    else:
        st.error("Koi data nahi mila. Thodi der baad dobara try karo.")

st.caption("Sirf education ke liye. End-of-day delayed data (Yahoo Finance). Investment advice nahi hai.")
