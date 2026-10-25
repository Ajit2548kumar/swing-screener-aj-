import io, os, time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Backtest", layout="wide")
st.title("🧪 Backtest — 8-Point Rules")
st.caption("Purane data par check: agar in rules se roz trade lete, to kya hota. "
           "Ye guarantee nahi hai ki aage bhi aisa hoga.")

LISTS = {
    "Nifty 50": "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "Nifty 100": "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "Nifty 500": "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "Custom (khud likho)": None,
}
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
SWING_N = 5
LOOKBACK = 20
VOL_MULT = 1.5


@st.cache_data(ttl=86400, show_spinner=False)
def load_list(name):
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
    return sorted(df["SYMBOL"].astype(str).str.strip().unique().tolist())


@st.cache_data(ttl=21600, show_spinner=False)
def load_chunk(syms):
    tickers = [x + ".NS" for x in syms]
    return yf.download(tickers, period="3y", interval="1d", group_by="ticker",
                       progress=False, auto_adjust=True, threads=True)


def prepare(df):
    """Har din ke liye 8 checks (sirf us din tak ke data se, aage ka data nahi)."""
    o, c, h, l, v = (df[k] for k in ["Open", "Close", "High", "Low", "Volume"])
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    trend = (c > e20) & (c > e50)

    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    rsi_ok = (rsi >= 55) & (rsi <= 70)

    # swing high/low: pivot tab confirm hota hai jab SWING_N din baad dikhe
    w = 2 * SWING_N + 1
    ph = h.where(h == h.rolling(w, center=True).max())
    pl = l.where(l == l.rolling(w, center=True).min())
    kh = ph.shift(SWING_N).dropna()
    kl = pl.shift(SWING_N).dropna()
    sh_last = kh.reindex(df.index).ffill()
    sh_prev = kh.shift(1).reindex(df.index).ffill()
    sl_last = kl.reindex(df.index).ffill()
    sl_prev = kl.shift(1).reindex(df.index).ffill()
    hhhl = (sh_last > sh_prev) & (sl_last > sl_prev)

    res = h.shift(1).rolling(LOOKBACK).max()
    rng = (h - l).replace(0, np.nan)
    strong = (c > o) & ((c - l) / rng >= 0.7)
    brk = (c > res) & strong
    breakout = brk.astype(float).rolling(10).max() > 0
    vr = v / v.shift(1).rolling(20).mean().replace(0, np.nan)
    vol_ok = (brk & (vr >= VOL_MULT)).astype(float).rolling(10).max() > 0

    lvl = res.where(brk).ffill(limit=9)
    retest = (breakout & ((c - lvl).abs() / lvl <= 0.03)
              & (l.rolling(3).min() >= lvl * 0.97) & (c > c.shift(1)))

    sl_min = pd.concat([sl_last, sl_prev], axis=1).min(axis=1)
    stop = (sl_min * 0.995).fillna(l.rolling(10).min() * 0.995)
    risk = c - stop
           # sabse paas ka asli resistance = c se upar ka sabse chhota confirmed swing high (pichle ~250 din)
    ci = df.index.get_indexer(kh.index)
    pv = kh.values
    if len(pv):
        tt = np.arange(len(df))[:, None]
        valid = ((ci[None, :] <= tt) & (ci[None, :] > tt - 250)
                 & (pv[None, :] > c.values[:, None]))
        nearest = np.where(valid, pv[None, :], np.inf).min(axis=1)
    else:
        nearest = np.full(len(df), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = pd.Series(np.where(np.isfinite(nearest), (nearest - c.values) / risk.values, 2.0),
                       index=c.index)
    rr_ok = (risk > 0) & (rr >= 2)
           
           

    gap = (o / c.shift(1) - 1).abs().rolling(20).max() * 100
    sl_ok = (risk > 0) & (risk / c <= 0.08) & (gap < 4)

    checks = [trend, hhhl, breakout, vol_ok, rsi_ok, retest, rr_ok, sl_ok]
    score = sum(x.fillna(False).astype(int) for x in checks)
    return trend.fillna(False), score, stop


def simulate(sym, df, trend, score, stop, min_score, test_start, max_hold, tgt_r, cost_pct):
    o, h, l, c = (df[k].values for k in ["Open", "High", "Low", "Close"])
    dates = df.index
    n = len(df)
    sig = (trend & (score >= min_score)).values & (dates >= test_start)
    stp = stop.values
    out, free = [], 0
    for t in np.flatnonzero(sig):
        if t < free or t + 1 >= n or np.isnan(stp[t]):
            continue
        entry, s = o[t + 1], stp[t]
        if not entry > s:
            continue
        risk = entry - s
        target = entry + tgt_r * risk
        end = min(t + max_hold, n - 1)
        ex_i, ex_px, why = None, None, None
        for d in range(t + 1, end + 1):
            if o[d] <= s:
                ex_i, ex_px, why = d, o[d], "Stop (gap)"
            elif l[d] <= s:
                ex_i, ex_px, why = d, s, "Stop"
            elif h[d] >= target:
                ex_i, ex_px, why = d, max(o[d], target), "Target"
            if ex_i is not None:
                break
        if ex_i is None:
            if end < t + max_hold:      # data khatam, trade adhura — chhod do
                continue
            ex_i, ex_px, why = end, c[end], "Time exit"
        r = (ex_px - entry) / risk - (cost_pct / 100) * entry / risk
        out.append({"Stock": sym, "Entry date": dates[t + 1].date(), "Exit date": dates[ex_i].date(),
                    "Entry": round(entry, 2), "Exit": round(ex_px, 2), "Exit reason": why,
                    "Days": ex_i - t, "R": round(r, 2),
                    "Return %": round((ex_px / entry - 1) * 100 - cost_pct, 2),
                    "Score": int(score.values[t])})
        free = ex_i + 1
    return out


# ---------------- UI ----------------
universe = st.selectbox("Universe", list(LISTS.keys()), index=0)
if LISTS[universe] is None:
    custom = st.text_area("Symbols (comma se alag, bina .NS ke)", "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN")
years = st.selectbox("Kitne saal ka test?", [1, 2], index=1)
min_score = st.slider("Trade tab lo jab score kam-se-kam ...", 4, 8, 6)
tgt_r = st.number_input("Target (R mein, 2 = 1:2)", value=2.0, step=0.5)
max_hold = st.number_input("Zyada se zyada kitne din hold", value=20, step=5)
cost_pct = st.number_input("Kharcha + slippage (%, round trip)", value=0.2, step=0.1)

if st.button("Backtest chalao"):
    syms = [x.strip().upper() for x in custom.split(",") if x.strip()] if LISTS[universe] is None \
        else load_list(universe)
    if not syms:
        st.error("List nahi mili.")
        st.stop()
    trades, ok, bad = [], 0, 0
    chunks = [tuple(syms[i:i + 100]) for i in range(0, len(syms), 100)]
    bar = st.progress(0.0)
    for i, ch in enumerate(chunks):
        try:
            data = load_chunk(ch)
            for sym in ch:
                try:
                    df = data[sym + ".NS"].dropna(how="all")
                    if len(df) < 150:
                        bad += 1
                        continue
                    start = df.index[-1] - pd.Timedelta(days=365 * years)
                    trend, score, stop = prepare(df)
                    trades += simulate(sym, df, trend, score, stop, min_score, start,
                                       int(max_hold), float(tgt_r), float(cost_pct))
                    ok += 1
                except Exception:
                    bad += 1
        except Exception:
            bad += len(ch)
        bar.progress((i + 1) / len(chunks))
        time.sleep(1)
    st.session_state["bt"] = (pd.DataFrame(trades), ok, bad, min_score)

if "bt" in st.session_state:
    tr, ok, bad, ms = st.session_state["bt"]
    st.caption(f"{ok} stocks ka test hua, {bad} ka data nahi mila.")
    if tr.empty:
        st.warning("Is setting par koi trade nahi bana. Score kam karke ya Nifty 500 le kar try karo.")
    else:
        tr = tr.sort_values("Exit date").reset_index(drop=True)
        win = tr[tr["R"] > 0]
        loss = tr[tr["R"] <= 0]
        pf = win["R"].sum() / abs(loss["R"].sum()) if len(loss) and loss["R"].sum() != 0 else float("inf")
        cum = tr["R"].cumsum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Trades", len(tr))
        c2.metric("Jeet %", f"{len(win) / len(tr) * 100:.0f}%")
        c3.metric("Avg R (expectancy)", f"{tr['R'].mean():.2f}")
        c4, c5, c6 = st.columns(3)
        c4.metric("Profit factor", f"{pf:.2f}")
        c5.metric("Avg return/trade", f"{tr['Return %'].mean():.2f}%")
        c6.metric("Max drawdown (R)", f"{(cum - cum.cummax()).min():.1f}")
        st.caption("R = 1 matlab jitna risk liya utna profit. Avg R positive ho tabhi rules paisa banate hain. "
                   "Jeet % akeli kuch nahi batati.")
        st.subheader("Cumulative R (trade dar trade)")
        st.line_chart(cum)
        st.subheader("Score ke hisaab se")
        rows = []
        for k in range(ms, 9):
            sub = tr[tr["Score"] >= k]
            if len(sub):
                rows.append({"Score >=": k, "Trades": len(sub),
                             "Jeet %": round((sub["R"] > 0).mean() * 100),
                             "Avg R": round(sub["R"].mean(), 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.subheader("Exit kaise hua")
        st.dataframe(tr["Exit reason"].value_counts().rename("Trades").to_frame(), use_container_width=True)
        st.subheader("Sab trades")
        st.dataframe(tr, use_container_width=True, hide_index=True)
        st.download_button("CSV download", tr.to_csv(index=False), "backtest_trades.csv")

    st.warning("Dhyan rakho: (1) Nifty 500 ki AAJ ki list use hoti hai, jo stocks gir kar list se nikal gaye wo "
               "nahi hain, isliye result thoda achha dikh sakta hai. (2) Entry agle din ke open par, stop/target "
               "daily candle se; ek din mein dono lage to stop pehle maana. (3) Free Yahoo data mein galtiyan "
               "ho sakti hain. Ye sirf education ke liye hai.")
