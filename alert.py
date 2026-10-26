"""Roz shaam NSE Nifty 500 screen karke Telegram par shortlist bhejta hai.
GitHub Actions se chalta hai. Token/ID code mein NAHI likhna, GitHub Secrets mein rakhna."""
import io, os, sys, time
import numpy as np
import pandas as pd
import requests
import yfinance as yf

TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
UNIVERSE_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}

MIN_SCORE = 6        # kam-se-kam itna score (8 mein se)
MAX_ALERTS = 15      # ek message mein zyada se zyada kitne stocks
capital = 500000     # total capital (Qty nikalne ke liye)
risk_pct = 1.0       # per trade risk %
swing_n = 5
lookback = 20
vol_mult = 1.5


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
    overhead = [x for x in swing_points(h, "high") if x > price]
    if overhead:
        target = min(overhead)          # sabse paas ka asli resistance
        rr = (target - price) / risk if risk > 0 else 0
    else:
        target = price + 2 * risk       # upar koi resistance nahi
        rr = 2.0 if risk > 0 else 0
    rr_ok = rr >= 2

    # 8. Stop loss + position size + gap risk
    gap = (o / c.shift(1) - 1).abs().iloc[-20:].max() * 100
    qty = int((capital * risk_pct / 100) // risk) if risk > 0 else 0
    sl_ok = 0.02 <= risk / price <= 0.08 and gap < 4 and qty > 0

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


CHECKS = ["1 Trend (जरूरी)", "2 HH/HL", "3 Breakout", "4 Volume 1.5x",
          "5 RSI 55-70", "6 Retest", "7 R:R 1:2", "8 SL/Size"]


def get_symbols():
    try:
        txt = requests.get(UNIVERSE_URL, headers=HEADERS, timeout=20).text
        df = pd.read_csv(io.StringIO(txt))
    except Exception:
        path = os.path.join("lists", "nifty500.csv")
        if not os.path.exists(path):
            return []
        df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    return sorted(df["SYMBOL"].astype(str).str.strip().unique().tolist())


def send(text):
    if not TOKEN or not CHAT_ID:
        print(text)
        return
    for i in range(0, len(text), 3800):
        r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                          data={"chat_id": CHAT_ID, "text": text[i:i + 3800]}, timeout=20)
        if r.status_code != 200:
            print("Telegram error:", r.status_code, r.text[:200])
            sys.exit(1)


def build_message(picks, last_date, total):
    d = last_date.strftime("%d-%b-%Y") if last_date is not None else "?"
    if not picks:
        return (f"📈 Swing Screener (data: {d})\n"
                f"{total} stocks mein se koi {MIN_SCORE}/8 score aur trend ✅ pe nahi hai.\n"
                "Sirf education ke liye.")
    lines = [f"📈 Swing Screener (data: {d})",
             f"{len(picks)} stock score {MIN_SCORE}/8 ya zyada + trend ✅ ({total} mein se):", ""]
    for n, r in enumerate(picks[:MAX_ALERTS], 1):
        missing = [c[:1] + " " + c[2:].split(" (")[0] for c in CHECKS if not r[c]]
        lines.append(f"{n}) {r['Stock']}  ₹{r['Price']}  score {r['Score /8']}/8")
        lines.append(f"   SL {r['Stop Loss']} | Target {r['Target']} | R:R {r['R:R']} | Qty {r['Qty']}")
        if missing:
            lines.append("   Kami: " + ", ".join(missing))
    if len(picks) > MAX_ALERTS:
        lines.append(f"... aur {len(picks) - MAX_ALERTS} stocks")
    lines += ["", "Sirf education ke liye, investment advice nahi. Entry se pehle chart khud dekho."]
    return "\n".join(lines)


def main():
    syms = get_symbols()
    if not syms:
        send("⚠️ Alert: NSE list nahi mili.")
        sys.exit(1)
    rows, last_date = [], None
    for i in range(0, len(syms), 100):
        ch = syms[i:i + 100]
        try:
            data = yf.download([x + ".NS" for x in ch], period="1y", interval="1d",
                               group_by="ticker", progress=False, auto_adjust=True, threads=True)
        except Exception as e:
            print("chunk fail:", type(e).__name__)
            continue
        if len(data):
            ld = data.index.max()
            last_date = ld if last_date is None or ld > last_date else last_date
        for sym in ch:
            try:
                res = analyse(sym, data)
                if res:
                    rows.append(res)
            except Exception:
                pass
        time.sleep(1)
    if len(rows) < 0.5 * len(syms):
        send(f"⚠️ Yahoo se data kam aaya ({len(rows)}/{len(syms)}). Aaj ka alert skip.")
        sys.exit(1)
    picks = sorted([r for r in rows if r["1 Trend (जरूरी)"] and r["Score /8"] >= MIN_SCORE],
                   key=lambda r: -r["Score /8"])
    send(build_message(picks, last_date, len(rows)))


if __name__ == "__main__":
    main()
