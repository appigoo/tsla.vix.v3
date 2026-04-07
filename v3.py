import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════
def init_state():
    defaults = {
        'log':              [],
        'auto_scanning':    False,
        'scan_interval_s':  300,        # seconds
        'last_scan_time':   0.0,        # epoch
        'scan_results':     [],
        'next_scan_in':     0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="📈 智能股市監控系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.signal-buy {
    background: linear-gradient(135deg,#00c851,#007e33);
    color:white;padding:14px 20px;border-radius:12px;
    font-weight:bold;font-size:18px;text-align:center;
    box-shadow:0 4px 15px rgba(0,200,81,.4);
}
.signal-sell {
    background: linear-gradient(135deg,#ff4444,#cc0000);
    color:white;padding:14px 20px;border-radius:12px;
    font-weight:bold;font-size:18px;text-align:center;
    box-shadow:0 4px 15px rgba(255,68,68,.4);
}
.signal-neutral {
    background: linear-gradient(135deg,#ffbb33,#ff8800);
    color:white;padding:14px 20px;border-radius:12px;
    font-weight:bold;font-size:18px;text-align:center;
    box-shadow:0 4px 15px rgba(255,187,51,.4);
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════
def send_telegram(message: str) -> bool:
    try:
        token   = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10
        )
        return resp.status_code == 200
    except Exception as e:
        st.warning(f"Telegram 發送失敗: {e}")
        return False

# ══════════════════════════════════════════════════════
# DATA FETCH
# ══════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def fetch_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df.dropna()
    except Exception as e:
        st.error(f"資料獲取失敗: {e}")
        return pd.DataFrame()

# ══════════════════════════════════════════════════════
# PURE-PYTHON / PURE-PANDAS INDICATORS
# (zero C-extension dependencies)
# ══════════════════════════════════════════════════════
def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()

def calc_macd(close, fast=12, slow=26, signal=9):
    macd_line   = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_rsi(close, n=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/n, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_atr(high, low, close, n=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def calc_bbands(close, n=20, std_mult=2):
    mid   = sma(close, n)
    sigma = close.rolling(n).std()
    return mid + std_mult * sigma, mid, mid - std_mult * sigma

def calc_adx(high, low, close, n=14):
    tr    = calc_atr(high, low, close, n)
    up    = high.diff()
    down  = (-low.diff())
    pdm   = np.where((up > down) & (up > 0), up, 0.0)
    ndm   = np.where((down > up) & (down > 0), down, 0.0)
    pdm_s = pd.Series(pdm, index=high.index).ewm(alpha=1/n, adjust=False).mean()
    ndm_s = pd.Series(ndm, index=high.index).ewm(alpha=1/n, adjust=False).mean()
    pdi   = 100 * pdm_s / tr
    ndi   = 100 * ndm_s / tr
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx   = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx

def calc_stoch(high, low, close, k=14, d=3):
    low_k  = low.rolling(k).min()
    high_k = high.rolling(k).max()
    stoch_k = 100 * (close - low_k) / (high_k - low_k).replace(0, np.nan)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d

def calc_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def calc_supertrend(high, low, close, n=10, mult=3.0):
    atr   = calc_atr(high, low, close, n)
    hl2   = (high + low) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    st    = pd.Series(np.nan, index=close.index)
    dire  = pd.Series(0,      index=close.index)

    for i in range(1, len(close)):
        if pd.isna(atr.iloc[i]):
            continue
        # adjust bands
        upper.iloc[i] = upper.iloc[i] if close.iloc[i-1] > upper.iloc[i-1] else min(upper.iloc[i], upper.iloc[i-1])
        lower.iloc[i] = lower.iloc[i] if close.iloc[i-1] < lower.iloc[i-1] else max(lower.iloc[i], lower.iloc[i-1])
        # direction
        if dire.iloc[i-1] == 1:
            dire.iloc[i] = 1 if close.iloc[i] > lower.iloc[i] else -1
        else:
            dire.iloc[i] = -1 if close.iloc[i] < upper.iloc[i] else 1
        st.iloc[i] = lower.iloc[i] if dire.iloc[i] == 1 else upper.iloc[i]

    return st, dire

# ══════════════════════════════════════════════════════
# K-LINE PATTERN RECOGNITION  (pure Python)
# ══════════════════════════════════════════════════════
def cdl_hammer(o, h, l, c):
    body = abs(c - o); rng = h - l
    if rng == 0 or body == 0: return 0
    return 100 if (min(o,c)-l >= 2*body and h-max(o,c) <= 0.2*rng) else 0

def cdl_shooting_star(o, h, l, c):
    body = abs(c - o); rng = h - l
    if rng == 0 or body == 0: return 0
    return -100 if (h-max(o,c) >= 2*body and min(o,c)-l <= 0.2*rng) else 0

def cdl_doji(o, h, l, c):
    rng = h - l
    return 100 if rng > 0 and abs(c-o)/rng < 0.1 else 0

def cdl_engulfing(o1, c1, o2, c2):
    if c1 < o1 and c2 > o2 and c2 > o1 and o2 < c1: return  100
    if c1 > o1 and c2 < o2 and c2 < o1 and o2 > c1: return -100
    return 0

def cdl_morning_star(o1,c1, o2,h2,l2,c2, o3,c3):
    rng2 = h2-l2
    if rng2 > 0 and abs(c2-o2)/rng2 < 0.35 and c1<o1 and c3>o3 and c3>(o1+c1)/2: return 100
    return 0

def cdl_evening_star(o1,c1, o2,h2,l2,c2, o3,c3):
    rng2 = h2-l2
    if rng2 > 0 and abs(c2-o2)/rng2 < 0.35 and c1>o1 and c3<o3 and c3<(o1+c1)/2: return -100
    return 0

def cdl_3white_soldiers(O, C):  # O,C are arrays of 3
    if all(C[i]>O[i] and C[i]>C[i-1] for i in range(1,3)): return 100
    return 0

def cdl_3black_crows(O, C):
    if all(C[i]<O[i] and C[i]<C[i-1] for i in range(1,3)): return -100
    return 0

def cdl_harami(o1, c1, o2, c2):
    if c1<o1 and c2>o2 and o2>c1 and c2<o1: return  100
    if c1>o1 and c2<o2 and o2<c1 and c2>o1: return -100
    return 0

def cdl_breakaway(O, H, L, C):
    """
    脫離形態 (CDLBREAKAWAY) — 5根K棒反轉型態
    ─────────────────────────────────────────
    看漲脫離（下跌趨勢反轉向上）：
      K1: 長黑K（空頭趨勢延伸）
      K2: 向下跳空的黑K（gap down，O2 < C1）
      K3, K4: 小實體震盪K（可紅可黑，實體 < K1實體*0.6）
      K5: 長紅K，收盤價落在 C1 與 O2 之間
          即：C1 < C5 < O2（收回跳空區間）

    看跌脫離（上漲趨勢反轉向下）：
      K1: 長紅K（多頭趨勢延伸）
      K2: 向上跳空的紅K（gap up，O2 > C1）
      K3, K4: 小實體震盪K
      K5: 長黑K，收盤價落在 C1 與 O2 之間
          即：O2 < C5 < C1（收回跳空區間）

    回傳 100（看漲）/ -100（看跌）/ 0（無形態）
    """
    o1,o2,o3,o4,o5 = O
    h1,h2,h3,h4,h5 = H
    l1,l2,l3,l4,l5 = L
    c1,c2,c3,c4,c5 = C

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    body3 = abs(c3 - o3)
    body4 = abs(c4 - o4)
    body5 = abs(c5 - o5)

    if body1 == 0:
        return 0

    small_body_thresh = body1 * 0.6   # K3, K4 must be noticeably smaller

    # ── 看漲脫離 ──
    # K1: long bearish
    k1_bear  = c1 < o1 and body1 > 0
    # K2: gap-down bearish (open below K1 close)
    k2_gap_d = o2 < c1 and c2 < o2
    # K3, K4: small bodies (震盪)
    k3_small = body3 < small_body_thresh
    k4_small = body4 < small_body_thresh
    # K5: long bullish, close between C1 and O2 (fills into gap)
    k5_bull  = c5 > o5 and body5 >= body1 * 0.7 and c1 < c5 < o2

    if k1_bear and k2_gap_d and k3_small and k4_small and k5_bull:
        return 100

    # ── 看跌脫離 ──
    # K1: long bullish
    k1_bull  = c1 > o1 and body1 > 0
    # K2: gap-up bullish (open above K1 close)
    k2_gap_u = o2 > c1 and c2 > o2
    # K3, K4: small bodies
    k3_small2 = body3 < small_body_thresh
    k4_small2 = body4 < small_body_thresh
    # K5: long bearish, close between O2 and C1 (fills into gap)
    k5_bear  = c5 < o5 and body5 >= body1 * 0.7 and o2 < c5 < c1

    if k1_bull and k2_gap_u and k3_small2 and k4_small2 and k5_bear:
        return -100

    return 0

def cdl_ladder_bottom(O, H, L, C):
    """
    梯底形態 (CDLLADDERBOTTOM) — 5根K棒底部反轉
    ──────────────────────────────────────────────
    K1~K3: 連續三根黑K（綠K）
      - 每根開盤價低於前一根開盤價（逐步下降）
      - 每根收盤價低於前一根收盤價（持續下跌）
    K4: 倒錘頭形態
      - 黑K或小實體
      - 上影線長（≥ 實體 × 2）
      - 下影線短（≤ 實體 × 0.2 × 全幅）
    K5: 強勢紅K（多頭確認）
      - 開盤價高於 K4 開盤價
      - 收盤價高於前四根 K 棒的最高價
    回傳 100（看漲反轉）/ 0（無形態）
    """
    o1,o2,o3,o4,o5 = O
    h1,h2,h3,h4,h5 = H
    l1,l2,l3,l4,l5 = L
    c1,c2,c3,c4,c5 = C

    # K1~K3：三根黑K，開盤與收盤逐步走低
    three_bear = (
        c1 < o1 and c2 < o2 and c3 < o3 and   # 都是黑K
        o2 < o1 and o3 < o2 and                 # 開盤逐步低
        c2 < c1 and c3 < c2                     # 收盤逐步低
    )
    if not three_bear:
        return 0

    # K4：倒錘頭（上影線長、下影線短、小實體）
    body4        = abs(c4 - o4)
    rng4         = h4 - l4
    upper4       = h4 - max(o4, c4)
    lower4       = min(o4, c4) - l4
    inv_hammer4  = (
        rng4 > 0 and
        upper4 >= max(body4, rng4 * 0.4) and    # 上影線要夠長
        lower4 <= rng4 * 0.25                    # 下影線要短
    )
    if not inv_hammer4:
        return 0

    # K5：紅K，開盤 > K4 開盤，收盤 > 前四根最高
    prev4_high = max(h1, h2, h3, h4)
    k5_confirm = (
        c5 > o5 and          # 紅K
        o5 > o4 and          # 開盤高於K4開盤
        c5 > prev4_high      # 收盤突破前四根最高
    )
    if not k5_confirm:
        return 0

    return 100

def add_cdl_patterns(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    O = df['Open'].values.astype(float)
    H = df['High'].values.astype(float)
    L = df['Low'].values.astype(float)
    C = df['Close'].values.astype(float)

    p = {k: np.zeros(n) for k in [
        'CDL_HAMMER','CDL_SHOOTINGSTAR','CDL_DOJI',
        'CDL_ENGULFING','CDL_MORNINGSTAR','CDL_EVENINGSTAR',
        'CDL_3WHITESOLDIERS','CDL_3BLACKCROWS','CDL_HARAMI',
        'CDL_BREAKAWAY','CDL_LADDERBOTTOM',
    ]}

    for i in range(n):
        p['CDL_HAMMER'][i]       = cdl_hammer(O[i],H[i],L[i],C[i])
        p['CDL_SHOOTINGSTAR'][i] = cdl_shooting_star(O[i],H[i],L[i],C[i])
        p['CDL_DOJI'][i]         = cdl_doji(O[i],H[i],L[i],C[i])
        if i >= 1:
            p['CDL_ENGULFING'][i] = cdl_engulfing(O[i-1],C[i-1],O[i],C[i])
            p['CDL_HARAMI'][i]    = cdl_harami(O[i-1],C[i-1],O[i],C[i])
        if i >= 2:
            p['CDL_MORNINGSTAR'][i]    = cdl_morning_star(O[i-2],C[i-2],O[i-1],H[i-1],L[i-1],C[i-1],O[i],C[i])
            p['CDL_EVENINGSTAR'][i]    = cdl_evening_star(O[i-2],C[i-2],O[i-1],H[i-1],L[i-1],C[i-1],O[i],C[i])
            p['CDL_3WHITESOLDIERS'][i] = cdl_3white_soldiers(O[i-2:i+1],C[i-2:i+1])
            p['CDL_3BLACKCROWS'][i]    = cdl_3black_crows(O[i-2:i+1],C[i-2:i+1])
        if i >= 4:
            p['CDL_BREAKAWAY'][i] = cdl_breakaway(
                O[i-4:i+1], H[i-4:i+1], L[i-4:i+1], C[i-4:i+1]
            )
            p['CDL_LADDERBOTTOM'][i] = cdl_ladder_bottom(
                O[i-4:i+1], H[i-4:i+1], L[i-4:i+1], C[i-4:i+1]
            )

    for k, v in p.items():
        df[k] = v
    return df

# ══════════════════════════════════════════════════════
# COMPUTE ALL INDICATORS
# ══════════════════════════════════════════════════════
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l, v = d['Close'], d['High'], d['Low'], d['Volume']

    d['EMA9'],  d['EMA21'], d['EMA55'] = ema(c,9), ema(c,21), ema(c,55)
    d['MACD'], d['MACD_signal'], d['MACD_hist'] = calc_macd(c)
    d['RSI14'] = calc_rsi(c)
    d['BB_upper'], d['BB_mid'], d['BB_lower'] = calc_bbands(c)
    d['ATR14'] = calc_atr(h, l, c)
    d['ADX']   = calc_adx(h, l, c)
    d['STOCH_K'], d['STOCH_D'] = calc_stoch(h, l, c)
    d['OBV']   = calc_obv(c, v)
    d['SuperTrend'], d['ST_direction'] = calc_supertrend(h, l, c)
    d = add_cdl_patterns(d)
    return d

# ══════════════════════════════════════════════════════
# SIGNAL ENGINE
# ══════════════════════════════════════════════════════
def generate_signals(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 30:
        return {}

    def safe(val, default=0.0):
        try:
            f = float(val)
            return default if np.isnan(f) else f
        except:
            return default

    row   = df.iloc[-1]
    prev  = df.iloc[-2]
    sigs  = {}
    score = 0
    close = safe(row['Close'])

    # MACD
    m,ms,mh = safe(row['MACD']),safe(row['MACD_signal']),safe(row['MACD_hist'])
    pm,pms  = safe(prev['MACD']),safe(prev['MACD_signal'])
    cb = pm<pms and m>ms;  cd = pm>pms and m<ms
    sigs['MACD'] = ("🟢 金叉" if cb else "🔴 死叉" if cd else
                    "🟡 多頭擴張" if mh>0 else "🟡 空頭擴張" if mh<0 else "⚪ 中性")
    score += 2 if cb else (-2 if cd else (1 if mh>0 else (-1 if mh<0 else 0)))

    # RSI
    rsi = safe(row['RSI14'], 50)
    sigs['RSI'] = (f"🟢 超賣反彈 ({rsi:.1f})" if rsi<30 else
                   f"🔴 超買回調 ({rsi:.1f})" if rsi>70 else f"⚪ 中性 ({rsi:.1f})")
    score += 2 if rsi<30 else (-2 if rsi>70 else 0)

    # EMA
    e9,e21,e55   = safe(row['EMA9']),safe(row['EMA21']),safe(row['EMA55'])
    pe9,pe21     = safe(prev['EMA9']),safe(prev['EMA21'])
    bull = e9>e21>e55>0;  bear = 0<e9<e21<e55
    gc = pe9<pe21 and e9>e21;  dc = pe9>pe21 and e9<e21
    sigs['EMA'] = ("🟢 多頭排列" if bull else "🔴 空頭排列" if bear else
                   "🟢 黃金交叉" if gc else "🔴 死亡交叉" if dc else "⚪ 糾結")
    score += 2 if (bull or gc) else (-2 if (bear or dc) else 0)

    # SuperTrend
    std  = safe(row['ST_direction'])
    pstd = safe(prev['ST_direction'])
    fb = pstd==-1 and std==1;  fs = pstd==1 and std==-1
    sigs['SuperTrend'] = ("🟢 翻多訊號🔔" if fb else "🔴 翻空訊號🔔" if fs else
                          "🟢 多頭趨勢" if std==1 else "🔴 空頭趨勢" if std==-1 else "⚪ N/A")
    score += 3 if (fb or std==1) else (-3 if (fs or std==-1) else 0)

    # Bollinger Bands
    bbu = safe(row['BB_upper'], close)
    bbl = safe(row['BB_lower'], close)
    bb_pos = (close-bbl)/(bbu-bbl)*100 if (bbu-bbl)>0 else 50
    sigs['BBANDS'] = (f"🟢 突破下軌 ({bb_pos:.0f}%)" if close<bbl else
                      f"🔴 突破上軌 ({bb_pos:.0f}%)" if close>bbu else
                      f"⚪ 帶內運行 ({bb_pos:.0f}%)")
    score += 2 if close<bbl else (-2 if close>bbu else 0)

    # ATR
    atr     = safe(row['ATR14'], 1)
    atr_pct = atr/close*100 if close>0 else 0
    sigs['ATR'] = f"{'高' if atr_pct>3 else '中' if atr_pct>1.5 else '低'}波動 ({atr_pct:.2f}%)"

    # ADX
    adx = safe(row['ADX'], 20)
    sigs['ADX'] = f"{'強趨勢🔥' if adx>40 else '趨勢中等' if adx>25 else '震盪行情'} ({adx:.1f})"

    # Stochastic
    k,dv   = safe(row['STOCH_K'],50), safe(row['STOCH_D'],50)
    pk,pdv = safe(prev['STOCH_K'],50),safe(prev['STOCH_D'],50)
    sb = pk<pdv and k>dv and k<25;  ss = pk>pdv and k<dv and k>75
    sigs['Stochastic'] = (f"🟢 低位金叉 ({k:.1f}/{dv:.1f})" if sb else
                          f"🔴 高位死叉 ({k:.1f}/{dv:.1f})" if ss else f"⚪ ({k:.1f}/{dv:.1f})")
    score += 2 if sb else (-2 if ss else 0)

    # OBV
    obv5 = df['OBV'].dropna().iloc[-5:]
    obv_up   = bool(obv5.is_monotonic_increasing)
    obv_down = bool(obv5.is_monotonic_decreasing)
    sigs['OBV'] = ("🟢 量能持續流入" if obv_up else "🔴 量能持續流出" if obv_down else "⚪ 量能震盪")
    score += 1 if obv_up else (-1 if obv_down else 0)

    # CDL Patterns
    cdl_map = {
        'CDL_HAMMER':         ('錘頭',      True),
        'CDL_SHOOTINGSTAR':   ('流星線',    False),
        'CDL_DOJI':           ('十字星',    None),
        'CDL_ENGULFING':      ('吞噬',      None),
        'CDL_MORNINGSTAR':    ('晨星',      True),
        'CDL_EVENINGSTAR':    ('昏星',      False),
        'CDL_3WHITESOLDIERS': ('三白兵',    True),
        'CDL_3BLACKCROWS':    ('三烏鴉',    False),
        'CDL_HARAMI':         ('孕線',      None),
        'CDL_BREAKAWAY':      ('脫離形態',  None),   # +100 看漲 / -100 看跌
        'CDL_LADDERBOTTOM':   ('梯底',      True),   # 只有看漲反轉
    }
    cdl_parts = []
    for col, (name, bullish) in cdl_map.items():
        val = safe(row.get(col, 0))
        if val != 0:
            is_bull = val > 0 if bullish is None else bullish
            cdl_parts.append(f"{'🟢' if is_bull else '🔴'} {name}")
            score += 2 if is_bull else -2
    sigs['K線形態'] = ", ".join(cdl_parts) if cdl_parts else "⚪ 無明顯形態"

    sigs['_score']       = score
    sigs['_overall']     = 'BUY' if score>=5 else ('SELL' if score<=-5 else 'HOLD')
    sigs['_close']       = round(close, 4)
    sigs['_stop_loss']   = round(close - 2*atr, 2)
    sigs['_take_profit'] = round(close + 3*atr, 2)
    sigs['_atr']         = round(atr, 4)
    sigs['_rsi']         = round(rsi, 2)
    return sigs

# ══════════════════════════════════════════════════════
# TELEGRAM MESSAGE
# ══════════════════════════════════════════════════════
def build_msg(ticker: str, sigs: dict, name: str = "") -> str:
    overall = sigs.get('_overall','HOLD')
    emoji   = {"BUY":"🟢 買入","SELL":"🔴 賣出","HOLD":"🟡 觀望"}[overall]
    return f"""📊 <b>{name or ticker} ({ticker})</b>
🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}

━━━━━━━━━━━━━━━━━━
💡 <b>交易建議：{emoji}</b>
📈 信號評分：{sigs.get('_score',0):+d} 分

━━━━━━━━━━━━━━━━━━
💰 當前價格：<b>${sigs.get('_close')}</b>
🛑 建議停損：${sigs.get('_stop_loss')}
🎯 獲利目標：${sigs.get('_take_profit')}
📐 ATR(14)：{sigs.get('_atr')}  |  RSI：{sigs.get('_rsi')}

━━━━━━━━━━━━━━━━━━
🔍 <b>技術指標</b>
• MACD：{sigs.get('MACD','')}
• RSI：{sigs.get('RSI','')}
• EMA排列：{sigs.get('EMA','')}
• SuperTrend：{sigs.get('SuperTrend','')}
• 布林通道：{sigs.get('BBANDS','')}
• ADX：{sigs.get('ADX','')}
• Stochastic：{sigs.get('Stochastic','')}
• OBV：{sigs.get('OBV','')}
• K線形態：{sigs.get('K線形態','')}

⚠️ 僅供參考，投資請自行判斷風險。""".strip()

# ══════════════════════════════════════════════════════
# CHART
# ══════════════════════════════════════════════════════
def plot_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.5,0.15,0.15,0.2],
        subplot_titles=(f"{ticker} K線","成交量","RSI(14)","MACD(12,26,9)")
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="K線",
        increasing_line_color='#ef5350', decreasing_line_color='#26a69a'
    ), row=1, col=1)

    if 'BB_upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], showlegend=False,
            line=dict(color='rgba(160,160,255,.35)', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name="BB",
            fill='tonexty', fillcolor='rgba(160,160,255,.07)',
            line=dict(color='rgba(160,160,255,.35)', width=1)), row=1, col=1)

    for col, clr, dash in [('EMA9','#f9ca24','solid'),('EMA21','#6ab04c','dot'),('EMA55','#e17055','dash')]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                line=dict(color=clr, width=1.2, dash=dash)), row=1, col=1)

    if 'ST_direction' in df.columns and 'SuperTrend' in df.columns:
        for d_val, clr, lbl in [(1,'#00b894','ST↑'),(-1,'#d63031','ST↓')]:
            sub = df[df['ST_direction']==d_val]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub.index, y=sub['SuperTrend'], name=lbl,
                    mode='markers', marker=dict(color=clr, size=4)), row=1, col=1)

    vc = ['#ef5350' if c>=o else '#26a69a' for c,o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Vol",
        marker_color=vc, opacity=0.7), row=2, col=1)

    if 'RSI14' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], name="RSI",
            line=dict(color='#a29bfe', width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,68,68,.5)", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,200,81,.5)",  row=3, col=1)

    if 'MACD' in df.columns:
        hc = ['#26a69a' if v>=0 else '#ef5350' for v in df['MACD_hist'].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name="Hist",
            marker_color=hc, opacity=0.6), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD",
            line=dict(color='#fdcb6e', width=1.5)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name="Signal",
            line=dict(color='#e17055', width=1.5, dash='dot')), row=4, col=1)

    fig.update_layout(
        height=820, template='plotly_dark',
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

# ══════════════════════════════════════════════════════
# SHARED BATCH SCAN FUNCTION
# ══════════════════════════════════════════════════════
def run_batch_scan(watchlist, period, interval, sig_filter, show_progress=True):
    """Scan all tickers, push Telegram, store results in session_state."""
    results = []
    tg_sent = 0
    prog    = st.progress(0, text="掃描中…") if show_progress else None

    for idx, ticker in enumerate(watchlist):
        if prog:
            prog.progress((idx + 1) / len(watchlist), text=f"分析 {ticker}…")
        df_t = fetch_data(ticker, period, interval)
        if df_t.empty or len(df_t) < 30:
            continue
        df_t = compute_indicators(df_t)
        sgs  = generate_signals(df_t)
        if not sgs:
            continue
        r = df_t.iloc[-1]
        p = df_t.iloc[-2]['Close'] if len(df_t) > 1 else r['Close']
        results.append({
            '代號':       ticker,
            '收盤價':     round(r['Close'], 2),
            '漲跌%':      round((r['Close'] - p) / p * 100, 2),
            '訊號':       sgs['_overall'],
            '評分':       sgs['_score'],
            'RSI':        round(float(r.get('RSI14', 0)), 1),
            'MACD':       sgs.get('MACD', ''),
            'EMA':        sgs.get('EMA', ''),
            'SuperTrend': sgs.get('SuperTrend', ''),
            'K線形態':    sgs.get('K線形態', ''),
            '停損':       sgs['_stop_loss'],
            '目標':       sgs['_take_profit'],
        })
        if sgs['_overall'] in sig_filter:
            try:    nm = yf.Ticker(ticker).info.get('shortName', ticker)
            except: nm = ticker
            if send_telegram(build_msg(ticker, sgs, nm)):
                tg_sent += 1

    if prog:
        prog.empty()

    st.session_state.scan_results   = results
    st.session_state.last_scan_time = time.time()
    return results, tg_sent

# ══════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════
st.title("📈 智能股市監控系統")
st.caption("純Python技術指標引擎 × 9種K線形態辨識 × Telegram即時推播 | ✅ Streamlit Cloud 零依賴安裝")

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ 監控設定")
    raw = st.text_area("監控清單（每行一個代號）",
                       value="TSLA\nUVXY\nUVIX\nNIO\nTSLL\nXPEV\nGLD\nMETA\nGOOGLE\nAAPL\nNVDA\nAMZN\nTSM\nMSFT", height=150)
    watchlist = [s.strip().upper() for s in raw.splitlines() if s.strip()]
    selected  = st.selectbox("主要分析標的", watchlist)
    period    = st.selectbox("資料期間", ['1mo','3mo','6mo','1y','2y'], index=2)
    interval  = st.selectbox("K棒週期",  ['1d','1wk','1h'], index=0)

    st.divider()
    st.subheader("📡 Telegram")
    has_tg = "TELEGRAM_BOT_TOKEN" in st.secrets and "TELEGRAM_CHAT_ID" in st.secrets
    st.info("✅ Telegram 已設定" if has_tg else "❌ 未設定 secrets")
    with st.expander("設定方法"):
        st.code('TELEGRAM_BOT_TOKEN = "your_token"\nTELEGRAM_CHAT_ID   = "your_chat_id"', language="toml")

    st.divider()
    sig_filter = st.multiselect("自動推播訊號", ["BUY","SELL","HOLD"], default=["BUY","SELL"])

    # ── Manual scan ──
    scan_btn = st.button("🔍 立即掃描全部清單", use_container_width=True, type="primary")
    test_btn = st.button("📤 發送測試訊息", use_container_width=True)
    if test_btn:
        ok = send_telegram("📊 <b>智能股市監控系統</b>\n✅ Telegram 連線測試成功！")
        st.success("✅ 已送出！") if ok else st.error("❌ 請確認 Token / Chat ID")

    st.divider()

    # ── Auto-scan controls ──
    st.subheader("⏱ 自動掃描排程")

    interval_options = {"1 分鐘": 60, "5 分鐘": 300, "15 分鐘": 900, "30 分鐘": 1800}
    chosen_label = st.radio(
        "掃描間隔",
        list(interval_options.keys()),
        index=1,
        horizontal=True,
    )
    st.session_state.scan_interval_s = interval_options[chosen_label]

    col_on, col_off = st.columns(2)
    with col_on:
        if st.button("▶ 啟動", use_container_width=True,
                     disabled=st.session_state.auto_scanning):
            st.session_state.auto_scanning  = True
            st.session_state.last_scan_time = 0.0
            st.rerun()
    with col_off:
        if st.button("⏹ 停止", use_container_width=True,
                     disabled=not st.session_state.auto_scanning):
            st.session_state.auto_scanning = False
            st.rerun()

    # ── Status badge ──
    if st.session_state.auto_scanning:
        elapsed   = time.time() - st.session_state.last_scan_time
        remaining = max(0, int(st.session_state.scan_interval_s - elapsed))
        m, s      = divmod(remaining, 60)
        last_ts   = (datetime.fromtimestamp(st.session_state.last_scan_time).strftime("%H:%M:%S")
                     if st.session_state.last_scan_time > 0 else "—")
        st.success(f"🟢 自動掃描中 | 下次：{m:02d}:{s:02d}")
        st.caption(f"上次掃描：{last_ts} | 間隔：{chosen_label}")
    else:
        st.info("⏸ 自動掃描未啟動")

tab1, tab2, tab3 = st.tabs(["📊 技術分析","🔍 批量掃描","📋 訊號記錄"])

# ══ TAB 1 ══
with tab1:
    ch, cr = st.columns([5,1])
    with ch: st.subheader(f"🔎 {selected} 深度分析")
    with cr:
        if st.button("🔄 刷新", use_container_width=True):
            st.cache_data.clear()

    df = fetch_data(selected, period, interval)
    if df.empty:
        st.error("無法獲取資料，請確認股票代號。")
    else:
        with st.spinner("計算中…"):
            df   = compute_indicators(df)
        sigs = generate_signals(df)

        if not sigs:
            st.warning("資料不足，請選擇較長期間。")
        else:
            row = df.iloc[-1]
            pc  = df.iloc[-2]['Close'] if len(df)>1 else row['Close']
            chg = (row['Close']-pc)/pc*100

            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("收盤價",  f"${row['Close']:.2f}", f"{chg:+.2f}%")
            c2.metric("RSI(14)", f"{float(row.get('RSI14',0)):.1f}")
            c3.metric("ATR(14)", f"{float(row.get('ATR14',0)):.3f}")
            c4.metric("ADX(14)", f"{float(row.get('ADX',0)):.1f}")
            c5.metric("MACD柱",  f"{float(row.get('MACD_hist',0)):.4f}")
            st.divider()

            overall = sigs['_overall']
            css = {'BUY':'signal-buy','SELL':'signal-sell','HOLD':'signal-neutral'}[overall]
            lbl = {'BUY':'📈 建議買入','SELL':'📉 建議賣出','HOLD':'⏸ 建議觀望'}[overall]

            cs, ct = st.columns([1,2])
            with cs:
                st.markdown(f'<div class="{css}">{lbl}<br>評分：{sigs["_score"]:+d}</div>',
                            unsafe_allow_html=True)
                st.write("")
                if st.button(f"📤 推播 {selected} 到 Telegram", use_container_width=True):
                    try:    nm = yf.Ticker(selected).info.get('shortName', selected)
                    except: nm = selected
                    ok = send_telegram(build_msg(selected, sigs, nm))
                    st.success("✅ 已送出！") if ok else st.error("❌ 失敗")
            with ct:
                st.markdown("**💰 ATR 動態停損建議**")
                t1,t2,t3 = st.columns(3)
                t1.metric("進場價", f"${sigs['_close']}")
                t2.metric("🛑 停損", f"${sigs['_stop_loss']}",
                          f"{(sigs['_stop_loss']/sigs['_close']-1)*100:.1f}%", delta_color="inverse")
                t3.metric("🎯 目標", f"${sigs['_take_profit']}",
                          f"{(sigs['_take_profit']/sigs['_close']-1)*100:.1f}%")
                rr = (sigs['_take_profit']-sigs['_close'])/max(sigs['_close']-sigs['_stop_loss'],0.0001)
                st.caption(f"📐 風報比：**{rr:.2f}**（2×ATR停損 / 3×ATR目標）")

            st.divider()
            st.plotly_chart(plot_chart(df, selected), use_container_width=True)
            st.divider()

            st.subheader("🔬 指標詳情")
            items = [("MACD","MACD"),("RSI(14)","RSI"),("EMA排列","EMA"),
                     ("SuperTrend","SuperTrend"),("布林通道","BBANDS"),("ATR波動","ATR"),
                     ("ADX強度","ADX"),("Stochastic","Stochastic"),("OBV量能","OBV")]
            cols = st.columns(3)
            for i,(lbl2,key) in enumerate(items):
                with cols[i%3]:
                    st.markdown(f"**{lbl2}**")
                    st.write(sigs.get(key,''))

            st.divider()
            st.subheader("🕯 K線形態辨識")
            cdl = sigs.get('K線形態','')
            st.success(cdl) if cdl and '無明顯' not in cdl else st.info("⚪ 目前無明顯K線形態訊號")

# ══ TAB 2 ══
with tab2:
    st.subheader("🔍 批量掃描")

    # Auto-scan status banner inside tab
    if st.session_state.auto_scanning:
        elapsed   = time.time() - st.session_state.last_scan_time
        remaining = max(0, int(st.session_state.scan_interval_s - elapsed))
        m2, s2    = divmod(remaining, 60)
        last_ts2  = (datetime.fromtimestamp(st.session_state.last_scan_time).strftime("%Y-%m-%d %H:%M:%S")
                     if st.session_state.last_scan_time > 0 else "尚未掃描")
        st.info(f"🟢 自動掃描已啟動 | 下次掃描倒數：**{m2:02d}:{s2:02d}** | 上次：{last_ts2}")

    trigger_scan = scan_btn or st.button("▶ 立即掃描", use_container_width=True)

    if trigger_scan:
        results, tg_sent = run_batch_scan(watchlist, period, interval, sig_filter, show_progress=True)
        if tg_sent:
            st.success(f"✅ 已推播 {tg_sent} 個訊號到 Telegram")

    # Display cached results (from manual or auto scan)
    results = st.session_state.get('scan_results', [])
    if results:
        last_ts_disp = (datetime.fromtimestamp(st.session_state.last_scan_time).strftime("%H:%M:%S")
                        if st.session_state.last_scan_time > 0 else "")
        if last_ts_disp:
            st.caption(f"📋 掃描結果（{last_ts_disp}）— 共 {len(results)} 檔")

        df_r = pd.DataFrame(results).sort_values('評分', ascending=False)

        def csig(v):
            return {'BUY':  'background-color:#00c85133;color:#00c851',
                    'SELL': 'background-color:#ff444433;color:#ff4444',
                    'HOLD': 'background-color:#ffbb3333;color:#ffbb33'}.get(v, '')

        #st.dataframe(df_r.drop(columns=[], errors='ignore').style.applymap(csig, subset=['訊號']),
         st.dataframe(df_r.drop(columns=[], errors='ignore').style.map(csig, subset=['訊號']),
                     use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 買入", sum(1 for r in results if r['訊號'] == 'BUY'))
        c2.metric("🔴 賣出", sum(1 for r in results if r['訊號'] == 'SELL'))
        c3.metric("🟡 觀望", sum(1 for r in results if r['訊號'] == 'HOLD'))
    elif not trigger_scan:
        st.info("點擊「立即掃描」或在側邊欄啟動自動掃描排程。")

# ══ TAB 3 ══
with tab3:
    st.subheader("📋 本次會話訊號記錄")
    if 'log' not in st.session_state:
        st.session_state.log = []
    df_l = fetch_data(selected, period, interval)
    if not df_l.empty and len(df_l)>=30:
        df_l  = compute_indicators(df_l)
        sgs_l = generate_signals(df_l)
        if sgs_l:
            e = {'時間': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 '代號': selected, '價格': sgs_l['_close'],
                 '訊號': sgs_l['_overall'], '評分': sgs_l['_score'],
                 'RSI':  sgs_l['_rsi'], 'K線形態': sgs_l.get('K線形態','')}
            if not st.session_state.log or st.session_state.log[-1]['時間'] != e['時間']:
                st.session_state.log.append(e)
    if st.session_state.log:
        st.dataframe(pd.DataFrame(st.session_state.log[::-1]),
                     use_container_width=True, hide_index=True)
        if st.button("🗑 清除記錄"): st.session_state.log=[]; st.rerun()
    else:
        st.info("尚無記錄。切換到「技術分析」標籤即可自動記錄。")

# ══════════════════════════════════════════════════════
# AUTO-SCAN LOOP  — must be LAST (blocks then reruns)
# ══════════════════════════════════════════════════════
if st.session_state.auto_scanning:
    elapsed   = time.time() - st.session_state.last_scan_time
    remaining = max(0, st.session_state.scan_interval_s - elapsed)

    if remaining <= 0:
        # Time to scan — do it silently (no spinner blocks UI on rerun)
        run_batch_scan(watchlist, period, interval, sig_filter, show_progress=False)
        st.cache_data.clear()
        st.rerun()
    else:
        # Sleep in small chunks so the countdown updates every 5 s
        sleep_s = min(5, remaining)
        time.sleep(sleep_s)
        st.rerun()
