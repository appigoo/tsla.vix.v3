import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="📈 智能股市監控系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .signal-buy {
        background: linear-gradient(135deg, #00c851, #007e33);
        color: white; padding: 14px 20px; border-radius: 12px;
        font-weight: bold; font-size: 18px; text-align: center;
        box-shadow: 0 4px 15px rgba(0,200,81,0.4);
    }
    .signal-sell {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white; padding: 14px 20px; border-radius: 12px;
        font-weight: bold; font-size: 18px; text-align: center;
        box-shadow: 0 4px 15px rgba(255,68,68,0.4);
    }
    .signal-neutral {
        background: linear-gradient(135deg, #ffbb33, #ff8800);
        color: white; padding: 14px 20px; border-radius: 12px;
        font-weight: bold; font-size: 18px; text-align: center;
        box-shadow: 0 4px 15px rgba(255,187,51,0.4);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────
def send_telegram(message: str) -> bool:
    try:
        token   = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        url     = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": chat_id, "text": message,
            "parse_mode": "HTML", "disable_web_page_preview": True
        }, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        st.warning(f"Telegram 發送失敗: {e}")
        return False

# ─────────────────────────────────────────
# DATA
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# K-LINE PATTERN HELPERS (pure Python/numpy)
# ─────────────────────────────────────────
def cdl_hammer(o, h, l, c):
    body = abs(c - o); rng = h - l
    if rng == 0 or body == 0: return 0
    lower = min(o, c) - l
    upper = h - max(o, c)
    return 100 if lower >= 2 * body and upper <= 0.2 * rng else 0

def cdl_shooting_star(o, h, l, c):
    body = abs(c - o); rng = h - l
    if rng == 0 or body == 0: return 0
    upper = h - max(o, c)
    lower = min(o, c) - l
    return -100 if upper >= 2 * body and lower <= 0.2 * rng else 0

def cdl_doji(o, h, l, c):
    rng = h - l
    return 100 if rng > 0 and abs(c - o) / rng < 0.1 else 0

def cdl_engulfing(o1, c1, o2, c2):
    if c1 < o1 and c2 > o2 and c2 > o1 and o2 < c1: return 100
    if c1 > o1 and c2 < o2 and c2 < o1 and o2 > c1: return -100
    return 0

def cdl_morning_star(o1,c1, o2,h2,l2,c2, o3,c3):
    rng2 = h2 - l2
    small = rng2 > 0 and abs(c2-o2)/rng2 < 0.35
    if c1 < o1 and small and c3 > o3 and c3 > (o1+c1)/2: return 100
    return 0

def cdl_evening_star(o1,c1, o2,h2,l2,c2, o3,c3):
    rng2 = h2 - l2
    small = rng2 > 0 and abs(c2-o2)/rng2 < 0.35
    if c1 > o1 and small and c3 < o3 and c3 < (o1+c1)/2: return -100
    return 0

def cdl_3soldiers(opens, closes):
    if all(closes[i] > opens[i] and closes[i] > closes[i-1] for i in range(1,3)): return 100
    return 0

def cdl_3crows(opens, closes):
    if all(closes[i] < opens[i] and closes[i] < closes[i-1] for i in range(1,3)): return -100
    return 0

def compute_cdl_patterns(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    O = df['Open'].values.astype(float)
    H = df['High'].values.astype(float)
    L = df['Low'].values.astype(float)
    C = df['Close'].values.astype(float)

    hammer    = np.zeros(n); shoot = np.zeros(n)
    doji      = np.zeros(n); engulf= np.zeros(n)
    mstar     = np.zeros(n); estar = np.zeros(n)
    soldiers  = np.zeros(n); crows = np.zeros(n)

    for i in range(n):
        hammer[i] = cdl_hammer(O[i], H[i], L[i], C[i])
        shoot[i]  = cdl_shooting_star(O[i], H[i], L[i], C[i])
        doji[i]   = cdl_doji(O[i], H[i], L[i], C[i])
        if i >= 1:
            engulf[i] = cdl_engulfing(O[i-1], C[i-1], O[i], C[i])
        if i >= 2:
            mstar[i]   = cdl_morning_star(O[i-2],C[i-2], O[i-1],H[i-1],L[i-1],C[i-1], O[i],C[i])
            estar[i]   = cdl_evening_star(O[i-2],C[i-2], O[i-1],H[i-1],L[i-1],C[i-1], O[i],C[i])
            soldiers[i]= cdl_3soldiers(O[i-2:i+1], C[i-2:i+1])
            crows[i]   = cdl_3crows(O[i-2:i+1], C[i-2:i+1])

    df['CDL_HAMMER']         = hammer
    df['CDL_SHOOTINGSTAR']   = shoot
    df['CDL_DOJI']           = doji
    df['CDL_ENGULFING']      = engulf
    df['CDL_MORNINGSTAR']    = mstar
    df['CDL_EVENINGSTAR']    = estar
    df['CDL_3WHITESOLDIERS'] = soldiers
    df['CDL_3BLACKCROWS']    = crows
    return df

# ─────────────────────────────────────────
# INDICATORS  (pandas-ta — pip installable)
# ─────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Trend
    d['EMA9']  = ta.ema(d['Close'], length=9)
    d['EMA21'] = ta.ema(d['Close'], length=21)
    d['EMA55'] = ta.ema(d['Close'], length=55)

    # MACD
    macd = ta.macd(d['Close'], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        d['MACD']        = macd.iloc[:, 0]
        d['MACD_hist']   = macd.iloc[:, 1]
        d['MACD_signal'] = macd.iloc[:, 2]

    # RSI
    d['RSI14'] = ta.rsi(d['Close'], length=14)

    # Bollinger Bands
    bb = ta.bbands(d['Close'], length=20, std=2)
    if bb is not None and not bb.empty:
        d['BB_lower'] = bb.iloc[:, 0]
        d['BB_mid']   = bb.iloc[:, 1]
        d['BB_upper'] = bb.iloc[:, 2]

    # ATR
    d['ATR14'] = ta.atr(d['High'], d['Low'], d['Close'], length=14)

    # ADX
    adx = ta.adx(d['High'], d['Low'], d['Close'], length=14)
    if adx is not None and not adx.empty:
        d['ADX'] = adx.iloc[:, 0]

    # Stochastic
    stoch = ta.stoch(d['High'], d['Low'], d['Close'], k=14, d=3, smooth_k=3)
    if stoch is not None and not stoch.empty:
        d['STOCH_K'] = stoch.iloc[:, 0]
        d['STOCH_D'] = stoch.iloc[:, 1]

    # OBV
    d['OBV'] = ta.obv(d['Close'], d['Volume'])

    # CCI
    d['CCI'] = ta.cci(d['High'], d['Low'], d['Close'], length=14)

    # SuperTrend
    st_res = ta.supertrend(d['High'], d['Low'], d['Close'], length=10, multiplier=3.0)
    if st_res is not None and not st_res.empty:
        for col in st_res.columns:
            if 'SUPERTd' in col:
                d['ST_direction'] = st_res[col]
            elif col.startswith('SUPERT_'):
                d['SuperTrend'] = st_res[col]

    # K-Line Patterns (pure Python)
    d = compute_cdl_patterns(d)
    return d

# ─────────────────────────────────────────
# SIGNAL ENGINE
# ─────────────────────────────────────────
def generate_signals(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 30:
        return {}
    row  = df.iloc[-1]
    prev = df.iloc[-2]
    signals = {}
    score = 0

    def safe(val, default=0.0):
        return default if pd.isna(val) else float(val)

    close = safe(row['Close'])

    # MACD
    m, ms, mh = safe(row.get('MACD')), safe(row.get('MACD_signal')), safe(row.get('MACD_hist'))
    pm, pms   = safe(prev.get('MACD')), safe(prev.get('MACD_signal'))
    cross_bull = pm < pms and m > ms
    cross_bear = pm > pms and m < ms
    signals['MACD'] = ("🟢 金叉" if cross_bull else "🔴 死叉" if cross_bear else
                       "🟡 多頭擴張" if mh > 0 else "🟡 空頭擴張" if mh < 0 else "⚪ 中性")
    score += 2 if cross_bull else (-2 if cross_bear else (1 if mh > 0 else (-1 if mh < 0 else 0)))

    # RSI
    rsi = safe(row.get('RSI14'), 50)
    signals['RSI'] = (f"🟢 超賣反彈 ({rsi:.1f})" if rsi < 30 else
                      f"🔴 超買回調 ({rsi:.1f})" if rsi > 70 else
                      f"⚪ 中性 ({rsi:.1f})")
    score += 2 if rsi < 30 else (-2 if rsi > 70 else 0)

    # EMA
    e9, e21, e55 = safe(row.get('EMA9')), safe(row.get('EMA21')), safe(row.get('EMA55'))
    pe9, pe21    = safe(prev.get('EMA9')), safe(prev.get('EMA21'))
    ema_bull = e9 > e21 > e55 > 0
    ema_bear = e9 < e21 < e55 > 0
    gc = pe9 < pe21 and e9 > e21
    dc = pe9 > pe21 and e9 < e21
    signals['EMA'] = ("🟢 多頭排列" if ema_bull else "🔴 空頭排列" if ema_bear else
                      "🟢 黃金交叉" if gc else "🔴 死亡交叉" if dc else "⚪ 糾結")
    score += 2 if (ema_bull or gc) else (-2 if (ema_bear or dc) else 0)

    # SuperTrend
    std  = safe(row.get('ST_direction', 0))
    pstd = safe(prev.get('ST_direction', 0))
    flip_bull = pstd == -1 and std == 1
    flip_bear = pstd == 1  and std == -1
    signals['SuperTrend'] = ("🟢 翻多訊號🔔" if flip_bull else "🔴 翻空訊號🔔" if flip_bear else
                              "🟢 多頭趨勢" if std == 1 else "🔴 空頭趨勢" if std == -1 else "⚪ N/A")
    score += 3 if (flip_bull or std == 1) else (-3 if (flip_bear or std == -1) else 0)

    # Bollinger Bands
    bbu = safe(row.get('BB_upper'), close)
    bbl = safe(row.get('BB_lower'), close)
    bb_pos = (close - bbl) / (bbu - bbl) * 100 if (bbu - bbl) > 0 else 50
    signals['BBANDS'] = (f"🟢 突破下軌 ({bb_pos:.0f}%)" if close < bbl else
                         f"🔴 突破上軌 ({bb_pos:.0f}%)" if close > bbu else
                         f"⚪ 帶內運行 ({bb_pos:.0f}%)")
    score += 2 if close < bbl else (-2 if close > bbu else 0)

    # ATR
    atr = safe(row.get('ATR14'), 1)
    atr_pct = atr / close * 100 if close > 0 else 0
    signals['ATR'] = f"{'高' if atr_pct > 3 else '中' if atr_pct > 1.5 else '低'}波動 ({atr_pct:.2f}%)"

    # ADX
    adx = safe(row.get('ADX'), 20)
    signals['ADX'] = f"{'強趨勢🔥' if adx > 40 else '趨勢中等' if adx > 25 else '震盪行情'} ({adx:.1f})"

    # Stochastic
    k, dv = safe(row.get('STOCH_K'), 50), safe(row.get('STOCH_D'), 50)
    pk, pdv = safe(prev.get('STOCH_K'), 50), safe(prev.get('STOCH_D'), 50)
    s_bull = pk < pdv and k > dv and k < 25
    s_bear = pk > pdv and k < dv and k > 75
    signals['Stochastic'] = (f"🟢 低位金叉 ({k:.1f}/{dv:.1f})" if s_bull else
                              f"🔴 高位死叉 ({k:.1f}/{dv:.1f})" if s_bear else
                              f"⚪ ({k:.1f}/{dv:.1f})")
    score += 2 if s_bull else (-2 if s_bear else 0)

    # OBV
    obv5 = df['OBV'].dropna().iloc[-5:]
    obv_up   = obv5.is_monotonic_increasing
    obv_down = obv5.is_monotonic_decreasing
    signals['OBV'] = ("🟢 量能持續流入" if obv_up else "🔴 量能持續流出" if obv_down else "⚪ 量能震盪")
    score += 1 if obv_up else (-1 if obv_down else 0)

    # CDL Patterns
    cdl_map = {
        'CDL_HAMMER':         ('錘頭',   True),
        'CDL_SHOOTINGSTAR':   ('流星線', False),
        'CDL_DOJI':           ('十字星', None),
        'CDL_ENGULFING':      ('吞噬',   None),
        'CDL_MORNINGSTAR':    ('晨星',   True),
        'CDL_EVENINGSTAR':    ('昏星',   False),
        'CDL_3WHITESOLDIERS': ('三白兵', True),
        'CDL_3BLACKCROWS':    ('三烏鴉', False),
    }
    cdl_parts = []
    for col, (name, bullish) in cdl_map.items():
        val = safe(row.get(col), 0)
        if val != 0:
            is_bull = val > 0 if bullish is None else bullish
            cdl_parts.append(f"{'🟢' if is_bull else '🔴'} {name}")
            score += 2 if is_bull else -2
    signals['K線形態'] = ", ".join(cdl_parts) if cdl_parts else "⚪ 無明顯形態"

    # Final
    signals['_score']       = score
    signals['_overall']     = 'BUY' if score >= 5 else ('SELL' if score <= -5 else 'HOLD')
    signals['_close']       = round(close, 4)
    signals['_stop_loss']   = round(close - 2 * atr, 2)
    signals['_take_profit'] = round(close + 3 * atr, 2)
    signals['_atr']         = round(atr, 4)
    signals['_rsi']         = round(rsi, 2)
    return signals

# ─────────────────────────────────────────
# TELEGRAM MESSAGE
# ─────────────────────────────────────────
def build_telegram_msg(ticker: str, signals: dict, name: str = "") -> str:
    overall = signals.get('_overall', 'HOLD')
    emoji   = {"BUY":"🟢 買入","SELL":"🔴 賣出","HOLD":"🟡 觀望"}[overall]
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""📊 <b>{name or ticker} ({ticker})</b>
🕐 {ts}

━━━━━━━━━━━━━━━━━━
💡 <b>交易建議：{emoji}</b>
📈 信號評分：{signals.get('_score', 0):+d} 分

━━━━━━━━━━━━━━━━━━
💰 當前價格：<b>${signals.get('_close')}</b>
🛑 建議停損：${signals.get('_stop_loss')}
🎯 獲利目標：${signals.get('_take_profit')}
📐 ATR(14)：{signals.get('_atr')}  |  RSI：{signals.get('_rsi')}

━━━━━━━━━━━━━━━━━━
🔍 <b>技術指標</b>
• MACD：{signals.get('MACD','')}
• RSI：{signals.get('RSI','')}
• EMA排列：{signals.get('EMA','')}
• SuperTrend：{signals.get('SuperTrend','')}
• 布林通道：{signals.get('BBANDS','')}
• ADX：{signals.get('ADX','')}
• Stochastic：{signals.get('Stochastic','')}
• OBV：{signals.get('OBV','')}
• K線形態：{signals.get('K線形態','')}

⚠️ 僅供參考，投資請自行判斷風險。""".strip()

# ─────────────────────────────────────────
# CHART
# ─────────────────────────────────────────
def plot_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.5,0.15,0.15,0.2],
        subplot_titles=(f"{ticker} K線 + EMA + 布林通道", "成交量", "RSI(14)", "MACD(12,26,9)")
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="K線",
        increasing_line_color='#ef5350', decreasing_line_color='#26a69a'
    ), row=1, col=1)

    if 'BB_upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], showlegend=False,
            line=dict(color='rgba(160,160,255,0.35)', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name="布林通道",
            fill='tonexty', fillcolor='rgba(160,160,255,0.06)',
            line=dict(color='rgba(160,160,255,0.35)', width=1)), row=1, col=1)

    for ema, clr, dash in [('EMA9','#f9ca24','solid'),('EMA21','#6ab04c','dot'),('EMA55','#e17055','dash')]:
        if ema in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ema], name=ema,
                line=dict(color=clr, width=1.2, dash=dash)), row=1, col=1)

    if 'ST_direction' in df.columns and 'SuperTrend' in df.columns:
        for val, clr, lbl in [(1,'#00b894','ST↑'),(-1,'#d63031','ST↓')]:
            sub = df[df['ST_direction'] == val]
            fig.add_trace(go.Scatter(x=sub.index, y=sub['SuperTrend'], name=lbl,
                mode='markers', marker=dict(color=clr, size=4)), row=1, col=1)

    vol_colors = ['#ef5350' if c >= o else '#26a69a' for c,o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="成交量",
        marker_color=vol_colors, opacity=0.7), row=2, col=1)

    if 'RSI14' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], name="RSI",
            line=dict(color='#a29bfe', width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,68,68,0.5)", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,200,81,0.5)",  row=3, col=1)
        fig.add_hrect(y0=70,y1=100, fillcolor="rgba(255,68,68,0.04)", layer="below", row=3, col=1)
        fig.add_hrect(y0=0, y1=30,  fillcolor="rgba(0,200,81,0.04)",  layer="below", row=3, col=1)

    if 'MACD' in df.columns:
        hist_clrs = ['#26a69a' if v >= 0 else '#ef5350' for v in df['MACD_hist'].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name="MACD柱",
            marker_color=hist_clrs, opacity=0.6), row=4, col=1)
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

# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
st.title("📈 智能股市監控系統")
st.caption("pandas-ta 技術指標 × 純Python K線形態辨識 × Telegram 即時推播 | ✅ Streamlit Cloud 相容")

# ─── SIDEBAR ───
with st.sidebar:
    st.header("⚙️ 監控設定")
    watchlist_raw = st.text_area("監控清單（每行一個代號）",
        value="AAPL\nTSLA\nNVDA\nMSFT\nSPY", height=150)
    watchlist = [s.strip().upper() for s in watchlist_raw.splitlines() if s.strip()]
    selected  = st.selectbox("主要分析標的", watchlist)
    period    = st.selectbox("資料期間", ['1mo','3mo','6mo','1y','2y'], index=2)
    interval  = st.selectbox("K棒週期",  ['1d','1wk','1h'], index=0)

    st.divider()
    st.subheader("📡 Telegram")
    has_tg = ("TELEGRAM_BOT_TOKEN" in st.secrets and "TELEGRAM_CHAT_ID" in st.secrets)
    st.info("✅ Telegram 已設定" if has_tg else "❌ 未設定 — 請配置 secrets")
    with st.expander("設定方法"):
        st.code('# .streamlit/secrets.toml\nTELEGRAM_BOT_TOKEN = "xxx"\nTELEGRAM_CHAT_ID   = "xxx"', language="toml")

    st.divider()
    signal_filter = st.multiselect("自動推播訊號", ["BUY","SELL","HOLD"], default=["BUY","SELL"])
    scan_btn = st.button("🔍 掃描全部清單", use_container_width=True, type="primary")
    test_btn = st.button("📤 發送測試訊息",  use_container_width=True)
    if test_btn:
        ok = send_telegram("📊 <b>智能股市監控系統</b>\n✅ Telegram 連線測試成功！")
        st.success("✅ 已送出！") if ok else st.error("❌ 失敗，請檢查 Token/Chat ID")

# ─── TABS ───
tab1, tab2, tab3 = st.tabs(["📊 技術分析", "🔍 批量掃描", "📋 訊號記錄"])

# ══════════════════ TAB 1 ══════════════════
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
        with st.spinner("計算技術指標中…"):
            df = compute_indicators(df)
        signals = generate_signals(df)
        if not signals:
            st.warning("資料不足，請選擇較長的期間。")
        else:
            row  = df.iloc[-1]
            pc   = df.iloc[-2]['Close'] if len(df) > 1 else row['Close']
            chg  = (row['Close'] - pc) / pc * 100

            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("收盤價",  f"${row['Close']:.2f}", f"{chg:+.2f}%")
            m2.metric("RSI(14)", f"{row.get('RSI14', 0):.1f}")
            m3.metric("ATR(14)", f"{row.get('ATR14', 0):.3f}")
            m4.metric("ADX(14)", f"{row.get('ADX', 0):.1f}")
            m5.metric("MACD柱",  f"{row.get('MACD_hist', 0):.4f}")

            st.divider()

            overall = signals['_overall']
            css = {'BUY':'signal-buy','SELL':'signal-sell','HOLD':'signal-neutral'}[overall]
            lbl = {'BUY':'📈 建議買入','SELL':'📉 建議賣出','HOLD':'⏸ 建議觀望'}[overall]

            cs, ct = st.columns([1,2])
            with cs:
                st.markdown(f'<div class="{css}">{lbl}<br>評分：{signals["_score"]:+d}</div>',
                            unsafe_allow_html=True)
                st.write("")
                if st.button(f"📤 推播 {selected} 到 Telegram", use_container_width=True):
                    try: nm = yf.Ticker(selected).info.get('shortName', selected)
                    except: nm = selected
                    ok = send_telegram(build_telegram_msg(selected, signals, nm))
                    st.success("✅ 已送出！") if ok else st.error("❌ 失敗")
            with ct:
                st.markdown("**💰 ATR 動態停損建議**")
                t1,t2,t3 = st.columns(3)
                t1.metric("進場價", f"${signals['_close']}")
                t2.metric("🛑 停損", f"${signals['_stop_loss']}",
                          f"{(signals['_stop_loss']/signals['_close']-1)*100:.1f}%", delta_color="inverse")
                t3.metric("🎯 目標", f"${signals['_take_profit']}",
                          f"{(signals['_take_profit']/signals['_close']-1)*100:.1f}%")
                rr = (signals['_take_profit']-signals['_close']) / max(signals['_close']-signals['_stop_loss'],0.0001)
                st.caption(f"📐 風報比：**{rr:.2f}**（2×ATR 停損 / 3×ATR 目標）")

            st.divider()
            st.plotly_chart(plot_chart(df, selected), use_container_width=True)
            st.divider()

            st.subheader("🔬 指標詳情")
            ind_items = [
                ("MACD","MACD"),("RSI(14)","RSI"),("EMA排列","EMA"),
                ("SuperTrend","SuperTrend"),("布林通道","BBANDS"),("ATR波動","ATR"),
                ("ADX強度","ADX"),("Stochastic","Stochastic"),("OBV量能","OBV"),
            ]
            cols = st.columns(3)
            for i,(label,key) in enumerate(ind_items):
                with cols[i%3]:
                    st.markdown(f"**{label}**")
                    st.write(signals.get(key,''))

            st.divider()
            st.subheader("🕯 K線形態辨識（純Python實作）")
            cdl = signals.get('K線形態','')
            st.success(cdl) if cdl and '無明顯' not in cdl else st.info("⚪ 目前無明顯K線形態訊號")

# ══════════════════ TAB 2 ══════════════════
with tab2:
    st.subheader("🔍 批量掃描")
    if scan_btn or st.button("▶ 開始掃描", use_container_width=True):
        results = []
        prog = st.progress(0, text="掃描中…")
        tg_sent = 0
        for idx, ticker in enumerate(watchlist):
            prog.progress((idx+1)/len(watchlist), text=f"分析 {ticker}…")
            df_t = fetch_data(ticker, period, interval)
            if df_t.empty or len(df_t) < 30: continue
            df_t = compute_indicators(df_t)
            sigs = generate_signals(df_t)
            if not sigs: continue
            r = df_t.iloc[-1]; p = df_t.iloc[-2]['Close'] if len(df_t) > 1 else r['Close']
            results.append({
                '代號': ticker, '收盤價': round(r['Close'],2),
                '漲跌%': round((r['Close']-p)/p*100,2),
                '訊號': sigs['_overall'], '評分': sigs['_score'],
                'RSI': round(r.get('RSI14',0),1),
                'MACD': sigs.get('MACD',''), 'EMA': sigs.get('EMA',''),
                'SuperTrend': sigs.get('SuperTrend',''), 'K線形態': sigs.get('K線形態',''),
                '停損': sigs['_stop_loss'], '目標': sigs['_take_profit'],
            })
            if sigs['_overall'] in signal_filter:
                try: nm = yf.Ticker(ticker).info.get('shortName', ticker)
                except: nm = ticker
                if send_telegram(build_telegram_msg(ticker, sigs, nm)):
                    tg_sent += 1
        prog.empty()
        if results:
            df_res = pd.DataFrame(results).sort_values('評分', ascending=False)
            def csig(v):
                return {'BUY':'background-color:#00c85133;color:#00c851',
                        'SELL':'background-color:#ff444433;color:#ff4444',
                        'HOLD':'background-color:#ffbb3333;color:#ffbb33'}.get(v,'')
            st.dataframe(df_res.style.applymap(csig, subset=['訊號']),
                         use_container_width=True, hide_index=True)
            if tg_sent: st.success(f"✅ 已推播 {tg_sent} 個訊號到 Telegram")
            b = sum(1 for r in results if r['訊號']=='BUY')
            s = sum(1 for r in results if r['訊號']=='SELL')
            h = sum(1 for r in results if r['訊號']=='HOLD')
            c1,c2,c3 = st.columns(3)
            c1.metric("🟢 買入",b); c2.metric("🔴 賣出",s); c3.metric("🟡 觀望",h)
        else:
            st.warning("未能取得任何資料。")

# ══════════════════ TAB 3 ══════════════════
with tab3:
    st.subheader("📋 本次會話訊號記錄")
    if 'log' not in st.session_state:
        st.session_state.log = []
    df_l = fetch_data(selected, period, interval)
    if not df_l.empty and len(df_l) >= 30:
        df_l  = compute_indicators(df_l)
        sigs_l = generate_signals(df_l)
        if sigs_l:
            entry = {'時間': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     '代號': selected, '價格': sigs_l['_close'],
                     '訊號': sigs_l['_overall'], '評分': sigs_l['_score'],
                     'RSI': sigs_l['_rsi'], 'K線形態': sigs_l.get('K線形態','')}
            if not st.session_state.log or st.session_state.log[-1]['時間'] != entry['時間']:
                st.session_state.log.append(entry)
    if st.session_state.log:
        st.dataframe(pd.DataFrame(st.session_state.log[::-1]),
                     use_container_width=True, hide_index=True)
        if st.button("🗑 清除記錄"): st.session_state.log = []; st.rerun()
    else:
        st.info("尚無記錄。切換到「技術分析」標籤即可自動記錄。")
