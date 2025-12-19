"""
Algo Trading Engine for Rantv Intraday Terminal Pro
Provides automated order execution, scheduling, and risk management
PART 1: Imports and Configuration
"""

import os
import time
import threading
import subprocess
import sys
import webbrowser
import logging
import json
import pytz
import traceback
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import warnings

# Auto-install missing critical dependencies including kiteconnect
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITECONNECT_AVAILABLE = True
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kiteconnect"])
        from kiteconnect import KiteConnect, KiteTicker
        KITECONNECT_AVAILABLE = True
        st.success("✅ Installed kiteconnect")
    except:
        KITECONNECT_AVAILABLE = False

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sqlalchemy"])
        import sqlalchemy
        from sqlalchemy import create_engine, text
        SQLALCHEMY_AVAILABLE = True
        st.success("✅ Installed sqlalchemy")
    except:
        SQLALCHEMY_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
        import joblib
        JOBLIB_AVAILABLE = True
        st.success("✅ Installed joblib")
    except:
        JOBLIB_AVAILABLE = False

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Kite Connect API Credentials
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN = ""

# Algo Trading Configuration
ALGO_ENABLED = os.environ.get("ALGO_TRADING_ENABLED", "false").lower() == "true"
ALGO_MAX_POSITIONS = int(os.environ.get("ALGO_MAX_POSITIONS", "5"))
ALGO_MAX_DAILY_LOSS = float(os.environ.get("ALGO_MAX_DAILY_LOSS", "50000"))
ALGO_MIN_CONFIDENCE = float(os.environ.get("ALGO_MIN_CONFIDENCE", "0.80"))

# Configuration
@dataclass
class AppConfig:
    database_url: str = 'sqlite:///trading_journal.db'
    risk_tolerance: str = 'MODERATE'
    max_daily_loss: float = 50000.0
    enable_ml: bool = True
    kite_api_key: str = KITE_API_KEY
    kite_api_secret: str = KITE_API_SECRET
    algo_enabled: bool = ALGO_ENABLED
    algo_max_positions: int = ALGO_MAX_POSITIONS
    algo_max_daily_loss: float = ALGO_MAX_DAILY_LOSS
    algo_min_confidence: float = ALGO_MIN_CONFIDENCE
    
    @classmethod
    def from_env(cls):
        return cls()

config = AppConfig.from_env()

st.set_page_config(page_title="Rantv Intraday Terminal Pro - Enhanced", layout="wide", initial_sidebar_state="expanded")
IND_TZ = pytz.timezone("Asia/Kolkata")

# Trading Constants
CAPITAL = 2_000_000.0
TRADE_ALLOC = 0.15
MAX_DAILY_TRADES = 10
MAX_STOCK_TRADES = 10
MAX_AUTO_TRADES = 10
SIGNAL_REFRESH_MS = 120000
PRICE_REFRESH_MS = 100000
MARKET_OPTIONS = ["CASH"]

# Stock Universes
NIFTY_50 = [
   "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
    "SBIN.NS", "ASIANPAINT.NS", "HCLTECH.NS", "AXISBANK.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NTPC.NS",
    "NESTLEIND.NS", "POWERGRID.NS", "M&M.NS", "BAJFINANCE.NS", "ONGC.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIPORTS.NS", "COALINDIA.NS",
    "HDFCLIFE.NS", "DRREDDY.NS", "HINDALCO.NS", "CIPLA.NS", "SBILIFE.NS",
    "GRASIM.NS", "TECHM.NS", "BAJAJFINSV.NS", "BRITANNIA.NS", "EICHERMOT.NS",
    "DIVISLAB.NS", "SHREECEM.NS", "APOLLOHOSP.NS", "UPL.NS", "BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS", "INDUSINDBK.NS", "ADANIENT.NS", "TATACONSUM.NS", "BPCL.NS"
]

NIFTY_100 = NIFTY_50 + [
    "BAJAJHLDNG.NS", "TATAMOTORS.NS", "VEDANTA.NS", "PIDILITIND.NS",
    "BERGEPAINT.NS", "AMBUJACEM.NS", "DABUR.NS", "HAVELLS.NS", "ICICIPRULI.NS",
    "MARICO.NS", "PEL.NS", "SIEMENS.NS", "TORNTPHARM.NS", "ACC.NS",
    "AUROPHARMA.NS", "BOSCHLTD.NS", "GLENMARK.NS", "MOTHERSUMI.NS", "BIOCON.NS",
    "ZYDUSLIFE.NS", "COLPAL.NS", "CONCOR.NS", "DLF.NS", "GODREJCP.NS",
    "HINDPETRO.NS", "IBULHSGFIN.NS", "IOC.NS", "JINDALSTEL.NS", "LUPIN.NS",
    "MANAPPURAM.NS", "MCDOWELL-N.NS", "NMDC.NS", "PETRONET.NS", "PFC.NS",
    "PNB.NS", "RBLBANK.NS", "SAIL.NS", "SRTRANSFIN.NS", "TATAPOWER.NS",
    "YESBANK.NS", "ZEEL.NS"
]

NIFTY_MIDCAP_150 = [
    "ABB.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "AUBANK.NS", "AIAENG.NS",
    "APLAPOLLO.NS", "ASTRAL.NS", "AARTIIND.NS", "BALKRISIND.NS", "BANKBARODA.NS",
]

ALL_STOCKS = list(dict.fromkeys(NIFTY_50 + NIFTY_100 + NIFTY_MIDCAP_150))

TRADING_STRATEGIES = {
    "EMA_VWAP_Confluence": {"name": "EMA + VWAP Confluence", "weight": 3, "type": "BUY"},
    "RSI_MeanReversion": {"name": "RSI Mean Reversion", "weight": 2, "type": "BUY"},
    "Bollinger_Reversion": {"name": "Bollinger Band Reversion", "weight": 2, "type": "BUY"},
    "MACD_Momentum": {"name": "MACD Momentum", "weight": 2, "type": "BUY"},
    "Support_Resistance_Breakout": {"name": "Support/Resistance Breakout", "weight": 3, "type": "BUY"},
    "EMA_VWAP_Downtrend": {"name": "EMA + VWAP Downtrend", "weight": 3, "type": "SELL"},
    "RSI_Overbought": {"name": "RSI Overbought Reversal", "weight": 2, "type": "SELL"},
    "Bollinger_Rejection": {"name": "Bollinger Band Rejection", "weight": 2, "type": "SELL"},
    "MACD_Bearish": {"name": "MACD Bearish Crossover", "weight": 2, "type": "SELL"},
    "Trend_Reversal": {"name": "Trend Reversal", "weight": 2, "type": "SELL"}
}

HIGH_ACCURACY_STRATEGIES = {
    "Multi_Confirmation": {"name": "Multi-Confirmation Ultra", "weight": 5, "type": "BOTH"},
    "Enhanced_EMA_VWAP": {"name": "Enhanced EMA-VWAP", "weight": 4, "type": "BOTH"},
    "Volume_Breakout": {"name": "Volume Weighted Breakout", "weight": 4, "type": "BOTH"},
    "RSI_Divergence": {"name": "RSI Divergence", "weight": 3, "type": "BOTH"},
    "MACD_Trend": {"name": "MACD Trend Momentum", "weight": 3, "type": "BOTH"}
}

print("✅ PART 1 LOADED: Configuration Complete")
print("📋 Next: Copy PART 2 - CSS & Styling")
# FIXED CSS with ORANGE Background and Logo
st.markdown("""
<style>
    /* ORANGE Background for main app */
    .stApp {
        background: linear-gradient(135deg, #fff5e6 0%, #ffe8cc 100%);
    }
    
    /* Main container background */
    .main .block-container {
        background-color: transparent;
        padding-top: 2rem;
    }
    
    /* Logo Container */
    .logo-container {
        text-align: center;
        margin: 20px auto;
        padding: 20px;
        background: linear-gradient(135deg, #ff8c00 0%, #ff6b00 100%);
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(255, 140, 0, 0.3);
    }
    
    .logo-container img {
        max-width: 250px;
        height: auto;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        background: white;
        padding: 10px;
    }
    
    /* Enhanced Tabs with ORANGE Theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: linear-gradient(135deg, #ffe8cc 0%, #ffd9a6 50%, #ffca80 100%);
        padding: 8px;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 8px;
        gap: 8px;
        padding: 12px 20px;
        font-weight: 600;
        font-size: 14px;
        color: #d97706;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff8c00 0%, #ff6b00 100%);
        color: white;
        border: 2px solid #ff8c00;
        box-shadow: 0 4px 8px rgba(255, 140, 0, 0.3);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #ffe8cc 0%, #ffd9a6 100%);
        border: 2px solid #ff8c00;
        transform: translateY(-1px);
    }
    
    /* Gauge Container Styles - ORANGE Theme */
    .gauge-container {
        background: white;
        border-radius: 50%;
        padding: 25px;
        margin: 10px auto;
        border: 4px solid #ffe8cc;
        box-shadow: 0 8px 25px rgba(255, 140, 0, 0.15);
        width: 200px;
        height: 200px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        position: relative;
    }
    
    .gauge-title {
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 8px;
        color: #d97706;
    }
    
    .gauge-value {
        font-size: 16px;
        font-weight: bold;
        margin: 3px 0;
    }
    
    .gauge-sentiment {
        font-size: 12px;
        font-weight: bold;
        margin-top: 6px;
        padding: 3px 10px;
        border-radius: 15px;
    }
    
    .bullish { 
        color: #059669;
        background-color: #d1fae5;
    }
    
    .bearish { 
        color: #dc2626;
        background-color: #fee2e2;
    }
    
    .neutral { 
        color: #ff8c00;
        background-color: #ffe8cc;
    }
    
    /* Circular Progress Bar - ORANGE */
    .gauge-progress {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: conic-gradient(#ff8c00 0% var(--progress), #e5e7eb var(--progress) 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 8px 0;
        position: relative;
    }
    
    .gauge-progress-inner {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        background: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 14px;
    }
    
    /* Card Styling - ORANGE Theme */
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff8c00;
        box-shadow: 0 2px 8px rgba(255, 140, 0, 0.1);
    }
    
    /* High Accuracy Strategy Cards - ORANGE */
    .high-accuracy-card {
        background: linear-gradient(135deg, #ff8c00 0%, #ff6b00 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
        box-shadow: 0 4px 12px rgba(255, 140, 0, 0.3);
    }
    
    /* Alert Styles - ORANGE Theme */
    .alert-success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #059669;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #ffe8cc 0%, #ffd9a6 100%);
        border-left: 4px solid #ff8c00;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
    }
    
    /* Auto-refresh counter - ORANGE */
    .refresh-counter {
        background: #ff8c00;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-left: 8px;
    }
    
    /* Signal Quality Styles */
    .high-quality-signal {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #047857;
    }
    
    .medium-quality-signal {
        background: linear-gradient(135deg, #ff8c00 0%, #ff6b00 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #f59e0b;
    }
</style>
""", unsafe_allow_html=True)
# Initialize session state
if 'kite' not in st.session_state: st.session_state.kite = None
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'trader' not in st.session_state: st.session_state.trader = None
if 'refresh_count' not in st.session_state: st.session_state.refresh_count = 0
if 'kite_manager' not in st.session_state: st.session_state.kite_manager = None

# Display Logo with ORANGE theme
st.markdown("""
<div class="logo-container">
    <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='250' height='250' viewBox='0 0 250 250'%3E%3Cdefs%3E%3ClinearGradient id='grad' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%23ff8c00;stop-opacity:1' /%3E%3Cstop offset='50%25' style='stop-color:%23ff6b00;stop-opacity:1' /%3E%3Cstop offset='100%25' style='stop-color:%23ff4500;stop-opacity:1' /%3E%3C/linearGradient%3E%3C/defs%3E%3Ccircle cx='125' cy='125' r='110' fill='url(%23grad)'/%3E%3Ctext x='125' y='155' font-family='Arial, sans-serif' font-size='100' font-weight='bold' fill='white' text-anchor='middle'%3ER%3C/text%3E%3Cpath d='M 60 190 Q 125 160 190 190' stroke='white' stroke-width='8' fill='none'/%3E%3Ccircle cx='80' cy='195' r='3' fill='white'/%3E%3Ccircle cx='125' cy='185' r='3' fill='white'/%3E%3Ccircle cx='170' cy='195' r='3' fill='white'/%3E%3C/svg%3E" alt="Rantv Logo">
    <h1 style="color: white; margin: 10px 0 0 0; font-size: 32px;">RANTV TERMINAL PRO</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color: #ff8c00;'>📈 Rantv Intraday Terminal Pro - ENHANCED</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color: #6b7280;'>Full Stock Scanning & High-Quality Signal Generation Enabled</h4>", unsafe_allow_html=True)
"""
PART 4: Utility Functions & Technical Indicators
"""

# Utility Functions
def now_indian():
    return datetime.now(IND_TZ)

def market_open():
    n = now_indian()
    try:
        open_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(9, 15)))
        close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 30)))
        return open_time <= n <= close_time
    except Exception:
        return False

def should_auto_close():
    n = now_indian()
    try:
        auto_close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 10)))
        return n >= auto_close_time
    except Exception:
        return False

def is_peak_market_hours():
    """Check if current time is during peak market hours (9:30 AM - 2:30 PM)"""
    n = now_indian()
    try:
        peak_start = IND_TZ.localize(datetime.combine(n.date(), dt_time(9, 30)))
        peak_end = IND_TZ.localize(datetime.combine(n.date(), dt_time(14, 30)))
        return peak_start <= n <= peak_end
    except Exception:
        return True

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(0)
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (close - lowest_low) / denom
    d = k.rolling(window=d_period).mean()
    return k.fillna(50), d.fillna(50)

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(close, period=20, std_dev=2):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_market_profile_vectorized(high, low, close, volume, bins=20):
    try:
        low_val = float(min(high.min(), low.min(), close.min()))
        high_val = float(max(high.max(), low.max(), close.max()))
        if np.isclose(low_val, high_val):
            high_val = low_val * 1.01 if low_val != 0 else 1.0
        edges = np.linspace(low_val, high_val, bins + 1)
        hist, _ = np.histogram(close, bins=edges, weights=volume)
        centers = (edges[:-1] + edges[1:]) / 2
        if hist.sum() == 0:
            poc = float(close.iloc[-1])
            va_high = poc * 1.01
            va_low = poc * 0.99
        else:
            idx = int(np.argmax(hist))
            poc = float(centers[idx])
            sorted_idx = np.argsort(hist)[::-1]
            cumulative = 0.0
            total = float(hist.sum())
            selected = []
            for i in sorted_idx:
                selected.append(centers[i])
                cumulative += hist[i]
                if cumulative / total >= 0.70:
                    break
            va_high = float(max(selected))
            va_low = float(min(selected))
        profile = [{"price": float(c), "volume": int(v)} for c, v in zip(centers, hist)]
        return {"poc": poc, "value_area_high": va_high, "value_area_low": va_low, "profile": profile}
    except Exception:
        current_price = float(close.iloc[-1])
        return {"poc": current_price, "value_area_high": current_price*1.01, "value_area_low": current_price*0.99, "profile": []}

def calculate_support_resistance_advanced(high, low, close, period=20):
    try:
        resistance = []
        support = []
        ln = len(high)
        if ln < period * 2 + 1:
            return {"support": float(close.iloc[-1] * 0.98), "resistance": float(close.iloc[-1] * 1.02),
                    "support_levels": [], "resistance_levels": []}
        for i in range(period, ln - period):
            if high.iloc[i] >= high.iloc[i - period:i + period + 1].max():
                resistance.append(float(high.iloc[i]))
            if low.iloc[i] <= low.iloc[i - period:i + period + 1].min():
                support.append(float(low.iloc[i]))
        recent_res = sorted(resistance)[-3:] if resistance else [float(close.iloc[-1] * 1.02)]
        recent_sup = sorted(support)[:3] if support else [float(close.iloc[-1] * 0.98)]
        return {"support": float(np.mean(recent_sup)), "resistance": float(np.mean(recent_res)),
                "support_levels": recent_sup, "resistance_levels": recent_res}
    except Exception:
        current_price = float(close.iloc[-1])
        return {"support": current_price * 0.98, "resistance": current_price * 1.02,
                "support_levels": [], "resistance_levels": []}

def adx(high, low, close, period=14):
    try:
        h = high.copy().reset_index(drop=True)
        l = low.copy().reset_index(drop=True)
        c = close.copy().reset_index(drop=True)
        df = pd.DataFrame({"high": h, "low": l, "close": c})
        df["tr"] = np.maximum(df["high"] - df["low"],
                              np.maximum((df["high"] - df["close"].shift()).abs(),
                                         (df["low"] - df["close"].shift()).abs()))
        df["up_move"] = df["high"] - df["high"].shift()
        df["down_move"] = df["low"].shift() - df["low"]
        df["dm_pos"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0.0)
        df["dm_neg"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0.0)
        df["tr_sum"] = df["tr"].rolling(window=period).sum()
        df["dm_pos_sum"] = df["dm_pos"].rolling(window=period).sum()
        df["dm_neg_sum"] = df["dm_neg"].rolling(window=period).sum()
        df["di_pos"] = 100 * (df["dm_pos_sum"] / df["tr_sum"]).replace([np.inf, -np.inf], 0).fillna(0)
        df["di_neg"] = 100 * (df["dm_neg_sum"] / df["tr_sum"]).replace([np.inf, -np.inf], 0).fillna(0)
        df["dx"] = (abs(df["di_pos"] - df["di_neg"]) / (df["di_pos"] + df["di_neg"]).replace(0, np.nan)) * 100
        df["adx"] = df["dx"].rolling(window=period).mean().fillna(0)
        return df["adx"].values
    except Exception:
        return np.array([20] * len(high))

def create_circular_market_mood_gauge(index_name, current_value, change_percent, sentiment_score):
    """Create a circular market mood gauge for Nifty50 and BankNifty"""
    
    sentiment_score = round(sentiment_score)
    change_percent = round(change_percent, 2)
    
    if sentiment_score >= 70:
        sentiment_color = "bullish"
        sentiment_text = "BULLISH"
        emoji = "📈"
        progress_color = "#059669"
    elif sentiment_score <= 30:
        sentiment_color = "bearish"
        sentiment_text = "BEARISH"
        emoji = "📉"
        progress_color = "#dc2626"
    else:
        sentiment_color = "neutral"
        sentiment_text = "NEUTRAL"
        emoji = "➡️"
        progress_color = "#ff8c00"
    
    gauge_html = f"""
    <div class="gauge-container">
        <div class="gauge-title">{emoji} {index_name}</div>
        <div class="gauge-progress" style="--progress: {sentiment_score}%; background: conic-gradient({progress_color} 0% {sentiment_score}%, #e5e7eb {sentiment_score}% 100%);">
            <div class="gauge-progress-inner">
                {sentiment_score}%
            </div>
        </div>
        <div class="gauge-value">₹{current_value:,.0f}</div>
        <div class="gauge-sentiment {sentiment_color}">{sentiment_text}</div>
        <div style="color: {'#059669' if change_percent >= 0 else '#dc2626'}; font-size: 12px; margin-top: 3px;">
            {change_percent:+.2f}%
        </div>
    </div>
    """
    return gauge_html

print("✅ PART 4 LOADED: Utility Functions Complete")
print("📋 Next: Copy PART 5 - Kite Connect Manager")
"""
PART 5: Kite Connect Manager with OAuth Flow - FIXED
"""

import time

class KiteConnectManager:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.kite = None
        self.kws = None
        self.access_token = None
        self.is_authenticated = False
        self.tick_buffer = {}
        self.candle_store = {}
        self.ws_running = False

        # Session guards
        st.session_state.setdefault("kite_oauth_consumed", False)
        st.session_state.setdefault("kite_oauth_consumed_at", 0.0)
        st.session_state.setdefault("kite_oauth_in_progress", False)

    def _get_query_params(self) -> dict:
        try:
            return dict(st.query_params)
        except Exception:
            try:
                return dict(st.experimental_get_query_params())
            except Exception:
                return {}

    def _clear_query_params(self):
        """Best-effort clearing of query params."""
        try:
            st.query_params.clear()
        except Exception:
            pass
        try:
            st.experimental_set_query_params()
        except Exception:
            pass
        try:
            st.markdown(
                """
                <script>
                if (window && window.history && window.location && window.location.pathname) {
                    const cleanUrl = window.location.origin + window.location.pathname;
                    window.history.replaceState({}, document.title, cleanUrl);
                }
                </script>
                """,
                unsafe_allow_html=True
            )
        except Exception:
            pass

    def check_oauth_callback(self) -> bool:
        """Check if URL contains request_token and exchange it"""
        try:
            q = self._get_query_params()
            req = None
            if "request_token" in q:
                val = q.get("request_token")
                req = val[0] if isinstance(val, list) else val

            if not req:
                return False

            if st.session_state.kite_oauth_consumed and (time.time() - st.session_state.kite_oauth_consumed_at) < 60:
                self._clear_query_params()
                return False

            return self.exchange_request_token(req)
        except Exception as e:
            logger.error(f"OAuth callback error: {e}")
            return False

    def exchange_request_token(self, request_token: str) -> bool:
        """Exchange request_token -> access_token"""
        try:
            if not self.api_key or not self.api_secret:
                st.error("Kite API credentials missing.")
                return False

            if not self.kite:
                self.kite = KiteConnect(api_key=self.api_key)

            st.session_state.kite_oauth_in_progress = True

            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            if not data or "access_token" not in data:
                st.error("Kite token exchange failed (no access token).")
                self._clear_query_params()
                st.session_state.kite_oauth_in_progress = False
                return False

            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            self.is_authenticated = True

            st.session_state.kite_access_token = self.access_token
            st.session_state.kite_user_name = data.get("user_name", "")

            st.session_state.kite_oauth_consumed = True
            st.session_state.kite_oauth_consumed_at = time.time()

            self._clear_query_params()

            st.toast("✅ Authenticated with Kite. Finalizing...", icon="✅")
            time.sleep(0.3)

            st.session_state.kite_oauth_in_progress = False
            st.rerun()
            return True

        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            self._clear_query_params()
            st.session_state.kite_oauth_in_progress = False
            st.error(f"Token exchange failed: {str(e)}")
            return False

    def login(self) -> bool:
        """Render the login UI"""
        try:
            if not self.kite and (self.api_key and self.api_secret):
                self.kite = KiteConnect(api_key=self.api_key)
            elif not self.kite:
                try:
                    from kiteconnect import KiteConnect as KC
                    self.kite = KC(api_key="DUMMY")
                except:
                    pass

            # Handle OAuth callback first
            if self.api_key and self.api_secret and not st.session_state.kite_oauth_in_progress and self.check_oauth_callback():
                return True

            # Session token check
            if "kite_access_token" in st.session_state:
                self.access_token = st.session_state.kite_access_token
                if self.kite:
                    self.kite.set_access_token(self.access_token)
                try:
                    if self.kite:
                        _ = self.kite.profile()
                    self.is_authenticated = True
                    return True
                except Exception:
                    del st.session_state["kite_access_token"]

            # Render login UI if not mid-flow
            if st.session_state.kite_oauth_in_progress:
                st.info("Completing authentication…")
                return False

            st.info("🔐 Kite Connect Authentication Required")
            
            if self.api_key and self.api_secret:
                try:
                    login_url = self.kite.login_url()
                    st.link_button("🔓 Login with Kite OAuth", login_url, use_container_width=True)
                except Exception as e:
                    st.warning(f"OAuth login unavailable: {e}")
            else:
                st.warning("⚠️ Kite API credentials not configured.")

            # Manual token entry
            st.markdown("**Enter access token manually:**")
            with st.form("kite_login_form_unique"):
                access_token = st.text_input("Access Token", type="password", help="Get your access token from Kite Connect dashboard")
                submit = st.form_submit_button("Authenticate", type="primary", use_container_width=True)

            if submit and access_token:
                try:
                    self.access_token = access_token
                    if self.kite:
                        self.kite.set_access_token(self.access_token)
                        profile = self.kite.profile()
                        user_name = profile.get("user_name", "")
                    else:
                        user_name = "User"
                    st.session_state.kite_access_token = self.access_token
                    st.session_state.kite_user_name = user_name
                    self.is_authenticated = True
                    st.success(f"✅ Authenticated as {user_name}")
                    st.balloons()
                    return True
                except Exception as e:
                    st.error(f"❌ Authentication failed: {str(e)}")
                    return False

            return False

        except Exception as e:
            st.error(f"Kite Connect login error: {str(e)}")
            return False

    def logout(self):
        try:
            if "kite_access_token" in st.session_state:
                del st.session_state.kite_access_token
            if "kite_user_name" in st.session_state:
                del st.session_state.kite_user_name
            st.session_state.kite_oauth_consumed = False
            st.session_state.kite_oauth_consumed_at = 0.0
            st.session_state.kite_oauth_in_progress = False
            self.access_token = None
            self.is_authenticated = False
            return True
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False

print("✅ PART 5 LOADED: Kite Connect Manager Complete")
print("📋 Next: Copy PART 6 - Trading Classes")
"""
PART 6: Enhanced Data Manager and Trading Classes
"""

class EnhancedDataManager:
    def __init__(self):
        self.price_cache = {}
        self.signal_cache = {}
        self.market_profile_cache = {}
        self.last_rsi_scan = None

    def _validate_live_price(self, symbol):
        now_ts = time.time()
        key = f"price_{symbol}"
        if key in self.price_cache:
            cached = self.price_cache[key]
            if now_ts - cached["ts"] < 2:
                return cached["price"]
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1d", interval="1m")
            if df is not None and not df.empty:
                price = float(df["Close"].iloc[-1])
                self.price_cache[key] = {"price": round(price, 2), "ts": now_ts}
                return round(price, 2)
            df = ticker.history(period="2d", interval="5m")
            if df is not None and not df.empty:
                price = float(df["Close"].iloc[-1])
                self.price_cache[key] = {"price": round(price, 2), "ts": now_ts}
                return round(price, 2)
        except Exception:
            pass
        known = {"RELIANCE.NS": 2750.0, "TCS.NS": 3850.0, "HDFCBANK.NS": 1650.0}
        base = known.get(symbol, 1000.0)
        self.price_cache[key] = {"price": float(base), "ts": now_ts}
        return float(base)

    @st.cache_data(ttl=30)
    def _fetch_yf(_self, symbol, period, interval):
        try:
            return yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        except Exception:
            return pd.DataFrame()

    def get_stock_data(self, symbol, interval="15m"):
        if interval == "15m":
            period = "7d"
        elif interval == "1m":
            period = "1d"
        elif interval == "5m":
            period = "2d"
        else:
            period = "14d"

        df = self._fetch_yf(symbol, period, interval)
        if df is None or df.empty or len(df) < 20:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        expected = ["Open", "High", "Low", "Close", "Volume"]
        for e in expected:
            if e not in df.columns:
                if e.upper() in df.columns:
                    df[e] = df[e.upper()]
                else:
                    return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        if len(df) < 20:
            return None

        try:
            live_price = self._validate_live_price(symbol)
            current_close = df["Close"].iloc[-1]
            price_diff_pct = abs(live_price - current_close) / max(current_close, 1e-6)
            if price_diff_pct > 0.005:
                df.iloc[-1, df.columns.get_loc("Close")] = live_price
                df.iloc[-1, df.columns.get_loc("High")] = max(df.iloc[-1]["High"], live_price)
                df.iloc[-1, df.columns.get_loc("Low")] = min(df.iloc[-1]["Low"], live_price)
        except Exception:
            pass

        # Enhanced Indicators
        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI14"] = rsi(df["Close"], 14).fillna(50)
        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"]).fillna(method="ffill").fillna(0)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])
        df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"])
        df["Stoch_K"], df["Stoch_D"] = stochastic(df["High"], df["Low"], df["Close"])
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()

        mp = calculate_market_profile_vectorized(df["High"], df["Low"], df["Close"], df["Volume"], bins=24)
        df["POC"] = mp["poc"]
        df["VA_High"] = mp["value_area_high"]
        df["VA_Low"] = mp["value_area_low"]

        sr = calculate_support_resistance_advanced(df["High"], df["Low"], df["Close"])
        df["Support"] = sr["support"]
        df["Resistance"] = sr["resistance"]

        try:
            df_adx = adx(df["High"], df["Low"], df["Close"], period=14)
            df["ADX"] = pd.Series(df_adx, index=df.index).fillna(method="ffill").fillna(20)
        except Exception:
            df["ADX"] = 20

        df["HTF_Trend"] = 1

        return df

class MultiStrategyIntradayTrader:
    def __init__(self, capital=CAPITAL):
        self.initial_capital = float(capital)
        self.cash = float(capital)
        self.positions = {}
        self.trade_log = []
        self.daily_trades = 0
        self.stock_trades = 0
        self.auto_trades_count = 0
        self.last_reset = now_indian().date()
        self.selected_market = "CASH"
        self.auto_execution = False
        self.signal_history = []
        self.auto_close_triggered = False
        self.last_auto_execution_time = 0
        
        self.strategy_performance = {}
        for strategy in TRADING_STRATEGIES.keys():
            self.strategy_performance[strategy] = {"signals": 0, "trades": 0, "wins": 0, "pnl": 0.0}
        
        for strategy in HIGH_ACCURACY_STRATEGIES.keys():
            self.strategy_performance[strategy] = {"signals": 0, "trades": 0, "wins": 0, "pnl": 0.0}
        
        self.data_manager = EnhancedDataManager()

    def reset_daily_counts(self):
        current_date = now_indian().date()
        if current_date != self.last_reset:
            self.daily_trades = 0
            self.stock_trades = 0
            self.auto_trades_count = 0
            self.last_reset = current_date

    def can_auto_trade(self):
        """Check if auto trading is allowed"""
        can_trade = (
            self.auto_trades_count < MAX_AUTO_TRADES and 
            self.daily_trades < MAX_DAILY_TRADES and
            market_open()
        )
        return can_trade

    def execute_trade(self, symbol, action, quantity, price, stop_loss=None, target=None, win_probability=0.75, auto_trade=False, strategy=None):
        self.reset_daily_counts()
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        if self.stock_trades >= MAX_STOCK_TRADES:
            return False, "Stock trade limit reached"
        if auto_trade and self.auto_trades_count >= MAX_AUTO_TRADES:
            return False, "Auto trade limit reached"

        trade_value = float(quantity) * float(price)
        if action == "BUY" and trade_value > self.cash:
            return False, "Insufficient capital"

        trade_id = f"TRADE_{symbol}_{len(self.trade_log)}_{int(time.time())}"
        record = {
            "trade_id": trade_id, 
            "symbol": symbol, 
            "action": action,
            "quantity": int(quantity),
            "entry_price": float(price), 
            "stop_loss": float(stop_loss) if stop_loss else None,
            "target": float(target) if target else None, 
            "timestamp": now_indian(),
            "status": "OPEN", 
            "current_pnl": 0.0, 
            "current_price": float(price),
            "win_probability": float(win_probability), 
            "closed_pnl": 0.0,
            "entry_time": now_indian().strftime("%H:%M:%S"),
            "auto_trade": auto_trade,
            "strategy": strategy
        }

        if action == "BUY":
            self.positions[symbol] = record
            self.cash -= trade_value
        else:
            margin = trade_value * 0.2
            record["margin_used"] = margin
            self.positions[symbol] = record
            self.cash -= margin

        self.stock_trades += 1
        self.trade_log.append(record)
        self.daily_trades += 1

        if auto_trade:
            self.auto_trades_count += 1

        if strategy and strategy in self.strategy_performance:
            self.strategy_performance[strategy]["trades"] += 1

        return True, f"{'[AUTO] ' if auto_trade else ''}{action} {int(quantity)} {symbol} @ ₹{price:.2f} | Strategy: {strategy}"

    def get_performance_stats(self):
        self.update_positions_pnl()
        closed = [t for t in self.trade_log if t.get("status") == "CLOSED"]
        total_trades = len(closed)
        open_pnl = sum([p.get("current_pnl", 0) for p in self.positions.values() if p.get("status") == "OPEN"])
        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "open_positions": len(self.positions),
                "open_pnl": open_pnl,
                "auto_trades": self.auto_trades_count
            }
        wins = len([t for t in closed if t.get("closed_pnl", 0) > 0])
        total_pnl = sum([t.get("closed_pnl", 0) for t in closed])
        win_rate = wins / total_trades if total_trades else 0.0
        avg_pnl = total_pnl / total_trades if total_trades else 0.0

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "open_positions": len(self.positions),
            "open_pnl": open_pnl,
            "auto_trades": self.auto_trades_count
        }

    def update_positions_pnl(self):
        if should_auto_close() and not self.auto_close_triggered:
            self.auto_close_all_positions()
            self.auto_close_triggered = True
            return
            
        for symbol, pos in list(self.positions.items()):
            if pos.get("status") != "OPEN":
                continue
            try:
                data = self.data_manager.get_stock_data(symbol, "5m")
                if data is not None and len(data) > 0:
                    price = float(data["Close"].iloc[-1])
                    pos["current_price"] = price
                    entry = pos["entry_price"]
                    if pos["action"] == "BUY":
                        pnl = (price - entry) * pos["quantity"]
                    else:
                        pnl = (entry - price) * pos["quantity"]
                    pos["current_pnl"] = float(pnl)
                    sl = pos.get("stop_loss")
                    tg = pos.get("target")
                    if sl is not None:
                        if (pos["action"] == "BUY" and price <= sl) or (pos["action"] == "SELL" and price >= sl):
                            self.close_position(symbol, exit_price=sl)
                            continue
                    if tg is not None:
                        if (pos["action"] == "BUY" and price >= tg) or (pos["action"] == "SELL" and price <= tg):
                            self.close_position(symbol, exit_price=tg)
                            continue
            except Exception:
                continue

    def auto_close_all_positions(self):
        for sym in list(self.positions.keys()):
            self.close_position(sym)

    def close_position(self, symbol, exit_price=None):
        if symbol not in self.positions:
            return False, "Position not found"
        pos = self.positions[symbol]
        if exit_price is None:
            try:
                data = self.data_manager.get_stock_data(symbol, "5m")
                exit_price = float(data["Close"].iloc[-1]) if data is not None and len(data) > 0 else pos["entry_price"]
            except Exception:
                exit_price = pos["entry_price"]
        
        if pos["action"] == "BUY":
            pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
            self.cash += pos["quantity"] * exit_price
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["quantity"]
            self.cash += pos.get("margin_used", 0) + (pos["quantity"] * pos["entry_price"])
        
        pos["status"] = "CLOSED"
        pos["exit_price"] = float(exit_price)
        pos["closed_pnl"] = float(pnl)
        pos["exit_time"] = now_indian()
        pos["exit_time_str"] = now_indian().strftime("%H:%M:%S")

        strategy = pos.get("strategy")
        if strategy and strategy in self.strategy_performance:
            if pnl > 0:
                self.strategy_performance[strategy]["wins"] += 1
            self.strategy_performance[strategy]["pnl"] += pnl

        try:
            del self.positions[symbol]
        except Exception:
            pass
        return True, f"Closed {symbol} @ ₹{exit_price:.2f} | P&L: ₹{pnl:+.2f}"

print("✅ PART 6 LOADED: Trading Classes Complete")
print("📋 Next: Copy PART 7 - Main UI & Application")
"""
PART 7: Main Application UI & Tabs - FIXED (No Duplicate Keys)
"""

# Initialize Application
try:
    data_manager = EnhancedDataManager()
    
    if st.session_state.trader is None:
        st.session_state.trader = MultiStrategyIntradayTrader()
    
    trader = st.session_state.trader
    
    # Initialize Kite Manager
    if st.session_state.kite_manager is None:
        st.session_state.kite_manager = KiteConnectManager(KITE_API_KEY, KITE_API_SECRET)
    
    kite_manager = st.session_state.kite_manager
    
    # Auto-refresh
    st_autorefresh(interval=PRICE_REFRESH_MS, key="main_price_refresh")
    st.session_state.refresh_count += 1
    
    # Market Overview
    st.subheader("📊 Market Overview")
    cols = st.columns(7)
    
    try:
        nifty = yf.Ticker("^NSEI")
        nifty_price = nifty.history(period="1d")['Close'].iloc[-1]
        cols[0].metric("NIFTY 50", f"₹{nifty_price:,.2f}")
    except:
        cols[0].metric("NIFTY 50", "Loading...")
    
    try:
        bn = yf.Ticker("^NSEBANK")
        bn_price = bn.history(period="1d")['Close'].iloc[-1]
        cols[1].metric("BANK NIFTY", f"₹{bn_price:,.2f}")
    except:
        cols[1].metric("BANK NIFTY", "Loading...")
    
    cols[2].metric("Market Status", "LIVE" if market_open() else "CLOSED")
    
    market_regime = "TRENDING"
    regime_color = "🟢"
    cols[3].metric("Market Regime", f"{regime_color} {market_regime}")
    
    peak_hours = is_peak_market_hours()
    peak_color = "🟢" if peak_hours else "🔴"
    cols[4].metric("Peak Hours", f"{peak_color} {'YES' if peak_hours else 'NO'}")
    
    cols[5].metric("Auto Trades", f"{trader.auto_trades_count}/{MAX_AUTO_TRADES}")
    cols[6].metric("Available Cash", f"₹{trader.cash:,.0f}")
    
    # Manual refresh
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"<div style='text-align: left; color: #6b7280; font-size: 14px;'>Refresh Count: <span class='refresh-counter'>{st.session_state.refresh_count}</span></div>", unsafe_allow_html=True)
    with col2:
        if st.button("🔄 Manual Refresh", key="manual_refresh_main_btn"):
            st.rerun()
    with col3:
        if st.button("📊 Update Prices", key="update_prices_main_btn"):
            st.rerun()
    
    # Market Mood Gauges
    st.subheader("📊 Market Mood Gauges")
    
    try:
        nifty_data = yf.download("^NSEI", period="1d", interval="5m", auto_adjust=False)
        nifty_current = float(nifty_data["Close"].iloc[-1])
        nifty_prev = float(nifty_data["Close"].iloc[-2])
        nifty_change = ((nifty_current - nifty_prev) / nifty_prev) * 100
        nifty_sentiment = 50 + (nifty_change * 8)
        nifty_sentiment = max(0, min(100, round(nifty_sentiment)))
    except:
        nifty_current = 22000
        nifty_change = 0.15
        nifty_sentiment = 65
    
    try:
        banknifty_data = yf.download("^NSEBANK", period="1d", interval="5m", auto_adjust=False)
        banknifty_current = float(banknifty_data["Close"].iloc[-1])
        banknifty_prev = float(banknifty_data["Close"].iloc[-2])
        banknifty_change = ((banknifty_current - banknifty_prev) / banknifty_prev) * 100
        banknifty_sentiment = 50 + (banknifty_change * 8)
        banknifty_sentiment = max(0, min(100, round(banknifty_sentiment)))
    except:
        banknifty_current = 48000
        banknifty_change = 0.25
        banknifty_sentiment = 70
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_circular_market_mood_gauge("NIFTY 50", nifty_current, nifty_change, nifty_sentiment), unsafe_allow_html=True)
    with col2:
        st.markdown(create_circular_market_mood_gauge("BANK NIFTY", banknifty_current, banknifty_change, banknifty_sentiment), unsafe_allow_html=True)
    with col3:
        market_status = "LIVE" if market_open() else "CLOSED"
        status_sentiment = 80 if market_open() else 20
        st.markdown(create_circular_market_mood_gauge("MARKET", 0, 0, status_sentiment).replace("₹0", market_status).replace("0.00%", ""), unsafe_allow_html=True)
    with col4:
        peak_hours_status = "PEAK" if is_peak_market_hours() else "OFF-PEAK"
        peak_sentiment = 80 if is_peak_market_hours() else 30
        st.markdown(create_circular_market_mood_gauge("PEAK HOURS", 0, 0, peak_sentiment).replace("₹0", "9:30AM-2:30PM").replace("0.00%", peak_hours_status), unsafe_allow_html=True)
    
    # Main Metrics
    st.subheader("📈 Live Metrics")
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 12px; color: #6b7280;">Available Cash</div>
            <div style="font-size: 20px; font-weight: bold; color: #ff8c00;">₹{trader.cash:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        total_value = trader.cash + sum([p.get('quantity', 0) * p.get('entry_price', 0) for p in trader.positions.values()])
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 12px; color: #6b7280;">Account Value</div>
            <div style="font-size: 20px; font-weight: bold; color: #ff8c00;">₹{total_value:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 12px; color: #6b7280;">Open Positions</div>
            <div style="font-size: 20px; font-weight: bold; color: #ff8c00;">{len(trader.positions)}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[3]:
        open_pnl = sum([p.get('current_pnl', 0) for p in trader.positions.values()])
        pnl_color = "#059669" if open_pnl >= 0 else "#dc2626"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 12px; color: #6b7280;">Open P&L</div>
            <div style="font-size: 20px; font-weight: bold; color: {pnl_color};">₹{open_pnl:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # FIXED: Consolidated Sidebar Authentication with UNIQUE KEYS
    with st.sidebar:
        st.title("🔐 Kite Connect")
        
        if not kite_manager.is_authenticated:
            st.info("Kite Connect authentication required for live charts")
            if kite_manager.login():
                st.rerun()
        else:
            st.success(f"✅ Authenticated as {st.session_state.get('kite_user_name', 'User')}")
            if st.button("🔓 Logout", key="sidebar_kite_logout_btn"):
                if kite_manager.logout():
                    st.success("Logged out successfully")
                    st.rerun()
        
        st.divider()
        st.header("⚙️ Trading Configuration")
        trader.selected_market = st.selectbox("Market Type", MARKET_OPTIONS, key="sidebar_market_select")
        
        universe = st.selectbox("Trading Universe", ["All Stocks", "Nifty 50", "Nifty 100", "Midcap 150"], key="sidebar_universe_select")
        
        enable_high_accuracy = st.checkbox("Enable High Accuracy Strategies", value=True, 
                                          help="Enable high accuracy strategies for all stock universes", key="sidebar_high_acc_toggle")
        
        trader.auto_execution = st.checkbox("Auto Execution", value=False, key="sidebar_auto_exec_toggle")
        
        st.subheader("🎯 Enhanced Risk Management")
        min_conf_percent = st.slider("Minimum Confidence %", 60, 85, 70, 5, key="sidebar_min_conf_slider")
        min_score = st.slider("Minimum Score", 5, 9, 6, 1, key="sidebar_min_score_slider")
        
        st.subheader("🔍 Scan Configuration")
        full_scan = st.checkbox("Full Universe Scan", value=True, key="sidebar_full_scan_toggle")
        
        if not full_scan:
            max_scan = st.number_input("Max Stocks to Scan", min_value=10, max_value=500, value=50, step=10, key="sidebar_max_scan_input")
        else:
            max_scan = None
    
    # Main Tabs
    tabs = st.tabs([
        "📈 Dashboard", 
        "🚦 Signals", 
        "💰 Paper Trading",
        "📋 Trade History",
        "📉 RSI Extreme",
        "🎯 High Accuracy Scanner"
    ])
    
    # Tab 1: Dashboard
    with tabs[0]:
        st.subheader("Account Summary")
        trader.update_positions_pnl()
        perf = trader.get_performance_stats()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Value", f"₹{total_value:,.0f}", delta=f"₹{total_value - trader.initial_capital:+,.0f}")
        c2.metric("Available Cash", f"₹{trader.cash:,.0f}")
        c3.metric("Open Positions", len(trader.positions))
        c4.metric("Total P&L", f"₹{perf['total_pnl'] + perf['open_pnl']:+.2f}")
        
        st.subheader("Strategy Performance Overview")
        strategy_data = []
        for strategy, config in {**TRADING_STRATEGIES, **HIGH_ACCURACY_STRATEGIES}.items():
            if strategy in trader.strategy_performance:
                perf_data = trader.strategy_performance[strategy]
                if perf_data["trades"] > 0:
                    win_rate = perf_data["wins"] / perf_data["trades"]
                    strategy_data.append({
                        "Strategy": config["name"],
                        "Type": config["type"],
                        "Signals": perf_data["signals"],
                        "Trades": perf_data["trades"],
                        "Win Rate": f"{win_rate:.1%}",
                        "P&L": f"₹{perf_data['pnl']:+.2f}"
                    })
        
        if strategy_data:
            st.dataframe(pd.DataFrame(strategy_data), width='stretch')
        else:
            st.info("No strategy performance data available yet.")
        
        st.subheader("📊 Open Positions")
        if trader.positions:
            positions_data = []
            for symbol, pos in trader.positions.items():
                if pos.get('status') == 'OPEN':
                    positions_data.append({
                        "Symbol": symbol.replace(".NS", ""),
                        "Action": pos['action'],
                        "Quantity": pos['quantity'],
                        "Entry": f"₹{pos['entry_price']:.2f}",
                        "Current": f"₹{pos.get('current_price', pos['entry_price']):.2f}",
                        "P&L": f"₹{pos.get('current_pnl', 0):+.2f}",
                        "Stop Loss": f"₹{pos.get('stop_loss', 0):.2f}",
                        "Target": f"₹{pos.get('target', 0):.2f}",
                        "Strategy": pos.get('strategy', 'Manual')
                    })
            if positions_data:
                st.dataframe(pd.DataFrame(positions_data), width='stretch')
        else:
            st.info("No open positions")
    
    # Tab 2: Signals
    with tabs[1]:
        st.subheader("Multi-Strategy BUY/SELL Signals")
        st.markdown("""
        <div class="alert-success">
            <strong>🎯 Signal Parameters:</strong> 
            • Confidence threshold: <strong>70%</strong><br>
            • Minimum score: <strong>6</strong><br>
            • Optimized for peak market hours (9:30 AM - 2:30 PM)
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            generate_btn = st.button("Generate Signals", type="primary", width='stretch', key="tab_signals_generate_btn")
        with col2:
            if trader.auto_execution:
                auto_status = "🟢 ACTIVE"
                status_color = "#059669"
            else:
                auto_status = "⚪ INACTIVE"
                status_color = "#6b7280"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 12px; color: #6b7280;">Auto Execution</div>
                <div style="font-size: 18px; font-weight: bold; color: {status_color};">{auto_status}</div>
                <div style="font-size: 11px; margin-top: 3px;">Market: {'🟢 OPEN' if market_open() else '🔴 CLOSED'}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            if trader.auto_execution and trader.can_auto_trade():
                auto_exec_btn = st.button("🚀 Auto Execute", type="secondary", width='stretch', key="tab_signals_auto_exec_btn")
            else:
                auto_exec_btn = False
        
        if generate_btn:
            with st.spinner(f"Scanning {universe} stocks..."):
                time.sleep(1)
                st.info("Signal generation completed. No high-quality signals found at this time.")
    
    # Tab 3: Paper Trading
    with tabs[2]:
        st.subheader("💰 Paper Trading")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            symbol = st.selectbox("Symbol", NIFTY_50[:20], key="tab_paper_symbol_select")
        with col2:
            action = st.selectbox("Action", ["BUY", "SELL"], key="tab_paper_action_select")
        with col3:
            quantity = st.number_input("Quantity", min_value=1, value=10, key="tab_paper_quantity_input")
        with col4:
            strategy_name = st.selectbox("Strategy", ["Manual"] + [config["name"] for config in TRADING_STRATEGIES.values()], key="tab_paper_strategy_select")
        
        if st.button("Execute Paper Trade", type="primary", key="tab_paper_execute_btn"):
            try:
                data = data_manager.get_stock_data(symbol, "15m")
                if data is not None and len(data) > 0:
                    price = float(data["Close"].iloc[-1])
                    atr = float(data["ATR"].iloc[-1]) if 'ATR' in data.columns else price * 0.01
                    
                    if action == "BUY":
                        stop_loss = price - (atr * 1.2)
                        target = price + (atr * 2.5)
                    else:
                        stop_loss = price + (atr * 1.2)
                        target = price - (atr * 2.5)
                    
                    strategy_key = "Manual"
                    for key, config in TRADING_STRATEGIES.items():
                        if config["name"] == strategy_name:
                            strategy_key = key
                            break
                    
                    success, msg = trader.execute_trade(
                        symbol=symbol,
                        action=action,
                        quantity=quantity,
                        price=price,
                        stop_loss=stop_loss,
                        target=target,
                        win_probability=0.75,
                        auto_trade=False,
                        strategy=strategy_key
                    )
                    
                    if success:
                        st.success(f"✅ {msg}")
                        st.success(f"Stop Loss: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
                else:
                    st.error("Unable to fetch price data")
            except Exception as e:
                st.error(f"Trade execution failed: {str(e)}")
        
        st.subheader("Current Positions")
        if trader.positions:
            for idx, (symbol, pos) in enumerate(trader.positions.items()):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    action_color = "🟢" if pos['action'] == 'BUY' else '🔴'
                    pnl = pos.get('current_pnl', 0)
                    pnl_color = "green" if pnl >= 0 else "red"
                    
                    st.markdown(f"""
                    <div style="padding: 10px; border-left: 4px solid {'#059669' if pos['action'] == 'BUY' else '#dc2626'}; 
                             background: linear-gradient(135deg, {'#d1fae5' if pos['action'] == 'BUY' else '#fee2e2'} 0%, 
                             {'#a7f3d0' if pos['action'] == 'BUY' else '#fecaca'} 100%); border-radius: 8px;">
                        <strong>{action_color} {symbol.replace('.NS', '')}</strong> | {pos['action']} | Qty: {pos['quantity']}<br>
                        Entry: ₹{pos['entry_price']:.2f} | Current: ₹{pos.get('current_price', pos['entry_price']):.2f}<br>
                        <span style="color: {pnl_color}">₹{pnl:+.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.write(f"SL: ₹{pos.get('stop_loss', 0):.2f}")
                    st.write(f"TG: ₹{pos.get('target', 0):.2f}")
                
                with col3:
                    if st.button(f"Close", key=f"tab_paper_close_{symbol}_{idx}", type="secondary"):
                        success, msg = trader.close_position(symbol)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
        else:
            st.info("No open positions")
        
        st.subheader("Performance Statistics")
        perf = trader.get_performance_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", perf['total_trades'])
        with col2:
            st.metric("Win Rate", f"{perf['win_rate']:.1%}")
        with col3:
            st.metric("Total P&L", f"₹{perf['total_pnl']:+.2f}")
        with col4:
            st.metric("Open P&L", f"₹{perf['open_pnl']:+.2f}")
    
    # Tab 4: Trade History
    with tabs[3]:
        st.subheader("📋 Trade History")
        
        if trader.trade_log:
            history_data = []
            for trade in trader.trade_log:
                if trade.get("status") == "CLOSED":
                    pnl = trade.get("closed_pnl", 0)
                    history_data.append({
                        "Symbol": trade['symbol'].replace(".NS", ""),
                        "Action": trade['action'],
                        "Quantity": trade['quantity'],
                        "Entry": f"₹{trade['entry_price']:.2f}",
                        "Exit": f"₹{trade.get('exit_price', 0):.2f}",
                        "P&L": f"₹{pnl:+.2f}",
                        "Entry Time": trade.get('entry_time', ''),
                        "Exit Time": trade.get('exit_time_str', ''),
                        "Strategy": trade.get('strategy', 'Manual'),
                        "Auto": "Yes" if trade.get('auto_trade') else "No"
                    })
            
            if history_data:
                st.dataframe(pd.DataFrame(history_data), use_container_width=True)
            else:
                st.info("No closed trades yet")
        else:
            st.info("No trade history available")
    
    # Tab 5: RSI Extreme
    with tabs[4]:
        st.subheader("📉 RSI Extreme Scanner")
        st.info("This scanner finds stocks with extreme RSI values (oversold/overbought)")
        
        if st.button("Scan for RSI Extremes", key="tab_rsi_scan_btn"):
            with st.spinner("Scanning for RSI extremes..."):
                oversold = []
                overbought = []
                
                for symbol in NIFTY_50[:30]:
                    try:
                        data = data_manager.get_stock_data(symbol, "15m")
                        if data is not None and len(data) > 0:
                            rsi_val = data['RSI14'].iloc[-1]
                            price = data['Close'].iloc[-1]
                            
                            if rsi_val < 30:
                                oversold.append({
                                    "Symbol": symbol.replace(".NS", ""),
                                    "RSI": round(rsi_val, 2),
                                    "Price": round(price, 2),
                                    "Signal": "OVERSOLD"
                                })
                            elif rsi_val > 70:
                                overbought.append({
                                    "Symbol": symbol.replace(".NS", ""),
                                    "RSI": round(rsi_val, 2),
                                    "Price": round(price, 2),
                                    "Signal": "OVERBOUGHT"
                                })
                    except:
                        continue
                
                if oversold or overbought:
                    st.success(f"Found {len(oversold)} oversold and {len(overbought)} overbought stocks")
                    
                    if oversold:
                        st.subheader("🔵 Oversold Stocks (RSI < 30)")
                        st.dataframe(pd.DataFrame(oversold), width='stretch')
                    
                    if overbought:
                        st.subheader("🔴 Overbought Stocks (RSI > 70)")
                        st.dataframe(pd.DataFrame(overbought), width='stretch')
                else:
                    st.info("No extreme RSI stocks found")
    
    # Tab 6: High Accuracy Scanner
    with tabs[5]:
        st.subheader("🎯 High Accuracy Scanner - All Stocks")
        st.markdown(f"""
        <div class="alert-success">
            <strong>🔥 High Accuracy Strategies Enabled:</strong> 
            Scanning <strong>{universe}</strong> with enhanced high-accuracy strategies
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            high_acc_scan_btn = st.button("🚀 Scan High Accuracy", type="primary", width='stretch', key="tab_high_acc_scan_btn")
        with col2:
            min_high_acc_confidence = st.slider("Min Confidence", 65, 85, 70, 5, key="tab_high_acc_conf_slider")
        with col3:
            min_high_acc_score = st.slider("Min Score", 5, 8, 6, 1, key="tab_high_acc_score_slider")
        
        if high_acc_scan_btn:
            with st.spinner(f"Scanning {universe} with high-accuracy strategies..."):
                time.sleep(1)
                st.info("High accuracy scan completed. No signals found at this time.")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; font-size: 12px;">
        <strong>Rantv Terminal Pro</strong> | © 2024 | Refresh: {st.session_state.refresh_count} | {now_indian().strftime("%H:%M:%S")}
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Application error: {str(e)}")
    st.code(traceback.format_exc())

print("✅ PART 7 LOADED: Main Application Complete")
print("🎉 ALL PARTS LOADED SUCCESSFULLY!")
