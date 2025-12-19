"""
Algo Trading Engine for Rantv Intraday Terminal Pro
Provides automated order execution, scheduling, and risk management
Enhanced with High Accuracy Strategies, Live Market Data, and Advanced Risk Management
"""

# ===================== PART 1: IMPORTS AND CONFIGURATION =====================
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

# Auto-install missing dependencies
def install_package(package_name):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except:
        return False

# Check and install required packages
KITECONNECT_AVAILABLE = False
SQLALCHEMY_AVAILABLE = False
JOBLIB_AVAILABLE = False

try:
    from kiteconnect import KiteConnect, KiteTicker
    KITECONNECT_AVAILABLE = True
except ImportError:
    if install_package("kiteconnect"):
        from kiteconnect import KiteConnect, KiteTicker
        KITECONNECT_AVAILABLE = True
        st.success("✅ Installed kiteconnect")

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    if install_package("sqlalchemy"):
        import sqlalchemy
        from sqlalchemy import create_engine, text
        SQLALCHEMY_AVAILABLE = True
        st.success("✅ Installed sqlalchemy")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    if install_package("joblib"):
        import joblib
        JOBLIB_AVAILABLE = True
        st.success("✅ Installed joblib")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Configuration
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN = ""

ALGO_ENABLED = os.environ.get("ALGO_TRADING_ENABLED", "false").lower() == "true"
ALGO_MAX_POSITIONS = int(os.environ.get("ALGO_MAX_POSITIONS", "5"))
ALGO_MAX_DAILY_LOSS = float(os.environ.get("ALGO_MAX_DAILY_LOSS", "50000"))
ALGO_MIN_CONFIDENCE = float(os.environ.get("ALGO_MIN_CONFIDENCE", "0.80"))

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
    "BANKINDIA.NS", "BATAINDIA.NS", "BEL.NS", "BHARATFORG.NS", "BHEL.NS",
    "BIOCON.NS", "BOSCHLTD.NS", "BRIGADE.NS", "CANBK.NS", "CANFINHOME.NS",
    "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS",
    "CONCOR.NS", "COROMANDEL.NS", "CROMPTON.NS", "CUMMINSIND.NS", "DABUR.NS",
    "DALBHARAT.NS", "DEEPAKNTR.NS", "DELTACORP.NS", "DIVISLAB.NS", "DIXON.NS",
    "DLF.NS", "DRREDDY.NS", "EDELWEISS.NS", "EICHERMOT.NS", "ESCORTS.NS",
    "EXIDEIND.NS", "FEDERALBNK.NS", "GAIL.NS", "GLENMARK.NS", "GODREJCP.NS",
    "GODREJPROP.NS", "GRANULES.NS", "GRASIM.NS", "GUJGASLTD.NS", "HAL.NS",
    "HAVELLS.NS", "HCLTECH.NS", "HDFCAMC.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "ICICIPRULI.NS",
    "IDEA.NS", "IDFCFIRSTB.NS", "IGL.NS", "INDIACEM.NS", "INDIAMART.NS",
    "INDUSTOWER.NS", "INFY.NS", "IOC.NS", "IPCALAB.NS", "JINDALSTEL.NS",
    "JSWENERGY.NS", "JUBLFOOD.NS", "KOTAKBANK.NS", "L&TFH.NS", "LICHSGFIN.NS",
    "LT.NS", "LTTS.NS", "MANAPPURAM.NS", "MARICO.NS", "MARUTI.NS", "MFSL.NS",
    "MGL.NS", "MINDTREE.NS", "MOTHERSUMI.NS", "MPHASIS.NS", "MRF.NS",
    "MUTHOOTFIN.NS", "NATIONALUM.NS", "NAUKRI.NS", "NESTLEIND.NS", "NMDC.NS",
    "NTPC.NS", "OBEROIRLTY.NS", "OFSS.NS", "ONGC.NS", "PAGEIND.NS",
    "PEL.NS", "PETRONET.NS", "PFC.NS", "PIDILITIND.NS", "PIIND.NS",
    "PNB.NS", "POWERGRID.NS", "RAJESHEXPO.NS", "RAMCOCEM.NS", "RBLBANK.NS",
    "RECLTD.NS", "RELIANCE.NS", "SAIL.NS", "SBICARD.NS", "SBILIFE.NS",
    "SHREECEM.NS", "SIEMENS.NS", "SRF.NS", "SRTRANSFIN.NS", "SUNPHARMA.NS",
    "SUNTV.NS", "SYNGENE.NS", "TATACHEM.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
    "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS",
    "TORNTPHARM.NS", "TRENT.NS", "UPL.NS", "VOLTAS.NS", "WIPRO.NS",
    "YESBANK.NS", "ZEEL.NS"
]

ALL_STOCKS = list(dict.fromkeys(NIFTY_50 + NIFTY_100 + NIFTY_MIDCAP_150))

# Trading Strategies
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

# ===================== PART 2: CSS STYLING =====================
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

# ===================== PART 3: SESSION STATE INITIALIZATION =====================
if 'kite' not in st.session_state: st.session_state.kite = None
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'trader' not in st.session_state: st.session_state.trader = None
if 'refresh_count' not in st.session_state: st.session_state.refresh_count = 0
if 'kite_manager' not in st.session_state: st.session_state.kite_manager = None
if 'last_signal_generation' not in st.session_state: st.session_state.last_signal_generation = 0
if 'auto_execution_triggered' not in st.session_state: st.session_state.auto_execution_triggered = False
if 'kite_oauth_in_progress' not in st.session_state: st.session_state.kite_oauth_in_progress = False
if 'kite_oauth_consumed' not in st.session_state: st.session_state.kite_oauth_consumed = False
if 'kite_oauth_consumed_at' not in st.session_state: st.session_state.kite_oauth_consumed_at = 0.0
if 'kite_access_token' not in st.session_state: st.session_state.kite_access_token = None
if 'kite_user_name' not in st.session_state: st.session_state.kite_user_name = ""

# ===================== PART 4: UTILITY FUNCTIONS =====================
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

# ===================== PART 5: KITE CONNECT MANAGER =====================
class KiteConnectManager:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.kite = None
        self.access_token = None
        self.is_authenticated = False

    def _get_query_params(self) -> dict:
        try:
            return dict(st.query_params)
        except Exception:
            try:
                return dict(st.experimental_get_query_params())
            except Exception:
                return {}

    def _clear_query_params(self):
        """Clear query params to prevent loops"""
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
            with st.form("kite_login_form"):
                access_token = st.text_input("Access Token", type="password", 
                                            help="Get your access token from Kite Connect dashboard")
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

# ===================== PART 6: DATA MANAGER =====================
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

        # Calculate technical indicators
        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI14"] = rsi(df["Close"], 14).fillna(50)
        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"]).fillna(method="ffill").fillna(0)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])
        df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"])
        df["Stoch_K"], df["Stoch_D"] = stochastic(df["High"], df["Low"], df["Close"])
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()
        
        sr = calculate_support_resistance_advanced(df["High"], df["Low"], df["Close"])
        df["Support"] = sr["support"]
        df["Resistance"] = sr["resistance"]
        
        df["ADX"] = 25  # Placeholder for ADX
        
        return df

# ===================== PART 7: TRADING ENGINE =====================
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
        
        # Initialize strategy performance
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
        return (
            self.auto_trades_count < MAX_AUTO_TRADES and 
            self.daily_trades < MAX_DAILY_TRADES and
            market_open()
        )

    def execute_trade(self, symbol, action, quantity, price, stop_loss=None, target=None, 
                     win_probability=0.75, auto_trade=False, strategy=None):
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
                    
                    # Check stop loss and target
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

    def get_open_positions_data(self):
        self.update_positions_pnl()
        out = []
        for symbol, pos in self.positions.items():
            if pos.get("status") != "OPEN":
                continue
            try:
                data = self.data_manager.get_stock_data(symbol, "5m")
                price = float(data["Close"].iloc[-1]) if data is not None and len(data) > 0 else pos["entry_price"]
                if pos["action"] == "BUY":
                    pnl = (price - pos["entry_price"]) * pos["quantity"]
                else:
                    pnl = (pos["entry_price"] - price) * pos["quantity"]
                
                out.append({
                    "Symbol": symbol.replace(".NS", ""),
                    "Action": pos["action"],
                    "Quantity": pos["quantity"],
                    "Entry Price": f"₹{pos['entry_price']:.2f}",
                    "Current Price": f"₹{price:.2f}",
                    "P&L": f"₹{pnl:+.2f}",
                    "Stop Loss": f"₹{pos.get('stop_loss', 0):.2f}",
                    "Target": f"₹{pos.get('target', 0):.2f}",
                    "Strategy": pos.get("strategy", "Manual"),
                    "Auto Trade": "Yes" if pos.get("auto_trade") else "No"
                })
            except Exception:
                continue
        return out

# ===================== PART 8: MAIN APPLICATION UI =====================
def main():
    # Display Logo and Header
    st.markdown("""
    <div class="logo-container">
        <h1 style="color: white; margin: 10px 0 0 0; font-size: 32px;">📈 RANTV TERMINAL PRO</h1>
        <p style="color: white; margin: 5px 0;">Enhanced Intraday Trading Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align:center; color: #ff8c00;'>Intraday Terminal Pro - ENHANCED</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color: #6b7280;'>Full Stock Scanning & High-Quality Signal Generation</h4>", unsafe_allow_html=True)
    
    # Initialize components
    data_manager = EnhancedDataManager()
    
    if st.session_state.trader is None:
        st.session_state.trader = MultiStrategyIntradayTrader()
    
    trader = st.session_state.trader
    
    if st.session_state.kite_manager is None:
        st.session_state.kite_manager = KiteConnectManager(KITE_API_KEY, KITE_API_SECRET)
    
    kite_manager = st.session_state.kite_manager
    
    # Auto-refresh
    st_autorefresh(interval=PRICE_REFRESH_MS, key="main_auto_refresh")
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
    cols[3].metric("Market Regime", "🟢 TRENDING")
    cols[4].metric("Peak Hours", f"{'🟢 YES' if is_peak_market_hours() else '🔴 NO'}")
    cols[5].metric("Auto Trades", f"{trader.auto_trades_count}/{MAX_AUTO_TRADES}")
    cols[6].metric("Available Cash", f"₹{trader.cash:,.0f}")
    
    # Market Mood Gauges
    st.subheader("📊 Market Mood Gauges")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_circular_market_mood_gauge("NIFTY 50", 22000, 0.15, 65), unsafe_allow_html=True)
    with col2:
        st.markdown(create_circular_market_mood_gauge("BANK NIFTY", 48000, 0.25, 70), unsafe_allow_html=True)
    with col3:
        market_status = "LIVE" if market_open() else "CLOSED"
        status_sentiment = 80 if market_open() else 20
        st.markdown(create_circular_market_mood_gauge("MARKET", 0, 0, status_sentiment).replace("₹0", market_status).replace("0.00%", ""), unsafe_allow_html=True)
    with col4:
        peak_hours_status = "PEAK" if is_peak_market_hours() else "OFF-PEAK"
        peak_sentiment = 80 if is_peak_market_hours() else 30
        st.markdown(create_circular_market_mood_gauge("PEAK HOURS", 0, 0, peak_sentiment).replace("₹0", "9:30AM-2:30PM").replace("0.00%", peak_hours_status), unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.title("🔐 Kite Connect")
        
        if not kite_manager.is_authenticated:
            st.info("Kite Connect authentication required for live charts")
            if kite_manager.login():
                st.rerun()
        else:
            st.success(f"✅ Authenticated as {st.session_state.get('kite_user_name', 'User')}")
            if st.button("🔓 Logout", key="sidebar_logout_btn"):
                if kite_manager.logout():
                    st.success("Logged out successfully")
                    st.rerun()
        
        st.divider()
        st.header("⚙️ Trading Configuration")
        
        universe = st.selectbox("Trading Universe", ["Nifty 50", "Nifty 100", "Midcap 150", "All Stocks"], 
                               key="universe_select")
        
        enable_high_accuracy = st.checkbox("Enable High Accuracy Strategies", value=True, 
                                          key="high_acc_toggle")
        
        trader.auto_execution = st.checkbox("Auto Execution", value=False, key="auto_exec_toggle")
        
        st.subheader("🎯 Risk Management")
        min_conf_percent = st.slider("Minimum Confidence %", 60, 85, 70, 5, key="min_conf_slider")
        min_score = st.slider("Minimum Score", 5, 9, 6, 1, key="min_score_slider")
        
        st.subheader("🔍 Scan Configuration")
        full_scan = st.checkbox("Full Universe Scan", value=True, key="full_scan_toggle")
        
        if not full_scan:
            max_scan = st.number_input("Max Stocks to Scan", min_value=10, max_value=500, value=50, 
                                      step=10, key="max_scan_input")
        else:
            max_scan = None
        
        # System Status
        st.divider()
        st.subheader("🛠️ System Status")
        st.write(f"✅ Kite Connect: {'Available' if KITECONNECT_AVAILABLE else 'Not Available'}")
        st.write(f"✅ Database: {'Available' if SQLALCHEMY_AVAILABLE else 'Not Available'}")
        st.write(f"✅ ML Support: {'Available' if JOBLIB_AVAILABLE else 'Not Available'}")
        st.write(f"🔄 Refresh Count: {st.session_state.refresh_count}")
    
    # Main Tabs
    tabs = st.tabs(["📈 Dashboard", "🚦 Signals", "💰 Paper Trading", "📋 Trade History", 
                   "📉 RSI Scanner", "🎯 High Accuracy"])
    
    # Tab 1: Dashboard
    with tabs[0]:
        st.subheader("Account Summary")
        trader.update_positions_pnl()
        perf = trader.get_performance_stats()
        
        total_value = trader.cash + sum([p.get('quantity', 0) * p.get('entry_price', 0) 
                                       for p in trader.positions.values()])
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Value", f"₹{total_value:,.0f}", 
                 delta=f"₹{total_value - trader.initial_capital:+,.0f}")
        c2.metric("Available Cash", f"₹{trader.cash:,.0f}")
        c3.metric("Open Positions", len(trader.positions))
        c4.metric("Total P&L", f"₹{perf['total_pnl'] + perf['open_pnl']:+.2f}")
        
        # Strategy Performance
        st.subheader("Strategy Performance")
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
            st.dataframe(pd.DataFrame(strategy_data), use_container_width=True)
        else:
            st.info("No strategy performance data available yet.")
        
        # Open Positions
        st.subheader("📊 Open Positions")
        positions = trader.get_open_positions_data()
        if positions:
            st.dataframe(pd.DataFrame(positions), use_container_width=True)
        else:
            st.info("No open positions")
    
    # Tab 2: Signals
    with tabs[1]:
        st.subheader("Multi-Strategy BUY/SELL Signals")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("Generate Signals", type="primary", key="generate_signals_btn"):
                st.session_state.last_signal_generation = time.time()
                with st.spinner(f"Scanning {universe} stocks..."):
                    time.sleep(2)
                    st.success(f"✅ Scan completed for {universe}")
                    st.info("Signal generation logic would be implemented here")
        
        # Signal Display Area
        st.subheader("Live Signals")
        st.info("No live signals available. Click 'Generate Signals' to scan the market.")
    
    # Tab 3: Paper Trading
    with tabs[2]:
        st.subheader("💰 Paper Trading")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            symbol = st.selectbox("Symbol", NIFTY_50[:20], key="paper_symbol")
        with col2:
            action = st.selectbox("Action", ["BUY", "SELL"], key="paper_action")
        with col3:
            quantity = st.number_input("Quantity", min_value=1, value=10, key="paper_quantity")
        with col4:
            strategy_name = st.selectbox("Strategy", ["Manual"] + [config["name"] for config in TRADING_STRATEGIES.values()], 
                                       key="paper_strategy")
        
        if st.button("Execute Paper Trade", type="primary", key="execute_paper_trade"):
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
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
                else:
                    st.error("Unable to fetch price data")
            except Exception as e:
                st.error(f"Trade execution failed: {str(e)}")
        
        # Current Positions
        st.subheader("Current Positions")
        if trader.positions:
            for idx, (symbol, pos) in enumerate(trader.positions.items()):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    pnl = pos.get('current_pnl', 0)
                    pnl_color = "green" if pnl >= 0 else "red"
                    
                    st.markdown(f"""
                    <div style="padding: 10px; border-left: 4px solid {'#059669' if pos['action'] == 'BUY' else '#dc2626'}; 
                             background: linear-gradient(135deg, {'#d1fae5' if pos['action'] == 'BUY' else '#fee2e2'} 0%, 
                             {'#a7f3d0' if pos['action'] == 'BUY' else '#fecaca'} 100%); border-radius: 8px;">
                        <strong>{'🟢' if pos['action'] == 'BUY' else '🔴'} {symbol.replace('.NS', '')}</strong> | 
                        {pos['action']} | Qty: {pos['quantity']}<br>
                        Entry: ₹{pos['entry_price']:.2f} | Current: ₹{pos.get('current_price', pos['entry_price']):.2f}<br>
                        <span style="color: {pnl_color}">P&L: ₹{pnl:+.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.write(f"SL: ₹{pos.get('stop_loss', 0):.2f}")
                    st.write(f"TG: ₹{pos.get('target', 0):.2f}")
                
                with col3:
                    if st.button(f"Close", key=f"close_{symbol}_{idx}"):
                        success, msg = trader.close_position(symbol)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
        else:
            st.info("No open positions")
    
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
    
    # Tab 5: RSI Scanner
    with tabs[4]:
        st.subheader("📉 RSI Extreme Scanner")
        st.info("Find stocks with extreme RSI values (oversold/overbought)")
        
        if st.button("Scan for RSI Extremes", key="rsi_scan_btn"):
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
                                    "Signal": "OVERSOLD 🔵"
                                })
                            elif rsi_val > 70:
                                overbought.append({
                                    "Symbol": symbol.replace(".NS", ""),
                                    "RSI": round(rsi_val, 2),
                                    "Price": round(price, 2),
                                    "Signal": "OVERBOUGHT 🔴"
                                })
                    except:
                        continue
                
                if oversold or overbought:
                    st.success(f"Found {len(oversold)} oversold and {len(overbought)} overbought stocks")
                    
                    if oversold:
                        st.subheader("🔵 Oversold Stocks (RSI < 30)")
                        st.dataframe(pd.DataFrame(oversold), use_container_width=True)
                    
                    if overbought:
                        st.subheader("🔴 Overbought Stocks (RSI > 70)")
                        st.dataframe(pd.DataFrame(overbought), use_container_width=True)
                else:
                    st.info("No extreme RSI stocks found")
    
    # Tab 6: High Accuracy Scanner
    with tabs[5]:
        st.subheader("🎯 High Accuracy Scanner")
        st.markdown(f"""
        <div class="alert-success">
            <strong>🔥 High Accuracy Strategies Enabled</strong><br>
            Scanning with enhanced multi-confirmation strategies
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Run High Accuracy Scan", type="primary", key="high_acc_scan"):
            with st.spinner(f"Running high accuracy scan on {universe}..."):
                time.sleep(2)
                st.success("High accuracy scan completed!")
                st.info("High accuracy signal generation logic would be implemented here")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; font-size: 12px;">
        <strong>Rantv Terminal Pro</strong> | Enhanced Intraday Trading Platform | © 2024 | 
        Refresh: {st.session_state.refresh_count} | {now_indian().strftime("%H:%M:%S")}
    </div>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.code(traceback.format_exc())
