"""
Algo Trading Engine for Rantv Intraday Terminal Pro
Provides automated order execution, scheduling, and risk management
Enhanced with Kite Live Charts, Algo Trading, and High Accuracy Strategies
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
import smtplib
import random
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import websocket

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import warnings
import re

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
TALIB_AVAILABLE = False
WEBSOCKET_AVAILABLE = False

try:
    from kiteconnect import KiteConnect, KiteTicker
    KITECONNECT_AVAILABLE = True
except ImportError:
    if install_package("kiteconnect"):
        from kiteconnect import KiteConnect, KiteTicker
        KITECONNECT_AVAILABLE = True

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    if install_package("sqlalchemy"):
        import sqlalchemy
        from sqlalchemy import create_engine, text
        SQLALCHEMY_AVAILABLE = True

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    if install_package("joblib"):
        import joblib
        JOBLIB_AVAILABLE = True

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    if install_package("TA-Lib"):
        try:
            import talib
            TALIB_AVAILABLE = True
        except:
            TALIB_AVAILABLE = False
            st.warning("TA-Lib not available. Using custom indicators.")
    else:
        TALIB_AVAILABLE = False

# Check for websocket-client
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    if install_package("websocket-client"):
        import websocket
        WEBSOCKET_AVAILABLE = True
    else:
        WEBSOCKET_AVAILABLE = False
        st.warning("websocket-client not available. Real-time features disabled.")

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

# Email configuration for daily reports
EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "your_email@gmail.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = "rantv2002@gmail.com"

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
    
    /* Algo Trading Styles */
    .algo-status-running {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #059669;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .algo-status-stopped {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .algo-status-paused {
        background: linear-gradient(135deg, #ffe8cc 0%, #ffd9a6 100%);
        border-left: 4px solid #ff8c00;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
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
if 'algo_engine' not in st.session_state: st.session_state.algo_engine = None
if 'api_key_input' not in st.session_state: st.session_state.api_key_input = ""
if 'api_secret_input' not in st.session_state: st.session_state.api_secret_input = ""
if 'request_token_input' not in st.session_state: st.session_state.request_token_input = ""
if 'login_url_generated' not in st.session_state: st.session_state.login_url_generated = None
if 'generated_signals' not in st.session_state: st.session_state.generated_signals = []
if 'signal_quality' not in st.session_state: st.session_state.signal_quality = 0
if 'kite_ticker' not in st.session_state: st.session_state.kite_ticker = None
if 'algo_scheduler_started' not in st.session_state: st.session_state.algo_scheduler_started = False
if 'last_email_sent' not in st.session_state: st.session_state.last_email_sent = None

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

def should_exit_all_positions():
    """Check if it's time to exit all positions (3:35 PM)"""
    n = now_indian()
    try:
        exit_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 35)))
        return n >= exit_time
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

def send_daily_report_email(trader, algo_engine):
    """Send daily trading report email"""
    try:
        if not EMAIL_SENDER or not EMAIL_PASSWORD:
            logger.warning("Email credentials not configured")
            return False
        
        # Check if email was already sent today
        today = now_indian().date()
        if st.session_state.last_email_sent == today:
            logger.info("Daily report already sent today")
            return True
        
        # Prepare email content
        perf_stats = trader.get_performance_stats()
        algo_stats = algo_engine.get_status() if algo_engine else {}
        
        # Get today's trades
        today_trades = []
        for trade in trader.trade_log:
            trade_time = trade.get('timestamp')
            if trade_time and trade_time.date() == today:
                today_trades.append(trade)
        
        # Create HTML email
        subject = f"Daily Trading Report - {today.strftime('%Y-%m-%d')}"
        
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6;">
            <h2 style="color: #ff8c00;">📊 Daily Trading Report</h2>
            <p><strong>Date:</strong> {today.strftime('%Y-%m-%d')}</p>
            
            <h3 style="color: #333;">📈 Performance Summary</h3>
            <table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Trades</td>
                    <td>{perf_stats['total_trades']}</td>
                </tr>
                <tr>
                    <td>Win Rate</td>
                    <td>{perf_stats['win_rate']:.1%}</td>
                </tr>
                <tr>
                    <td>Total P&L</td>
                    <td style="color: {'green' if perf_stats['total_pnl'] >= 0 else 'red'}">
                        ₹{perf_stats['total_pnl']:+.2f}
                    </td>
                </tr>
                <tr>
                    <td>Open P&L</td>
                    <td style="color: {'green' if perf_stats['open_pnl'] >= 0 else 'red'}">
                        ₹{perf_stats['open_pnl']:+.2f}
                    </td>
                </tr>
                <tr>
                    <td>Average P&L per Trade</td>
                    <td>₹{perf_stats['avg_pnl']:+.2f}</td>
                </tr>
                <tr>
                    <td>Open Positions</td>
                    <td>{perf_stats['open_positions']}</td>
                </tr>
                <tr>
                    <td>Auto Trades</td>
                    <td>{perf_stats['auto_trades']}</td>
                </tr>
            </table>
            
            <h3 style="color: #333;">🤖 Algo Engine Status</h3>
            <table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>State</td>
                    <td>{algo_stats.get('state', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Active Positions</td>
                    <td>{algo_stats.get('active_positions', 0)}</td>
                </tr>
                <tr>
                    <td>Total Orders</td>
                    <td>{algo_stats.get('total_orders', 0)}</td>
                </tr>
                <tr>
                    <td>Realized P&L</td>
                    <td>₹{algo_stats.get('realized_pnl', 0):+.2f}</td>
                </tr>
                <tr>
                    <td>Trades Today</td>
                    <td>{algo_stats.get('trades_today', 0)}</td>
                </tr>
            </table>
            
            <h3 style="color: #333;">📋 Today's Trades ({len(today_trades)})</h3>
        """
        
        if today_trades:
            html_content += """
            <table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th>Symbol</th>
                    <th>Action</th>
                    <th>Quantity</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>P&L</th>
                    <th>Strategy</th>
                    <th>Auto</th>
                </tr>
            """
            
            for trade in today_trades:
                pnl = trade.get('closed_pnl', trade.get('current_pnl', 0))
                html_content += f"""
                <tr>
                    <td>{trade['symbol'].replace('.NS', '')}</td>
                    <td>{trade['action']}</td>
                    <td>{trade['quantity']}</td>
                    <td>₹{trade['entry_price']:.2f}</td>
                    <td>₹{trade.get('exit_price', trade.get('current_price', 0)):.2f}</td>
                    <td style="color: {'green' if pnl >= 0 else 'red'}">₹{pnl:+.2f}</td>
                    <td>{trade.get('strategy', 'Manual')}</td>
                    <td>{'Yes' if trade.get('auto_trade') else 'No'}</td>
                </tr>
                """
            
            html_content += "</table>"
        else:
            html_content += "<p>No trades executed today.</p>"
        
        html_content += """
            <hr>
            <p style="color: #666; font-size: 12px;">
                This is an automated daily report from Rantv Terminal Pro.<br>
                Generated at: """ + now_indian().strftime("%Y-%m-%d %H:%M:%S") + """
            </p>
        </body>
        </html>
        """
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        
        # Attach HTML
        msg.attach(MIMEText(html_content, 'html'))
        
        # Send email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Daily report email sent to {EMAIL_RECEIVER}")
        st.session_state.last_email_sent = today
        return True
        
    except Exception as e:
        logger.error(f"Failed to send daily report email: {e}")
        return False

# ===================== KITE LIVE TICKER =====================
class KiteLiveTicker:
    """Live WebSocket ticker for Kite Connect"""
    
    def __init__(self, kite_manager):
        self.kite_manager = kite_manager
        self.ws = None
        self.tick_data = {}
        self.subscribed_tokens = set()
        self.is_connected = False
        self.last_update = {}
        self.candle_data = {}  # Store candle data for each token
        self.thread = None
        
    def connect(self):
        """Connect to Kite WebSocket"""
        if not self.kite_manager.is_authenticated or not self.kite_manager.access_token:
            return False
            
        try:
            access_token = self.kite_manager.access_token
            api_key = self.kite_manager.api_key
            
            # WebSocket URL
            ws_url = f"wss://ws.kite.trade?api_key={api_key}&access_token={access_token}"
            
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start WebSocket in a separate thread
            self.thread = threading.Thread(target=self.ws.run_forever)
            self.thread.daemon = True
            self.thread.start()
            
            self.is_connected = True
            logger.info("Kite WebSocket connected")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect WebSocket"""
        if self.ws:
            self.ws.close()
        self.is_connected = False
        self.subscribed_tokens.clear()
        logger.info("Kite WebSocket disconnected")
    
    def _on_open(self, ws):
        logger.info("WebSocket connection opened")
        # Subscribe to default indices
        self.subscribe([256265, 260105])  # NIFTY 50, BANKNIFTY
    
    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            
            if isinstance(data, dict) and 'type' in data:
                if data['type'] == 'ticks':
                    self._process_ticks(data)
                elif data['type'] == 'order_update':
                    logger.info(f"Order update: {data}")
            
        except Exception as e:
            logger.error(f"WebSocket message error: {e}")
    
    def _process_ticks(self, data):
        """Process tick data"""
        ticks = data.get('ticks', [])
        
        for tick in ticks:
            token = tick.get('instrument_token')
            if not token:
                continue
                
            # Store tick data
            self.tick_data[token] = tick
            self.last_update[token] = time.time()
            
            # Update candle data
            self._update_candle_data(token, tick)
    
    def _update_candle_data(self, token, tick):
        """Update candle data from ticks"""
        if token not in self.candle_data:
            self.candle_data[token] = {
                'open': [],
                'high': [],
                'low': [],
                'close': [],
                'volume': [],
                'timestamp': []
            }
        
        current_time = time.time()
        candle_interval = 60  # 1-minute candles
        
        # Get or create current candle
        if not self.candle_data[token]['timestamp']:
            # First candle
            candle_start = current_time - (current_time % candle_interval)
        else:
            candle_start = self.candle_data[token]['timestamp'][-1]
        
        if current_time - candle_start >= candle_interval:
            # Create new candle
            new_candle_start = current_time - (current_time % candle_interval)
            self.candle_data[token]['timestamp'].append(new_candle_start)
            self.candle_data[token]['open'].append(tick.get('last_price', 0))
            self.candle_data[token]['high'].append(tick.get('last_price', 0))
            self.candle_data[token]['low'].append(tick.get('last_price', 0))
            self.candle_data[token]['close'].append(tick.get('last_price', 0))
            self.candle_data[token]['volume'].append(tick.get('volume_traded', 0))
        else:
            # Update current candle
            if self.candle_data[token]['high']:
                idx = -1
                current_price = tick.get('last_price', 0)
                
                # Update high
                if current_price > self.candle_data[token]['high'][idx]:
                    self.candle_data[token]['high'][idx] = current_price
                
                # Update low
                if current_price < self.candle_data[token]['low'][idx]:
                    self.candle_data[token]['low'][idx] = current_price
                
                # Update close
                self.candle_data[token]['close'][idx] = current_price
                
                # Update volume
                self.candle_data[token]['volume'][idx] = tick.get('volume_traded', 0)
    
    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.is_connected = False
        self.subscribed_tokens.clear()
    
    def subscribe(self, tokens):
        """Subscribe to instruments"""
        if not self.is_connected or not self.ws:
            return False
            
        try:
            # Add to subscription list
            for token in tokens:
                self.subscribed_tokens.add(token)
            
            # Send subscription message
            subscribe_msg = {
                "a": "subscribe",
                "v": tokens
            }
            
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to tokens: {tokens}")
            return True
            
        except Exception as e:
            logger.error(f"Subscription error: {e}")
            return False
    
    def unsubscribe(self, tokens):
        """Unsubscribe from instruments"""
        if not self.is_connected or not self.ws:
            return False
            
        try:
            # Remove from subscription list
            for token in tokens:
                self.subscribed_tokens.discard(token)
            
            # Send unsubscribe message
            unsubscribe_msg = {
                "a": "unsubscribe",
                "v": tokens
            }
            
            self.ws.send(json.dumps(unsubscribe_msg))
            logger.info(f"Unsubscribed from tokens: {tokens}")
            return True
            
        except Exception as e:
            logger.error(f"Unsubscribe error: {e}")
            return False
    
    def get_latest_tick(self, token):
        """Get latest tick for a token"""
        return self.tick_data.get(token)
    
    def get_candle_data(self, token, num_candles=100):
        """Get candle data for a token"""
        if token not in self.candle_data:
            return None
        
        data = self.candle_data[token]
        if not data['timestamp']:
            return None
        
        # Return last N candles
        n = min(num_candles, len(data['timestamp']))
        
        return {
            'timestamp': data['timestamp'][-n:],
            'open': data['open'][-n:],
            'high': data['high'][-n:],
            'low': data['low'][-n:],
            'close': data['close'][-n:],
            'volume': data['volume'][-n:]
        }
    
    def get_last_update_time(self, token):
        """Get last update time for a token"""
        return self.last_update.get(token, 0)

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

            st.success(f"✅ Authenticated as {data.get('user_name', 'User')}")
            st.balloons()
            st.session_state.kite_oauth_in_progress = False
            time.sleep(1)
            st.rerun()
            return True

        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            self._clear_query_params()
            st.session_state.kite_oauth_in_progress = False
            st.error(f"Token exchange failed: {str(e)}")
            return False

    def get_login_url(self) -> Optional[str]:
        """Get Kite login URL"""
        try:
            if not self.api_key:
                return None
            if not self.kite:
                self.kite = KiteConnect(api_key=self.api_key)
            return self.kite.login_url()
        except Exception as e:
            logger.error(f"Error getting login URL: {e}")
            return None

    def login_ui(self):
        """Render Kite login UI in sidebar"""
        with st.sidebar:
            st.title("🔐 Kite Connect Login")
            
            # Check for OAuth callback
            if self.api_key and self.api_secret and not st.session_state.kite_oauth_in_progress:
                self.check_oauth_callback()
            
            # If already authenticated, show user info
            if self.is_authenticated:
                st.success(f"✅ Authenticated as {st.session_state.get('kite_user_name', 'User')}")
                if st.button("🔓 Logout", key="kite_logout_btn"):
                    if self.logout():
                        st.success("Logged out successfully")
                        st.rerun()
                return
            
            # Show login form
            st.info("Enter Kite Connect credentials to access live charts")
            
            api_key = st.text_input("API Key", value=st.session_state.get("api_key_input", ""), 
                                   type="password", key="kite_api_key_input")
            api_secret = st.text_input("API Secret", value=st.session_state.get("api_secret_input", ""), 
                                      type="password", key="kite_api_secret_input")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Generate Login URL", type="primary", key="generate_login_url_btn"):
                    if api_key:
                        st.session_state.api_key_input = api_key
                        st.session_state.api_secret_input = api_secret
                        self.api_key = api_key
                        self.api_secret = api_secret
                        
                        login_url = self.get_login_url()
                        if login_url:
                            st.session_state.login_url_generated = login_url
                            st.success("Login URL generated! Click the link below:")
                            st.markdown(f"[🔗 Click here to login to Kite]({login_url})")
                            st.code(login_url)
                            
                            # Try to open browser automatically
                            try:
                                webbrowser.open(login_url, new=2)
                                st.info("Browser opened automatically. If not, click the link above.")
                            except:
                                st.info("Please copy the URL above and open in your browser.")
                        else:
                            st.error("Failed to generate login URL. Check API key.")
                    else:
                        st.error("Please enter API Key")
            
            with col2:
                st.markdown("**Or enter token manually:**")
                request_token = st.text_input("Request Token", key="kite_request_token_input")
                
                if st.button("Complete Authentication", key="complete_auth_btn"):
                    if api_key and api_secret and request_token:
                        success = self.exchange_request_token(request_token)
                        if success:
                            st.rerun()
                    else:
                        st.error("Please fill all fields")

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

    def get_historical_data(self, instrument_token, interval="minute", days=7):
        """Get historical data from Kite"""
        if not self.is_authenticated or not self.kite:
            return None
        
        try:
            from_date = datetime.now().date() - timedelta(days=days)
            to_date = datetime.now().date()
            
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date.strftime("%Y-%m-%d"),
                to_date=to_date.strftime("%Y-%m-%d"),
                interval=interval,
                continuous=False,
                oi=False
            )
            
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
            return None
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None

# ===================== PART 6: ALGO TRADING ENGINE =====================
class AlgoState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"

class OrderStatus(Enum):
    PENDING = "pending"
    PLACED = "placed"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class AlgoOrder:
    order_id: str
    symbol: str
    action: str
    quantity: int
    price: float
    stop_loss: float
    target: float
    strategy: str
    confidence: float
    status: OrderStatus = OrderStatus.PENDING
    placed_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    error_message: Optional[str] = None

@dataclass
class RiskLimits:
    max_positions: int = 5
    max_daily_loss: float = 50000.0
    max_position_size: float = 100000.0
    min_confidence: float = 0.70
    max_trades_per_day: int = 10
    max_trades_per_stock: int = 2

@dataclass
class AlgoStats:
    total_orders: int = 0
    filled_orders: int = 0
    rejected_orders: int = 0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    daily_loss: float = 0.0
    trades_today: int = 0

class AlgoEngine:
    """Algorithmic Trading Engine with Daily Exit at 3:35 PM"""
    
    def __init__(self, trader=None):
        self.state = AlgoState.STOPPED
        self.trader = trader
        self.risk_limits = RiskLimits()
        self.stats = AlgoStats()
        self.orders: Dict[str, AlgoOrder] = {}
        self.active_positions: Dict[str, AlgoOrder] = {}
        self.order_history: List[AlgoOrder] = []
        
        self._stop_event = threading.Event()
        self._scheduler_thread = None
        self._lock = threading.Lock()
        
        self.daily_exit_completed = False
        self.last_signal_scan = 0
        
        logger.info("AlgoEngine initialized")
    
    def start(self) -> bool:
        if self.state == AlgoState.RUNNING:
            return False
        
        self.state = AlgoState.RUNNING
        self._stop_event.clear()
        
        self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        
        logger.info("AlgoEngine started")
        return True
    
    def stop(self):
        logger.info("Stopping AlgoEngine...")
        self.state = AlgoState.STOPPED
        self._stop_event.set()
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        
        logger.info("AlgoEngine stopped")
    
    def pause(self):
        self.state = AlgoState.PAUSED
        logger.info("AlgoEngine paused")
    
    def resume(self):
        if self.state == AlgoState.PAUSED:
            self.state = AlgoState.RUNNING
            logger.info("AlgoEngine resumed")
    
    def emergency_stop(self, reason: str = "Manual trigger"):
        logger.critical(f"EMERGENCY STOP: {reason}")
        self.state = AlgoState.EMERGENCY_STOP
        self._stop_event.set()
    
    def _run_scheduler(self):
        logger.info("Algo scheduler thread started")
        
        while not self._stop_event.is_set():
            try:
                if self.state != AlgoState.RUNNING:
                    time.sleep(1)
                    continue
                
                # Check market hours
                if not market_open():
                    time.sleep(10)
                    continue
                
                # Check if it's time to exit all positions (3:35 PM)
                if should_exit_all_positions() and not self.daily_exit_completed:
                    logger.info("3:35 PM - Exiting all positions")
                    self._exit_all_positions()
                    self.daily_exit_completed = True
                    
                    # Send daily report email
                    if self.trader:
                        send_daily_report_email(self.trader, self)
                    
                    time.sleep(60)  # Sleep for a minute after exit
                    continue
                
                # Reset daily exit flag at market open
                current_time = now_indian()
                if current_time.hour == 9 and current_time.minute < 30:
                    self.daily_exit_completed = False
                
                # Update positions P&L
                if self.trader:
                    self.trader.update_positions_pnl()
                
                # Check risk limits
                self._check_risk_limits()
                
                # Generate and process signals periodically
                current_time_ts = time.time()
                if current_time_ts - self.last_signal_scan > 300:  # Every 5 minutes
                    self._scan_and_process_signals()
                    self.last_signal_scan = current_time_ts
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(10)
        
        logger.info("Algo scheduler thread stopped")
    
    def _scan_and_process_signals(self):
        """Scan for signals and process them"""
        if not self.trader or not self.trader.signal_generator:
            return
        
        try:
            # Generate signals for Nifty 50
            signals = self.trader.signal_generator.scan_stock_universe(
                universe="Nifty 50",
                max_stocks=30,
                min_confidence=self.risk_limits.min_confidence
            )
            
            # Process signals
            if signals:
                executed_signals = self.process_signals(signals)
                if executed_signals:
                    logger.info(f"Executed {len(executed_signals)} signals")
        
        except Exception as e:
            logger.error(f"Error in signal scanning: {e}")
    
    def _exit_all_positions(self):
        """Exit all active positions"""
        if not self.trader:
            return
        
        logger.info("Exiting all algo positions")
        
        # Close all positions through trader
        for symbol in list(self.trader.positions.keys()):
            try:
                success, msg = self.trader.close_position(symbol)
                if success:
                    logger.info(f"Closed position: {symbol} - {msg}")
                    
                    # Update algo order status
                    if symbol in self.active_positions:
                        order = self.active_positions[symbol]
                        order.status = OrderStatus.FILLED
                        order.filled_at = now_indian()
                        
                        # Calculate P&L
                        if order.filled_price:
                            if order.action == "BUY":
                                pnl = (order.filled_price - order.price) * order.quantity
                            else:
                                pnl = (order.price - order.filled_price) * order.quantity
                            
                            self.stats.realized_pnl += pnl
                            self.stats.total_pnl += pnl
                            
                            if pnl > 0:
                                self.stats.win_count += 1
                            else:
                                self.stats.loss_count += 1
                else:
                    logger.warning(f"Failed to close position: {symbol} - {msg}")
            except Exception as e:
                logger.error(f"Error closing position {symbol}: {e}")
        
        # Clear active positions
        self.active_positions.clear()
    
    def _check_risk_limits(self):
        """Check if risk limits are breached"""
        if not self.trader:
            return
        
        # Calculate current P&L
        total_pnl = sum([p.get('current_pnl', 0) for p in self.trader.positions.values()])
        
        # Check daily loss limit
        if total_pnl < -self.risk_limits.max_daily_loss:
            self.emergency_stop(f"Daily loss limit exceeded: ₹{total_pnl:.2f}")
    
    def process_signals(self, signals):
        """Process generated signals for algo trading"""
        if self.state != AlgoState.RUNNING:
            return []
        
        executed_signals = []
        
        for signal in signals:
            # Check if we already have position in this symbol
            if signal['symbol'] in self.active_positions:
                continue
            
            # Check risk limits
            if len(self.active_positions) >= self.risk_limits.max_positions:
                logger.info("Max positions limit reached")
                break
            
            if self.stats.trades_today >= self.risk_limits.max_trades_per_day:
                logger.info("Daily trade limit reached")
                break
            
            # Check confidence threshold
            if signal['confidence'] < self.risk_limits.min_confidence:
                continue
            
            # Execute trade through trader
            if self.trader:
                success, msg = self.trader.execute_auto_trade_from_signal(signal)
                
                if success:
                    # Create algo order record
                    order_id = f"ALGO_{signal['symbol']}_{int(time.time())}"
                    order = AlgoOrder(
                        order_id=order_id,
                        symbol=signal['symbol'],
                        action=signal['action'],
                        quantity=signal.get('quantity', 10),
                        price=signal['price'],
                        stop_loss=signal['stop_loss'],
                        target=signal['target'],
                        strategy=signal['strategy'],
                        confidence=signal['confidence'],
                        status=OrderStatus.PLACED,
                        placed_at=datetime.now()
                    )
                    
                    self.orders[order_id] = order
                    self.active_positions[signal['symbol']] = order
                    self.order_history.append(order)
                    
                    self.stats.total_orders += 1
                    self.stats.trades_today += 1
                    
                    executed_signals.append(signal)
                    
                    logger.info(f"Algo executed: {signal['symbol']} {signal['action']}")
        
        return executed_signals
    
    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "active_positions": len(self.active_positions),
            "total_orders": self.stats.total_orders,
            "filled_orders": self.stats.filled_orders,
            "trades_today": self.stats.trades_today,
            "realized_pnl": self.stats.realized_pnl,
            "unrealized_pnl": self.stats.unrealized_pnl,
            "daily_loss": self.stats.daily_loss,
            "daily_exit_completed": self.daily_exit_completed
        }
    
    def update_risk_limits(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.risk_limits, key):
                setattr(self.risk_limits, key, value)
                logger.info(f"Updated risk limit: {key} = {value}")

# ===================== PART 7: DATA MANAGER =====================
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
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()
        
        sr = calculate_support_resistance_advanced(df["High"], df["Low"], df["Close"])
        df["Support"] = sr["support"]
        df["Resistance"] = sr["resistance"]
        
        return df

# ===================== PART 8: TRADING ENGINE =====================
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
        
        self.data_manager = EnhancedDataManager()
        self.signal_generator = SignalGenerator(self.data_manager)
        
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

    def execute_auto_trade_from_signal(self, signal, max_quantity=50):
        """Execute auto trade based on generated signal"""
        if not self.can_auto_trade():
            return False, "Auto trading limits reached"
        
        symbol = signal['symbol']
        action = signal['action']
        price = signal['price']
        stop_loss = signal['stop_loss']
        target = signal['target']
        strategy = signal['strategy']
        confidence = signal['confidence']
        
        # Calculate position size based on confidence and risk
        position_size_pct = min(0.2, confidence * 0.25)  # Max 20% per trade
        max_trade_value = self.cash * position_size_pct
        quantity = int(max_trade_value / price)
        
        # Apply limits
        quantity = min(quantity, max_quantity)
        
        if quantity < 1:
            return False, "Position size too small"
        
        # Execute trade
        success, msg = self.execute_trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            target=target,
            win_probability=signal['win_probability'],
            auto_trade=True,
            strategy=strategy
        )
        
        if success:
            logger.info(f"Auto trade executed: {msg}")
        
        return success, msg

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

# ===================== SIGNAL GENERATOR CLASS =====================
class SignalGenerator:
    """Generate trading signals for algo paper trading"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.signals_generated = 0
        self.last_scan_time = None
        self.signal_cache = {}
        
    def calculate_signal_quality(self, signals):
        """Calculate overall quality score for signals"""
        if not signals:
            return 0.0
        
        avg_confidence = sum(s.get('confidence', 0) for s in signals) / len(signals)
        avg_score = sum(s.get('score', 0) for s in signals) / len(signals)
        
        # Multiplier based on market conditions
        market_multiplier = 1.0
        if is_peak_market_hours():
            market_multiplier = 1.2
        if not market_open():
            market_multiplier = 0.5
        
        return min(100.0, (avg_confidence * 10 + avg_score * 10) * market_multiplier)
    
    def generate_multi_strategy_signal(self, symbol, data):
        """Generate signal using multiple strategy confluence"""
        if data is None or len(data) < 50:
            return None
        
        try:
            signals = []
            current_price = float(data['Close'].iloc[-1])
            
            # 1. EMA Strategy
            ema_signal = self._check_ema_strategy(data)
            if ema_signal:
                signals.append(ema_signal)
            
            # 2. RSI Strategy
            rsi_signal = self._check_rsi_strategy(data)
            if rsi_signal:
                signals.append(rsi_signal)
            
            # 3. MACD Strategy
            macd_signal = self._check_macd_strategy(data)
            if macd_signal:
                signals.append(macd_signal)
            
            # 4. Bollinger Bands Strategy
            bb_signal = self._check_bollinger_strategy(data)
            if bb_signal:
                signals.append(bb_signal)
            
            # 5. Volume Strategy
            volume_signal = self._check_volume_strategy(data)
            if volume_signal:
                signals.append(volume_signal)
            
            # 6. Support/Resistance Strategy
            sr_signal = self._check_support_resistance_strategy(data, current_price)
            if sr_signal:
                signals.append(sr_signal)
            
            if not signals:
                return None
            
            # Combine signals
            buy_signals = [s for s in signals if s['action'] == 'BUY']
            sell_signals = [s for s in signals if s['action'] == 'SELL']
            
            # Calculate weighted score
            buy_score = sum(s['weight'] * s['confidence'] for s in buy_signals)
            sell_score = sum(s['weight'] * s['confidence'] for s in sell_signals)
            
            # Determine final action
            if len(buy_signals) > len(sell_signals) and buy_score > sell_score:
                action = 'BUY'
                confidence = buy_score / max(1, len(buy_signals))
                score = len(buy_signals)
                strategy = "Multi-Strategy Confluence"
            elif len(sell_signals) > len(buy_signals) and sell_score > buy_score:
                action = 'SELL'
                confidence = sell_score / max(1, len(sell_signals))
                score = len(sell_signals)
                strategy = "Multi-Strategy Confluence"
            else:
                # Neutral or conflicting signals
                return None
            
            # Calculate stop loss and target
            atr = float(data['ATR'].iloc[-1]) if 'ATR' in data.columns else current_price * 0.02
            
            if action == 'BUY':
                stop_loss = current_price - (atr * 1.5)
                target = current_price + (atr * 3.0)
            else:
                stop_loss = current_price + (atr * 1.5)
                target = current_price - (atr * 3.0)
            
            # Calculate win probability based on confidence and market conditions
            base_prob = min(0.95, confidence * 0.8)
            if is_peak_market_hours():
                base_prob = min(0.97, base_prob * 1.1)
            
            return {
                'symbol': symbol,
                'action': action,
                'price': current_price,
                'stop_loss': round(stop_loss, 2),
                'target': round(target, 2),
                'confidence': round(confidence, 3),
                'score': score,
                'strategy': strategy,
                'win_probability': round(base_prob, 3),
                'timestamp': now_indian(),
                'atr': round(atr, 2),
                'signal_count': len(signals),
                'rsi': float(data['RSI14'].iloc[-1]) if 'RSI14' in data.columns else 50
            }
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _check_ema_strategy(self, data):
        """Check EMA crossover strategy"""
        try:
            if len(data) < 50:
                return None
            
            ema8 = data['EMA8'].iloc[-1]
            ema21 = data['EMA21'].iloc[-1]
            ema50 = data['EMA50'].iloc[-1]
            price = data['Close'].iloc[-1]
            
            # Bullish: Price above all EMAs and EMA8 > EMA21 > EMA50
            if price > ema8 > ema21 > ema50:
                return {
                    'action': 'BUY',
                    'confidence': 0.85,
                    'weight': 2,
                    'strategy': 'EMA Golden Cross'
                }
            
            # Bearish: Price below all EMAs and EMA8 < EMA21 < EMA50
            elif price < ema8 < ema21 < ema50:
                return {
                    'action': 'SELL',
                    'confidence': 0.85,
                    'weight': 2,
                    'strategy': 'EMA Death Cross'
                }
            
            # EMA8 crossover EMA21
            ema8_prev = data['EMA8'].iloc[-2]
            ema21_prev = data['EMA21'].iloc[-2]
            
            if ema8 > ema21 and ema8_prev <= ema21_prev:
                return {
                    'action': 'BUY',
                    'confidence': 0.75,
                    'weight': 1,
                    'strategy': 'EMA8/21 Crossover'
                }
            elif ema8 < ema21 and ema8_prev >= ema21_prev:
                return {
                    'action': 'SELL',
                    'confidence': 0.75,
                    'weight': 1,
                    'strategy': 'EMA8/21 Crossover'
                }
            
            return None
            
        except Exception:
            return None
    
    def _check_rsi_strategy(self, data):
        """Check RSI strategy"""
        try:
            if 'RSI14' not in data.columns:
                return None
            
            rsi = data['RSI14'].iloc[-1]
            rsi_prev = data['RSI14'].iloc[-2] if len(data) > 1 else rsi
            
            # Oversold bounce
            if rsi < 30 and rsi > rsi_prev:
                return {
                    'action': 'BUY',
                    'confidence': 0.80,
                    'weight': 1.5,
                    'strategy': 'RSI Oversold Bounce'
                }
            
            # Overbought reversal
            elif rsi > 70 and rsi < rsi_prev:
                return {
                    'action': 'SELL',
                    'confidence': 0.80,
                    'weight': 1.5,
                    'strategy': 'RSI Overbought Reversal'
                }
            
            # RSI divergence (simplified)
            if len(data) > 20:
                prices = data['Close'].iloc[-20:]
                rsis = data['RSI14'].iloc[-20:]
                
                # Bullish divergence: Lower lows in price, higher lows in RSI
                if (prices.iloc[-1] < prices.iloc[-5] and 
                    rsis.iloc[-1] > rsis.iloc[-5] and 
                    rsi < 45):
                    return {
                        'action': 'BUY',
                        'confidence': 0.85,
                        'weight': 2,
                        'strategy': 'RSI Bullish Divergence'
                    }
                
                # Bearish divergence: Higher highs in price, lower highs in RSI
                elif (prices.iloc[-1] > prices.iloc[-5] and 
                      rsis.iloc[-1] < rsis.iloc[-5] and 
                      rsi > 55):
                    return {
                        'action': 'SELL',
                        'confidence': 0.85,
                        'weight': 2,
                        'strategy': 'RSI Bearish Divergence'
                    }
            
            return None
            
        except Exception:
            return None
    
    def _check_macd_strategy(self, data):
        """Check MACD strategy"""
        try:
            if 'MACD' not in data.columns or 'MACD_Signal' not in data.columns:
                return None
            
            macd = data['MACD'].iloc[-1]
            signal = data['MACD_Signal'].iloc[-1]
            hist = data['MACD_Hist'].iloc[-1]
            
            macd_prev = data['MACD'].iloc[-2] if len(data) > 1 else macd
            signal_prev = data['MACD_Signal'].iloc[-2] if len(data) > 1 else signal
            hist_prev = data['MACD_Hist'].iloc[-2] if len(data) > 1 else hist
            
            # MACD crossover signal line
            if macd > signal and macd_prev <= signal_prev:
                return {
                    'action': 'BUY',
                    'confidence': 0.75,
                    'weight': 1,
                    'strategy': 'MACD Bullish Crossover'
                }
            elif macd < signal and macd_prev >= signal_prev:
                return {
                    'action': 'SELL',
                    'confidence': 0.75,
                    'weight': 1,
                    'strategy': 'MACD Bearish Crossover'
                }
            
            # MACD histogram turning positive/negative
            if hist > 0 and hist_prev <= 0:
                return {
                    'action': 'BUY',
                    'confidence': 0.70,
                    'weight': 0.8,
                    'strategy': 'MACD Histogram Turn'
                }
            elif hist < 0 and hist_prev >= 0:
                return {
                    'action': 'SELL',
                    'confidence': 0.70,
                    'weight': 0.8,
                    'strategy': 'MACD Histogram Turn'
                }
            
            return None
            
        except Exception:
            return None
    
    def _check_bollinger_strategy(self, data):
        """Check Bollinger Bands strategy"""
        try:
            if 'BB_Upper' not in data.columns or 'BB_Lower' not in data.columns:
                return None
            
            price = data['Close'].iloc[-1]
            bb_upper = data['BB_Upper'].iloc[-1]
            bb_lower = data['BB_Lower'].iloc[-1]
            bb_middle = data['BB_Middle'].iloc[-1]
            
            # Price touches lower band and starts moving up
            if price <= bb_lower * 1.005 and data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                return {
                    'action': 'BUY',
                    'confidence': 0.80,
                    'weight': 1.5,
                    'strategy': 'Bollinger Band Bounce'
                }
            
            # Price touches upper band and starts moving down
            elif price >= bb_upper * 0.995 and data['Close'].iloc[-1] < data['Close'].iloc[-2]:
                return {
                    'action': 'SELL',
                    'confidence': 0.80,
                    'weight': 1.5,
                    'strategy': 'Bollinger Band Rejection'
                }
            
            # Bollinger Band squeeze breakout
            if len(data) > 20:
                bb_width = (bb_upper - bb_lower) / bb_middle
                bb_width_prev = (data['BB_Upper'].iloc[-2] - data['BB_Lower'].iloc[-2]) / data['BB_Middle'].iloc[-2]
                
                # Squeeze followed by expansion
                if bb_width_prev < 0.05 and bb_width > bb_width_prev * 1.2:
                    if price > bb_middle:
                        return {
                            'action': 'BUY',
                            'confidence': 0.85,
                            'weight': 2,
                            'strategy': 'BB Squeeze Breakout Up'
                        }
                    else:
                        return {
                            'action': 'SELL',
                            'confidence': 0.85,
                            'weight': 2,
                            'strategy': 'BB Squeeze Breakout Down'
                        }
            
            return None
            
        except Exception:
            return None
    
    def _check_volume_strategy(self, data):
        """Check volume-based strategies"""
        try:
            if len(data) < 10:
                return None
            
            volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            price_change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
            
            # High volume breakout
            if volume > avg_volume * 1.5 and abs(price_change) > 1:
                if price_change > 0:
                    return {
                        'action': 'BUY',
                        'confidence': 0.75,
                        'weight': 1.2,
                        'strategy': 'Volume Breakout Up'
                    }
                else:
                    return {
                        'action': 'SELL',
                        'confidence': 0.75,
                        'weight': 1.2,
                        'strategy': 'Volume Breakout Down'
                    }
            
            return None
            
        except Exception:
            return None
    
    def _check_support_resistance_strategy(self, data, current_price):
        """Check support/resistance strategy"""
        try:
            if 'Support' not in data.columns or 'Resistance' not in data.columns:
                return None
            
            support = data['Support'].iloc[-1]
            resistance = data['Resistance'].iloc[-1]
            
            # Near support with bounce
            if current_price <= support * 1.01 and current_price > data['Close'].iloc[-2]:
                return {
                    'action': 'BUY',
                    'confidence': 0.80,
                    'weight': 1.5,
                    'strategy': 'Support Bounce'
                }
            
            # Near resistance with rejection
            elif current_price >= resistance * 0.99 and current_price < data['Close'].iloc[-2]:
                return {
                    'action': 'SELL',
                    'confidence': 0.80,
                    'weight': 1.5,
                    'strategy': 'Resistance Rejection'
                }
            
            # Breakout above resistance
            if current_price > resistance and data['Close'].iloc[-2] <= resistance:
                return {
                    'action': 'BUY',
                    'confidence': 0.85,
                    'weight': 2,
                    'strategy': 'Resistance Breakout'
                }
            
            # Breakdown below support
            elif current_price < support and data['Close'].iloc[-2] >= support:
                return {
                    'action': 'SELL',
                    'confidence': 0.85,
                    'weight': 2,
                    'strategy': 'Support Breakdown'
                }
            
            return None
            
        except Exception:
            return None
    
    def scan_stock_universe(self, universe, max_stocks=50, min_confidence=0.70):
        """Scan stock universe for trading signals"""
        signals = []
        cache_key = f"scan_{universe}_{max_stocks}"
        
        # Check cache
        if cache_key in self.signal_cache:
            cached = self.signal_cache[cache_key]
            if (time.time() - cached['timestamp']) < 300:  # 5 minute cache
                return cached['signals']
        
        # Determine which stocks to scan
        if universe == "Nifty 50":
            stocks = NIFTY_50[:min(max_stocks, len(NIFTY_50))]
        elif universe == "Nifty 100":
            stocks = NIFTY_100[:min(max_stocks, len(NIFTY_100))]
        elif universe == "Midcap 150":
            stocks = NIFTY_MIDCAP_150[:min(max_stocks, len(NIFTY_MIDCAP_150))]
        else:
            stocks = ALL_STOCKS[:min(max_stocks, len(ALL_STOCKS))]
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(stocks):
            status_text.text(f"Scanning {symbol.replace('.NS', '')}... ({i+1}/{len(stocks)})")
            
            try:
                # Get stock data
                data = self.data_manager.get_stock_data(symbol, "15m")
                
                if data is not None and len(data) > 50:
                    # Generate signal
                    signal = self.generate_multi_strategy_signal(symbol, data)
                    
                    if signal and signal['confidence'] >= min_confidence:
                        signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
            
            # Update progress
            progress_bar.progress((i + 1) / len(stocks))
        
        progress_bar.empty()
        status_text.empty()
        
        # Sort signals by confidence score
        signals.sort(key=lambda x: x['confidence'] * x['score'], reverse=True)
        
        # Cache results
        self.signal_cache[cache_key] = {
            'signals': signals,
            'timestamp': time.time(),
            'count': len(signals)
        }
        
        self.signals_generated += len(signals)
        self.last_scan_time = now_indian()
        
        return signals

# ===================== UPDATED KITE LIVE CHARTS =====================
def create_kite_live_charts(kite_manager):
    """Create Kite Connect Live Charts Tab with Real-Time Ticks"""
    st.subheader("📈 Kite Connect Live Charts (Real-Time)")
    
    if not kite_manager.is_authenticated:
        st.info("Please authenticate with Kite Connect in the sidebar to view live charts")
        return
    
    st.success(f"✅ Authenticated as {st.session_state.get('kite_user_name', 'User')}")
    
    # Initialize WebSocket ticker
    if st.session_state.kite_ticker is None:
        st.session_state.kite_ticker = KiteLiveTicker(kite_manager)
    
    ticker = st.session_state.kite_ticker
    
    # WebSocket connection controls
    col1, col2 = st.columns(2)
    
    with col1:
        if not ticker.is_connected:
            if st.button("🔗 Connect Live Feed", type="primary", key="connect_ws_btn"):
                if ticker.connect():
                    st.success("Live feed connected!")
                    st.rerun()
                else:
                    st.error("Failed to connect live feed")
        else:
            if st.button("🔴 Disconnect Live Feed", type="secondary", key="disconnect_ws_btn"):
                ticker.disconnect()
                st.info("Live feed disconnected")
                st.rerun()
    
    with col2:
        if ticker.is_connected:
            status_color = "🟢"
            status_text = "CONNECTED"
        else:
            status_color = "🔴"
            status_text = "DISCONNECTED"
        
        st.metric("Live Feed Status", f"{status_color} {status_text}")
    
    # Index mapping
    INDEX_MAP = {
        "NIFTY 50": {"token": 256265, "symbol": "NSE:NIFTY 50"},
        "BANK NIFTY": {"token": 260105, "symbol": "NSE:NIFTY BANK"},
        "FIN NIFTY": {"token": 257801, "symbol": "NSE:NIFTY FIN SERVICE"},
        "SENSEX": {"token": 265, "symbol": "BSE:SENSEX"}
    }
    
    # Stock mapping
    STOCK_MAP = {
        "RELIANCE": {"token": 738561, "symbol": "NSE:RELIANCE"},
        "TCS": {"token": 2953217, "symbol": "NSE:TCS"},
        "HDFCBANK": {"token": 341249, "symbol": "NSE:HDFCBANK"},
        "INFY": {"token": 408065, "symbol": "NSE:INFY"},
        "ICICIBANK": {"token": 1270529, "symbol": "NSE:ICICIBANK"}
    }
    
    # Create tabs for different chart types
    chart_tabs = st.tabs(["📊 Live Index Charts", "📈 Live Stock Charts", "📉 Historical Charts"])
    
    # Tab 1: Live Index Charts
    with chart_tabs[0]:
        st.subheader("📊 Live Index Charts (Real-Time)")
        
        if not ticker.is_connected:
            st.warning("Connect to live feed to see real-time charts")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_index = st.selectbox("Select Index", list(INDEX_MAP.keys()), key="live_index_select")
            
            with col2:
                interval = st.selectbox("Candle Interval", ["1m", "5m", "15m", "30m", "1h"], key="live_interval_select")
            
            with col3:
                num_candles = st.slider("Number of Candles", 10, 200, 50, key="live_num_candles")
            
            # Get index info
            index_info = INDEX_MAP[selected_index]
            token = index_info["token"]
            
            # Subscribe to this token
            ticker.subscribe([token])
            
            # Create placeholder for live updates
            chart_placeholder = st.empty()
            stats_placeholder = st.empty()
            
            # Auto-refresh for live data
            if st.button("🔄 Update Live Chart", key="update_live_chart_btn") or ticker.is_connected:
                try:
                    # Get candle data
                    candle_data = ticker.get_candle_data(token, num_candles)
                    
                    if candle_data and len(candle_data['timestamp']) > 0:
                        # Convert timestamps to datetime
                        timestamps = [datetime.fromtimestamp(ts) for ts in candle_data['timestamp']]
                        
                        # Create candlestick chart
                        fig = go.Figure(data=[go.Candlestick(
                            x=timestamps,
                            open=candle_data['open'],
                            high=candle_data['high'],
                            low=candle_data['low'],
                            close=candle_data['close'],
                            name=selected_index,
                            increasing_line_color='#26a69a',
                            decreasing_line_color='#ef5350'
                        )])
                        
                        # Add volume as subplot
                        fig = make_subplots(
                            rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.03,
                            row_heights=[0.7, 0.3]
                        )
                        
                        # Candlestick chart
                        fig.add_trace(go.Candlestick(
                            x=timestamps,
                            open=candle_data['open'],
                            high=candle_data['high'],
                            low=candle_data['low'],
                            close=candle_data['close'],
                            name=selected_index,
                            increasing_line_color='#26a69a',
                            decreasing_line_color='#ef5350'
                        ), row=1, col=1)
                        
                        # Volume chart
                        fig.add_trace(go.Bar(
                            x=timestamps,
                            y=candle_data['volume'],
                            name='Volume',
                            marker_color='#ff8c00',
                            opacity=0.7
                        ), row=2, col=1)
                        
                        # Add moving averages
                        if len(candle_data['close']) >= 20:
                            close_prices = np.array(candle_data['close'])
                            sma20 = talib.SMA(close_prices, timeperiod=20) if TALIB_AVAILABLE else pd.Series(close_prices).rolling(20).mean()
                            sma50 = talib.SMA(close_prices, timeperiod=50) if TALIB_AVAILABLE else pd.Series(close_prices).rolling(50).mean()
                            
                            fig.add_trace(go.Scatter(
                                x=timestamps,
                                y=sma20,
                                mode='lines',
                                name='SMA 20',
                                line=dict(color='orange', width=1)
                            ), row=1, col=1)
                            
                            fig.add_trace(go.Scatter(
                                x=timestamps,
                                y=sma50,
                                mode='lines',
                                name='SMA 50',
                                line=dict(color='blue', width=1)
                            ), row=1, col=1)
                        
                        # Update layout
                        fig.update_layout(
                            title=f'{selected_index} Live Chart ({interval}) - Real-Time',
                            xaxis_title='Time',
                            yaxis_title='Price (₹)',
                            height=600,
                            template='plotly_white',
                            showlegend=True,
                            xaxis_rangeslider_visible=False
                        )
                        
                        # Update axes
                        fig.update_xaxes(title_text="Time", row=2, col=1)
                        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
                        fig.update_yaxes(title_text="Volume", row=2, col=1)
                        
                        # Display chart
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        # Get latest tick
                        latest_tick = ticker.get_latest_tick(token)
                        
                        if latest_tick:
                            # Display live stats
                            current_price = latest_tick.get('last_price', 0)
                            open_price = latest_tick.get('ohlc', {}).get('open', 0)
                            high_price = latest_tick.get('ohlc', {}).get('high', 0)
                            low_price = latest_tick.get('ohlc', {}).get('low', 0)
                            change = current_price - open_price
                            change_pct = (change / open_price * 100) if open_price > 0 else 0
                            
                            stats_cols = st.columns(6)
                            stats_cols[0].metric("Current", f"₹{current_price:,.2f}", 
                                               f"{change:+.2f} ({change_pct:+.2f}%)",
                                               delta_color="normal" if change >= 0 else "inverse")
                            stats_cols[1].metric("Open", f"₹{open_price:,.2f}")
                            stats_cols[2].metric("High", f"₹{high_price:,.2f}")
                            stats_cols[3].metric("Low", f"₹{low_price:,.2f}")
                            stats_cols[4].metric("Volume", f"{latest_tick.get('volume_traded', 0):,}")
                            stats_cols[5].metric("Last Update", 
                                               datetime.fromtimestamp(ticker.get_last_update_time(token)).strftime("%H:%M:%S") 
                                               if ticker.get_last_update_time(token) > 0 else "N/A")
                            
                            # Order book simulation
                            with st.expander("📊 Order Book (Simulated)", expanded=False):
                                ob_cols = st.columns(2)
                                
                                with ob_cols[0]:
                                    st.markdown("**Bid Side**")
                                    for i in range(5, 0, -1):
                                        bid_price = current_price - (i * 0.05 * current_price / 100)
                                        bid_qty = random.randint(100, 1000)
                                        st.markdown(f"₹{bid_price:.2f} | {bid_qty:>6}")
                                
                                with ob_cols[1]:
                                    st.markdown("**Ask Side**")
                                    for i in range(1, 6):
                                        ask_price = current_price + (i * 0.05 * current_price / 100)
                                        ask_qty = random.randint(100, 1000)
                                        st.markdown(f"₹{ask_price:.2f} | {ask_qty:>6}")
                        
                        # Auto-refresh toggle
                        auto_refresh = st.checkbox("🔄 Auto-Refresh (5s)", value=True, key="auto_refresh_live")
                        
                        if auto_refresh:
                            time.sleep(5)
                            st.rerun()
                    
                    else:
                        st.info("Waiting for live data...")
                        time.sleep(2)
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error updating live chart: {str(e)}")
    
    # Tab 2: Live Stock Charts
    with chart_tabs[1]:
        st.subheader("📈 Live Stock Charts (Real-Time)")
        
        if not ticker.is_connected:
            st.warning("Connect to live feed to see real-time stock charts")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_stock = st.selectbox("Select Stock", list(STOCK_MAP.keys()), key="live_stock_select")
            
            with col2:
                stock_interval = st.selectbox("Chart Interval", ["1m", "5m", "15m"], key="live_stock_interval")
            
            # Get stock info
            stock_info = STOCK_MAP[selected_stock]
            token = stock_info["token"]
            
            # Subscribe to this stock
            ticker.subscribe([token])
            
            # Create placeholder
            stock_chart_placeholder = st.empty()
            stock_stats_placeholder = st.empty()
            
            if st.button("📊 Load Live Stock Chart", key="load_live_stock_btn"):
                try:
                    # Get candle data
                    candle_data = ticker.get_candle_data(token, 50)
                    
                    if candle_data and len(candle_data['timestamp']) > 0:
                        # Convert timestamps
                        timestamps = [datetime.fromtimestamp(ts) for ts in candle_data['timestamp']]
                        
                        # Create chart
                        fig = go.Figure(data=[go.Candlestick(
                            x=timestamps,
                            open=candle_data['open'],
                            high=candle_data['high'],
                            low=candle_data['low'],
                            close=candle_data['close'],
                            name=selected_stock,
                            increasing_line_color='#26a69a',
                            decreasing_line_color='#ef5350'
                        )])
                        
                        # Add indicators
                        if len(candle_data['close']) >= 20:
                            close_prices = np.array(candle_data['close'])
                            
                            # EMA
                            ema9 = talib.EMA(close_prices, timeperiod=9) if TALIB_AVAILABLE else pd.Series(close_prices).ewm(span=9).mean()
                            ema21 = talib.EMA(close_prices, timeperiod=21) if TALIB_AVAILABLE else pd.Series(close_prices).ewm(span=21).mean()
                            
                            fig.add_trace(go.Scatter(
                                x=timestamps,
                                y=ema9,
                                mode='lines',
                                name='EMA 9',
                                line=dict(color='orange', width=1)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=timestamps,
                                y=ema21,
                                mode='lines',
                                name='EMA 21',
                                line=dict(color='blue', width=1)
                            ))
                        
                        fig.update_layout(
                            title=f'{selected_stock} Live Chart ({stock_interval})',
                            height=400,
                            xaxis_rangeslider_visible=False
                        )
                        
                        stock_chart_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        # Get latest tick
                        latest_tick = ticker.get_latest_tick(token)
                        
                        if latest_tick:
                            current_price = latest_tick.get('last_price', 0)
                            prev_close = latest_tick.get('ohlc', {}).get('open', current_price)
                            change = current_price - prev_close
                            change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                            
                            # Display stats
                            stats_cols = st.columns(4)
                            stats_cols[0].metric("Price", f"₹{current_price:.2f}", 
                                               f"{change:+.2f} ({change_pct:+.2f}%)")
                            stats_cols[1].metric("Volume", f"{latest_tick.get('volume_traded', 0):,}")
                            stats_cols[2].metric("Bid Qty", f"{latest_tick.get('depth', {}).get('buy', [{}])[0].get('quantity', 0):,}")
                            stats_cols[3].metric("Ask Qty", f"{latest_tick.get('depth', {}).get('sell', [{}])[0].get('quantity', 0):,}")
                    
                except Exception as e:
                    st.error(f"Error loading stock chart: {str(e)}")
    
    # Tab 3: Historical Charts (Original functionality)
    with chart_tabs[2]:
        st.subheader("📉 Historical Charts")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_index = st.selectbox("Select Index", list(INDEX_MAP.keys()), key="hist_index_select")
        
        with col2:
            interval = st.selectbox("Interval", ["minute", "5minute", "15minute", "30minute", "60minute", "day"], 
                                   key="hist_interval_select")
        
        with col3:
            days_back = st.slider("Days Back", 1, 30, 7, key="hist_days_slider")
        
        if st.button("📊 Load Historical Chart", type="primary", key="load_hist_chart_btn"):
            try:
                index_info = INDEX_MAP[selected_index]
                with st.spinner(f"Fetching {selected_index} data..."):
                    # Try to get data from Kite
                    data = kite_manager.get_historical_data(index_info["token"], interval, days_back)
                    
                    if data is not None and len(data) > 0:
                        # Create candlestick chart
                        fig = go.Figure(data=[go.Candlestick(
                            x=data.index,
                            open=data['open'],
                            high=data['high'],
                            low=data['low'],
                            close=data['close'],
                            name=selected_index
                        )])
                        
                        # Add moving averages
                        data['SMA20'] = data['close'].rolling(window=20).mean()
                        data['SMA50'] = data['close'].rolling(window=50).mean()
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['SMA20'],
                            mode='lines',
                            name='SMA 20',
                            line=dict(color='orange', width=1)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['SMA50'],
                            mode='lines',
                            name='SMA 50',
                            line=dict(color='blue', width=1)
                        ))
                        
                        fig.update_layout(
                            title=f'{selected_index} Historical Chart ({interval})',
                            xaxis_title='Date',
                            yaxis_title='Price (₹)',
                            height=500,
                            template='plotly_white',
                            showlegend=True
                        )
                        
                        fig.update_xaxes(
                            rangeslider_visible=False,
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=1, label="1D", step="day", stepmode="backward"),
                                    dict(count=7, label="1W", step="day", stepmode="backward"),
                                    dict(count=1, label="1M", step="month", stepmode="backward"),
                                    dict(count=3, label="3M", step="month", stepmode="backward"),
                                    dict(step="all")
                                ])
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show current stats
                        if len(data) > 0:
                            current_price = data['close'].iloc[-1]
                            prev_close = data['close'].iloc[-2] if len(data) > 1 else current_price
                            change_pct = ((current_price - prev_close) / prev_close) * 100
                            
                            st.metric(f"Current {selected_index}", 
                                     f"₹{current_price:,.2f}", 
                                     f"{change_pct:+.2f}%")
                            
                            # Show additional stats
                            cols = st.columns(4)
                            cols[0].metric("Open", f"₹{data['open'].iloc[-1]:,.2f}")
                            cols[1].metric("High", f"₹{data['high'].max():,.2f}")
                            cols[2].metric("Low", f"₹{data['low'].min():,.2f}")
                            cols[3].metric("Volume", f"{data['volume'].sum():,}")
                            
                            # Download button
                            csv = data.to_csv()
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv,
                                file_name=f"{selected_index}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key="download_hist_data"
                            )
                        
                    else:
                        # Fallback to yfinance
                        st.info("Using yfinance data (Kite data unavailable)")
                        ticker_map = {
                            "NIFTY 50": "^NSEI",
                            "BANK NIFTY": "^NSEBANK",
                            "FIN NIFTY": "^NIFTY_FIN_SERVICE",
                            "SENSEX": "^BSESN"
                        }
                        
                        yf_ticker = ticker_map.get(selected_index, "^NSEI")
                        period_map = {
                            "minute": "1d", "5minute": "5d", "15minute": "7d",
                            "30minute": "15d", "60minute": "30d", "day": f"{days_back}d"
                        }
                        
                        df = yf.download(yf_ticker, period=period_map.get(interval, "7d"), 
                                        interval=interval.replace("minute", "m").replace("60minute", "60m"))
                        
                        if not df.empty:
                            fig = go.Figure(data=[go.Candlestick(
                                x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close']
                            )])
                            
                            fig.update_layout(
                                title=f'{selected_index} Chart ({interval}) - via Yahoo Finance',
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            current_price = df['Close'].iloc[-1]
                            prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
                            change_pct = ((current_price - prev_close) / prev_close) * 100
                            
                            st.metric(f"Current {selected_index}", f"₹{current_price:,.2f}", f"{change_pct:+.2f}%")
                
            except Exception as e:
                st.error(f"Error loading chart: {str(e)}")
    
    # Market Depth and Advanced Features
    st.markdown("---")
    st.subheader("🎯 Advanced Features")
    
    adv_cols = st.columns(3)
    
    with adv_cols[0]:
        if st.button("📊 Market Watch", key="market_watch_btn"):
            st.info("Market watch feature would show multiple instruments simultaneously")
    
    with adv_cols[1]:
        if st.button("🔔 Price Alerts", key="price_alerts_btn"):
            st.info("Price alert system for notifications")
    
    with adv_cols[2]:
        if st.button("📈 Technical Scanner", key="tech_scanner_btn"):
            st.info("Technical scanner for pattern recognition")
    
    # WebSocket status info
    if ticker.is_connected:
        st.markdown("""
        <div class="alert-success">
            <strong>✅ Live Feed Active</strong><br>
            Real-time data streaming from Kite Connect WebSocket
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-warning">
            <strong>⚠️ Live Feed Inactive</strong><br>
            Click "Connect Live Feed" to enable real-time tick-by-tick data
        </div>
        """, unsafe_allow_html=True)

# ===================== PART 10: ALGO TRADING TAB =====================
def create_algo_trading_tab(algo_engine):
    """Create Algo Trading Control Tab"""
    st.subheader("🤖 Algorithmic Trading Engine")
    
    if algo_engine is None:
        st.warning("Algo Engine not initialized")
        return
    
    status = algo_engine.get_status()
    
    # Status Display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        state = status["state"]
        if state == "running":
            status_color = "🟢"
            status_class = "algo-status-running"
        elif state == "stopped":
            status_color = "🔴"
            status_class = "algo-status-stopped"
        elif state == "paused":
            status_color = "🟡"
            status_class = "algo-status-paused"
        else:
            status_color = "⚫"
            status_class = "algo-status-stopped"
        
        st.markdown(f"""
        <div class="{status_class}">
            <div style="font-size: 14px; color: #6b7280;">Engine Status</div>
            <div style="font-size: 20px; font-weight: bold;">{status_color} {state.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Active Positions", status["active_positions"])
    
    with col3:
        st.metric("Total Orders", status["total_orders"])
    
    with col4:
        st.metric("Today's P&L", f"₹{status['realized_pnl'] + status['unrealized_pnl']:+.2f}")
    
    # Control Buttons
    st.subheader("Engine Controls")
    ctrl_cols = st.columns(5)
    
    with ctrl_cols[0]:
        if st.button("▶️ Start Engine", type="primary", key="start_algo_btn", 
                    disabled=status["state"] == "running"):
            if algo_engine.start():
                st.success("Algo Engine started!")
                st.rerun()
    
    with ctrl_cols[1]:
        if st.button("⏸️ Pause Engine", key="pause_algo_btn", 
                    disabled=status["state"] != "running"):
            algo_engine.pause()
            st.info("Algo Engine paused")
            st.rerun()
    
    with ctrl_cols[2]:
        if st.button("▶️ Resume Engine", key="resume_algo_btn", 
                    disabled=status["state"] != "paused"):
            algo_engine.resume()
            st.success("Algo Engine resumed")
            st.rerun()
    
    with ctrl_cols[3]:
        if st.button("⏹️ Stop Engine", key="stop_algo_btn", 
                    disabled=status["state"] == "stopped"):
            algo_engine.stop()
            st.info("Algo Engine stopped")
            st.rerun()
    
    with ctrl_cols[4]:
        if st.button("🚨 Emergency Stop", type="secondary", key="emergency_stop_btn"):
            algo_engine.emergency_stop("Manual emergency stop")
            st.error("EMERGENCY STOP ACTIVATED")
            st.rerun()
    
    # Daily Exit Info
    st.subheader("🕒 Daily Schedule")
    
    schedule_cols = st.columns(4)
    with schedule_cols[0]:
        st.metric("Market Open", "9:15 AM")
    with schedule_cols[1]:
        st.metric("Peak Hours", "9:30 AM - 2:30 PM")
    with schedule_cols[2]:
        st.metric("Market Close", "3:30 PM")
    with schedule_cols[3]:
        st.metric("Auto Exit", "3:35 PM")
    
    # Manual Daily Exit Button
    if st.button("📤 Force Daily Exit Now", type="secondary", key="force_daily_exit_btn"):
        if algo_engine.trader and algo_engine.trader.positions:
            algo_engine._exit_all_positions()
            st.success("All positions exited!")
            st.rerun()
        else:
            st.info("No positions to exit")
    
    # Send Daily Report Button
    if st.button("📧 Send Daily Report", type="primary", key="send_daily_report_btn"):
        if algo_engine.trader:
            success = send_daily_report_email(algo_engine.trader, algo_engine)
            if success:
                st.success("Daily report sent to rantv2002@gmail.com")
            else:
                st.error("Failed to send daily report")
        else:
            st.error("Trader not initialized")
    
    # Risk Settings
    st.subheader("Risk Management Settings")
    
    with st.expander("Configure Risk Limits", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            new_max_positions = st.number_input(
                "Max Positions",
                min_value=1, max_value=20,
                value=algo_engine.risk_limits.max_positions,
                key="algo_max_positions_input"
            )
            
            new_max_daily_loss = st.number_input(
                "Max Daily Loss (₹)",
                min_value=1000, max_value=500000,
                value=int(algo_engine.risk_limits.max_daily_loss),
                key="algo_max_daily_loss_input"
            )
        
        with col2:
            new_min_confidence = st.slider(
                "Min Signal Confidence",
                min_value=0.5, max_value=0.99,
                value=algo_engine.risk_limits.min_confidence,
                step=0.01,
                key="algo_min_confidence_slider"
            )
            
            new_max_trades = st.number_input(
                "Max Trades/Day",
                min_value=1, max_value=50,
                value=algo_engine.risk_limits.max_trades_per_day,
                key="algo_max_trades_input"
            )
        
        if st.button("Update Risk Settings", key="update_risk_settings_btn"):
            algo_engine.update_risk_limits(
                max_positions=new_max_positions,
                max_daily_loss=float(new_max_daily_loss),
                min_confidence=new_min_confidence,
                max_trades_per_day=new_max_trades
            )
            st.success("Risk settings updated!")
            st.rerun()
    
    # Performance Metrics
    st.subheader("Performance Metrics")
    perf_cols = st.columns(4)
    
    with perf_cols[0]:
        st.metric("Filled Orders", status["filled_orders"])
    
    with perf_cols[1]:
        st.metric("Rejected Orders", status.get("rejected_orders", 0))
    
    with perf_cols[2]:
        st.metric("Realized P&L", f"₹{status['realized_pnl']:+.2f}")
    
    with perf_cols[3]:
        st.metric("Unrealized P&L", f"₹{status['unrealized_pnl']:+.2f}")
    
    # Active Positions
    if algo_engine.active_positions:
        st.subheader("Active Algo Positions")
        positions_data = []
        for symbol, order in algo_engine.active_positions.items():
            positions_data.append({
                "Symbol": symbol.replace(".NS", ""),
                "Action": order.action,
                "Quantity": order.quantity,
                "Entry Price": f"₹{order.price:.2f}",
                "Current Price": f"₹{order.filled_price or order.price:.2f}",
                "Stop Loss": f"₹{order.stop_loss:.2f}",
                "Target": f"₹{order.target:.2f}",
                "Strategy": order.strategy,
                "Confidence": f"{order.confidence:.1%}"
            })
        
        st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
    
    # Strategy Configuration
    st.subheader("Strategy Configuration")
    
    strategy_cols = st.columns(2)
    
    with strategy_cols[0]:
        st.markdown("**Standard Strategies**")
        for strategy, config in TRADING_STRATEGIES.items():
            enabled = st.checkbox(config["name"], value=True, key=f"std_{strategy}_checkbox")
    
    with strategy_cols[1]:
        st.markdown("**Signal Settings**")
        signal_frequency = st.selectbox("Signal Scan Frequency", 
                                       ["1 min", "5 min", "15 min", "30 min"],
                                       key="signal_frequency_select")
        
        max_signals_per_scan = st.slider("Max Signals per Scan", 1, 20, 5, 
                                        key="max_signals_slider")
    
    # Market Conditions
    st.subheader("Market Conditions")
    
    market_cols = st.columns(4)
    
    with market_cols[0]:
        market_open_status = "🟢 OPEN" if market_open() else "🔴 CLOSED"
        st.metric("Market Status", market_open_status)
    
    with market_cols[1]:
        peak_hours_status = "🟢 PEAK" if is_peak_market_hours() else "⚪ OFF-PEAK"
        st.metric("Peak Hours", peak_hours_status)
    
    with market_cols[2]:
        daily_exit_status = "🟢 PENDING" if not status.get('daily_exit_completed', False) else "🔴 COMPLETED"
        st.metric("Daily Exit", daily_exit_status)
    
    with market_cols[3]:
        st.metric("Trades Today", f"{status['trades_today']}/{algo_engine.risk_limits.max_trades_per_day}")
    
    # Info Box
    st.markdown("---")
    st.markdown("""
    <div class="alert-warning">
        <strong>⚠️ Important:</strong> Algorithmic trading involves significant risk. 
        The system will automatically exit all positions at 3:35 PM daily.
        A detailed report will be sent to rantv2002@gmail.com after market close.
        Always test with paper trading first. Set appropriate risk limits. 
        Monitor the system regularly. Past performance does not guarantee future results.
    </div>
    """, unsafe_allow_html=True)

# ===================== PART 11: MAIN APPLICATION =====================
def main():
    # Display Logo and Header
    st.markdown("""
    <div class="logo-container">
        <h1 style="color: white; margin: 10px 0 0 0; font-size: 32px;">📈 RANTV TERMINAL PRO</h1>
        <p style="color: white; margin: 5px 0;">Enhanced Intraday Trading Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align:center; color: #ff8c00;'>Intraday Terminal Pro - ENHANCED</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color: #6b7280;'>Complete Trading Suite with Algo Engine & Live Charts</h4>", unsafe_allow_html=True)
    
    # Initialize components
    data_manager = EnhancedDataManager()
    
    if st.session_state.trader is None:
        st.session_state.trader = MultiStrategyIntradayTrader()
    
    trader = st.session_state.trader
    
    if st.session_state.kite_manager is None:
        st.session_state.kite_manager = KiteConnectManager(KITE_API_KEY, KITE_API_SECRET)
    
    kite_manager = st.session_state.kite_manager
    
    if st.session_state.algo_engine is None:
        st.session_state.algo_engine = AlgoEngine(trader=trader)
    
    algo_engine = st.session_state.algo_engine
    
    # Auto-refresh
    st_autorefresh(interval=PRICE_REFRESH_MS, key="main_auto_refresh")
    st.session_state.refresh_count += 1
    
    # Sidebar - Kite Login
    kite_manager.login_ui()
    
    # Sidebar - Trading Configuration
    with st.sidebar:
        st.divider()
        st.header("⚙️ Trading Configuration")
        
        universe = st.selectbox("Trading Universe", ["Nifty 50", "Nifty 100", "Midcap 150", "All Stocks"], 
                               key="universe_select")
        
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
        st.write(f"✅ TA-Lib: {'Available' if TALIB_AVAILABLE else 'Not Available'}")
        st.write(f"✅ WebSocket: {'Available' if WEBSOCKET_AVAILABLE else 'Not Available'}")
        st.write(f"✅ Algo Engine: {'Ready' if algo_engine else 'Not Initialized'}")
        st.write(f"🔄 Refresh Count: {st.session_state.refresh_count}")
        st.write(f"📊 Market: {'Open' if market_open() else 'Closed'}")
        st.write(f"⏰ Peak Hours: {'Active' if is_peak_market_hours() else 'Inactive'}")
        st.write(f"🕒 Auto Exit: {'3:35 PM Daily'}")
        
        # Manual daily report button in sidebar
        if st.button("📧 Send Test Report", key="sidebar_test_report_btn"):
            if algo_engine.trader:
                success = send_daily_report_email(algo_engine.trader, algo_engine)
                if success:
                    st.success("Test report sent!")
                else:
                    st.error("Failed to send test report")
            else:
                st.error("Trader not initialized")
    
    # Market Overview
    st.subheader("📊 Market Overview")
    cols = st.columns(7)
    
    try:
        nifty = yf.Ticker("^NSEI")
        nifty_history = nifty.history(period="1d")
        if not nifty_history.empty:
            nifty_price = nifty_history['Close'].iloc[-1]
            nifty_prev = nifty_history['Close'].iloc[-2] if len(nifty_history) > 1 else nifty_price
            nifty_change = ((nifty_price - nifty_prev) / nifty_prev) * 100
            cols[0].metric("NIFTY 50", f"₹{nifty_price:,.2f}", f"{nifty_change:+.2f}%")
        else:
            cols[0].metric("NIFTY 50", "Loading...")
    except:
        cols[0].metric("NIFTY 50", "Loading...")
    
    try:
        bn = yf.Ticker("^NSEBANK")
        bn_history = bn.history(period="1d")
        if not bn_history.empty:
            bn_price = bn_history['Close'].iloc[-1]
            bn_prev = bn_history['Close'].iloc[-2] if len(bn_history) > 1 else bn_price
            bn_change = ((bn_price - bn_prev) / bn_prev) * 100
            cols[1].metric("BANK NIFTY", f"₹{bn_price:,.2f}", f"{bn_change:+.2f}%")
        else:
            cols[1].metric("BANK NIFTY", "Loading...")
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
    
    try:
        nifty_data = yf.download("^NSEI", period="1d", interval="5m", auto_adjust=False)
        if not nifty_data.empty:
            nifty_current = float(nifty_data["Close"].iloc[-1])
            nifty_prev = float(nifty_data["Close"].iloc[-2])
            nifty_change = ((nifty_current - nifty_prev) / nifty_prev) * 100
            nifty_sentiment = 50 + (nifty_change * 8)
            nifty_sentiment = max(0, min(100, round(nifty_sentiment)))
        else:
            nifty_current = 22000
            nifty_change = 0.15
            nifty_sentiment = 65
    except:
        nifty_current = 22000
        nifty_change = 0.15
        nifty_sentiment = 65
    
    try:
        banknifty_data = yf.download("^NSEBANK", period="1d", interval="5m", auto_adjust=False)
        if not banknifty_data.empty:
            banknifty_current = float(banknifty_data["Close"].iloc[-1])
            banknifty_prev = float(banknifty_data["Close"].iloc[-2])
            banknifty_change = ((banknifty_current - banknifty_prev) / banknifty_prev) * 100
            banknifty_sentiment = 50 + (banknifty_change * 8)
            banknifty_sentiment = max(0, min(100, round(banknifty_sentiment)))
        else:
            banknifty_current = 48000
            banknifty_change = 0.25
            banknifty_sentiment = 70
    except:
        banknifty_current = 48000
        banknifty_change = 0.25
        banknifty_sentiment = 70
    
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
    
    # Refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"<div style='text-align: left; color: #6b7280; font-size: 14px;'>Refresh Count: <span class='refresh-counter'>{st.session_state.refresh_count}</span></div>", unsafe_allow_html=True)
    with col2:
        if st.button("🔄 Manual Refresh", key="manual_refresh_main_btn"):
            st.rerun()
    with col3:
        if st.button("📊 Update Prices", key="update_prices_main_btn"):
            st.rerun()
    
    # Main Tabs - REMOVED HIGH ACCURACY TAB
    tabs = st.tabs(["📈 Dashboard", "🚦 Signals", "💰 Paper Trading", "📋 Trade History", 
                   "📉 RSI Scanner", "📊 Kite Charts", "🤖 Algo Trading"])
    
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
        for strategy, config in TRADING_STRATEGIES.items():
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
        st.subheader("📊 Multi-Strategy Signal Scanner")
        
        # Signal generator controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            scan_mode = st.radio(
                "Scan Mode",
                ["Quick Scan (Top 30)", "Full Universe Scan"],
                horizontal=True,
                key="scan_mode"
            )
        
        with col2:
            min_confidence = st.slider(
                "Min Confidence",
                min_value=0.60,
                max_value=0.95,
                value=0.70,
                step=0.05,
                key="min_conf_signal"
            )
        
        with col3:
            max_signals = st.number_input(
                "Max Signals",
                min_value=1,
                max_value=20,
                value=10,
                key="max_signals_input"
            )
        
        # Generate signals button
        if st.button("🚀 Generate Trading Signals", type="primary", key="generate_signals_btn"):
            st.session_state.last_signal_generation = time.time()
            
            with st.spinner(f"Scanning {universe} for trading signals..."):
                # Determine scan size
                scan_size = 30 if scan_mode == "Quick Scan (Top 30)" else 100
                
                # Generate signals
                signals = trader.signal_generator.scan_stock_universe(
                    universe=universe,
                    max_stocks=scan_size,
                    min_confidence=min_confidence
                )
                
                # Store in session state
                st.session_state.generated_signals = signals[:max_signals]
                st.session_state.signal_quality = trader.signal_generator.calculate_signal_quality(signals)
                
                st.success(f"✅ Generated {len(signals)} signals (showing top {min(max_signals, len(signals))})")
        
        # Display signals if available
        if 'generated_signals' in st.session_state and st.session_state.generated_signals:
            signals = st.session_state.generated_signals
            
            # Signal quality indicator
            quality = st.session_state.get('signal_quality', 0)
            
            if quality >= 70:
                quality_class = "high-quality-signal"
                quality_text = "HIGH QUALITY"
            elif quality >= 50:
                quality_class = "medium-quality-signal"
                quality_text = "MEDIUM QUALITY"
            else:
                quality_class = "alert-warning"
                quality_text = "LOW QUALITY"
            
            st.markdown(f"""
            <div class="{quality_class}">
                <strong>📊 Signal Quality: {quality_text}</strong> | 
                Score: {quality:.1f}/100 | 
                Generated: {trader.signal_generator.last_scan_time.strftime('%H:%M:%S') if trader.signal_generator.last_scan_time else 'N/A'}
            </div>
            """, unsafe_allow_html=True)
            
            # Display signals in a table
            signal_data = []
            for i, signal in enumerate(signals):
                signal_data.append({
                    "#": i+1,
                    "Symbol": signal['symbol'].replace('.NS', ''),
                    "Action": f"{'🟢 BUY' if signal['action'] == 'BUY' else '🔴 SELL'}",
                    "Price": f"₹{signal['price']:.2f}",
                    "Stop Loss": f"₹{signal['stop_loss']:.2f}",
                    "Target": f"₹{signal['target']:.2f}",
                    "Confidence": f"{signal['confidence']:.1%}",
                    "Score": signal['score'],
                    "Win Prob": f"{signal['win_probability']:.1%}",
                    "Strategy": signal['strategy'],
                    "RSI": f"{signal.get('rsi', 0):.1f}"
                })
            
            # Create dataframe
            df_signals = pd.DataFrame(signal_data)
            
            # Display with formatting
            st.dataframe(
                df_signals,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Action": st.column_config.TextColumn(width="small"),
                    "Confidence": st.column_config.ProgressColumn(
                        format="%.1f%%",
                        min_value=0,
                        max_value=1.0
                    ),
                    "Win Prob": st.column_config.ProgressColumn(
                        format="%.1f%%",
                        min_value=0,
                        max_value=1.0
                    )
                }
            )
            
            # Auto-trade controls
            st.subheader("🤖 Auto-Trade Execution")
            
            if not trader.auto_execution:
                st.warning("Auto execution is disabled. Enable in sidebar to auto-trade.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📈 Execute All BUY Signals", type="secondary", 
                            disabled=not trader.auto_execution or not market_open()):
                    executed = 0
                    for signal in [s for s in signals if s['action'] == 'BUY']:
                        success, msg = trader.execute_auto_trade_from_signal(signal)
                        if success:
                            executed += 1
                            st.success(f"Executed: {signal['symbol']}")
                        else:
                            st.warning(f"Failed: {signal['symbol']} - {msg}")
                    
                    if executed > 0:
                        st.balloons()
                        st.success(f"Executed {executed} BUY signals!")
                        st.rerun()
            
            with col2:
                if st.button("📉 Execute All SELL Signals", type="secondary",
                            disabled=not trader.auto_execution or not market_open()):
                    executed = 0
                    for signal in [s for s in signals if s['action'] == 'SELL']:
                        success, msg = trader.execute_auto_trade_from_signal(signal)
                        if success:
                            executed += 1
                            st.success(f"Executed: {signal['symbol']}")
                        else:
                            st.warning(f"Failed: {signal['symbol']} - {msg}")
                    
                    if executed > 0:
                        st.balloons()
                        st.success(f"Executed {executed} SELL signals!")
                        st.rerun()
            
            with col3:
                if st.button("🎯 Execute Top 3 Signals", type="primary",
                            disabled=not trader.auto_execution or not market_open()):
                    executed = 0
                    for signal in signals[:3]:
                        success, msg = trader.execute_auto_trade_from_signal(signal)
                        if success:
                            executed += 1
                            st.success(f"Executed: {signal['symbol']}")
                        else:
                            st.warning(f"Failed: {signal['symbol']} - {msg}")
                    
                    if executed > 0:
                        st.balloons()
                        st.success(f"Executed {executed} signals!")
                        st.rerun()
            
            # Manual trade execution for individual signals
            st.subheader("🎯 Manual Signal Execution")
            
            selected_signal_idx = st.selectbox(
                "Select Signal to Trade",
                range(len(signals)),
                format_func=lambda x: f"{signals[x]['symbol'].replace('.NS', '')} - {signals[x]['action']} @ ₹{signals[x]['price']:.2f} ({signals[x]['confidence']:.1%})",
                key="signal_select"
            )
            
            if selected_signal_idx is not None:
                selected_signal = signals[selected_signal_idx]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Action", selected_signal['action'])
                with col2:
                    st.metric("Price", f"₹{selected_signal['price']:.2f}")
                with col3:
                    st.metric("Confidence", f"{selected_signal['confidence']:.1%}")
                with col4:
                    st.metric("Win Probability", f"{selected_signal['win_probability']:.1%}")
                
                st.markdown(f"""
                **Strategy:** {selected_signal['strategy']}  
                **Stop Loss:** ₹{selected_signal['stop_loss']:.2f} ({abs(selected_signal['price'] - selected_signal['stop_loss']):.2f} points)  
                **Target:** ₹{selected_signal['target']:.2f} ({abs(selected_signal['target'] - selected_signal['price']):.2f} points)  
                **Risk/Reward:** 1:{abs((selected_signal['target'] - selected_signal['price']) / (selected_signal['price'] - selected_signal['stop_loss'])):.2f}
                """)
                
                # Manual execution controls
                exec_col1, exec_col2 = st.columns([1, 2])
                
                with exec_col1:
                    quantity = st.number_input(
                        "Quantity",
                        min_value=1,
                        max_value=100,
                        value=min(10, int(trader.cash * 0.1 / selected_signal['price'])),
                        key="signal_quantity"
                    )
                
                with exec_col2:
                    exec_col2a, exec_col2b = st.columns(2)
                    with exec_col2a:
                        if st.button("📈 Execute Trade", type="primary", key="execute_signal_trade"):
                            success, msg = trader.execute_trade(
                                symbol=selected_signal['symbol'],
                                action=selected_signal['action'],
                                quantity=quantity,
                                price=selected_signal['price'],
                                stop_loss=selected_signal['stop_loss'],
                                target=selected_signal['target'],
                                win_probability=selected_signal['win_probability'],
                                auto_trade=False,
                                strategy=selected_signal['strategy']
                            )
                            
                            if success:
                                st.success(f"✅ {msg}")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(f"❌ {msg}")
                    
                    with exec_col2b:
                        if st.button("🤖 Auto-Trade This", type="secondary", key="auto_trade_signal",
                                    disabled=not trader.auto_execution):
                            success, msg = trader.execute_auto_trade_from_signal(selected_signal)
                            
                            if success:
                                st.success(f"✅ Auto-trade executed: {msg}")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(f"❌ Auto-trade failed: {msg}")
        
        else:
            # No signals generated yet
            st.info("""
            ### 📊 Signal Generation Ready
            
            Click **"Generate Trading Signals"** to scan the market for opportunities.
            
            The scanner will:
            1. Analyze technical indicators (EMA, RSI, MACD, Bollinger Bands)
            2. Check volume patterns
            3. Identify support/resistance levels
            4. Generate confidence scores for each signal
            5. Filter by minimum confidence threshold
            
            **Requirements:**
            - Market must be open for live signals
            - Stock data available (15m interval)
            - Minimum 50 data points per stock
            """)
            
            # Quick stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market Status", "OPEN" if market_open() else "CLOSED")
            with col2:
                st.metric("Peak Hours", "YES" if is_peak_market_hours() else "NO")
            with col3:
                st.metric("Signals Generated", trader.signal_generator.signals_generated)
    
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
    
    # Tab 6: Kite Charts - UPDATED WITH WEB SOCKET
    with tabs[5]:
        create_kite_live_charts(kite_manager)
    
    # Tab 7: Algo Trading - UPDATED WITH DAILY EXIT
    with tabs[6]:
        create_algo_trading_tab(algo_engine)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; font-size: 12px;">
        <strong>Rantv Terminal Pro</strong> | Complete Trading Suite with Algo Engine & Live Charts | © 2024 | 
        Refresh: {st.session_state.refresh_count} | {now_indian().strftime("%H:%M:%S")} | 
        Auto Exit: 3:35 PM | Daily Report: rantv2002@gmail.com
    </div>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.code(traceback.format_exc())
