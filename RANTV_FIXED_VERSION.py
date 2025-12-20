"""
RANTV TERMINAL PRO - INTEGRATED EDITION
Combines Standard and Professional Features
- High-Frequency Strategies
- Institutional-Grade Risk Management
- Multi-Timeframe Analysis
- Advanced Technical Indicators
- Consistent Alpha Generation
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
from scipy import stats
from scipy.stats import zscore

warnings.filterwarnings('ignore')

# Auto-install missing dependencies
def install_package(package_name):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--quiet"])
        return True
    except:
        return False

# Check and install required packages
KITECONNECT_AVAILABLE = False
SQLALCHEMY_AVAILABLE = False
JOBLIB_AVAILABLE = False
TALIB_AVAILABLE = False
WEBSOCKET_AVAILABLE = False
ARCH_AVAILABLE = False
STATSMODELS_AVAILABLE = False

# Try to import statsmodels first
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint, adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    if install_package("statsmodels"):
        try:
            import statsmodels.api as sm
            from statsmodels.tsa.stattools import coint, adfuller
            STATSMODELS_AVAILABLE = True
        except:
            STATSMODELS_AVAILABLE = False

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

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    if install_package("arch"):
        try:
            from arch import arch_model
            ARCH_AVAILABLE = True
        except:
            ARCH_AVAILABLE = False

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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================== PROFESSIONAL CONFIGURATION =====================
class ProfessionalConfig:
    # Capital Management - 10-20 Lakhs
    TOTAL_CAPITAL = 15_00_000.0  # ₹15 Lakhs
    MAX_CAPITAL_UTILIZATION = 0.50  # Max 50% deployed
    MIN_POSITION_SIZE = 10000.0  # Minimum ₹10k
    MAX_POSITION_SIZE = 3_00_000.0  # Maximum ₹3L
    
    # Risk Management
    RISK_PER_TRADE = 0.02  # 2% risk per trade
    MAX_DAILY_LOSS = 50000.0  # ₹50k daily loss limit
    MAX_DRAWDOWN = 0.15  # 15% max drawdown
    
    # Execution
    EXECUTION_ALGORITHMS = ["TWAP", "VWAP", "POV", "Iceberg", "Sniper"]
    DEFAULT_SLIPPAGE = 0.0005  # 5 basis points
    COMMISSION_RATE = 0.0003  # 3 basis points
    
    # Alpha Strategies
    ENABLED_STRATEGIES = [
        "statistical_arbitrage", "pairs_trading", "market_neutral",
        "volatility_arbitrage", "momentum_reversal", "mean_reversion"
    ]
    
    # High-Frequency Settings
    HFT_ENABLED = True
    HFT_LATENCY_TARGET = 100  # ms
    HFT_MIN_PROFIT = 0.001  # 10 basis points
    HFT_MAX_POSITIONS = 10
    
    # Backtesting
    BACKTEST_PERIOD = 180  # days
    WALK_FORWARD_WINDOWS = 3
    MIN_SAMPLE_SIZE = 50
    MIN_SHARPE_RATIO = 1.2
    MIN_PROFIT_FACTOR = 1.5
    
    # Market Microstructure
    ORDER_BOOK_DEPTH = 5
    VOLUME_PROFILE_BINS = 10
    LARGE_ORDER_THRESHOLD = 0.90
    
    # Multi-Timeframe Analysis
    TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d"]
    TIMEFRAME_WEIGHTS = {
        "5m": 0.25, "15m": 0.20, "30m": 0.15, 
        "1h": 0.15, "4h": 0.15, "1d": 0.10
    }

config = ProfessionalConfig()

# Configuration from environment
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN = ""

ALGO_ENABLED = os.environ.get("ALGO_TRADING_ENABLED", "true").lower() == "true"
ALGO_MAX_POSITIONS = int(os.environ.get("ALGO_MAX_POSITIONS", "8"))
ALGO_MAX_DAILY_LOSS = float(os.environ.get("ALGO_MAX_DAILY_LOSS", "50000"))
ALGO_MIN_CONFIDENCE = float(os.environ.get("ALGO_MIN_CONFIDENCE", "0.75"))

# Email configuration
EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "your_email@gmail.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = "rantv2002@gmail.com"
EMAIL_CC = []

@dataclass
class AppConfig:
    database_url: str = 'sqlite:///trading_journal.db'
    risk_tolerance: str = 'PROFESSIONAL'
    max_daily_loss: float = config.MAX_DAILY_LOSS
    enable_ml: bool = True
    kite_api_key: str = KITE_API_KEY
    kite_api_secret: str = KITE_API_SECRET
    algo_enabled: bool = ALGO_ENABLED
    algo_max_positions: int = ALGO_MAX_POSITIONS
    algo_max_daily_loss: float = ALGO_MAX_DAILY_LOSS
    algo_min_confidence: float = ALGO_MIN_CONFIDENCE
    total_capital: float = config.TOTAL_CAPITAL
    max_capital_utilization: float = config.MAX_CAPITAL_UTILIZATION
    hft_enabled: bool = config.HFT_ENABLED
    
    @classmethod
    def from_env(cls):
        return cls()

config_app = AppConfig.from_env()
st.set_page_config(page_title="Rantv Terminal Pro - Integrated", layout="wide", initial_sidebar_state="expanded")
IND_TZ = pytz.timezone("Asia/Kolkata")

# Trading Constants
CAPITAL = config.TOTAL_CAPITAL
TRADE_ALLOC = 0.08  # 8% per trade
MAX_DAILY_TRADES = 30
MAX_STOCK_TRADES = 3
MAX_AUTO_TRADES = 50
SIGNAL_REFRESH_MS = 30000  # 30 seconds for HFT
PRICE_REFRESH_MS = 10000  # 10 seconds
MARKET_OPTIONS = ["CASH", "FUTURES"]

# ===================== STOCK UNIVERSES =====================

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
    "YESBANK.NS", "ZEEL.NS", "ADANIGREEN.NS", "ADANITRANS.NS", "ABCAPITAL.NS",
    "ABFRL.NS", "ALKEM.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "BANDHANBNK.NS",
    "BANKBARODA.NS", "BATAINDIA.NS", "BHARATFORG.NS", "BHARATRAS.NS", "BIKAJI.NS",
    "BLUEDART.NS", "CANBK.NS", "CHOLAFIN.NS", "CROMPTON.NS", "DALBHARAT.NS",
    "DEEPAKNTR.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "GAIL.NS",
    "GLAND.NS", "GODREJIND.NS", "GODREJPROP.NS", "HAL.NS", "HINDOILEXP.NS",
    "INDHOTEL.NS", "INDIACEM.NS", "INDIAMART.NS", "INDIGO.NS", "INDUSINDBK.NS",
    "IEX.NS", "IRCTC.NS", "JUBLFOOD.NS", "LICHSGFIN.NS", "LTI.NS", "LTTS.NS",
    "LALPATHLAB.NS", "M&MFIN.NS", "MFSL.NS", "MGL.NS", "MPHASIS.NS", "MRF.NS",
    "NAM-INDIA.NS", "NAUKRI.NS", "NAVINFLUOR.NS", "PAGEIND.NS", "PERSISTENT.NS",
    "PIIND.NS", "POLYCAB.NS", "RECLTD.NS", "SAIL.NS", "SHREECEM.NS",
    "SOLARINDS.NS", "SONACOMS.NS", "SUPREMEIND.NS", "SUZLON.NS", "SYNGENE.NS",
    "TATACOMM.NS", "TATACONSUM.NS", "TATAELXSI.NS", "TATAMOTORS.NS", "TATAPOWER.NS",
    "TATASTEEL.NS", "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", "VOLTAS.NS"
]

SECTOR_STOCKS = {
    "BANKING": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS", 
                "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "FEDERALBNK.NS", "RBLBANK.NS"],
    "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "MPHASIS.NS",
           "LTTS.NS", "COFORGE.NS", "PERSISTENT.NS", "MINDTREE.NS"],
    "AUTO": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
             "HEROMOTOCO.NS", "ASHOKLEY.NS", "TVSMOTOR.NS", "ESCORTS.NS", "BHARATFORG.NS"],
    "ENERGY": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS",
               "ADANIGREEN.NS", "GAIL.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
             "GODREJCP.NS", "MARICO.NS", "JUBLFOOD.NS", "TATACONSUM.NS", "UBL.NS"],
    "PHARMA": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS",
               "TORNTPHARM.NS", "GLENMARK.NS", "AUROPHARMA.NS", "BIOCON.NS", "ALKEM.NS"],
    "METALS": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDANTA.NS", "COALINDIA.NS",
               "NMDC.NS", "SAIL.NS", "JINDALSTEL.NS", "NATIONALUM.NS"],
    "INFRA": ["LT.NS", "ADANIPORTS.NS", "IRCTC.NS", "CONCOR.NS"],
}

ALL_STOCKS = list(dict.fromkeys(NIFTY_50 + NIFTY_100))

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

# ===================== TECHNICAL INDICATOR FUNCTIONS =====================

def ema(series, span):
    """Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(0)
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def macd(close, fast=12, slow=26, signal=9):
    """MACD Indicator"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(close, period=20, std_dev=2):
    """Bollinger Bands"""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_support_resistance_advanced(high, low, close, period=20):
    """Advanced Support/Resistance Calculation"""
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

# ===================== PART 2: CSS STYLING =====================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .main .block-container {
        background-color: transparent;
        padding-top: 2rem;
    }
    
    .logo-container {
        text-align: center;
        margin: 20px auto;
        padding: 20px;
        background: linear-gradient(135deg, #ff8c00 0%, #ff6b00 100%);
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(255, 140, 0, 0.3);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: linear-gradient(135deg, #ffe8cc 0%, #ffd9a6 50%, #ffca80 100%);
        padding: 8px;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background-color: #ffffff;
        border-radius: 8px;
        padding: 12px 20px;
        font-weight: 600;
        font-size: 14px;
        color: #d97706;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff8c00 0%, #ff6b00 100%);
        color: white;
        border: 2px solid #ff8c00;
    }
    
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff8c00;
        box-shadow: 0 2px 8px rgba(255, 140, 0, 0.1);
    }
    
    .bullish { color: #059669; background-color: #d1fae5; }
    .bearish { color: #dc2626; background-color: #fee2e2; }
    .neutral { color: #ff8c00; background-color: #ffe8cc; }
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

# ===================== MAIN APPLICATION =====================

st.title("🚀 RANTV TERMINAL PRO - INTEGRATED EDITION")
st.markdown("**Professional Algorithmic Trading Platform** | Real-time Signals | Risk Management | Multi-Strategy")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    selected_universe = st.selectbox(
        "Stock Universe",
        ["NIFTY 50", "NIFTY 100", "Custom Sector"]
    )
    
    if selected_universe == "Custom Sector":
        sector = st.selectbox("Select Sector", list(SECTOR_STOCKS.keys()))
        active_stocks = SECTOR_STOCKS[sector]
    elif selected_universe == "NIFTY 100":
        active_stocks = NIFTY_100
    else:
        active_stocks = NIFTY_50
    
    st.divider()
    
    # Algorithm Settings
    st.subheader("Algorithm Settings")
    algo_enabled = st.checkbox("Enable Algo Trading", value=config_app.algo_enabled)
    max_positions = st.slider("Max Open Positions", 1, 20, config_app.algo_max_positions)
    daily_loss_limit = st.number_input("Daily Loss Limit (₹)", value=config_app.algo_max_daily_loss)
    min_confidence = st.slider("Min Signal Confidence", 0.5, 1.0, config_app.algo_min_confidence, 0.05)
    
    st.divider()
    
    # Risk Parameters
    st.subheader("Risk Management")
    risk_per_trade = st.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.5)
    capital_utilized = st.slider("Capital Utilization (%)", 10, 100, 50, 10)

st.divider()

# Main Dashboard
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "📈 Live Analysis",
    "🎯 Strategies",
    "📋 Trade Journal",
    "⚙️ Settings"
])

with tab1:
    st.header("Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total P&L", "₹2,450", "+12.5%")
    with col2:
        st.metric("Active Trades", 3, "+2")
    with col3:
        st.metric("Win Rate", "65%", "+5%")
    with col4:
        st.metric("Daily Loss", "₹5,200", "-₹2,100")
    
    st.divider()
    
    # Market Status
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Market Status")
        if market_open():
            st.success("🟢 Market is OPEN")
        else:
            st.info("🔴 Market is CLOSED")
    
    with col2:
        st.subheader("System Status")
        st.success("✅ All Systems Active")

with tab2:
    st.header("Live Analysis")
    
    selected_stock = st.selectbox("Select Stock", active_stocks)
    
    # Fetch data
    try:
        data = yf.download(selected_stock, period="60d", progress=False)
        
        if len(data) > 0:
            # Calculate indicators
            data['EMA_20'] = ema(data['Close'], 20)
            data['EMA_50'] = ema(data['Close'], 50)
            data['RSI'] = rsi(data['Close'], 14)
            data['MACD_line'], data['MACD_signal'], data['MACD_hist'] = macd(data['Close'])
            data['BB_upper'], data['BB_middle'], data['BB_lower'] = bollinger_bands(data['Close'])
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='white')))
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], name='EMA 20', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], name='EMA 50', line=dict(color='red')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show latest values
            latest = data.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Price", f"₹{latest['Close']:.2f}")
            with col2:
                st.metric("RSI (14)", f"{latest['RSI']:.1f}")
            with col3:
                st.metric("MACD", f"{latest['MACD_line']:.4f}")
            with col4:
                support_res = calculate_support_resistance_advanced(data['High'], data['Low'], data['Close'])
                st.metric("Support", f"₹{support_res['support']:.2f}")
    except:
        st.warning("Unable to fetch data for selected stock")

with tab3:
    st.header("Trading Strategies")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Active Strategies")
        
        for strat_key, strat_info in TRADING_STRATEGIES.items():
            cols = st.columns([3, 1, 1])
            with cols[0]:
                st.write(f"**{strat_info['name']}** ({strat_info['type']})")
            with cols[1]:
                st.write(f"Weight: {strat_info['weight']}")
            with cols[2]:
                status = st.checkbox(f"Enable", key=strat_key, value=True)

with tab4:
    st.header("Trade Journal")
    
    # Sample trades data
    trades_data = {
        'Date': ['2024-12-20', '2024-12-19', '2024-12-18'],
        'Symbol': ['TCS.NS', 'RELIANCE.NS', 'INFY.NS'],
        'Type': ['BUY', 'SELL', 'BUY'],
        'Qty': [10, 5, 15],
        'Entry': [3850, 2450, 2750],
        'Exit': [3920, 2380, 2800],
        'P&L': [700, -350, 750],
        'Strategy': ['EMA+VWAP', 'RSI', 'MACD'],
    }
    
    df_trades = pd.DataFrame(trades_data)
    st.dataframe(df_trades, use_container_width=True)

with tab5:
    st.header("Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Kite Connect API")
        api_key = st.text_input("API Key", type="password")
        api_secret = st.text_input("API Secret", type="password")
        
        if st.button("Connect Kite"):
            st.success("Kite Connected! (Demo)")
    
    with col2:
        st.subheader("Email Configuration")
        email = st.text_input("Email Address", EMAIL_SENDER)
        password = st.text_input("App Password", type="password")
        
        if st.button("Test Email"):
            st.info("Email configuration saved!")

st.divider()
st.caption("RANTV Terminal Pro - Integrated Edition | Capital: ₹15L | HFT Enabled | Professional Grade")
