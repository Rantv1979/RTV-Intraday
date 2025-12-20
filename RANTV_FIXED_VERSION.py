"""
RANTV TERMINAL PRO - PROFESSIONAL EDITION
Enhanced with Institutional-Grade Features for Professional Trading
- High-Frequency Strategies
- Large Capital Deployment
- Consistent Alpha Generation
- Advanced Risk Management
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
import warnings
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
            st.warning("statsmodels not available. Some advanced features disabled.")

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
            st.warning("ARCH not available. Basic volatility modeling.")

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

# ===================== PROFESSIONAL CONFIGURATION =====================
class ProfessionalConfig:
    # Capital Management - 10-20 Lakhs
    TOTAL_CAPITAL = 15_00_000.0  # ₹15 Lakhs (middle of 10-20L range)
    MAX_CAPITAL_UTILIZATION = 0.50  # Max 50% deployed for safety
    MIN_POSITION_SIZE = 10000.0  # Minimum ₹10k per position
    MAX_POSITION_SIZE = 3_00_000.0  # Maximum ₹3L per position
    
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
        "statistical_arbitrage",
        "pairs_trading", 
        "market_neutral",
        "volatility_arbitrage",
        "momentum_reversal",
        "mean_reversion"
    ]
    
    # High-Frequency Settings
    HFT_ENABLED = True
    HFT_LATENCY_TARGET = 100  # ms
    HFT_MIN_PROFIT = 0.001  # 10 basis points minimum profit
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
    LARGE_ORDER_THRESHOLD = 0.90  # 90th percentile
    
    # Multi-Timeframe Analysis
    TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d"]
    TIMEFRAME_WEIGHTS = {
        "5m": 0.25, "15m": 0.20, "30m": 0.15, 
        "1h": 0.15, "4h": 0.15, "1d": 0.10
    }

config = ProfessionalConfig()

# Update existing configuration
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN = ""

ALGO_ENABLED = os.environ.get("ALGO_TRADING_ENABLED", "true").lower() == "true"
ALGO_MAX_POSITIONS = int(os.environ.get("ALGO_MAX_POSITIONS", "8"))
ALGO_MAX_DAILY_LOSS = float(os.environ.get("ALGO_MAX_DAILY_LOSS", "50000"))
ALGO_MIN_CONFIDENCE = float(os.environ.get("ALGO_MIN_CONFIDENCE", "0.75"))

# Email configuration for professional reports
EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "your_email@gmail.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = "rantv2002@gmail.com"
EMAIL_CC = []  # Add professional team emails

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
st.set_page_config(page_title="Rantv Terminal Pro - Professional Edition", layout="wide", initial_sidebar_state="expanded")
IND_TZ = pytz.timezone("Asia/Kolkata")

# Trading Constants - UPDATED FOR 10-20 LAKHS CAPITAL
CAPITAL = config.TOTAL_CAPITAL
TRADE_ALLOC = 0.08  # 8% per trade maximum
MAX_DAILY_TRADES = 30  # Increased for HFT
MAX_STOCK_TRADES = 3
MAX_AUTO_TRADES = 50
SIGNAL_REFRESH_MS = 30000  # 30 seconds for HFT
PRICE_REFRESH_MS = 10000  # 10 seconds for real-time
MARKET_OPTIONS = ["CASH", "FUTURES"]

# ===================== COMPLETE STOCK UNIVERSES =====================

# NIFTY 50 - Complete List
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

# NIFTY 100 - Additional stocks beyond NIFTY 50
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

# NIFTY MIDCAP 150 - Complete List
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
    "YESBANK.NS", "ZEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "AMBUJACEM.NS",
    "ASHOKLEY.NS", "AUROPHARMA.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS",
    "BAJFINANCE.NS", "BANDHANBNK.NS", "BERGEPAINT.NS", "BHARTIARTL.NS",
    "BRITANNIA.NS", "CADILAHC.NS", "COLPAL.NS", "DABUR.NS", "EICHERMOT.NS",
    "GLENMARK.NS", "GODREJCP.NS", "HAVELLS.NS", "HDFCBANK.NS", "HINDALCO.NS",
    "ICICIBANK.NS", "INDUSINDBK.NS", "JSWSTEEL.NS", "LUPIN.NS", "MARICO.NS",
    "MCDOWELL-N.NS", "NESTLEIND.NS", "PEL.NS", "PIDILITIND.NS", "PNB.NS",
    "SIEMENS.NS", "SRTRANSFIN.NS", "SUNTV.NS", "TATACONSUM.NS", "TORNTPHARM.NS",
    "UBL.NS", "ULTRACEMCO.NS", "VEDANTA.NS"
]

# Create complete universe by removing duplicates
ALL_STOCKS = list(dict.fromkeys(NIFTY_50 + NIFTY_100 + NIFTY_MIDCAP_150))

# Create sector-wise groupings for better organization
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
               "NMDC.NS", "SAIL.NS", "JINDALSTEL.NS", "NATIONALUM.NS", "HINDCOPPER.NS"],
    
    "INFRA": ["LT.NS", "ADANIPORTS.NS", "ADANIENSOL.NS", "IRCTC.NS", "CONCOR.NS",
              "KEC.NS", "NBCC.NS", "PNCINFRA.NS", "NCC.NS", "IRB.NS"],
    
    "REALTY": ["DLF.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "SUNTEKKREALTY.NS", "BRIGADE.NS",
               "GODREJPROP.NS", "PHOENIXLTD.NS", "SOBHA.NS", "MAHINDRAFORG.NS", "INDIAFORG.NS"],
    
    "MEDIA": ["ZEEL.NS", "SUNTV.NS", "TV18BRDCST.NS", "NETWORK18.NS", "PVR.NS",
              "INOXLEISURE.NS", "HATHWAY.NS", "DEN.NS", "SITINET.NS", "TVTODAY.NS"]
}

# Create market cap based groupings
MARKET_CAP_GROUPS = {
    "LARGE_CAP": NIFTY_50[:30],  # Top 30 Nifty 50
    "MID_CAP": NIFTY_MIDCAP_150[:50],  # Top 50 Midcap
    "SMALL_CAP": [stock for stock in NIFTY_MIDCAP_150[50:100] if stock not in NIFTY_50 + NIFTY_100]
}

# ===================== UTILITY FUNCTIONS =====================
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

# ===================== SESSION STATE INITIALIZATION =====================
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

# ===================== CSS STYLING =====================
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
    
    /* Professional Badges */
    .pro-badge {
        background: linear-gradient(135deg, #ff8c00 0%, #ff6b00 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin: 2px;
    }
    
    .alpha-badge {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin: 2px;
    }
    
    .hft-badge {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ===================== SIMPLIFIED PROFESSIONAL MODULES =====================

class DynamicCapitalAllocator:
    """Professional capital allocation for 10-20 Lakhs capital"""
    
    def __init__(self, total_capital):
        self.total_capital = total_capital
        self.deployed_capital = 0.0
        self.position_sizing_mode = "half_kelly"
        self.max_position_size_pct = 0.08  # 8% max per position for 10-20L
        self.min_position_size_pct = 0.02  # 2% min per position
        self.position_history = []
        
    def calculate_kelly_fraction(self, win_prob, win_loss_ratio):
        """Calculate Kelly Criterion fraction"""
        if win_loss_ratio <= 0:
            return 0.0
        
        kelly_f = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        
        # Apply conservative approach
        if self.position_sizing_mode == "half_kelly":
            return kelly_f * 0.5
        elif self.position_sizing_mode == "quarter_kelly":
            return kelly_f * 0.25
        elif self.position_sizing_mode == "fixed_fractional":
            return 0.02
        else:
            return max(0.01, min(kelly_f, 0.10))
    
    def calculate_position_size(self, symbol, entry_price, stop_loss, confidence, strategy_stats):
        """Calculate optimal position size for 10-20L capital"""
        
        # Calculate win probability
        win_prob = strategy_stats.get("win_rate", confidence)
        avg_win = strategy_stats.get("avg_win", 1.5)
        avg_loss = strategy_stats.get("avg_loss", 1.0)
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.5
        
        # Calculate Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(win_prob, win_loss_ratio)
        
        # Calculate risk per trade (2% of capital for 10-20L)
        risk_per_trade = self.total_capital * 0.02
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            risk_per_share = entry_price * 0.02  # Default 2% risk
        
        # Maximum shares based on risk
        max_shares_risk = int(risk_per_trade / risk_per_share)
        
        # Maximum shares based on capital allocation
        max_shares_capital = int((self.total_capital * kelly_fraction) / entry_price)
        
        # Choose minimum of the two
        shares = min(max_shares_risk, max_shares_capital)
        
        # Apply position size limits for 10-20L
        min_shares = int((self.total_capital * self.min_position_size_pct) / entry_price)
        max_shares = int((self.total_capital * self.max_position_size_pct) / entry_price)
        
        shares = max(min_shares, min(shares, max_shares))
        
        # Check available capital
        available_capital = self.total_capital - self.deployed_capital
        max_shares_available = int(available_capital / entry_price)
        shares = min(shares, max_shares_available)
        
        position_value = shares * entry_price
        
        return {
            "shares": shares,
            "position_value": position_value,
            "risk_amount": shares * risk_per_share,
            "kelly_fraction": kelly_fraction,
            "capital_utilization": position_value / self.total_capital
        }
    
    def update_deployed_capital(self, position_value, action="add"):
        """Track deployed capital"""
        if action == "add":
            self.deployed_capital += position_value
        else:
            self.deployed_capital -= position_value
        
        self.deployed_capital = max(0.0, self.deployed_capital)
    
    def get_capital_utilization(self):
        """Get current capital utilization"""
        return self.deployed_capital / self.total_capital

class AlphaGenerator:
    """Alpha Generation Strategies without statsmodels dependency"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.strategies = {
            "statistical_arbitrage": self.statistical_arbitrage,
            "pairs_trading": self.pairs_trading,
            "market_neutral": self.market_neutral,
            "momentum_reversal": self.momentum_reversal,
            "mean_reversion": self.mean_reversion
        }
        self.pairs_cache = {}
        
    def statistical_arbitrage(self, symbols=None, lookback_days=30):
        """Multi-stock mean reversion strategy"""
        if symbols is None:
            symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
        
        # Get historical data
        price_data = {}
        for symbol in symbols:
            data = self.data_manager.get_stock_data(symbol, "1h")
            if data is not None and len(data) > 50:
                price_data[symbol] = data["Close"].values[-50:]
        
        if len(price_data) < 3:
            return []
        
        # Create price matrix
        price_df = pd.DataFrame(price_data)
        
        # Calculate z-scores for each stock
        z_scores = {}
        for symbol in price_df.columns:
            prices = price_df[symbol].values
            if len(prices) > 20:
                mean = np.mean(prices[-20:])
                std = np.std(prices[-20:])
                if std > 0:
                    z_scores[symbol] = (prices[-1] - mean) / std
        
        # Generate signals
        signals = []
        buy_threshold = -1.5
        sell_threshold = 1.5
        
        for symbol, z in z_scores.items():
            if z < buy_threshold:
                data = self.data_manager.get_stock_data(symbol, "15m")
                if data is not None:
                    current_price = data["Close"].iloc[-1]
                    atr = data["ATR"].iloc[-1] if "ATR" in data.columns else current_price * 0.02
                    
                    signals.append({
                        "symbol": symbol,
                        "action": "BUY",
                        "price": current_price,
                        "confidence": min(0.95, 0.7 + abs(z) * 0.1),
                        "strategy": "Statistical_Arbitrage",
                        "stop_loss": current_price - (atr * 1.5),
                        "target": current_price + (atr * 3),
                        "z_score": z,
                        "type": "ALPHA"
                    })
            elif z > sell_threshold:
                data = self.data_manager.get_stock_data(symbol, "15m")
                if data is not None:
                    current_price = data["Close"].iloc[-1]
                    atr = data["ATR"].iloc[-1] if "ATR" in data.columns else current_price * 0.02
                    
                    signals.append({
                        "symbol": symbol,
                        "action": "SELL",
                        "price": current_price,
                        "confidence": min(0.95, 0.7 + abs(z) * 0.1),
                        "strategy": "Statistical_Arbitrage",
                        "stop_loss": current_price + (atr * 1.5),
                        "target": current_price - (atr * 3),
                        "z_score": z,
                        "type": "ALPHA"
                    })
        
        return signals
    
    def pairs_trading(self, symbol1="RELIANCE.NS", symbol2="TCS.NS"):
        """Simplified pairs trading without cointegration"""
        cache_key = f"{symbol1}_{symbol2}"
        
        if cache_key in self.pairs_cache:
            cached_data = self.pairs_cache[cache_key]
            if time.time() - cached_data["timestamp"] < 3600:
                return cached_data["signals"]
        
        # Get historical data
        data1 = self.data_manager.get_stock_data(symbol1, "1h")
        data2 = self.data_manager.get_stock_data(symbol2, "1h")
        
        if data1 is None or data2 is None or len(data1) < 50 or len(data2) < 50:
            return []
        
        # Align data
        prices1 = data1["Close"].values[-50:]
        prices2 = data2["Close"].values[-50:]
        
        # Calculate ratio
        ratio = prices1 / prices2
        
        # Calculate z-score of ratio
        ratio_mean = np.mean(ratio[-20:])
        ratio_std = np.std(ratio[-20:])
        
        if ratio_std == 0:
            return []
        
        current_ratio = ratio[-1]
        z_score = (current_ratio - ratio_mean) / ratio_std
        
        # Generate signals
        signals = []
        entry_threshold = 1.5
        
        if z_score > entry_threshold:
            # Ratio too high - sell symbol1, buy symbol2
            signals.append({
                "symbol": symbol1,
                "action": "SELL",
                "price": prices1[-1],
                "confidence": min(0.90, 0.6 + abs(z_score) * 0.1),
                "strategy": "Pairs_Trading",
                "pair": symbol2,
                "z_score": z_score,
                "type": "ALPHA_PAIR"
            })
            signals.append({
                "symbol": symbol2,
                "action": "BUY",
                "price": prices2[-1],
                "confidence": min(0.90, 0.6 + abs(z_score) * 0.1),
                "strategy": "Pairs_Trading",
                "pair": symbol1,
                "z_score": z_score,
                "type": "ALPHA_PAIR"
            })
        elif z_score < -entry_threshold:
            # Ratio too low - buy symbol1, sell symbol2
            signals.append({
                "symbol": symbol1,
                "action": "BUY",
                "price": prices1[-1],
                "confidence": min(0.90, 0.6 + abs(z_score) * 0.1),
                "strategy": "Pairs_Trading",
                "pair": symbol2,
                "z_score": z_score,
                "type": "ALPHA_PAIR"
            })
            signals.append({
                "symbol": symbol2,
                "action": "SELL",
                "price": prices2[-1],
                "confidence": min(0.90, 0.6 + abs(z_score) * 0.1),
                "strategy": "Pairs_Trading",
                "pair": symbol1,
                "z_score": z_score,
                "type": "ALPHA_PAIR"
            })
        
        # Cache results
        self.pairs_cache[cache_key] = {
            "signals": signals,
            "timestamp": time.time(),
            "z_score": z_score
        }
        
        return signals
    
    def market_neutral(self, universe="Nifty 50"):
        """Market-neutral portfolio construction"""
        if universe == "Nifty 50":
            symbols = NIFTY_50[:20]
        elif universe == "Nifty 100":
            symbols = NIFTY_100[:30]
        else:
            symbols = NIFTY_MIDCAP_150[:30]
        
        signals = []
        
        for symbol in symbols:
            data = self.data_manager.get_stock_data(symbol, "1h")
            if data is None or len(data) < 50:
                continue
            
            # Calculate momentum and volatility
            momentum = data["Close"].iloc[-1] / data["Close"].iloc[-20] - 1
            volatility = data["Close"].pct_change().std() * np.sqrt(252)
            
            # Simple scoring system
            score = 0
            if momentum > 0.05:  # 5% momentum
                score += 2
            elif momentum > 0.02:
                score += 1
            
            if volatility < 0.30:  # 30% annualized vol
                score += 2
            elif volatility < 0.40:
                score += 1
            
            # Volume confirmation
            volume_ratio = data["Volume"].iloc[-1] / data["Volume"].rolling(20).mean().iloc[-1]
            if volume_ratio > 1.2:
                score += 1
            
            if score >= 3:
                current_price = data["Close"].iloc[-1]
                atr = data["ATR"].iloc[-1] if "ATR" in data.columns else current_price * 0.02
                
                signals.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "price": current_price,
                    "confidence": 0.7 + (score - 3) * 0.1,
                    "strategy": "Market_Neutral",
                    "stop_loss": current_price - (atr * 1.5),
                    "target": current_price + (atr * 3),
                    "type": "ALPHA"
                })
        
        return signals
    
    def momentum_reversal(self, symbol):
        """Momentum reversal strategy"""
        data = self.data_manager.get_stock_data(symbol, "15m")
        if data is None or len(data) < 100:
            return []
        
        # Calculate RSI
        if 'RSI14' in data.columns:
            rsi = data['RSI14'].iloc[-1]
            rsi_prev = data['RSI14'].iloc[-2] if len(data) > 1 else rsi
            
            current_price = data["Close"].iloc[-1]
            atr = data["ATR"].iloc[-1] if "ATR" in data.columns else current_price * 0.02
            
            signals = []
            
            # RSI oversold bounce
            if rsi < 30 and rsi > rsi_prev:
                signals.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "price": current_price,
                    "confidence": 0.75,
                    "strategy": "Momentum_Reversal",
                    "stop_loss": current_price - (atr * 1.5),
                    "target": current_price + (atr * 2.5),
                    "type": "ALPHA"
                })
            
            # RSI overbought reversal
            if rsi > 70 and rsi < rsi_prev:
                signals.append({
                    "symbol": symbol,
                    "action": "SELL",
                    "price": current_price,
                    "confidence": 0.75,
                    "strategy": "Momentum_Reversal",
                    "stop_loss": current_price + (atr * 1.5),
                    "target": current_price - (atr * 2.5),
                    "type": "ALPHA"
                })
            
            return signals
        return []
    
    def mean_reversion(self, symbol):
        """Mean reversion strategy using Bollinger Bands"""
        data = self.data_manager.get_stock_data(symbol, "15m")
        if data is None or len(data) < 100:
            return []
        
        if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
            current_price = data["Close"].iloc[-1]
            bb_upper = data['BB_Upper'].iloc[-1]
            bb_lower = data['BB_Lower'].iloc[-1]
            atr = data["ATR"].iloc[-1] if "ATR" in data.columns else current_price * 0.02
            
            signals = []
            
            # Price at lower Bollinger Band
            if current_price <= bb_lower * 1.01:
                signals.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "price": current_price,
                    "confidence": 0.80,
                    "strategy": "Mean_Reversion",
                    "stop_loss": current_price - (atr * 1.5),
                    "target": current_price + (atr * 3),
                    "type": "ALPHA"
                })
            
            # Price at upper Bollinger Band
            if current_price >= bb_upper * 0.99:
                signals.append({
                    "symbol": symbol,
                    "action": "SELL",
                    "price": current_price,
                    "confidence": 0.80,
                    "strategy": "Mean_Reversion",
                    "stop_loss": current_price + (atr * 1.5),
                    "target": current_price - (atr * 3),
                    "type": "ALPHA"
                })
            
            return signals
        return []
    
    def generate_all_alpha_signals(self, universe="Nifty 50"):
        """Generate signals from all alpha strategies"""
        all_signals = []
        
        # Determine symbols based on universe
        if universe == "Nifty 50":
            symbols = NIFTY_50[:30]
        elif universe == "Nifty 100":
            symbols = NIFTY_100[:40]
        elif universe == "Midcap 150":
            symbols = NIFTY_MIDCAP_150[:40]
        else:
            symbols = ALL_STOCKS[:50]
        
        # Statistical Arbitrage
        stat_arb_signals = self.statistical_arbitrage(symbols[:10])
        all_signals.extend(stat_arb_signals)
        
        # Pairs Trading (top pairs)
        pairs = [("RELIANCE.NS", "TCS.NS"), ("HDFCBANK.NS", "ICICIBANK.NS"), 
                ("INFY.NS", "TCS.NS"), ("HINDUNILVR.NS", "ITC.NS")]
        
        for pair in pairs:
            pair_signals = self.pairs_trading(pair[0], pair[1])
            all_signals.extend(pair_signals)
        
        # Market Neutral
        neutral_signals = self.market_neutral(universe)
        all_signals.extend(neutral_signals)
        
        # Momentum Reversal for top stocks
        for symbol in symbols[:20]:
            momentum_signals = self.momentum_reversal(symbol)
            all_signals.extend(momentum_signals)
            
            mean_rev_signals = self.mean_reversion(symbol)
            all_signals.extend(mean_rev_signals)
        
        # Filter and rank signals
        ranked_signals = self._rank_signals(all_signals)
        
        return ranked_signals[:20]  # Return top 20 signals
    
    def _rank_signals(self, signals):
        """Rank signals by confidence and strategy weight"""
        if not signals:
            return []
        
        # Strategy weights
        strategy_weights = {
            "Pairs_Trading": 1.5,
            "Statistical_Arbitrage": 1.3,
            "Market_Neutral": 1.2,
            "Momentum_Reversal": 1.4,
            "Mean_Reversion": 1.4,
            "Standard": 1.0
        }
        
        for signal in signals:
            strategy = signal.get("strategy", "Standard")
            weight = strategy_weights.get(strategy, 1.0)
            
            # Adjust confidence by strategy weight
            signal["weighted_confidence"] = signal.get("confidence", 0.5) * weight
            signal["score"] = signal.get("confidence", 0.5) * 10 * weight
        
        # Sort by weighted confidence
        signals.sort(key=lambda x: x.get("weighted_confidence", 0), reverse=True)
        
        return signals

# ===================== ENHANCED DATA MANAGER =====================
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

# ===================== SIGNAL GENERATOR =====================
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
            
            macd_prev = data['MACD'].iloc[-2] if len(data) > 1 else macd
            signal_prev = data['MACD_Signal'].iloc[-2] if len(data) > 1 else signal
            
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
        elif universe == "All Stocks":
            stocks = ALL_STOCKS[:min(max_stocks, len(ALL_STOCKS))]
        else:
            stocks = NIFTY_50[:min(max_stocks, 30)]
        
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

# ===================== TRADING ENGINE =====================
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
        
        # Professional components
        self.capital_allocator = DynamicCapitalAllocator(capital)
        self.alpha_generator = AlphaGenerator(self.data_manager)
        
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

    def execute_professional_trade(self, signal, algorithm="TWAP"):
        """Execute trade with professional position sizing"""
        symbol = signal['symbol']
        action = signal['action']
        
        # Calculate optimal position size
        position_size = self.capital_allocator.calculate_position_size(
            symbol=symbol,
            entry_price=signal['price'],
            stop_loss=signal['stop_loss'],
            confidence=signal['confidence'],
            strategy_stats=self.strategy_performance.get(signal.get('strategy', 'Manual'), {})
        )
        
        quantity = position_size['shares']
        
        if quantity < 1:
            return False, f"Position size too small: {quantity} shares"
        
        # Execute trade
        success, msg = self.execute_trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=signal['price'],
            stop_loss=signal['stop_loss'],
            target=signal['target'],
            win_probability=signal.get('win_probability', 0.75),
            auto_trade=signal.get('type', 'STANDARD') != 'MANUAL',
            strategy=signal.get('strategy', 'Professional')
        )
        
        if success:
            # Update deployed capital
            self.capital_allocator.update_deployed_capital(
                position_size['position_value'], 
                action="add"
            )
            
            # Add professional details to signal
            signal['execution_details'] = {
                'algorithm': algorithm,
                'position_size': position_size
            }
        
        return success, f"{msg} | Position: ₹{position_size['position_value']:,.0f} | Risk: ₹{position_size['risk_amount']:,.0f}"

    def execute_auto_trade_from_signal(self, signal, max_quantity=100):
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
        
        # Calculate position size based on confidence
        position_size_pct = min(0.15, confidence * 0.20)  # Max 15% per trade
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
            
            # Update deployed capital
            position_value = pos["quantity"] * pos["entry_price"]
            self.capital_allocator.update_deployed_capital(position_value, action="remove")
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

    def scan_alpha_signals(self, universe="Nifty 50"):
        """Scan for alpha generation signals"""
        signals = self.alpha_generator.generate_all_alpha_signals(universe)
        return signals
    
    def get_professional_stats(self):
        """Get professional trading statistics"""
        basic_stats = self.get_performance_stats()
        
        # Add capital management stats
        professional_stats = {
            **basic_stats,
            "total_capital": self.capital_allocator.total_capital,
            "deployed_capital": self.capital_allocator.deployed_capital,
            "capital_utilization": self.capital_allocator.get_capital_utilization(),
            "available_capital": self.capital_allocator.total_capital - self.capital_allocator.deployed_capital
        }
        
        return professional_stats

# ===================== KITE CONNECT MANAGER =====================
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

# ===================== ALGO TRADING ENGINE =====================
class AlgoState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"

class AlgoEngine:
    """Algorithmic Trading Engine with Daily Exit at 3:35 PM"""
    
    def __init__(self, trader=None):
        self.state = AlgoState.STOPPED
        self.trader = trader
        self.risk_limits = {
            "max_positions": 8,
            "max_daily_loss": 50000.0,
            "max_position_size": 200000.0,
            "min_confidence": 0.75,
            "max_trades_per_day": 20,
            "max_trades_per_stock": 2
        }
        self.stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "rejected_orders": 0,
            "total_pnl": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "daily_loss": 0.0,
            "trades_today": 0
        }
        self.orders = {}
        self.active_positions = {}
        self.order_history = []
        
        self._stop_event = threading.Event()
        self._scheduler_thread = None
        
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
                    time.sleep(60)
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
                max_stocks=20,
                min_confidence=self.risk_limits["min_confidence"]
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
        if total_pnl < -self.risk_limits["max_daily_loss"]:
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
            if len(self.active_positions) >= self.risk_limits["max_positions"]:
                logger.info("Max positions limit reached")
                break
            
            if self.stats["trades_today"] >= self.risk_limits["max_trades_per_day"]:
                logger.info("Daily trade limit reached")
                break
            
            # Check confidence threshold
            if signal['confidence'] < self.risk_limits["min_confidence"]:
                continue
            
            # Execute trade through trader
            if self.trader:
                success, msg = self.trader.execute_auto_trade_from_signal(signal)
                
                if success:
                    # Create algo order record
                    order_id = f"ALGO_{signal['symbol']}_{int(time.time())}"
                    order = {
                        "order_id": order_id,
                        "symbol": signal['symbol'],
                        "action": signal['action'],
                        "quantity": signal.get('quantity', 10),
                        "price": signal['price'],
                        "stop_loss": signal['stop_loss'],
                        "target": signal['target'],
                        "strategy": signal['strategy'],
                        "confidence": signal['confidence'],
                        "status": "PLACED",
                        "placed_at": datetime.now()
                    }
                    
                    self.orders[order_id] = order
                    self.active_positions[signal['symbol']] = order
                    self.order_history.append(order)
                    
                    self.stats["total_orders"] += 1
                    self.stats["trades_today"] += 1
                    
                    executed_signals.append(signal)
                    
                    logger.info(f"Algo executed: {signal['symbol']} {signal['action']}")
        
        return executed_signals
    
    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "active_positions": len(self.active_positions),
            "total_orders": self.stats["total_orders"],
            "filled_orders": self.stats["filled_orders"],
            "trades_today": self.stats["trades_today"],
            "realized_pnl": self.stats["realized_pnl"],
            "unrealized_pnl": self.stats["unrealized_pnl"],
            "daily_loss": self.stats["daily_loss"],
            "daily_exit_completed": self.daily_exit_completed
        }

# ===================== MAIN APPLICATION =====================
def main():
    # Display Logo and Header
    st.markdown("""
    <div class="logo-container">
        <h1 style="color: white; margin: 10px 0 0 0; font-size: 32px;">📈 RANTV TERMINAL PRO</h1>
        <p style="color: white; margin: 5px 0;">Professional Edition - Complete Stock Universe (10-20L Capital)</p>
        <div style="display: flex; justify-content: center; gap: 15px; margin-top: 10px;">
            <span class="pro-badge">NIFTY 50</span>
            <span class="pro-badge">NIFTY 100</span>
            <span class="pro-badge">MIDCAP 150</span>
            <span class="alpha-badge">Alpha Generation</span>
            <span class="hft-badge">₹15L Capital</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align:center; color: #ff8c00;'>Professional Trading Suite</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color: #6b7280;'>Complete Stock Universe with 10-20 Lakhs Capital Deployment</h4>", unsafe_allow_html=True)
    
    # Initialize components
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
    
    # Sidebar
    with st.sidebar:
        # Kite Login
        kite_manager.login_ui()
        
        st.divider()
        st.header("⚙️ Trading Configuration")
        
        # Universe Selection
        universe = st.selectbox(
            "Stock Universe", 
            ["Nifty 50", "Nifty 100", "Midcap 150", "All Stocks", "Banking", "IT", "Auto", "FMCG", "Pharma"],
            key="universe_select"
        )
        
        # Capital Management
        st.subheader("💰 Capital Management")
        
        capital_display = st.empty()
        capital_display.metric("Total Capital", f"₹{trader.capital_allocator.total_capital:,.0f}")
        
        utilization = trader.capital_allocator.get_capital_utilization()
        st.metric("Capital Utilization", f"{utilization:.1%}")
        
        # Trading Settings
        st.subheader("🎯 Trading Settings")
        
        trader.auto_execution = st.checkbox("Auto Execution", value=False, key="auto_exec_toggle")
        
        min_conf_percent = st.slider("Minimum Confidence %", 60, 90, 75, 5, key="min_conf_slider")
        min_confidence = min_conf_percent / 100
        
        # Strategy Selection
        st.subheader("📊 Strategy Selection")
        
        st.checkbox("EMA Strategies", value=True, key="ema_strategies")
        st.checkbox("RSI Strategies", value=True, key="rsi_strategies")
        st.checkbox("MACD Strategies", value=True, key="macd_strategies")
        st.checkbox("Bollinger Bands", value=True, key="bb_strategies")
        st.checkbox("Alpha Strategies", value=True, key="alpha_strategies")
        
        # System Status
        st.divider()
        st.subheader("🛠️ System Status")
        
        st.write(f"✅ Statsmodels: {'Available' if STATSMODELS_AVAILABLE else 'Not Available'}")
        st.write(f"✅ Kite Connect: {'Available' if KITECONNECT_AVAILABLE else 'Not Available'}")
        st.write(f"✅ TA-Lib: {'Available' if TALIB_AVAILABLE else 'Not Available'}")
        st.write(f"✅ Alpha Engine: {'Active' if STATSMODELS_AVAILABLE else 'Basic'}")
        st.write(f"🔄 Refresh Count: {st.session_state.refresh_count}")
        st.write(f"📊 Market: {'Open' if market_open() else 'Closed'}")
        st.write(f"⏰ Peak Hours: {'Active' if is_peak_market_hours() else 'Inactive'}")
        st.write(f"🕒 Auto Exit: {'3:35 PM Daily'}")
        
        # Quick Actions
        st.divider()
        st.subheader("⚡ Quick Actions")
        
        if st.button("🎯 Run Alpha Scan", key="sidebar_alpha_scan"):
            with st.spinner("Running alpha scan..."):
                signals = trader.scan_alpha_signals(universe)
                if signals:
                    st.success(f"Found {len(signals)} alpha signals")
                else:
                    st.info("No alpha signals found")
        
        if st.button("📊 Update All", key="sidebar_update_all"):
            st.rerun()
    
    # Main Content
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
    cols[3].metric("Selected Universe", universe)
    cols[4].metric("Total Stocks", len(ALL_STOCKS))
    cols[5].metric("Auto Trades", f"{trader.auto_trades_count}/{MAX_AUTO_TRADES}")
    cols[6].metric("Available Cash", f"₹{trader.cash:,.0f}")
    
    # Stock Universe Stats
    st.subheader("📈 Stock Universe Statistics")
    
    universe_cols = st.columns(4)
    
    with universe_cols[0]:
        st.metric("NIFTY 50 Stocks", len(NIFTY_50))
    
    with universe_cols[1]:
        st.metric("NIFTY 100 Stocks", len(NIFTY_100))
    
    with universe_cols[2]:
        st.metric("MIDCAP 150 Stocks", len(NIFTY_MIDCAP_150))
    
    with universe_cols[3]:
        st.metric("Total Unique Stocks", len(ALL_STOCKS))
    
    # Main Tabs
    tabs = st.tabs(["📊 Dashboard", "🚦 Signals", "🎯 Alpha Strategies", "💰 Paper Trading", 
                   "📋 Trade History", "📉 RSI Scanner", "🤖 Algo Trading"])
    
    # Tab 1: Dashboard
    with tabs[0]:
        st.subheader("Account Summary")
        trader.update_positions_pnl()
        perf = trader.get_performance_stats()
        pro_stats = trader.get_professional_stats()
        
        total_value = trader.cash + sum([p.get('quantity', 0) * p.get('entry_price', 0) 
                                       for p in trader.positions.values()])
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Value", f"₹{total_value:,.0f}", 
                 delta=f"₹{total_value - trader.initial_capital:+,.0f}")
        c2.metric("Available Cash", f"₹{trader.cash:,.0f}")
        c3.metric("Open Positions", len(trader.positions))
        c4.metric("Total P&L", f"₹{perf['total_pnl'] + perf['open_pnl']:+.2f}")
        
        # Capital Management
        st.subheader("💰 Capital Management")
        
        cap_cols = st.columns(3)
        cap_cols[0].metric("Total Capital", f"₹{pro_stats['total_capital']:,.0f}")
        cap_cols[1].metric("Deployed Capital", f"₹{pro_stats['deployed_capital']:,.0f}")
        cap_cols[2].metric("Capital Utilization", f"{pro_stats['capital_utilization']:.1%}")
        
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
                value=0.75,
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
                scan_size = 30 if scan_mode == "Quick Scan (Top 30)" else 50
                
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
                    "Strategy": signal['strategy']
                })
            
            # Create dataframe
            df_signals = pd.DataFrame(signal_data)
            
            # Display with formatting
            st.dataframe(
                df_signals,
                use_container_width=True,
                hide_index=True,
                column_config={
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
            
            exec_cols = st.columns(4)
            
            with exec_cols[0]:
                if st.button("📈 Execute Top Signal", type="primary", 
                            disabled=not trader.auto_execution or not market_open()):
                    if signals:
                        success, msg = trader.execute_auto_trade_from_signal(signals[0])
                        if success:
                            st.success(f"✅ {msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
            
            with exec_cols[1]:
                if st.button("📈 Execute All BUY", type="secondary",
                            disabled=not trader.auto_execution or not market_open()):
                    buy_signals = [s for s in signals if s['action'] == 'BUY']
                    executed = 0
                    for signal in buy_signals[:3]:  # Limit to 3
                        success, msg = trader.execute_auto_trade_from_signal(signal)
                        if success:
                            executed += 1
                    if executed > 0:
                        st.success(f"Executed {executed} BUY signals!")
                        st.rerun()
            
            with exec_cols[2]:
                if st.button("📉 Execute All SELL", type="secondary",
                            disabled=not trader.auto_execution or not market_open()):
                    sell_signals = [s for s in signals if s['action'] == 'SELL']
                    executed = 0
                    for signal in sell_signals[:3]:
                        success, msg = trader.execute_auto_trade_from_signal(signal)
                        if success:
                            executed += 1
                    if executed > 0:
                        st.success(f"Executed {executed} SELL signals!")
                        st.rerun()
            
            with exec_cols[3]:
                if st.button("🎯 Professional Execute Top", type="primary",
                            disabled=not trader.auto_execution or not market_open()):
                    if signals:
                        success, msg = trader.execute_professional_trade(signals[0])
                        if success:
                            st.success(f"✅ {msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
    
    # Tab 3: Alpha Strategies
    with tabs[2]:
        st.subheader("🎯 Alpha Generation Strategies")
        
        if not STATSMODELS_AVAILABLE:
            st.warning("⚠️ statsmodels not available. Using simplified alpha strategies.")
        
        # Alpha Strategy Controls
        col1, col2 = st.columns(2)
        
        with col1:
            alpha_universe = st.selectbox(
                "Alpha Scan Universe",
                ["Nifty 50", "Nifty 100", "Midcap 150"],
                key="alpha_universe"
            )
        
        with col2:
            min_alpha_confidence = st.slider(
                "Min Alpha Confidence",
                min_value=0.65,
                max_value=0.95,
                value=0.75,
                step=0.05,
                key="min_alpha_conf"
            )
        
        # Run Alpha Scan
        if st.button("🚀 Run Alpha Strategies", type="primary", key="run_alpha_strategies"):
            with st.spinner(f"Running alpha strategies on {alpha_universe}..."):
                signals = trader.scan_alpha_signals(alpha_universe)
                
                if signals:
                    # Filter by confidence
                    filtered_signals = [s for s in signals if s.get('confidence', 0) >= min_alpha_confidence]
                    
                    # Display alpha signals
                    alpha_data = []
                    for i, signal in enumerate(filtered_signals[:15]):  # Show top 15
                        signal_type = signal.get('type', 'ALPHA')
                        alpha_data.append({
                            "#": i+1,
                            "Symbol": signal['symbol'].replace('.NS', ''),
                            "Type": signal_type,
                            "Action": signal['action'],
                            "Price": f"₹{signal['price']:.2f}",
                            "Confidence": f"{signal['confidence']:.1%}",
                            "Strategy": signal['strategy'],
                            "Z-Score": f"{signal.get('z_score', 'N/A')}"
                        })
                    
                    df_alpha = pd.DataFrame(alpha_data)
                    st.dataframe(df_alpha, use_container_width=True)
                    
                    # Alpha execution
                    st.subheader("Alpha Execution")
                    
                    alpha_exec_cols = st.columns(3)
                    
                    with alpha_exec_cols[0]:
                        if st.button("🎯 Execute Top Alpha", key="execute_top_alpha"):
                            if filtered_signals:
                                success, msg = trader.execute_professional_trade(filtered_signals[0])
                                if success:
                                    st.success(f"✅ {msg}")
                                    st.rerun()
                    
                    with alpha_exec_cols[1]:
                        if st.button("📊 Execute Pairs Trade", key="execute_pairs_trade"):
                            # Find pairs trading signals
                            pairs_signals = [s for s in filtered_signals if s.get('strategy') == 'Pairs_Trading']
                            if pairs_signals:
                                success, msg = trader.execute_professional_trade(pairs_signals[0])
                                if success:
                                    st.success(f"✅ {msg}")
                                    st.rerun()
                    
                    with alpha_exec_cols[2]:
                        if st.button("📈 Execute Statistical Arb", key="execute_stat_arb"):
                            # Find statistical arbitrage signals
                            stat_arb_signals = [s for s in filtered_signals if s.get('strategy') == 'Statistical_Arbitrage']
                            if stat_arb_signals:
                                success, msg = trader.execute_professional_trade(stat_arb_signals[0])
                                if success:
                                    st.success(f"✅ {msg}")
                                    st.rerun()
                
                else:
                    st.info("No alpha signals generated. Try adjusting parameters.")
        
        # Alpha Strategy Info
        st.subheader("📊 Alpha Strategy Information")
        
        info_cols = st.columns(3)
        
        with info_cols[0]:
            st.markdown("""
            **Statistical Arbitrage**
            - Multi-stock mean reversion
            - Z-score based entry/exit
            - Market neutral approach
            """)
        
        with info_cols[1]:
            st.markdown("""
            **Pairs Trading**
            - Ratio-based trading
            - Simplified cointegration
            - Mean reversion of spreads
            """)
        
        with info_cols[2]:
            st.markdown("""
            **Market Neutral**
            - Factor-based scoring
            - Momentum + volatility
            - Volume confirmation
            """)
    
    # Tab 4: Paper Trading
    with tabs[3]:
        st.subheader("💰 Professional Paper Trading")
        
        # Professional Trading Interface
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            symbol = st.selectbox("Symbol", ALL_STOCKS[:30], key="paper_symbol")
        with col2:
            action = st.selectbox("Action", ["BUY", "SELL"], key="paper_action")
        with col3:
            quantity = st.number_input("Quantity", min_value=1, value=50, key="paper_quantity")
        with col4:
            strategy_name = st.selectbox("Strategy", 
                                        ["Professional Alpha", "Statistical Arbitrage", "Pairs Trading", 
                                         "Market Neutral", "Momentum Reversal", "Mean Reversion", "Manual"],
                                        key="paper_strategy")
        
        # Professional execution options
        exec_option = st.radio(
            "Execution Type",
            ["Standard Trade", "Professional Position Sizing", "Alpha Strategy"],
            horizontal=True,
            key="exec_option"
        )
        
        if st.button("🚀 Execute Professional Trade", type="primary", key="execute_pro_trade"):
            try:
                data = trader.data_manager.get_stock_data(symbol, "15m")
                if data is not None and len(data) > 0:
                    price = float(data["Close"].iloc[-1])
                    atr = float(data["ATR"].iloc[-1]) if 'ATR' in data.columns else price * 0.02
                    
                    if action == "BUY":
                        stop_loss = price - (atr * 1.5)
                        target = price + (atr * 3)
                    else:
                        stop_loss = price + (atr * 1.5)
                        target = price - (atr * 3)
                    
                    # Create signal based on execution type
                    if exec_option == "Professional Position Sizing":
                        # Use professional position sizing
                        signal = {
                            'symbol': symbol,
                            'action': action,
                            'price': price,
                            'stop_loss': stop_loss,
                            'target': target,
                            'confidence': 0.8,
                            'strategy': strategy_name,
                            'win_probability': 0.75,
                            'type': 'PROFESSIONAL'
                        }
                        
                        success, msg = trader.execute_professional_trade(signal)
                    
                    elif exec_option == "Alpha Strategy":
                        # Generate alpha signal
                        if strategy_name == "Statistical Arbitrage":
                            # Get z-score for statistical arbitrage
                            symbols = [symbol, "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
                            price_data = {}
                            for sym in symbols:
                                sym_data = trader.data_manager.get_stock_data(sym, "1h")
                                if sym_data is not None and len(sym_data) > 50:
                                    price_data[sym] = sym_data["Close"].values[-50:]
                            
                            if len(price_data) >= 3:
                                price_df = pd.DataFrame(price_data)
                                if symbol in price_df.columns:
                                    prices = price_df[symbol].values
                                    if len(prices) > 20:
                                        mean = np.mean(prices[-20:])
                                        std = np.std(prices[-20:])
                                        if std > 0:
                                            z_score = (prices[-1] - mean) / std
                                            confidence = min(0.95, 0.7 + abs(z_score) * 0.1)
                                            
                                            signal = {
                                                'symbol': symbol,
                                                'action': action,
                                                'price': price,
                                                'stop_loss': stop_loss,
                                                'target': target,
                                                'confidence': confidence,
                                                'strategy': 'Statistical_Arbitrage',
                                                'win_probability': 0.75,
                                                'z_score': z_score,
                                                'type': 'ALPHA'
                                            }
                                            
                                            success, msg = trader.execute_professional_trade(signal)
                                        else:
                                            success, msg = False, "Insufficient data for alpha strategy"
                                    else:
                                        success, msg = False, "Insufficient data for alpha strategy"
                                else:
                                    success, msg = False, "Symbol not in price data"
                            else:
                                success, msg = False, "Insufficient data for statistical arbitrage"
                        else:
                            # Fall back to standard trade
                            success, msg = trader.execute_trade(
                                symbol=symbol,
                                action=action,
                                quantity=quantity,
                                price=price,
                                stop_loss=stop_loss,
                                target=target,
                                win_probability=0.75,
                                auto_trade=False,
                                strategy=strategy_name
                            )
                    else:
                        # Standard trade
                        success, msg = trader.execute_trade(
                            symbol=symbol,
                            action=action,
                            quantity=quantity,
                            price=price,
                            stop_loss=stop_loss,
                            target=target,
                            win_probability=0.75,
                            auto_trade=False,
                            strategy=strategy_name
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
        
        # Current Positions with Professional Details
        st.subheader("Current Professional Positions")
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
                        <span style="color: {pnl_color}">P&L: ₹{pnl:+.2f}</span> | Strategy: {pos.get('strategy', 'Manual')}
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
    
    # Tab 5: Trade History
    with tabs[4]:
        st.subheader("📋 Professional Trade History")
        
        if trader.trade_log:
            history_data = []
            for trade in trader.trade_log[-50:]:  # Last 50 trades
                if trade.get("status") == "CLOSED":
                    pnl = trade.get("closed_pnl", 0)
                    history_data.append({
                        "Symbol": trade['symbol'].replace('.NS', ""),
                        "Action": trade['action'],
                        "Quantity": trade['quantity'],
                        "Entry": f"₹{trade['entry_price']:.2f}",
                        "Exit": f"₹{trade.get('exit_price', 0):.2f}",
                        "P&L": f"₹{pnl:+.2f}",
                        "Entry Time": trade.get('entry_time', ''),
                        "Exit Time": trade.get('exit_time_str', ''),
                        "Strategy": trade.get('strategy', 'Manual'),
                        "Type": "Professional" if trade.get('strategy') in ['Statistical_Arbitrage', 'Pairs_Trading', 'Market_Neutral'] else "Standard"
                    })
            
            if history_data:
                df_history = pd.DataFrame(history_data)
                st.dataframe(df_history, use_container_width=True)
                
                # Summary statistics
                st.subheader("📊 Performance Summary")
                
                if history_data:
                    total_pnl = sum([float(h['P&L'].replace('₹', '').replace(',', '')) for h in history_data])
                    winning_trades = sum(1 for h in history_data if float(h['P&L'].replace('₹', '').replace(',', '')) > 0)
                    total_trades = len(history_data)
                    win_rate = winning_trades / total_trades if total_trades > 0 else 0
                    
                    summary_cols = st.columns(4)
                    summary_cols[0].metric("Total Trades", total_trades)
                    summary_cols[1].metric("Win Rate", f"{win_rate:.1%}")
                    summary_cols[2].metric("Total P&L", f"₹{total_pnl:+.2f}")
                    summary_cols[3].metric("Avg P&L per Trade", f"₹{total_pnl/total_trades:+.2f}" if total_trades > 0 else "₹0.00")
            else:
                st.info("No closed trades yet")
        else:
            st.info("No trade history available")
    
    # Tab 6: RSI Scanner
    with tabs[5]:
        st.subheader("📉 RSI Extreme Scanner")
        
        # Scanner controls
        col1, col2 = st.columns(2)
        
        with col1:
            rsi_universe = st.selectbox(
                "Scan Universe",
                ["Nifty 50", "Nifty 100", "Midcap 150", "All Stocks"],
                key="rsi_universe"
            )
        
        with col2:
            rsi_threshold = st.slider(
                "RSI Threshold",
                min_value=20,
                max_value=80,
                value=30,
                step=5,
                key="rsi_threshold"
            )
        
        if st.button("🔍 Scan for RSI Extremes", key="rsi_scan_btn"):
            with st.spinner(f"Scanning {rsi_universe} for RSI extremes..."):
                # Determine stocks to scan
                if rsi_universe == "Nifty 50":
                    stocks = NIFTY_50[:50]
                elif rsi_universe == "Nifty 100":
                    stocks = NIFTY_100[:80]
                elif rsi_universe == "Midcap 150":
                    stocks = NIFTY_MIDCAP_150[:100]
                else:
                    stocks = ALL_STOCKS[:150]
                
                oversold = []
                overbought = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, symbol in enumerate(stocks):
                    status_text.text(f"Scanning {symbol.replace('.NS', '')}... ({i+1}/{len(stocks)})")
                    
                    try:
                        data = trader.data_manager.get_stock_data(symbol, "15m")
                        if data is not None and len(data) > 0:
                            if 'RSI14' in data.columns:
                                rsi_val = data['RSI14'].iloc[-1]
                                price = data['Close'].iloc[-1]
                                
                                if rsi_val < rsi_threshold:
                                    oversold.append({
                                        "Symbol": symbol.replace('.NS', ''),
                                        "RSI": round(rsi_val, 2),
                                        "Price": round(price, 2),
                                        "Signal": "OVERSOLD 🔵",
                                        "ATR": round(data['ATR'].iloc[-1], 2) if 'ATR' in data.columns else "N/A"
                                    })
                                elif rsi_val > (100 - rsi_threshold):
                                    overbought.append({
                                        "Symbol": symbol.replace('.NS', ''),
                                        "RSI": round(rsi_val, 2),
                                        "Price": round(price, 2),
                                        "Signal": "OVERBOUGHT 🔴",
                                        "ATR": round(data['ATR'].iloc[-1], 2) if 'ATR' in data.columns else "N/A"
                                    })
                    except:
                        continue
                    
                    progress_bar.progress((i + 1) / len(stocks))
                
                progress_bar.empty()
                status_text.empty()
                
                if oversold or overbought:
                    st.success(f"Found {len(oversold)} oversold and {len(overbought)} overbought stocks")
                    
                    if oversold:
                        st.subheader("🔵 Oversold Stocks (RSI < {})".format(rsi_threshold))
                        st.dataframe(pd.DataFrame(oversold), use_container_width=True)
                        
                        # Quick action for oversold
                        if oversold and st.button("📈 Trade Top Oversold", key="trade_oversold_btn"):
                            top_oversold = oversold[0]
                            symbol = top_oversold["Symbol"] + ".NS"
                            data = trader.data_manager.get_stock_data(symbol, "15m")
                            if data is not None:
                                price = data["Close"].iloc[-1]
                                atr = data["ATR"].iloc[-1] if 'ATR' in data.columns else price * 0.02
                                
                                signal = {
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'price': price,
                                    'stop_loss': price - (atr * 1.5),
                                    'target': price + (atr * 3),
                                    'confidence': 0.8 - (top_oversold["RSI"] / 100),
                                    'strategy': 'RSI_Mean_Reversion',
                                    'win_probability': 0.75,
                                    'type': 'ALPHA'
                                }
                                
                                success, msg = trader.execute_professional_trade(signal)
                                if success:
                                    st.success(f"✅ {msg}")
                                    st.rerun()
                    
                    if overbought:
                        st.subheader("🔴 Overbought Stocks (RSI > {})".format(100 - rsi_threshold))
                        st.dataframe(pd.DataFrame(overbought), use_container_width=True)
                else:
                    st.info("No extreme RSI stocks found")
    
    # Tab 7: Algo Trading
    with tabs[6]:
        st.subheader("🤖 Algorithmic Trading Engine")
        
        if algo_engine is None:
            st.warning("Algo Engine not initialized")
        else:
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
            
            # Algo Settings
            st.subheader("⚙️ Algo Settings")
            
            settings_cols = st.columns(3)
            
            with settings_cols[0]:
                algo_universe = st.selectbox(
                    "Algo Universe",
                    ["Nifty 50", "Nifty 100", "Midcap 150"],
                    key="algo_universe"
                )
            
            with settings_cols[1]:
                algo_min_conf = st.slider(
                    "Min Confidence",
                    min_value=0.60,
                    max_value=0.95,
                    value=0.75,
                    step=0.05,
                    key="algo_min_conf"
                )
            
            with settings_cols[2]:
                max_algo_positions = st.number_input(
                    "Max Positions",
                    min_value=1,
                    max_value=20,
                    value=8,
                    key="max_algo_positions"
                )
            
            # Active Positions
            if algo_engine.active_positions:
                st.subheader("Active Algo Positions")
                positions_data = []
                for symbol, order in algo_engine.active_positions.items():
                    positions_data.append({
                        "Symbol": symbol.replace('.NS', ''),
                        "Action": order["action"],
                        "Quantity": order["quantity"],
                        "Entry Price": f"₹{order['price']:.2f}",
                        "Stop Loss": f"₹{order['stop_loss']:.2f}",
                        "Target": f"₹{order['target']:.2f}",
                        "Strategy": order["strategy"],
                        "Confidence": f"{order['confidence']:.1%}"
                    })
                
                st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
            
            # Manual trigger for signal scan
            if st.button("🔍 Manual Signal Scan", key="manual_algo_scan"):
                with st.spinner("Running algo signal scan..."):
                    signals = trader.signal_generator.scan_stock_universe(
                        universe=algo_universe,
                        max_stocks=30,
                        min_confidence=algo_min_conf
                    )
                    
                    if signals:
                        executed = algo_engine.process_signals(signals[:max_algo_positions])
                        st.success(f"Processed {len(executed)} signals")
                        st.rerun()
                    else:
                        st.info("No signals found")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; font-size: 12px;">
        <strong>Rantv Terminal Pro - Professional Edition</strong> | Complete Stock Universe (Nifty50/Nifty100/Midcap150) | 
        Capital: ₹{trader.capital_allocator.total_capital:,.0f} | Refresh: {st.session_state.refresh_count} | 
        {now_indian().strftime("%H:%M:%S")} | Auto Exit: 3:35 PM | Stocks: {len(ALL_STOCKS)}
    </div>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.code(traceback.format_exc())
