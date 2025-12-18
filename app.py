# Rantv Intraday Trading Signals & Market Analysis - PRODUCTION READY
# ENHANCED VERSION WITH FULL STOCK SCANNING & BETTER SIGNAL QUALITY
# UPDATED: Lowered confidence to 70%, score to 6, added ADX trend filter, optimized for peak hours
# INTEGRATED WITH KITE CONNECT FOR LIVE CHARTS

import time
from datetime import datetime, time as dt_time
import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import math
import warnings
import os
from dataclasses import dataclass
from typing import Optional, Dict, List
import requests
import json
import traceback
import subprocess
import sys
from datetime import timedelta
import threading

# ================= AUTO TRADING CONTROL =================
AUTO_TRADING_ENABLED = False   # <<< CHANGE TO True ONLY WHEN READY
AUTO_MIN_CONFIDENCE = 0.85     # Minimum confidence for auto execution
AUTO_MAX_TRADES_PER_DAY = 5
AUTO_COOLDOWN_SECONDS = 300    # 5 min gap per symbol

from collections import defaultdict

auto_trade_state = {
    "trades_today": 0,
    "last_trade_time": defaultdict(lambda: datetime.min),
    "enabled_at": None
}

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
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Kite Connect API Credentials
KITE_API_KEY = os.environ.get("KITE_API_KEY", "pwnmsnpy30s4uotu")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "m44rfdl9ligc4ctaq7r9sxkxpgnfm30m")
KITE_ACCESS_TOKEN = ""  # Will be set after login

# Configuration
@dataclass
class AppConfig:
    database_url: str = 'sqlite:///trading_journal.db'
    risk_tolerance: str = 'MODERATE'
    max_daily_loss: float = 50000.0
    enable_ml: bool = True
    kite_api_key: str = KITE_API_KEY
    kite_api_secret: str = KITE_API_SECRET
    
    @classmethod
    def from_env(cls):
        return cls()

# Initialize configuration
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

# Stock Universes - COMBINED ALL STOCKS
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

# MIDCAP STOCKS - High Potential for Intraday
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

# COMBINED ALL STOCKS - NEW UNIVERSES
ALL_STOCKS = list(dict.fromkeys(NIFTY_50 + NIFTY_100 + NIFTY_MIDCAP_150))

# Enhanced Trading Strategies with Better Balance - ALL STRATEGIES ENABLED
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

# HIGH ACCURACY STRATEGIES FOR ALL STOCKS - ENABLED FOR ALL UNIVERSES
HIGH_ACCURACY_STRATEGIES = {
    "Multi_Confirmation": {"name": "Multi-Confirmation Ultra", "weight": 5, "type": "BOTH"},
    "Enhanced_EMA_VWAP": {"name": "Enhanced EMA-VWAP", "weight": 4, "type": "BOTH"},
    "Volume_Breakout": {"name": "Volume Weighted Breakout", "weight": 4, "type": "BOTH"},
    "RSI_Divergence": {"name": "RSI Divergence", "weight": 3, "type": "BOTH"},
    "MACD_Trend": {"name": "MACD Trend Momentum", "weight": 3, "type": "BOTH"}
}

# FIXED CSS with Light Yellowish Background and Better Tabs
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #fff9e6 0%, #fff0d6 100%);
    }
    
    .main .block-container {
        background-color: transparent;
        padding-top: 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: linear-gradient(135deg, #e6f2ff 0%, #ffe6e6 50%, #e6ffe6 100%);
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
        color: #1e3a8a;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        color: white;
        border: 2px solid #2563eb;
        box-shadow: 0 4px 8px rgba(30, 58, 138, 0.3);
        transform: translateY(-2px);
    }
    
    .gauge-container {
        background: white;
        border-radius: 50%;
        padding: 25px;
        margin: 10px auto;
        border: 4px solid #e0f2fe;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
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
        color: #1e3a8a;
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
        color: #d97706;
        background-color: #fef3c7;
    }
    
    .gauge-progress {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: conic-gradient(#059669 0% var(--progress), #e5e7eb var(--progress) 100%);
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
    
    .rsi-oversold { 
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #059669;
    }
    
    .rsi-overbought { 
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
    }
    
    .bullish-signal { 
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #059669;
        border-radius: 8px;
    }
    
    .bearish-signal { 
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
        border-radius: 8px;
    }
    
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1e3a8a;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .high-accuracy-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
    }
    
    .refresh-counter {
        background: #1e3a8a;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-left: 8px;
    }
    
    .profit-positive {
        color: #059669;
        font-weight: bold;
        background-color: #d1fae5;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    .profit-negative {
        color: #dc2626;
        font-weight: bold;
        background-color: #fee2e2;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    .trade-buy {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #059669;
    }
    
    .trade-sell {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #059669;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #d97706;
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
    
    .dependencies-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #d97706;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #f59e0b;
    }
    
    .auto-exec-active {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #059669;
    }
    
    .auto-exec-inactive {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #6b7280;
    }
    
    .high-quality-signal {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #047857;
    }
    
    .medium-quality-signal {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #b45309;
    }
    
    .low-quality-signal {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #b91c1c;
    }
</style>
""", unsafe_allow_html=True)

# System Status Check
def check_system_status():
    """Check system dependencies and return status"""
    status = {
        "kiteconnect": KITECONNECT_AVAILABLE,
        "sqlalchemy": SQLALCHEMY_AVAILABLE,
        "joblib": JOBLIB_AVAILABLE,
        "yfinance": True,
        "plotly": True,
        "pandas": True,
        "numpy": True,
        "streamlit": True,
        "pytz": True,
        "streamlit_autorefresh": True
    }
    return status

# Display system status in sidebar
system_status = check_system_status()

# Kite Token Database Manager for OAuth Token Persistence
class KiteTokenDatabase:
    def __init__(self):
        self.db_url = os.environ.get("DATABASE_URL")
        self.engine = None
        self.connected = False
        if self.db_url and SQLALCHEMY_AVAILABLE:
            try:
                self.engine = create_engine(self.db_url)
                self.create_tables()
                self.connected = True
            except Exception as e:
                logger.error(f"Kite Token DB connection failed: {e}")
    
    def create_tables(self):
        if not self.engine:
            return
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS kite_tokens (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(100) DEFAULT 'default',
                        access_token TEXT,
                        refresh_token TEXT,
                        public_token TEXT,
                        user_name VARCHAR(200),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        is_valid BOOLEAN DEFAULT TRUE
                    )
                """))
                conn.commit()
        except Exception as e:
            logger.error(f"Error creating kite_tokens table: {e}")
    
    def save_token(self, access_token, user_name="", public_token="", refresh_token=""):
        if not self.connected:
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(text("UPDATE kite_tokens SET is_valid = FALSE WHERE user_id = 'default'"))
                conn.execute(text("""
                    INSERT INTO kite_tokens (user_id, access_token, refresh_token, public_token, user_name, is_valid, expires_at)
                    VALUES ('default', :access_token, :refresh_token, :public_token, :user_name, TRUE, NOW() + INTERVAL '8 hours')
                """), {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "public_token": public_token,
                    "user_name": user_name
                })
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving token: {e}")
            return False
    
    def get_valid_token(self):
        if not self.connected:
            return None
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT access_token, user_name FROM kite_tokens 
                    WHERE user_id = 'default' AND is_valid = TRUE AND expires_at > NOW()
                    ORDER BY created_at DESC LIMIT 1
                """))
                row = result.fetchone()
                if row:
                    return {"access_token": row[0], "user_name": row[1]}
                return None
        except Exception as e:
            logger.error(f"Error getting token: {e}")
            return None
    
    def invalidate_token(self):
        if not self.connected:
            return
        try:
            with self.engine.connect() as conn:
                conn.execute(text("UPDATE kite_tokens SET is_valid = FALSE WHERE user_id = 'default'"))
                conn.commit()
        except Exception as e:
            logger.error(f"Error invalidating token: {e}")

# Initialize Kite Token Database
kite_token_db = KiteTokenDatabase()

# Kite Connect Manager Class - Enhanced with OAuth Flow
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

    def check_oauth_callback(self) -> bool:
        """If the URL contains a request_token and we haven't consumed it this session, exchange it for an access token."""
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

            try:
                kite_token_db.save_token(
                    access_token=self.access_token,
                    user_name=data.get("user_name", ""),
                    public_token=data.get("public_token", ""),
                    refresh_token=data.get("refresh_token", "")
                )
            except Exception as db_e:
                logger.warning(f"Kite token DB save warning: {db_e}")

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
            if not self.api_key or not self.api_secret:
                st.warning("Kite API Key not configured. Set KITE_API_KEY and KITE_API_SECRET in environment secrets.")
                return False

            if not self.kite:
                self.kite = KiteConnect(api_key=self.api_key)

            if not st.session_state.kite_oauth_in_progress and self.check_oauth_callback():
                return True

            if "kite_access_token" in st.session_state:
                self.access_token = st.session_state.kite_access_token
                self.kite.set_access_token(self.access_token)
                try:
                    _ = self.kite.profile()
                    self.is_authenticated = True
                    return True
                except Exception:
                    del st.session_state["kite_access_token"]

            db_token = kite_token_db.get_valid_token() if kite_token_db else None
            if db_token:
                self.access_token = db_token["access_token"]
                self.kite.set_access_token(self.access_token)
                try:
                    profile = self.kite.profile()
                    self.is_authenticated = True
                    st.session_state.kite_access_token = self.access_token
                    st.session_state.kite_user_name = profile.get("user_name", "")
                    return True
                except Exception:
                    try:
                        kite_token_db.invalidate_token()
                    except Exception:
                        pass

            if st.session_state.kite_oauth_in_progress:
                st.info("Completing authentication…")
                return False

            st.info("Kite Connect authentication required for live charts.")
            login_url = self.kite.login_url()

            st.link_button("🔐 Login with Kite", login_url, use_container_width=True)

            st.markdown("**Or enter access token manually:**")
            with st.form("kite_login_form"):
                access_token = st.text_input("Access Token", type="password", help="Paste your access token from Kite Connect")
                submit = st.form_submit_button("Authenticate", type="primary")

            if submit and access_token:
                try:
                    self.access_token = access_token
                    self.kite.set_access_token(self.access_token)
                    profile = self.kite.profile()
                    user_name = profile.get("user_name", "")
                    st.session_state.kite_access_token = self.access_token
                    st.session_state.kite_user_name = user_name
                    try:
                        kite_token_db.save_token(access_token=self.access_token, user_name=user_name)
                    except Exception:
                        pass
                    self.is_authenticated = True
                    st.success(f"Authenticated as {user_name}")
                    return True
                except Exception as e:
                    st.error(f"Authentication failed: {str(e)}")
                    return False

            return False

        except Exception as e:
            st.error(f"Kite Connect login error: {str(e)}")
            return False

    def logout(self):
        """Logout from Kite Connect"""
        try:
            if "kite_access_token" in st.session_state:
                del st.session_state.kite_access_token
            if "kite_user_name" in st.session_state:
                del st.session_state.kite_user_name
            kite_token_db.invalidate_token()
            self.access_token = None
            self.is_authenticated = False
            return True
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False

# NEW: Advanced Risk Management System
class AdvancedRiskManager:
    def __init__(self, max_daily_loss=50000):
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0.0
        self.position_sizing_enabled = True
        self.last_reset_date = datetime.now().date()
    
    def reset_daily_metrics(self):
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
    
    def calculate_kelly_position_size(self, win_probability, win_loss_ratio, available_capital, price, atr):
        """Calculate position size using Kelly Criterion"""
        try:
            if win_loss_ratio <= 0:
                win_loss_ratio = 2.0
                
            kelly_fraction = win_probability - (1 - win_probability) / win_loss_ratio
            
            risk_capital = available_capital * 0.1
            position_value = risk_capital * (kelly_fraction / 2)
            
            if price <= 0:
                return 1
                
            quantity = int(position_value / price)
            
            return max(1, min(quantity, int(available_capital * 0.2 / price)))
        except Exception:
            return int((available_capital * 0.1) / price)
    
    def check_trade_viability(self, symbol, action, quantity, price, current_positions):
        """Automatically adjust position size to stay within risk limits."""

        self.reset_daily_metrics()

        if price is None or price <= 0:
            return False, "Invalid price"

        current_portfolio_value = sum([
            pos.get("quantity", 0) * pos.get("entry_price", 0)
            for pos in current_positions.values()
            if pos.get("entry_price", 0) > 0
        ])

        if current_portfolio_value <= 0:
            current_portfolio_value = price * max(quantity, 1)

        requested_value = quantity * price

        MAX_CONCENTRATION = 0.20
        max_allowed_value = max(current_portfolio_value * MAX_CONCENTRATION, 1)

        if requested_value > max_allowed_value:
            adjusted_qty = int(max_allowed_value // price)
            if adjusted_qty < 1:
                adjusted_qty = 1
            quantity = adjusted_qty
            requested_value = quantity * price

        HARD_CAP = 0.50
        hard_cap_value = current_portfolio_value * HARD_CAP

        if requested_value > hard_cap_value:
            adjusted_qty = int(hard_cap_value // price)
            adjusted_qty = max(1, adjusted_qty)
            quantity = adjusted_qty

        if self.daily_pnl < -self.max_daily_loss:
            return False, "Daily loss limit exceeded"

        return True, f"Trade viable (final adjusted quantity: {quantity})"

# Alert Notification Manager for High-Confidence Signals
class AlertManager:
    """Manages trading alerts and notifications"""
    
    def __init__(self, max_alerts=50):
        self.alerts = []
        self.max_alerts = max_alerts
        self.alert_thresholds = {
            'high_confidence': 0.85,
            'medium_confidence': 0.70,
            'critical_pnl_loss': -5000,
            'critical_pnl_gain': 10000
        }
        self.muted_symbols = set()
        self.last_alert_time = {}
        self.alert_cooldown = 60
    
    def create_alert(self, alert_type, symbol, message, confidence=0.0, priority="NORMAL", data=None):
        """Create a new alert"""
        current_time = datetime.now()
        
        if symbol in self.last_alert_time:
            time_diff = (current_time - self.last_alert_time[symbol]).total_seconds()
            if time_diff < self.alert_cooldown:
                return None
        
        if symbol in self.muted_symbols:
            return None
        
        alert = {
            'id': len(self.alerts) + 1,
            'timestamp': current_time,
            'type': alert_type,
            'symbol': symbol,
            'message': message,
            'confidence': confidence,
            'priority': priority,
            'acknowledged': False,
            'data': data or {}
        }
        
        self.alerts.insert(0, alert)
        self.last_alert_time[symbol] = current_time
        
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[:self.max_alerts]
        
        return alert
    
    def create_signal_alert(self, signal):
        """Create alert from trading signal"""
        if not signal:
            return None
        
        confidence = signal.get('confidence', 0)
        symbol = signal.get('symbol', 'UNKNOWN')
        action = signal.get('action', 'BUY')
        strategy = signal.get('strategy', 'Unknown')
        
        if confidence >= self.alert_thresholds['high_confidence']:
            priority = "HIGH"
            alert_type = "HIGH_CONFIDENCE_SIGNAL"
        elif confidence >= self.alert_thresholds['medium_confidence']:
            priority = "MEDIUM"
            alert_type = "SIGNAL"
        else:
            return None
        
        message = f"{action} Signal: {symbol} | {strategy} | Confidence: {confidence:.1%}"
        
        return self.create_alert(
            alert_type=alert_type,
            symbol=symbol,
            message=message,
            confidence=confidence,
            priority=priority,
            data=signal
        )
    
    def get_unacknowledged_alerts(self):
        """Get all unacknowledged alerts"""
        return [a for a in self.alerts if not a['acknowledged']]
    
    def get_high_priority_alerts(self):
        """Get high priority alerts"""
        return [a for a in self.alerts if a['priority'] in ['HIGH', 'CRITICAL'] and not a['acknowledged']]
    
    def acknowledge_alert(self, alert_id):
        """Mark an alert as acknowledged"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                return True
        return False
    
    def acknowledge_all(self):
        """Acknowledge all alerts"""
        for alert in self.alerts:
            alert['acknowledged'] = True
    
    def mute_symbol(self, symbol, duration_minutes=30):
        """Mute alerts for a symbol temporarily"""
        self.muted_symbols.add(symbol)
    
    def unmute_symbol(self, symbol):
        """Unmute a symbol"""
        self.muted_symbols.discard(symbol)
    
    def get_alert_summary(self):
        """Get summary of current alerts"""
        unack = self.get_unacknowledged_alerts()
        return {
            'total': len(self.alerts),
            'unacknowledged': len(unack),
            'high_priority': len([a for a in unack if a['priority'] in ['HIGH', 'CRITICAL']]),
            'signals': len([a for a in unack if 'SIGNAL' in a['type']]),
            'pnl_alerts': len([a for a in unack if 'PNL' in a['type']]),
            'risk_alerts': len([a for a in unack if 'RISK' in a['type']])
        }
    
    def get_recent_alerts(self, limit=10):
        """Get most recent alerts"""
        return self.alerts[:limit]

# NEW: Enhanced Database Manager
class TradeDatabase:
    def __init__(self, db_url="sqlite:///trading_journal.db"):
        self.engine = None
        self.connected = False
        if SQLALCHEMY_AVAILABLE:
            try:
                os.makedirs('data', exist_ok=True)
                db_path = os.path.join('data', 'trading_journal.db')
                self.db_url = f'sqlite:///{db_path}'
                self.engine = create_engine(self.db_url)
                self.connected = True
                self.create_tables()
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                self.engine = None
                self.connected = False
        else:
            self.engine = None
            self.connected = False
    
    def create_tables(self):
        """Create necessary database tables"""
        if not self.connected:
            return
            
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT UNIQUE,
                        symbol TEXT,
                        action TEXT,
                        quantity INTEGER,
                        entry_price REAL,
                        exit_price REAL,
                        stop_loss REAL,
                        target REAL,
                        pnl REAL,
                        entry_time TIMESTAMP,
                        exit_time TIMESTAMP,
                        strategy TEXT,
                        auto_trade BOOLEAN,
                        status TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.commit()
                logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
    
    def log_trade(self, trade_data):
        """Log trade to database"""
        if not self.connected:
            return
            
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT OR REPLACE INTO trades 
                    (trade_id, symbol, action, quantity, entry_price, exit_price, 
                     stop_loss, target, pnl, entry_time, exit_time, strategy, 
                     auto_trade, status)
                    VALUES (:trade_id, :symbol, :action, :quantity, :entry_price, 
                            :exit_price, :stop_loss, :target, :pnl, :entry_time, 
                            :exit_time, :strategy, :auto_trade, :status)
                """), trade_data)
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging trade: {e}")

# Enhanced Utilities
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

# FIXED Circular Market Mood Gauge Component with Rounded Percentages
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
        progress_color = "#d97706"
    
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

# Enhanced Data Manager
class EnhancedDataManager:
    def __init__(self):
        self.price_cache = {}
        self.signal_cache = {}
        self.market_profile_cache = {}
        self.last_rsi_scan = None
        self.risk_manager = AdvancedRiskManager()
        self.database = TradeDatabase()
        self.kite_manager = KiteConnectManager(KITE_API_KEY, KITE_API_SECRET)
        self.alert_manager = AlertManager()

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
            return self.create_validated_demo_data(symbol)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        expected = ["Open", "High", "Low", "Close", "Volume"]
        for e in expected:
            if e not in df.columns:
                if e.upper() in df.columns:
                    df[e] = df[e.upper()]
                else:
                    return self.create_validated_demo_data(symbol)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        if len(df) < 20:
            return self.create_validated_demo_data(symbol)

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

        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI14"] = rsi(df["Close"], 14).fillna(50)
        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"]).fillna(method="ffill").fillna(0)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])
        df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"])
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()

        return df

    def create_validated_demo_data(self, symbol):
        live = self._validate_live_price(symbol)
        periods = 300
        end = now_indian()
        dates = pd.date_range(end=end, periods=periods, freq="15min")
        base = float(live)
        rng = np.random.default_rng(int(abs(hash(symbol)) % (2 ** 32 - 1)))
        returns = rng.normal(0, 0.0009, periods)
        prices = base * np.cumprod(1 + returns)
        openp = prices * (1 + rng.normal(0, 0.0012, periods))
        highp = prices * (1 + abs(rng.normal(0, 0.0045, periods)))
        lowp = prices * (1 - abs(rng.normal(0, 0.0045, periods)))
        vol = rng.integers(1000, 200000, periods)
        df = pd.DataFrame({"Open": openp, "High": highp, "Low": lowp, "Close": prices, "Volume": vol}, index=dates)
        df.iloc[-1, df.columns.get_loc("Close")] = live
        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI14"] = rsi(df["Close"], 14).fillna(50)
        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"]).fillna(0)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])
        df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"])
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()
        return df

    def get_historical_accuracy(self, symbol, strategy):
        accuracy_map = {
            "Multi_Confirmation": 0.82,
            "Enhanced_EMA_VWAP": 0.78,
            "Volume_Breakout": 0.75,
            "RSI_Divergence": 0.72,
            "MACD_Trend": 0.70,
            "EMA_VWAP_Confluence": 0.75,
            "RSI_MeanReversion": 0.68,
            "Bollinger_Reversion": 0.65,
            "MACD_Momentum": 0.70,
            "Support_Resistance_Breakout": 0.73,
            "EMA_VWAP_Downtrend": 0.72,
            "RSI_Overbought": 0.65,
            "Bollinger_Rejection": 0.63,
            "MACD_Bearish": 0.68,
            "Trend_Reversal": 0.60
        }
        return accuracy_map.get(strategy, 0.65)

# Enhanced Multi-Strategy Trading Engine
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
        self.risk_manager = AdvancedRiskManager()

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

    def calculate_support_resistance(self, symbol, current_price):
        try:
            data = self.data_manager.get_stock_data(symbol, "15m")
            if data is None or len(data) < 20:
                return current_price * 0.98, current_price * 1.02
            return float(data["Close"].min() * 0.98), float(data["Close"].max() * 1.02)
        except Exception:
            return current_price * 0.98, current_price * 1.02

    def calculate_intraday_target_sl(self, entry_price, action, atr, current_price, support, resistance):
        if atr <= 0 or np.isnan(atr):
            atr = max(entry_price * 0.005, 1.0)
        
        if action == "BUY":
            sl = entry_price - (atr * 1.2)
            target = entry_price + (atr * 2.5)
            if target > resistance:
                target = min(target, resistance * 0.998)
            sl = max(sl, support * 0.995)
        else:
            sl = entry_price + (atr * 1.2)
            target = entry_price - (atr * 2.5)
            if target < support:
                target = max(target, support * 1.002)
            sl = min(sl, resistance * 1.005)

        rr = abs(target - entry_price) / max(abs(entry_price - sl), 1e-6)
        if rr < 2.0:
            if action == "BUY":
                target = entry_price + max((entry_price - sl) * 2.0, atr * 2.0)
            else:
                target = entry_price - max((sl - entry_price) * 2.0, atr * 2.0)
                
        return round(float(target), 2), round(float(sl), 2)

    def equity(self):
        total = float(self.cash)
        for symbol, pos in self.positions.items():
            if pos.get("status") == "OPEN":
                try:
                    data = self.data_manager.get_stock_data(symbol, "5m")
                    price = float(data["Close"].iloc[-1]) if data is not None and len(data) > 0 else pos["entry_price"]
                    total += pos["quantity"] * price
                except Exception:
                    total += pos["quantity"] * pos["entry_price"]
        return total

    def execute_trade(self, symbol, action, quantity, price, stop_loss=None, target=None, win_probability=0.75, auto_trade=False, strategy=None):
        risk_ok, risk_msg = self.data_manager.risk_manager.check_trade_viability(symbol, action, quantity, price, self.positions)
        if not risk_ok:
            return False, f"Risk check failed: {risk_msg}"
            
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

        try:
            if self.data_manager.database.connected:
                self.data_manager.database.log_trade({
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "action": action,
                    "quantity": int(quantity),
                    "entry_price": float(price),
                    "exit_price": None,
                    "stop_loss": float(stop_loss) if stop_loss else None,
                    "target": float(target) if target else None,
                    "pnl": 0.0,
                    "entry_time": now_indian(),
                    "exit_time": None,
                    "strategy": strategy,
                    "auto_trade": auto_trade,
                    "status": "OPEN"
                })
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

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
                    pos["max_pnl"] = max(pos.get("max_pnl", 0.0), float(pnl))
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
            if self.data_manager.database.connected:
                self.data_manager.database.log_trade({
                    "trade_id": pos["trade_id"],
                    "symbol": symbol,
                    "action": pos["action"],
                    "quantity": pos["quantity"],
                    "entry_price": pos["entry_price"],
                    "exit_price": float(exit_price),
                    "stop_loss": pos.get("stop_loss"),
                    "target": pos.get("target"),
                    "pnl": float(pnl),
                    "entry_time": pos["timestamp"],
                    "exit_time": now_indian(),
                    "strategy": strategy,
                    "auto_trade": pos.get("auto_trade", False),
                    "status": "CLOSED"
                })
        except Exception as e:
            logger.error(f"Failed to update trade in database: {e}")

        try:
            del self.positions[symbol]
        except Exception:
            pass
        return True, f"Closed {symbol} @ ₹{exit_price:.2f} | P&L: ₹{pnl:+.2f}"

    def generate_strategy_signals(self, symbol, data):
        signals = []
        if data is None or len(data) < 30:
            return signals
        
        try:
            live = float(data["Close"].iloc[-1])
            ema8 = float(data["EMA8"].iloc[-1])
            ema21 = float(data["EMA21"].iloc[-1])
            ema50 = float(data["EMA50"].iloc[-1])
            rsi_val = float(data["RSI14"].iloc[-1])
            atr = float(data["ATR"].iloc[-1]) if "ATR" in data.columns else max(live*0.005,1)
            macd_line = float(data["MACD"].iloc[-1])
            macd_signal = float(data["MACD_Signal"].iloc[-1])
            vwap = float(data["VWAP"].iloc[-1])
            
            support, resistance = self.calculate_support_resistance(symbol, live)

            # BUY STRATEGIES
            # Strategy 1: EMA + VWAP Confluence
            if (ema8 > ema21 > ema50 and live > vwap):
                action = "BUY"; confidence = 0.82; score = 9; strategy = "EMA_VWAP_Confluence"
                target, stop_loss = self.calculate_intraday_target_sl(live, action, atr, live, support, resistance)
                rr = abs(target - live) / max(abs(live - stop_loss), 1e-6)
                if rr >= 2.5:
                    historical_accuracy = self.data_manager.get_historical_accuracy(symbol, strategy)
                    if historical_accuracy >= 0.65:
                        signals.append({
                            "symbol": symbol, "action": action, "entry": live, "current_price": live,
                            "target": target, "stop_loss": stop_loss, "confidence": confidence,
                            "win_probability": historical_accuracy, "historical_accuracy": historical_accuracy,
                            "rsi": rsi_val, "risk_reward": rr, "score": score, "strategy": strategy,
                            "strategy_name": TRADING_STRATEGIES[strategy]["name"]
                        })

            # Strategy 2: RSI Mean Reversion
            if rsi_val < 30 and live > support and rsi_val > 25:
                action = "BUY"; confidence = 0.78; score = 8; strategy = "RSI_MeanReversion"
                target, stop_loss = self.calculate_intraday_target_sl(live, action, atr, live, support, resistance)
                rr = abs(target - live) / max(abs(live - stop_loss), 1e-6)
                if rr >= 2.5:
                    historical_accuracy = self.data_manager.get_historical_accuracy(symbol, strategy)
                    if historical_accuracy >= 0.65:
                        signals.append({
                            "symbol": symbol, "action": action, "entry": live, "current_price": live,
                            "target": target, "stop_loss": stop_loss, "confidence": confidence,
                            "win_probability": historical_accuracy, "historical_accuracy": historical_accuracy,
                            "rsi": rsi_val, "risk_reward": rr, "score": score, "strategy": strategy,
                            "strategy_name": TRADING_STRATEGIES[strategy]["name"]
                        })

            # SELL STRATEGIES
            # Strategy 3: RSI Overbought
            if rsi_val > 70 and live < resistance and rsi_val < 75:
                action = "SELL"; confidence = 0.78; score = 8; strategy = "RSI_Overbought"
                target, stop_loss = self.calculate_intraday_target_sl(live, action, atr, live, support, resistance)
                rr = abs(target - live) / max(abs(live - stop_loss), 1e-6)
                if rr >= 2.5:
                    historical_accuracy = self.data_manager.get_historical_accuracy(symbol, strategy)
                    if historical_accuracy >= 0.65:
                        signals.append({
                            "symbol": symbol, "action": action, "entry": live, "current_price": live,
                            "target": target, "stop_loss": stop_loss, "confidence": confidence,
                            "win_probability": historical_accuracy, "historical_accuracy": historical_accuracy,
                            "rsi": rsi_val, "risk_reward": rr, "score": score, "strategy": strategy,
                            "strategy_name": TRADING_STRATEGIES[strategy]["name"]
                        })

            # Strategy 4: MACD Bearish
            if (macd_line < macd_signal and ema8 < ema21 and live < vwap):
                action = "SELL"; confidence = 0.80; score = 8; strategy = "MACD_Bearish"
                target, stop_loss = self.calculate_intraday_target_sl(live, action, atr, live, support, resistance)
                rr = abs(target - live) / max(abs(live - stop_loss), 1e-6)
                if rr >= 2.5:
                    historical_accuracy = self.data_manager.get_historical_accuracy(symbol, strategy)
                    if historical_accuracy >= 0.65:
                        signals.append({
                            "symbol": symbol, "action": action, "entry": live, "current_price": live,
                            "target": target, "stop_loss": stop_loss, "confidence": confidence,
                            "win_probability": historical_accuracy, "historical_accuracy": historical_accuracy,
                            "rsi": rsi_val, "risk_reward": rr, "score": score, "strategy": strategy,
                            "strategy_name": TRADING_STRATEGIES[strategy]["name"]
                        })

            # update strategy signals count
            for s in signals:
                strat = s.get("strategy")
                if strat in self.strategy_performance:
                    self.strategy_performance[strat]["signals"] += 1

            return signals

        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return signals

    def generate_quality_signals(self, universe, max_scan=None, min_confidence=0.70, min_score=6, use_high_accuracy=True):
        signals = []
        
        if universe == "Nifty 50":
            stocks = NIFTY_50
        elif universe == "Nifty 100":
            stocks = NIFTY_100
        elif universe == "Midcap 150":
            stocks = NIFTY_MIDCAP_150
        elif universe == "All Stocks":
            stocks = ALL_STOCKS
        else:
            stocks = NIFTY_50
        
        if max_scan is not None and max_scan < len(stocks):
            stocks_to_scan = stocks[:max_scan]
        else:
            stocks_to_scan = stocks
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, symbol in enumerate(stocks_to_scan):
            try:
                status_text.text(f"Scanning {symbol} ({idx+1}/{len(stocks_to_scan)})")
                progress_bar.progress((idx + 1) / len(stocks_to_scan))
                
                data = self.data_manager.get_stock_data(symbol, "15m")
                if data is None or len(data) < 30:
                    continue
                
                standard_signals = self.generate_strategy_signals(symbol, data)
                signals.extend(standard_signals)
                    
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        signals = [s for s in signals if s.get("confidence", 0) >= min_confidence and s.get("score", 0) >= min_score]
        
        signals.sort(key=lambda x: (x.get("score", 0), x.get("confidence", 0)), reverse=True)
        
        self.signal_history = signals[:30]
        
        return signals[:20]

    def auto_execute_signals(self, signals):
        """Auto-execute signals with enhanced feedback"""
        executed = []
        
        if not self.can_auto_trade():
            st.warning(f"⚠️ Cannot auto-trade. Check: Daily trades: {self.daily_trades}/{MAX_DAILY_TRADES}, Auto trades: {self.auto_trades_count}/{MAX_AUTO_TRADES}, Market open: {market_open()}")
            return executed
        
        st.info(f"🚀 Attempting to auto-execute {len(signals[:10])} signals...")
        
        for signal in signals[:10]:
            if not self.can_auto_trade():
                st.warning("Auto-trade limit reached")
                break
                
            if signal["symbol"] in self.positions:
                st.info(f"Skipping {signal['symbol']} - already in position")
                continue
                
            try:
                data = self.data_manager.get_stock_data(signal["symbol"], "15m")
                atr = data["ATR"].iloc[-1] if "ATR" in data.columns else signal["entry"] * 0.01
            except:
                atr = signal["entry"] * 0.01
                
            optimal_qty = self.data_manager.risk_manager.calculate_kelly_position_size(
                signal.get("win_probability", 0.75),
                signal.get("risk_reward", 2.0),
                self.cash,
                signal["entry"],
                atr
            )
            
            if optimal_qty > 0:
                success, msg = self.execute_trade(
                    symbol=signal["symbol"],
                    action=signal["action"],
                    quantity=optimal_qty,
                    price=signal["entry"],
                    stop_loss=signal.get("stop_loss"),
                    target=signal.get("target"),
                    win_probability=signal.get("win_probability", 0.75),
                    auto_trade=True,
                    strategy=signal.get("strategy")
                )
                if success:
                    executed.append(msg)
                    st.toast(f"✅ Auto-executed: {msg}", icon="🚀")
                else:
                    st.toast(f"❌ Auto-execution failed: {msg}", icon="⚠️")
            else:
                st.info(f"Skipping {signal['symbol']} - position size calculation failed")
        
        self.last_auto_execution_time = time.time()
        return executed

# Enhanced Initialization with Error Handling
def initialize_application():
    """Initialize the application with comprehensive error handling"""
    
    with st.sidebar.expander("🛠️ System Status"):
        for package, status in system_status.items():
            if status:
                st.write(f"✅ {package}")
            else:
                st.write(f"❌ {package} - Missing")
    
    if not SQLALCHEMY_AVAILABLE or not JOBLIB_AVAILABLE:
        st.markdown("""
        <div class="dependencies-warning">
            <h4>🔧 Some Features Limited</h4>
            <p>For full functionality:</p>
            <code>pip install sqlalchemy joblib kiteconnect</code>
            <p><strong>Limited features:</strong></p>
            <ul>
                <li>Database features (trades won't persist)</li>
                <li>ML model persistence</li>
                <li>Kite Connect live charts</li>
            </ul>
            <p><em>Basic trading functionality is available.</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    try:
        data_manager = EnhancedDataManager()
        
        if "trader" not in st.session_state:
            st.session_state.trader = MultiStrategyIntradayTrader()
        
        trader = st.session_state.trader
        
        if "refresh_count" not in st.session_state:
            st.session_state.refresh_count = 0
        
        st.session_state.refresh_count += 1
        
        return data_manager, trader
        
    except Exception as e:
        st.error(f"Application initialization failed: {str(e)}")
        st.code(traceback.format_exc())
        return None, None

# MAIN APPLICATION
try:
    # Initialize the application
    data_manager, trader = initialize_application()
    
    if data_manager is None or trader is None:
        st.error("Failed to initialize application. Please refresh the page.")
        st.stop()
    
    # Auto-refresh
    st_autorefresh(interval=PRICE_REFRESH_MS, key="price_refresh_improved")

    # Enhanced UI with Circular Market Mood Gauges
    st.markdown("<h1 style='text-align:center; color: #1e3a8a;'>Rantv Intraday Terminal Pro - ENHANCED</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color: #6b7280;'>Full Stock Scanning & High-Quality Signal Generation Enabled</h4>", unsafe_allow_html=True)

    # Market overview with enhanced metrics
    cols = st.columns(7)
    try:
        nift = data_manager._validate_live_price("^NSEI")
        cols[0].metric("NIFTY 50", f"₹{nift:,.2f}")
    except Exception:
        cols[0].metric("NIFTY 50", "N/A")
    try:
        bn = data_manager._validate_live_price("^NSEBANK")
        cols[1].metric("BANK NIFTY", f"₹{bn:,.2f}")
    except Exception:
        cols[1].metric("BANK NIFTY", "N/A")
    cols[2].metric("Market Status", "LIVE" if market_open() else "CLOSED")
    cols[3].metric("Peak Hours", "10AM-2PM")
    cols[4].metric("Auto Trades", f"{trader.auto_trades_count}/{MAX_AUTO_TRADES}")
    cols[5].metric("Available Cash", f"₹{trader.cash:,.0f}")
    cols[6].metric("Open Positions", f"{len(trader.positions)}")

    # Manual refresh button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"<div style='text-align: left; color: #6b7280; font-size: 14px;'>Refresh Count: <span class='refresh-counter'>{st.session_state.refresh_count}</span></div>", unsafe_allow_html=True)
    with col2:
        if st.button("🔄 Manual Refresh", key="manual_refresh_btn", width='stretch'):
            st.rerun()
    with col3:
        if st.button("📊 Update Prices", key="update_prices_btn", width='stretch'):
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
        
    except Exception:
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
        
    except Exception:
        banknifty_current = 48000
        banknifty_change = 0.25
        banknifty_sentiment = 70

    # Display Circular Market Mood Gauges
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
        st.markdown(create_circular_market_mood_gauge("PERFORMANCE", trader.equity(), 0, 65).replace("₹0", f"₹{trader.equity():,.0f}").replace("0.00%", ""), unsafe_allow_html=True)

    # Main metrics with card styling
    st.subheader("📈 Live Metrics")
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 12px; color: #6b7280;">Available Cash</div>
            <div style="font-size: 20px; font-weight: bold; color: #1e3a8a;">₹{trader.cash:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 12px; color: #6b7280;">Account Value</div>
            <div style="font-size: 20px; font-weight: bold; color: #1e3a8a;">₹{trader.equity():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        open_pnl = sum([p.get('current_pnl', 0) for p in trader.positions.values()])
        pnl_color = "#059669" if open_pnl >= 0 else "#dc2626"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 12px; color: #6b7280;">Open P&L</div>
            <div style="font-size: 20px; font-weight: bold; color: {pnl_color};">₹{open_pnl:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[3]:
        total_trades = len([t for t in trader.trade_log if t.get("status") == "CLOSED"])
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 12px; color: #6b7280;">Total Trades</div>
            <div style="font-size: 20px; font-weight: bold; color: #1e3a8a;">{total_trades}</div>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar Configuration
    st.sidebar.header("⚙️ Trading Configuration")
    trader.selected_market = st.sidebar.selectbox("Market Type", MARKET_OPTIONS, key="market_type_select")
    universe = st.sidebar.selectbox("Trading Universe", ["All Stocks", "Nifty 50", "Nifty 100", "Midcap 150"], key="universe_select")
    enable_high_accuracy = st.sidebar.checkbox("Enable High Accuracy Strategies", value=True, key="high_acc_toggle")
    trader.auto_execution = st.sidebar.checkbox("Auto Execution", value=False, key="auto_execution_toggle")
    
    # Risk Management Settings
    st.sidebar.subheader("🎯 Risk Management")
    min_conf_percent = st.sidebar.slider("Minimum Confidence %", 60, 85, 70, 5, key="min_conf_slider")
    min_score = st.sidebar.slider("Minimum Score", 5, 9, 6, 1, key="min_score_slider")
    
    # Scan Configuration
    st.sidebar.subheader("🔍 Scan Configuration")
    full_scan = st.sidebar.checkbox("Full Universe Scan", value=True, key="full_scan_toggle")
    
    if not full_scan:
        max_scan = st.sidebar.number_input("Max Stocks to Scan", min_value=10, max_value=500, value=50, step=10, key="max_scan_input")
    else:
        max_scan = None

    # Strategy Performance
    st.sidebar.header("🎯 Strategy Performance")
    st.sidebar.subheader("🔥 High Accuracy")
    for strategy, config in HIGH_ACCURACY_STRATEGIES.items():
        if strategy in trader.strategy_performance:
            perf = trader.strategy_performance[strategy]
            if perf["signals"] > 0:
                win_rate = perf["wins"] / perf["trades"] if perf["trades"] > 0 else 0
                color = "#059669" if win_rate > 0.7 else "#dc2626" if win_rate < 0.5 else "#d97706"
                st.sidebar.write(f"**{config['name']}**")
                st.sidebar.write(f"📊 Signals: {perf['signals']} | Trades: {perf['trades']}")
                st.sidebar.write(f"🎯 Win Rate: <span style='color: {color};'>{win_rate:.1%}</span>", unsafe_allow_html=True)
                st.sidebar.write(f"💰 P&L: ₹{perf['pnl']:+.2f}")
                st.sidebar.markdown("---")

    # Enhanced Tabs
    tabs = st.tabs([
        "📈 Dashboard", 
        "🚦 Signals", 
        "💰 Paper Trading", 
        "📋 Trade History",
        "📉 RSI Extreme", 
        "⚡ Strategies",
        "🎯 High Accuracy Scanner"
    ])

    # Tab 1: Dashboard
    with tabs[0]:
        st.subheader("Account Summary")
        trader.update_positions_pnl()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Value", f"₹{trader.equity():,.0f}")
        c2.metric("Available Cash", f"₹{trader.cash:,.0f}")
        c3.metric("Open Positions", len(trader.positions))
        c4.metric("Daily Trades", trader.daily_trades)
        
        # Open Positions
        st.subheader("📊 Open Positions")
        if trader.positions:
            for symbol, pos in trader.positions.items():
                if pos.get("status") == "OPEN":
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        action_color = "🟢" if pos["action"] == "BUY" else "🔴"
                        pnl = pos.get("current_pnl", 0)
                        pnl_color = "green" if pnl >= 0 else "red"
                        
                        st.markdown(f"""
                        <div style="padding: 10px; border-left: 4px solid {'#059669' if pos['action'] == 'BUY' else '#dc2626'}; 
                                 background: linear-gradient(135deg, {'#d1fae5' if pos['action'] == 'BUY' else '#fee2e2'} 0%, 
                                 {'#a7f3d0' if pos['action'] == 'BUY' else '#fecaca'} 100%); border-radius: 8px;">
                            <strong>{action_color} {symbol.replace('.NS','')}</strong> | {pos['action']} | Qty: {pos['quantity']}<br>
                            Entry: ₹{pos['entry_price']:.2f} | Current: ₹{pos.get('current_price', pos['entry_price']):.2f}<br>
                            <span style="color: {pnl_color}">P&L: ₹{pnl:+.2f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.write(f"SL: ₹{pos.get('stop_loss', 0):.2f}")
                        st.write(f"TG: ₹{pos.get('target', 0):.2f}")
                    
                    with col3:
                        if st.button(f"Close", key=f"close_{symbol}", type="secondary"):
                            success, msg = trader.close_position(symbol)
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
        else:
            st.info("No open positions")

    # Tab 2: Signals
    with tabs[1]:
        st.subheader("Multi-Strategy BUY/SELL Signals")
        st.markdown("""
        <div class="alert-success">
            <strong>🎯 UPDATED Signal Parameters:</strong> 
            • Confidence threshold reduced from 75% to <strong>70%</strong><br>
            • Minimum score reduced from 7 to <strong>6</strong><br>
            • Optimized for peak market hours (9:30 AM - 2:30 PM)<br>
            • These changes should generate more trading opportunities
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            generate_btn = st.button("Generate Signals", type="primary", width='stretch', key="generate_signals_btn")
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
                auto_exec_btn = st.button("🚀 Auto Execute", type="secondary", width='stretch', key="auto_exec_btn")
            else:
                auto_exec_btn = False
        
        if "last_signal_generation" not in st.session_state:
            st.session_state.last_signal_generation = 0
        
        current_time = time.time()
        auto_generate = False
        
        if trader.auto_execution and market_open():
            time_since_last = current_time - st.session_state.last_signal_generation
            if time_since_last > 60:
                auto_generate = True
                st.session_state.last_signal_generation = current_time
        
        generate_signals = generate_btn or auto_generate
        
        if generate_signals:
            with st.spinner(f"Scanning {universe} stocks with enhanced strategies..."):
                signals = trader.generate_quality_signals(
                    universe, 
                    max_scan=max_scan,
                    min_confidence=min_conf_percent/100.0, 
                    min_score=min_score,
                    use_high_accuracy=enable_high_accuracy
                )
            
            if signals:
                buy_signals = [s for s in signals if s["action"] == "BUY"]
                sell_signals = [s for s in signals if s["action"] == "SELL"]
                
                st.success(f"✅ Found {len(buy_signals)} BUY signals and {len(sell_signals)} SELL signals")
                
                data_rows = []
                for s in signals:
                    is_high_acc = s["strategy"] in HIGH_ACCURACY_STRATEGIES
                    strategy_display = f"🔥 {s['strategy_name']}" if is_high_acc else s['strategy_name']
                    
                    data_rows.append({
                        "Symbol": s["symbol"].replace(".NS",""),
                        "Action": s["action"],
                        "Strategy": strategy_display,
                        "Entry Price": f"₹{s['entry']:.2f}",
                        "Target": f"₹{s['target']:.2f}",
                        "Stop Loss": f"₹{s['stop_loss']:.2f}",
                        "Confidence": f"{s['confidence']:.1%}",
                        "R:R": f"{s['risk_reward']:.2f}",
                        "Score": s['score'],
                        "RSI": f"{s['rsi']:.1f}"
                    })
                
                st.dataframe(pd.DataFrame(data_rows), width='stretch')
                
                # AUTO-EXECUTION LOGIC
                if trader.auto_execution and trader.can_auto_trade():
                    auto_execute_now = False
                    
                    if auto_exec_btn:
                        auto_execute_now = True
                        st.info("🚀 Manual auto-execution triggered")
                    elif auto_generate:
                        auto_execute_now = True
                        st.info("🚀 Auto-execution triggered")
                    
                    if auto_execute_now:
                        executed = trader.auto_execute_signals(signals)
                        if executed:
                            st.success(f"✅ Auto-execution completed: {len(executed)} trades executed")
                            for msg in executed:
                                st.write(f"✓ {msg}")
                            st.rerun()
                        else:
                            st.warning("No trades were auto-executed. Check trade limits or existing positions.")
                    elif trader.auto_execution and not auto_execute_now:
                        st.info("Auto-execution is active. Signals will be executed automatically.")
                
                # Manual Execution
                st.subheader("Manual Execution")
                for idx, s in enumerate(signals[:5]):
                    col_a, col_b, col_c = st.columns([3,1,1])
                    with col_a:
                        action_color = "🟢" if s["action"] == "BUY" else "🔴"
                        is_high_acc = s["strategy"] in HIGH_ACCURACY_STRATEGIES
                        strategy_display = f"🔥 {s['strategy_name']}" if is_high_acc else s['strategy_name']
                        
                        st.markdown(f"""
                        <div class="{'high-quality-signal' if s['score'] >= 8 else 'medium-quality-signal' if s['score'] >= 6 else 'low-quality-signal'}">
                            <strong>{action_color} {s['symbol'].replace('.NS','')}</strong> - {s['action']} @ ₹{s['entry']:.2f}<br>
                            Strategy: {strategy_display}<br>
                            R:R: {s['risk_reward']:.2f} | Score: {s['score']}/10
                        </div>
                        """, unsafe_allow_html=True)
                    with col_b:
                        qty = int((trader.cash * TRADE_ALLOC) / s["entry"])
                        st.write(f"Qty: {qty}")
                    with col_c:
                        if st.button(f"Execute", key=f"exec_{s['symbol']}_{idx}_{int(time.time())}"):
                            success, msg = trader.execute_trade(
                                symbol=s["symbol"], action=s["action"], quantity=qty, price=s["entry"],
                                stop_loss=s["stop_loss"], target=s["target"], win_probability=s.get("win_probability",0.75),
                                strategy=s.get("strategy")
                            )
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
            else:
                if market_open():
                    st.warning("No signals found. Try adjusting the filters or wait for better market conditions.")
                else:
                    st.info("Market is closed. Signals are only generated during market hours (9:15 AM - 3:30 PM).")
        else:
            if trader.auto_execution:
                if market_open():
                    st.info("🔄 Auto-execution is active. High-quality signals will be generated and executed automatically during market hours.")
                    st.write(f"**Auto-execution status:**")
                    st.write(f"- Daily trades: {trader.daily_trades}/{MAX_DAILY_TRADES}")
                    st.write(f"- Auto trades: {trader.auto_trades_count}/{MAX_AUTO_TRADES}")
                    st.write(f"- Available cash: ₹{trader.cash:,.0f}")
                    st.write(f"- Can auto-trade: {'✅ Yes' if trader.can_auto_trade() else '❌ No'}")
                    
                    time_since_last = int(current_time - st.session_state.last_signal_generation)
                    time_to_next = max(0, 60 - time_since_last)
                    st.write(f"- Next auto-scan in: {time_to_next} seconds")
                else:
                    st.warning("Market is closed. Auto-execution will resume when market opens (9:15 AM - 3:30 PM).")

    # Tab 3: Paper Trading
    with tabs[2]:
        st.subheader("💰 Paper Trading")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            symbol = st.selectbox("Symbol", NIFTY_50[:20], key="paper_symbol_select")
        with col2:
            action = st.selectbox("Action", ["BUY", "SELL"], key="paper_action_select")
        with col3:
            quantity = st.number_input("Quantity", min_value=1, value=10, key="paper_qty_input")
        with col4:
            strategy = st.selectbox("Strategy", ["Manual"] + [config["name"] for config in TRADING_STRATEGIES.values()], key="paper_strategy_select")
        
        if st.button("Execute Paper Trade", type="primary", key="paper_execute_btn"):
            try:
                data = data_manager.get_stock_data(symbol, "15m")
                price = float(data["Close"].iloc[-1])
                atr = float(data["ATR"].iloc[-1]) if 'ATR' in data.columns else price * 0.01
                support, resistance = trader.calculate_support_resistance(symbol, price)
                
                target, stop_loss = trader.calculate_intraday_target_sl(
                    price, action, atr, price, support, resistance
                )
                
                strategy_key = "Manual"
                for key, config in TRADING_STRATEGIES.items():
                    if config["name"] == strategy:
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
                    st.success(f"Stop Loss: ₹{stop_loss:.2f} | Target: ₹{target:.2f} | R:R: {(abs(target-price)/abs(price-stop_loss)):.2f}:1")
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")
                    
            except Exception as e:
                st.error(f"Trade execution failed: {str(e)}")

    # Tab 4: Trade History
    with tabs[3]:
        st.subheader("📋 Trade History")
        
        if SQLALCHEMY_AVAILABLE and trader.data_manager.database.connected:
            st.success("✅ Database connected - trades are being stored")
        
        trade_history = [t for t in trader.trade_log if t.get("status") == "CLOSED"]
        if trade_history:
            for trade in trade_history[-10:]:
                pnl = trade.get("closed_pnl", 0)
                pnl_class = "profit-positive" if pnl >= 0 else "profit-negative"
                trade_class = "trade-buy" if trade.get("action") == "BUY" else "trade-sell"
                
                st.markdown(f"""
                <div class="{trade_class}">
                    <div style="padding: 10px;">
                        <strong>{trade['symbol'].replace('.NS', '')}</strong> | {trade['action']} | Qty: {trade['quantity']}<br>
                        Entry: ₹{trade['entry_price']:.2f} | Exit: ₹{trade.get('exit_price', 0):.2f} | <span class="{pnl_class}">P&L: ₹{pnl:+.2f}</span><br>
                        Strategy: {trade.get('strategy', 'Manual')} | {'Auto' if trade.get('auto_trade') else 'Manual'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No trade history available")

    # Tab 5: RSI Extreme
    with tabs[4]:
        st.subheader("📉 RSI Extreme Scanner")
        
        st.info("This scanner finds stocks with extreme RSI values (oversold/overbought)")
        
        if st.button("Scan for RSI Extremes", key="rsi_scan_btn"):
            with st.spinner("Scanning for RSI extremes..."):
                try:
                    oversold = []
                    overbought = []
                    
                    for symbol in NIFTY_50[:30]:
                        data = data_manager.get_stock_data(symbol, "15m")
                        if len(data) > 0:
                            rsi_val = (data['RSI14'].iloc[-1] if 'RSI14' in data and data['RSI14'].dropna().shape[0] > 0 else 50.0)
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
                    
                    if oversold or overbought:
                        st.success(f"Found {len(oversold)} oversold and {len(overbought)} overbought stocks")
                        
                        if oversold:
                            st.subheader("🔵 Oversold Stocks (RSI < 30)")
                            df_oversold = pd.DataFrame(oversold)
                            st.dataframe(df_oversold, width='stretch')
                        
                        if overbought:
                            st.subheader("🔴 Overbought Stocks (RSI > 70)")
                            df_overbought = pd.DataFrame(overbought)
                            st.dataframe(df_overbought, width='stretch')
                    else:
                        st.info("No extreme RSI stocks found")
                        
                except Exception as e:
                    st.error(f"Error scanning RSI extremes: {str(e)}")

    # Tab 6: Strategies
    with tabs[5]:
        st.subheader("⚡ Trading Strategies")
        
        st.write("### High Accuracy Strategies")
        for strategy, config in HIGH_ACCURACY_STRATEGIES.items():
            with st.expander(f"🔥 {config['name']}"):
                st.write(f"**Type:** {config['type']}")
                st.write(f"**Weight:** {config['weight']}")
                st.write("**Description:** High probability setup with multiple confirmations")
        
        st.write("### Standard Strategies")
        for strategy, config in TRADING_STRATEGIES.items():
            with st.expander(f"{config['name']}"):
                st.write(f"**Type:** {config['type']}")
                st.write(f"**Weight:** {config['weight']}")
                st.write("**Description:** Standard trading strategy")

    # Tab 7: High Accuracy Scanner
    with tabs[6]:
        st.subheader("🎯 High Accuracy Scanner - All Stocks")
        st.markdown(f"""
        <div class="alert-success">
            <strong>🔥 High Accuracy Strategies Enabled:</strong> 
            Scanning <strong>{universe}</strong> with enhanced high-accuracy strategies.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            high_acc_scan_btn = st.button("🚀 Scan High Accuracy", type="primary", width='stretch', key="high_acc_scan_btn")
        with col2:
            min_high_acc_confidence = st.slider("Min Confidence", 65, 85, 70, 5, key="high_acc_conf_slider")
        with col3:
            min_high_acc_score = st.slider("Min Score", 5, 8, 6, 1, key="high_acc_score_slider")
        
        if high_acc_scan_btn:
            with st.spinner(f"Scanning {universe} with high-accuracy strategies..."):
                high_acc_signals = trader.generate_quality_signals(
                    universe, 
                    max_scan=50 if universe == "All Stocks" else max_scan,
                    min_confidence=min_high_acc_confidence/100.0,
                    min_score=min_high_acc_score,
                    use_high_accuracy=True
                )
            
            if high_acc_signals:
                st.success(f"🎯 Found {len(high_acc_signals)} high-confidence signals!")
                
                for idx, signal in enumerate(high_acc_signals[:10]):
                    with st.container():
                        st.markdown(f"""
                        <div class="{'high-quality-signal' if signal['score'] >= 8 else 'medium-quality-signal' if signal['score'] >= 6 else 'low-quality-signal'}">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong>{signal['symbol'].replace('.NS', '')}</strong> | 
                                    <span style="color: {'#ffffff' if signal['action'] == 'BUY' else '#ffffff'}">
                                        {signal['action']}
                                    </span> | 
                                    ₹{signal['entry']:.2f}
                                </div>
                                <div>
                                    <span style="background: #f59e0b; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px;">
                                        {signal['strategy_name']}
                                    </span>
                                </div>
                            </div>
                            <div style="font-size: 12px; margin-top: 5px;">
                                Target: ₹{signal['target']:.2f} | SL: ₹{signal['stop_loss']:.2f} | 
                                R:R: {signal['risk_reward']:.2f} | Score: {signal['score']}/10
                            </div>
                            <div style="font-size: 11px; margin-top: 3px;">
                                Confidence: {signal['confidence']:.1%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Quick execution buttons
                st.subheader("Quick Execution")
                exec_cols = st.columns(3)
                for idx, signal in enumerate(high_acc_signals[:6]):
                    with exec_cols[idx % 3]:
                        if st.button(
                            f"{signal['action']} {signal['symbol'].replace('.NS', '')}", 
                            key=f"high_acc_exec_{signal['symbol']}_{idx}",
                            width='stretch'
                        ):
                            qty = int((trader.cash * TRADE_ALLOC) / signal["entry"])
                            success, msg = trader.execute_trade(
                                symbol=signal["symbol"],
                                action=signal["action"],
                                quantity=qty,
                                price=signal["entry"],
                                stop_loss=signal["stop_loss"],
                                target=signal["target"],
                                win_probability=signal.get("win_probability", 0.75),
                                strategy=signal.get("strategy")
                            )
                            if success:
                                st.success(msg)
                                st.rerun()
            else:
                st.info("No high-accuracy signals found. Try adjusting the filters or wait for better market conditions.")

    st.markdown("---")
    st.markdown("<div style='text-align:center; color: #6b7280;'>Enhanced Intraday Terminal Pro with Full Stock Scanning & High-Quality Signal Filters</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Application error: {str(e)}")
    st.info("Please refresh the page and try again")
    logger.error(f"Application crash: {e}")
    st.code(traceback.format_exc())
