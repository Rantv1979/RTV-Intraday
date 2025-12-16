# Rantv Intraday Trading Signals & Market Analysis - OPTIMIZED PRODUCTION READY
# ENHANCED VERSION WITH FULL STOCK SCANNING & BETTER SIGNAL QUALITY
# OPTIMIZED: Combined features, Kite Connect in sidebar, cleaner structure

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
import traceback
import subprocess
import sys
from datetime import timedelta
import threading

# Auto-install missing critical dependencies
def install_package(package_name):
    """Install a Python package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except:
        return False

# Check and install missing packages
PACKAGES = [
    ("kiteconnect", "KiteConnect"),
    ("sqlalchemy", "sqlalchemy"),
    ("joblib", "joblib"),
    ("plotly", "plotly"),
    ("streamlit-autorefresh", "st_autorefresh")
]

for package, import_name in PACKAGES:
    try:
        __import__(import_name)
        globals()[f"{import_name.upper()}_AVAILABLE"] = True
    except ImportError:
        if install_package(package):
            st.success(f"✅ Installed {package}")
            globals()[f"{import_name.upper()}_AVAILABLE"] = True
        else:
            globals()[f"{import_name.upper()}_AVAILABLE"] = False

# Import after installation attempts
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITECONNECT_AVAILABLE = True
except:
    KITECONNECT_AVAILABLE = False

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except:
    SQLALCHEMY_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except:
    JOBLIB_AVAILABLE = False

# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Kite Connect API Credentials
KITE_API_KEY = "pwnmsnpy30s4uotu"
KITE_API_SECRET = "m44rfdl9ligc4ctaq7r9sxkxpgnfm30m"
KITE_ACCESS_TOKEN = ""

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

# Clean and combine all stocks
def clean_stock_list(lst):
    """Clean stock list"""
    clean = []
    for s in lst:
        if isinstance(s, str):
            t = s.strip().upper()
            if not t.endswith(".NS"):
                t = t + ".NS"
            clean.append(t)
    return list(dict.fromkeys(clean))

NIFTY_50 = clean_stock_list(NIFTY_50)
NIFTY_100 = clean_stock_list(NIFTY_100)
NIFTY_MIDCAP_150 = clean_stock_list(NIFTY_MIDCAP_150)
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

# CSS Styles
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #fff9e6 0%, #fff0d6 100%); }
    .main .block-container { background-color: transparent; padding-top: 2rem; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: linear-gradient(135deg, #e6f2ff 0%, #ffe6e6 50%, #e6ffe6 100%);
        padding: 8px; border-radius: 12px; margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px; white-space: pre-wrap; background-color: #ffffff; border-radius: 8px;
        gap: 8px; padding: 12px 20px; font-weight: 600; font-size: 14px; color: #1e3a8a;
        border: 2px solid transparent; transition: all 0.3s ease; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); color: white;
        border: 2px solid #2563eb; box-shadow: 0 4px 8px rgba(30, 58, 138, 0.3); transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #dbeafe 0%, #e0f2fe 100%);
        border: 2px solid #93c5fd; transform: translateY(-1px);
    }
    
    .gauge-container {
        background: white; border-radius: 50%; padding: 25px; margin: 10px auto;
        border: 4px solid #e0f2fe; box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        width: 200px; height: 200px; display: flex; flex-direction: column;
        align-items: center; justify-content: center; text-align: center; position: relative;
    }
    
    .gauge-progress {
        width: 100px; height: 100px; border-radius: 50%;
        background: conic-gradient(#059669 0% var(--progress), #e5e7eb var(--progress) 100%);
        display: flex; align-items: center; justify-content: center; margin: 8px 0; position: relative;
    }
    
    .gauge-progress-inner {
        width: 70px; height: 70px; border-radius: 50%; background: white;
        display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px;
    }
    
    .high-accuracy-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); color: white;
        padding: 15px; border-radius: 10px; border-left: 4px solid #f59e0b;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
    }
    
    .metric-card {
        background: white; padding: 15px; border-radius: 10px; border-left: 4px solid #1e3a8a;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .alert-success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #059669; padding: 12px; border-radius: 8px; margin: 8px 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #d97706; padding: 12px; border-radius: 8px; margin: 8px 0;
    }
    
    .high-quality-signal {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white;
        padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #047857;
    }
    
    .medium-quality-signal {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white;
        padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #b45309;
    }
    
    .trade-buy {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #059669;
    }
    
    .trade-sell {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
    }
</style>
""", unsafe_allow_html=True)

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

def is_peak_market_hours():
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

def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (close - lowest_low) / denom
    d = k.rolling(window=d_period).mean()
    return k.fillna(50), d.fillna(50)

def create_circular_market_mood_gauge(index_name, current_value, change_percent, sentiment_score):
    """Create circular market mood gauge"""
    sentiment_score = round(sentiment_score)
    change_percent = round(change_percent, 2)
    
    if sentiment_score >= 70:
        sentiment_color = "bullish"; sentiment_text = "BULLISH"; emoji = "📈"; progress_color = "#059669"
    elif sentiment_score <= 30:
        sentiment_color = "bearish"; sentiment_text = "BEARISH"; emoji = "📉"; progress_color = "#dc2626"
    else:
        sentiment_color = "neutral"; sentiment_text = "NEUTRAL"; emoji = "➡️"; progress_color = "#d97706"
    
    gauge_html = f"""
    <div class="gauge-container">
        <div class="gauge-title">{emoji} {index_name}</div>
        <div class="gauge-progress" style="--progress: {sentiment_score}%; background: conic-gradient({progress_color} 0% {sentiment_score}%, #e5e7eb {sentiment_score}% 100%);">
            <div class="gauge-progress-inner">{sentiment_score}%</div>
        </div>
        <div class="gauge-value">₹{current_value:,.0f}</div>
        <div class="gauge-sentiment {sentiment_color}">{sentiment_text}</div>
        <div style="color: {'#059669' if change_percent >= 0 else '#dc2626'}; font-size: 12px; margin-top: 3px;">
            {change_percent:+.2f}%
        </div>
    </div>
    """
    return gauge_html

# Kite Connect Manager
class KiteConnectManager:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = None
        self.access_token = None
        self.is_authenticated = False
        
    def login(self):
        """Login to Kite Connect"""
        try:
            self.kite = KiteConnect(api_key=self.api_key)
            
            if "kite_access_token" in st.session_state:
                self.access_token = st.session_state.kite_access_token
                self.kite.set_access_token(self.access_token)
                self.is_authenticated = True
                return True
            
            st.warning("Kite Connect authentication required.")
            login_url = self.kite.login_url()
            st.markdown(f'<a href="{login_url}" target="_blank">Login to Kite Connect</a>', unsafe_allow_html=True)
            
            with st.form("kite_login_form"):
                st.write("**Enter Access Token:**")
                access_token = st.text_input("Access Token", type="password")
                submit = st.form_submit_button("Authenticate")
                
                if submit and access_token:
                    try:
                        self.access_token = access_token
                        self.kite.set_access_token(self.access_token)
                        profile = self.kite.profile()
                        st.session_state.kite_access_token = self.access_token
                        self.is_authenticated = True
                        st.success(f"✅ Authenticated as {profile['user_name']}")
                        return True
                    except Exception as e:
                        st.error(f"Authentication failed: {str(e)}")
                        return False
            return False
            
        except Exception as e:
            st.error(f"Kite Connect login error: {str(e)}")
            return False
    
    def get_historical_data(self, instrument_token, interval="minute", days=1):
        """Get historical data from Kite Connect"""
        if not self.is_authenticated:
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
            logger.error(f"Error fetching Kite data: {e}")
            return None
    
    def get_live_quote(self, instrument_token):
        """Get live quote for an instrument"""
        if not self.is_authenticated:
            return None
            
        try:
            quote = self.kite.quote([instrument_token])
            if instrument_token in quote:
                return quote[instrument_token]
            return None
        except Exception as e:
            logger.error(f"Error fetching live quote: {e}")
            return None

# Risk Manager
class AdvancedRiskManager:
    def __init__(self, max_daily_loss=50000):
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
    
    def reset_daily_metrics(self):
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
    
    def check_trade_viability(self, symbol, action, quantity, price, current_positions):
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
        
        return True, f"Trade viable (quantity: {quantity})"

# Data Manager
class EnhancedDataManager:
    def __init__(self):
        self.price_cache = {}
        self.signal_cache = {}
        self.market_profile_cache = {}
        self.last_rsi_scan = None
        self.risk_manager = AdvancedRiskManager()
        self.kite_manager = KiteConnectManager(KITE_API_KEY, KITE_API_SECRET)
        
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
            return self.create_demo_data(symbol)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        expected = ["Open", "High", "Low", "Close", "Volume"]
        
        for e in expected:
            if e not in df.columns:
                if e.upper() in df.columns:
                    df[e] = df[e.upper()]
                else:
                    return self.create_demo_data(symbol)
        
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        if len(df) < 20:
            return self.create_demo_data(symbol)
        
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
        
        # Add technical indicators
        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI14"] = rsi(df["Close"], 14).fillna(50)
        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"]).fillna(method="ffill").fillna(0)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])
        df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"])
        df["Stoch_K"], df["Stoch_D"] = stochastic(df["High"], df["Low"], df["Close"])
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()
        
        return df
    
    def create_demo_data(self, symbol):
        """Create demo data when real data is unavailable"""
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
        
        # Add indicators
        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI14"] = rsi(df["Close"], 14).fillna(50)
        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"]).fillna(0)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])
        df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"])
        df["Stoch_K"], df["Stoch_D"] = stochastic(df["High"], df["Low"], df["Close"])
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()
        
        return df

# Trading Engine
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
        for strategy in {**TRADING_STRATEGIES, **HIGH_ACCURACY_STRATEGIES}.keys():
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
        return (
            self.auto_trades_count < MAX_AUTO_TRADES and 
            self.daily_trades < MAX_DAILY_TRADES and
            market_open()
        )
    
    def calculate_support_resistance(self, symbol, current_price):
        try:
            data = self.data_manager.get_stock_data(symbol, "15m")
            if data is None or len(data) < 20:
                return current_price * 0.98, current_price * 1.02
            
            # Simple support/resistance calculation
            high = data["High"].tail(20).max()
            low = data["Low"].tail(20).min()
            close = data["Close"].iloc[-1]
            
            support = (low + close) / 2 * 0.99
            resistance = (high + close) / 2 * 1.01
            
            return float(support), float(resistance)
        except Exception:
            return current_price * 0.98, current_price * 1.02
    
    def calculate_stop_target(self, entry_price, action, atr, current_price, support, resistance):
        if action == "BUY":
            sl = entry_price - (atr * 1.5)
            target = entry_price + (atr * 3.0)
            sl = max(sl, support * 0.995)
            if target > resistance:
                target = min(target, resistance * 0.998)
        else:
            sl = entry_price + (atr * 1.5)
            target = entry_price - (atr * 3.0)
            sl = min(sl, resistance * 1.005)
            if target < support:
                target = max(target, support * 1.002)
        
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
    
    def execute_trade(self, symbol, action, quantity, price, stop_loss=None, target=None, 
                      win_probability=0.75, auto_trade=False, strategy=None):
        
        risk_ok, risk_msg = self.data_manager.risk_manager.check_trade_viability(
            symbol, action, quantity, price, self.positions
        )
        if not risk_ok:
            return False, f"Risk check failed: {risk_msg}"
            
        self.reset_daily_counts()
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        if auto_trade and self.auto_trades_count >= MAX_AUTO_TRADES:
            return False, "Auto trade limit reached"
        
        trade_value = float(quantity) * float(price)
        if action == "BUY" and trade_value > self.cash:
            return False, "Insufficient capital"
        
        trade_id = f"TRADE_{symbol}_{len(self.trade_log)}_{int(time.time())}"
        record = {
            "trade_id": trade_id, "symbol": symbol, "action": action, "quantity": int(quantity),
            "entry_price": float(price), "stop_loss": float(stop_loss) if stop_loss else None,
            "target": float(target) if target else None, "timestamp": now_indian(),
            "status": "OPEN", "current_pnl": 0.0, "current_price": float(price),
            "win_probability": float(win_probability), "closed_pnl": 0.0,
            "entry_time": now_indian().strftime("%H:%M:%S"), "auto_trade": auto_trade,
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
        
        return True, f"{'[AUTO] ' if auto_trade else ''}{action} {int(quantity)} {symbol} @ ₹{price:.2f}"
    
    def generate_signals(self, symbol, data):
        signals = []
        if data is None or len(data) < 30:
            return signals
        
        try:
            current_price = float(data["Close"].iloc[-1])
            ema8 = float(data["EMA8"].iloc[-1])
            ema21 = float(data["EMA21"].iloc[-1])
            ema50 = float(data["EMA50"].iloc[-1])
            rsi_val = float(data["RSI14"].iloc[-1])
            atr = float(data["ATR"].iloc[-1]) if "ATR" in data.columns else current_price * 0.01
            macd_line = float(data["MACD"].iloc[-1])
            macd_signal = float(data["MACD_Signal"].iloc[-1])
            vwap = float(data["VWAP"].iloc[-1])
            
            support, resistance = self.calculate_support_resistance(symbol, current_price)
            volume = float(data["Volume"].iloc[-1])
            volume_avg = float(data["Volume"].rolling(20).mean().iloc[-1]) if len(data["Volume"]) >= 20 else float(data["Volume"].mean())
            
            # BUY SIGNALS
            # 1. EMA + VWAP Confluence
            if ema8 > ema21 > ema50 and current_price > vwap:
                target, stop_loss = self.calculate_stop_target(current_price, "BUY", atr, current_price, support, resistance)
                rr = abs(target - current_price) / max(abs(current_price - stop_loss), 1e-6)
                if rr >= 2.5:
                    signals.append({
                        "symbol": symbol, "action": "BUY", "entry": current_price, "current_price": current_price,
                        "target": target, "stop_loss": stop_loss, "confidence": 0.82, "win_probability": 0.75,
                        "risk_reward": rr, "score": 9, "strategy": "EMA_VWAP_Confluence",
                        "strategy_name": TRADING_STRATEGIES["EMA_VWAP_Confluence"]["name"], "rsi": rsi_val
                    })
            
            # 2. RSI Mean Reversion
            if rsi_val < 30 and current_price > support:
                target, stop_loss = self.calculate_stop_target(current_price, "BUY", atr, current_price, support, resistance)
                rr = abs(target - current_price) / max(abs(current_price - stop_loss), 1e-6)
                if rr >= 2.5:
                    signals.append({
                        "symbol": symbol, "action": "BUY", "entry": current_price, "current_price": current_price,
                        "target": target, "stop_loss": stop_loss, "confidence": 0.78, "win_probability": 0.72,
                        "risk_reward": rr, "score": 8, "strategy": "RSI_MeanReversion",
                        "strategy_name": TRADING_STRATEGIES["RSI_MeanReversion"]["name"], "rsi": rsi_val
                    })
            
            # 3. High Accuracy: Multi-Confirmation
            if (ema8 > ema21 > ema50 and current_price > vwap and 
                rsi_val > 50 and rsi_val < 70 and volume > volume_avg * 1.5 and 
                macd_line > macd_signal):
                
                target, stop_loss = self.calculate_stop_target(current_price, "BUY", atr, current_price, support, resistance)
                rr = abs(target - current_price) / max(abs(current_price - stop_loss), 1e-6)
                if rr >= 2.5:
                    signals.append({
                        "symbol": symbol, "action": "BUY", "entry": current_price, "current_price": current_price,
                        "target": target, "stop_loss": stop_loss, "confidence": 0.88, "win_probability": 0.82,
                        "risk_reward": rr, "score": 9, "strategy": "Multi_Confirmation",
                        "strategy_name": HIGH_ACCURACY_STRATEGIES["Multi_Confirmation"]["name"], "rsi": rsi_val
                    })
            
            # SELL SIGNALS
            # 4. EMA + VWAP Downtrend
            if ema8 < ema21 < ema50 and current_price < vwap:
                target, stop_loss = self.calculate_stop_target(current_price, "SELL", atr, current_price, support, resistance)
                rr = abs(target - current_price) / max(abs(current_price - stop_loss), 1e-6)
                if rr >= 2.5:
                    signals.append({
                        "symbol": symbol, "action": "SELL", "entry": current_price, "current_price": current_price,
                        "target": target, "stop_loss": stop_loss, "confidence": 0.82, "win_probability": 0.75,
                        "risk_reward": rr, "score": 9, "strategy": "EMA_VWAP_Downtrend",
                        "strategy_name": TRADING_STRATEGIES["EMA_VWAP_Downtrend"]["name"], "rsi": rsi_val
                    })
            
            # 5. RSI Overbought
            if rsi_val > 70 and current_price < resistance:
                target, stop_loss = self.calculate_stop_target(current_price, "SELL", atr, current_price, support, resistance)
                rr = abs(target - current_price) / max(abs(current_price - stop_loss), 1e-6)
                if rr >= 2.5:
                    signals.append({
                        "symbol": symbol, "action": "SELL", "entry": current_price, "current_price": current_price,
                        "target": target, "stop_loss": stop_loss, "confidence": 0.78, "win_probability": 0.72,
                        "risk_reward": rr, "score": 8, "strategy": "RSI_Overbought",
                        "strategy_name": TRADING_STRATEGIES["RSI_Overbought"]["name"], "rsi": rsi_val
                    })
            
            # Update strategy counts
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
                
                stock_signals = self.generate_signals(symbol, data)
                signals.extend(stock_signals)
                    
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        signals = [s for s in signals if s.get("confidence", 0) >= min_confidence and s.get("score", 0) >= min_score]
        signals.sort(key=lambda x: (x.get("score", 0), x.get("confidence", 0)), reverse=True)
        self.signal_history = signals[:30]
        
        return signals[:20]
    
    def update_positions_pnl(self):
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
            except Exception:
                continue
    
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
                    "Entry Time": pos.get("entry_time"),
                    "Strategy": pos.get("strategy", "Manual")
                })
            except Exception:
                continue
        return out
    
    def get_trade_history_data(self):
        history_data = []
        for trade in self.trade_log:
            if trade.get("status") == "CLOSED":
                pnl = trade.get("closed_pnl", 0)
                pnl_class = "profit-positive" if pnl >= 0 else "profit-negative"
                trade_class = "trade-buy" if trade.get("action") == "BUY" else "trade-sell"
                
                history_data.append({
                    "Symbol": trade.get("symbol", "").replace(".NS", ""),
                    "Action": trade.get("action", ""),
                    "Quantity": trade.get("quantity", 0),
                    "Entry Price": f"₹{trade.get('entry_price', 0):.2f}",
                    "Exit Price": f"₹{trade.get('exit_price', 0):.2f}",
                    "P&L": f"<span class='{pnl_class}'>₹{pnl:+.2f}</span>",
                    "Entry Time": trade.get("entry_time", ""),
                    "Exit Time": trade.get("exit_time_str", ""),
                    "Strategy": trade.get("strategy", "Manual"),
                    "_row_class": trade_class
                })
        return history_data
    
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

# Kite Live Charts Function
def create_kite_live_charts():
    """Create Kite Connect Live Charts"""
    st.subheader("📈 Kite Connect Live Charts")
    
    # Initialize Kite Manager
    if "kite_manager" not in st.session_state:
        st.session_state.kite_manager = KiteConnectManager(KITE_API_KEY, KITE_API_SECRET)
    
    kite_manager = st.session_state.kite_manager
    
    # Authentication Section
    if not kite_manager.is_authenticated:
        st.info("Kite Connect authentication required for live charts.")
        if kite_manager.login():
            st.rerun()
        return
    
    # Chart Selection
    col1, col2 = st.columns(2)
    with col1:
        selected_index = st.selectbox("Select Index", ["NIFTY 50", "BANKNIFTY", "FINNIFTY", "SENSEX"])
    with col2:
        interval = st.selectbox("Interval", ["minute", "5minute", "15minute", "30minute", "hour"])
    
    days_back = st.slider("Days Back", 1, 30, 7)
    
    # Token Mapping
    token_map = {
        "NIFTY 50": 256265,
        "BANKNIFTY": 260105,
        "FINNIFTY": 257801,
        "SENSEX": 265
    }
    
    if st.button("Load Live Chart", type="primary"):
        token = token_map.get(selected_index)
        if token:
            with st.spinner(f"Fetching {selected_index} data..."):
                data = kite_manager.get_historical_data(token, interval, days_back)
                
                if data is not None and len(data) > 0:
                    # Create Candlestick Chart
                    fig = go.Figure(data=[go.Candlestick(
                        x=data.index,
                        open=data['open'],
                        high=data['high'],
                        low=data['low'],
                        close=data['close'],
                        name='Price'
                    )])
                    
                    # Add moving averages
                    data['EMA20'] = ema(data['close'], 20)
                    data['EMA50'] = ema(data['close'], 50)
                    
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['EMA20'],
                        mode='lines',
                        name='EMA20',
                        line=dict(color='orange', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['EMA50'],
                        mode='lines',
                        name='EMA50',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f'{selected_index} Live Chart ({interval})',
                        xaxis_title='Time',
                        yaxis_title='Price',
                        height=600,
                        template='plotly_white',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Current Statistics
                    current_price = data['close'].iloc[-1]
                    prev_close = data['close'].iloc[-2] if len(data) > 1 else current_price
                    change_pct = ((current_price - prev_close) / prev_close) * 100
                    
                    cols = st.columns(3)
                    cols[0].metric("Current Price", f"₹{current_price:.2f}")
                    cols[1].metric("Change", f"{change_pct:+.2f}%")
                    cols[2].metric("Period High", f"₹{data['high'].max():.2f}")
                    
                    # Volume Chart
                    fig_volume = go.Figure()
                    fig_volume.add_trace(go.Bar(
                        x=data.index,
                        y=data['volume'],
                        name='Volume',
                        marker_color='rgba(30, 58, 138, 0.7)'
                    ))
                    
                    fig_volume.update_layout(
                        title='Volume Profile',
                        xaxis_title='Time',
                        yaxis_title='Volume',
                        height=300,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_volume, use_container_width=True)
                    
                else:
                    st.error("Could not fetch data. Please check your Kite Connect permissions.")
        else:
            st.error("Invalid index selection")

# MAIN APPLICATION
def main():
    # Initialize
    st_autorefresh(interval=PRICE_REFRESH_MS, key="price_refresh")
    
    if "trader" not in st.session_state:
        st.session_state.trader = MultiStrategyIntradayTrader()
    
    if "refresh_count" not in st.session_state:
        st.session_state.refresh_count = 0
    
    st.session_state.refresh_count += 1
    
    trader = st.session_state.trader
    data_manager = trader.data_manager
    
    # Header
    st.markdown("<h1 style='text-align:center; color: #1e3a8a;'>Rantv Intraday Terminal Pro</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color: #6b7280;'>Full Stock Scanning & High-Quality Signal Generation</h4>", unsafe_allow_html=True)
    
    # Market Overview
    cols = st.columns(6)
    try:
        nift = data_manager._validate_live_price("^NSEI")
        cols[0].metric("NIFTY 50", f"₹{nift:,.2f}")
    except:
        cols[0].metric("NIFTY 50", "N/A")
    
    try:
        bn = data_manager._validate_live_price("^NSEBANK")
        cols[1].metric("BANK NIFTY", f"₹{bn:,.2f}")
    except:
        cols[1].metric("BANK NIFTY", "N/A")
    
    cols[2].metric("Market Status", "LIVE" if market_open() else "CLOSED")
    cols[3].metric("Peak Hours", "YES" if is_peak_market_hours() else "NO")
    cols[4].metric("Auto Trades", f"{trader.auto_trades_count}/{MAX_AUTO_TRADES}")
    cols[5].metric("Available Cash", f"₹{trader.cash:,.0f}")
    
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
    
    # Display Gauges
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
        st.markdown(create_circular_market_mood_gauge("PEAK HOURS", 0, 0, peak_sentiment).replace("₹0", "9:30-2:30").replace("0.00%", peak_hours_status), unsafe_allow_html=True)
    
    # Main Metrics
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
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 12px; color: #6b7280;">Open Positions</div>
            <div style="font-size: 20px; font-weight: bold; color: #1e3a8a;">{len(trader.positions)}</div>
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
    
    # SIDEBAR CONFIGURATION
    with st.sidebar:
        st.header("🎯 Kite Connect")
        
        # Kite Connect Status
        if KITECONNECT_AVAILABLE:
            kite_manager = data_manager.kite_manager
            if not kite_manager.is_authenticated:
                if st.button("Login to Kite Connect", key="kite_login_btn"):
                    if kite_manager.login():
                        st.rerun()
            else:
                st.success("✅ Kite Connect Authenticated")
                if st.button("Logout", key="kite_logout_btn"):
                    if "kite_access_token" in st.session_state:
                        del st.session_state.kite_access_token
                    kite_manager.is_authenticated = False
                    st.rerun()
                
                # Quick Kite Actions
                st.subheader("Quick Actions")
                if st.button("Test Connection", key="test_kite_btn"):
                    try:
                        profile = kite_manager.kite.profile()
                        st.success(f"Connected as: {profile['user_name']}")
                    except Exception as e:
                        st.error(f"Connection failed: {str(e)}")
        else:
            st.warning("Kite Connect not available")
        
        st.markdown("---")
        st.header("⚙️ Trading Configuration")
        
        # Universe Selection
        universe = st.selectbox("Trading Universe", ["All Stocks", "Nifty 50", "Nifty 100", "Midcap 150"], key="universe_select")
        enable_high_accuracy = st.checkbox("Enable High Accuracy Strategies", value=True, key="high_acc_toggle")
        trader.auto_execution = st.checkbox("Auto Execution", value=False, key="auto_exec_toggle")
        
        # Risk Management
        st.subheader("🎯 Risk Management")
        min_conf_percent = st.slider("Minimum Confidence %", 60, 85, 70, 5, key="min_conf_slider")
        min_score = st.slider("Minimum Score", 5, 9, 6, 1, key="min_score_slider")
        
        # Scan Configuration
        st.subheader("🔍 Scan Configuration")
        full_scan = st.checkbox("Full Universe Scan", value=True, key="full_scan_toggle")
        if not full_scan:
            max_scan = st.number_input("Max Stocks to Scan", min_value=10, max_value=500, value=50, step=10, key="max_scan_input")
        else:
            max_scan = None
        
        # Strategy Performance
        st.markdown("---")
        st.header("📊 Strategy Performance")
        
        # High Accuracy Strategies
        st.subheader("🔥 High Accuracy")
        for strategy, config in HIGH_ACCURACY_STRATEGIES.items():
            if strategy in trader.strategy_performance:
                perf = trader.strategy_performance[strategy]
                if perf["trades"] > 0:
                    win_rate = perf["wins"] / perf["trades"]
                    color = "#059669" if win_rate > 0.7 else "#dc2626" if win_rate < 0.5 else "#d97706"
                    st.write(f"**{config['name']}**")
                    st.write(f"Win Rate: <span style='color: {color};'>{win_rate:.1%}</span>", unsafe_allow_html=True)
                    st.write(f"P&L: ₹{perf['pnl']:+.2f}")
                    st.markdown("---")
        
        # Standard Strategies
        st.subheader("📊 Standard Strategies")
        for strategy, config in TRADING_STRATEGIES.items():
            if strategy in trader.strategy_performance:
                perf = trader.strategy_performance[strategy]
                if perf["trades"] > 0:
                    win_rate = perf["wins"] / perf["trades"]
                    st.write(f"**{config['name']}**")
                    st.write(f"Trades: {perf['trades']} | Win Rate: {win_rate:.1%}")
    
    # MAIN TABS
    tabs = st.tabs([
        "📈 Dashboard", 
        "🚦 Signals", 
        "💰 Paper Trading", 
        "📋 Trade History",
        "📉 RSI Scanner", 
        "⚡ Strategies",
        "🎯 High Accuracy",
        "📊 Kite Live Charts"
    ])
    
    # Tab 1: Dashboard
    with tabs[0]:
        st.subheader("Account Summary")
        perf = trader.get_performance_stats()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Value", f"₹{trader.equity():,.0f}")
        c2.metric("Available Cash", f"₹{trader.cash:,.0f}")
        c3.metric("Open Positions", len(trader.positions))
        c4.metric("Total P&L", f"₹{perf['total_pnl'] + perf['open_pnl']:+.2f}")
        
        # Open Positions
        st.subheader("📊 Open Positions")
        open_positions = trader.get_open_positions_data()
        if open_positions:
            st.dataframe(pd.DataFrame(open_positions), width='stretch')
        else:
            st.info("No open positions")
    
    # Tab 2: Signals
    with tabs[1]:
        st.subheader("Multi-Strategy BUY/SELL Signals")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            generate_btn = st.button("Generate Signals", type="primary", key="gen_sig_btn")
        
        if generate_btn:
            with st.spinner(f"Scanning {universe} stocks..."):
                signals = trader.generate_quality_signals(
                    universe, 
                    max_scan=max_scan,
                    min_confidence=min_conf_percent/100.0, 
                    min_score=min_score,
                    use_high_accuracy=enable_high_accuracy
                )
            
            if signals:
                st.success(f"✅ Found {len(signals)} signals")
                
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
                
                # Manual Execution
                st.subheader("Manual Execution")
                for idx, s in enumerate(signals[:5]):
                    col_a, col_b, col_c = st.columns([3,1,1])
                    with col_a:
                        action_color = "🟢" if s["action"] == "BUY" else "🔴"
                        is_high_acc = s["strategy"] in HIGH_ACCURACY_STRATEGIES
                        strategy_display = f"🔥 {s['strategy_name']}" if is_high_acc else s['strategy_name']
                        
                        st.markdown(f"""
                        <div class="{'high-quality-signal' if s['score'] >= 8 else 'medium-quality-signal'}">
                            <strong>{action_color} {s['symbol'].replace('.NS','')}</strong> - {s['action']} @ ₹{s['entry']:.2f}<br>
                            Strategy: {strategy_display} | R:R: {s['risk_reward']:.2f}
                        </div>
                        """, unsafe_allow_html=True)
                    with col_b:
                        qty = int((trader.cash * TRADE_ALLOC) / s["entry"])
                        st.write(f"Qty: {qty}")
                    with col_c:
                        if st.button(f"Execute", key=f"exec_{s['symbol']}_{idx}"):
                            success, msg = trader.execute_trade(
                                symbol=s["symbol"], action=s["action"], quantity=qty, price=s["entry"],
                                stop_loss=s["stop_loss"], target=s["target"], 
                                win_probability=s.get("win_probability",0.75),
                                strategy=s.get("strategy")
                            )
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
            else:
                st.warning("No signals found. Try adjusting filters or scanning during market hours.")
    
    # Tab 3: Paper Trading
    with tabs[2]:
        st.subheader("💰 Paper Trading")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.selectbox("Symbol", NIFTY_50[:20], key="paper_sym")
        with col2:
            action = st.selectbox("Action", ["BUY", "SELL"], key="paper_act")
        with col3:
            quantity = st.number_input("Quantity", min_value=1, value=10, key="paper_qty")
        
        if st.button("Execute Paper Trade", type="primary", key="paper_exec"):
            try:
                data = data_manager.get_stock_data(symbol, "15m")
                price = float(data["Close"].iloc[-1])
                atr = float(data["ATR"].iloc[-1]) if 'ATR' in data.columns else price * 0.01
                support, resistance = trader.calculate_support_resistance(symbol, price)
                
                target, stop_loss = trader.calculate_stop_target(
                    price, action, atr, price, support, resistance
                )
                
                success, msg = trader.execute_trade(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    price=price,
                    stop_loss=stop_loss,
                    target=target
                )
                
                if success:
                    st.success(f"✅ {msg}")
                    st.success(f"Stop Loss: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")
                    
            except Exception as e:
                st.error(f"Trade execution failed: {str(e)}")
        
        # Current Positions
        st.subheader("Current Positions")
        positions_df = trader.get_open_positions_data()
        if positions_df:
            for idx, pos in enumerate(positions_df):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    action_color = "🟢" if pos['Action'] == 'BUY' else "🔴"
                    st.markdown(f"""
                    <div style="padding: 10px; border-left: 4px solid {'#059669' if pos['Action'] == 'BUY' else '#dc2626'}; 
                             background: linear-gradient(135deg, {'#d1fae5' if pos['Action'] == 'BUY' else '#fee2e2'} 0%, 
                             {'#a7f3d0' if pos['Action'] == 'BUY' else '#fecaca'} 100%); border-radius: 8px;">
                        <strong>{action_color} {pos['Symbol']}</strong> | {pos['Action']} | Qty: {pos['Quantity']}<br>
                        Entry: {pos['Entry Price']} | Current: {pos['Current Price']}<br>
                        {pos['P&L']}
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    if st.button(f"Close", key=f"close_{pos['Symbol']}_{idx}"):
                        success, msg = trader.close_position(f"{pos['Symbol']}.NS")
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
        trade_history = trader.get_trade_history_data()
        if trade_history:
            for trade in trade_history:
                st.markdown(f"""
                <div class="{trade.get('_row_class', '')}">
                    <div style="padding: 10px;">
                        <strong>{trade['Symbol']}</strong> | {trade['Action']} | Qty: {trade['Quantity']}<br>
                        Entry: {trade['Entry Price']} | Exit: {trade['Exit Price']} | {trade['P&L']}<br>
                        Duration: {trade.get('Duration', 'N/A')} | Strategy: {trade['Strategy']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No trade history available")
    
    # Tab 5: RSI Scanner
    with tabs[4]:
        st.subheader("📉 RSI Extreme Scanner")
        
        if st.button("Scan for RSI Extremes", key="rsi_scan"):
            with st.spinner("Scanning for RSI extremes..."):
                oversold = []
                overbought = []
                
                for symbol in NIFTY_50[:30]:
                    data = data_manager.get_stock_data(symbol, "15m")
                    if len(data) > 0:
                        rsi_val = data['RSI14'].iloc[-1] if 'RSI14' in data and data['RSI14'].dropna().shape[0] > 0 else 50.0
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
                        st.dataframe(pd.DataFrame(oversold), width='stretch')
                    
                    if overbought:
                        st.subheader("🔴 Overbought Stocks (RSI > 70)")
                        st.dataframe(pd.DataFrame(overbought), width='stretch')
                else:
                    st.info("No extreme RSI stocks found")
    
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
        st.subheader("🎯 High Accuracy Scanner")
        
        col1, col2 = st.columns(2)
        with col1:
            high_acc_scan_btn = st.button("🚀 Scan High Accuracy", type="primary", key="high_acc_scan")
        with col2:
            min_high_acc_score = st.slider("Min Score", 5, 8, 6, 1, key="high_acc_score")
        
        if high_acc_scan_btn:
            with st.spinner(f"Scanning {universe} with high-accuracy strategies..."):
                high_acc_signals = trader.generate_quality_signals(
                    universe, 
                    max_scan=50 if universe == "All Stocks" else max_scan,
                    min_confidence=0.70,
                    min_score=min_high_acc_score,
                    use_high_accuracy=True
                )
            
            if high_acc_signals:
                st.success(f"🎯 Found {len(high_acc_signals)} high-confidence signals!")
                
                for idx, signal in enumerate(high_acc_signals[:10]):
                    with st.container():
                        st.markdown(f"""
                        <div class="high-quality-signal">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong>{signal['symbol'].replace('.NS', '')}</strong> | 
                                    <span style="color: #ffffff">
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
                                R:R: {signal['risk_reward']:.2f} | Confidence: {signal['confidence']:.1%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No high-accuracy signals found.")
    
    # Tab 8: Kite Live Charts
    with tabs[7]:
        create_kite_live_charts()
    
    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align:center; color: #6b7280;'>Enhanced Intraday Terminal Pro | Kite Connect Integrated | All Rights Reserved</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.code(traceback.format_exc())
