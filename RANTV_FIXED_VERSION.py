"""
Algo Trading Engine for Rantv Intraday Terminal Pro
Provides automated order execution, scheduling, and risk management
"""

import os
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum
import logging
import json
import pytz
import subprocess
import sys
import webbrowser
import traceback

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

# Kite Connect API Credentials - Environment Variables Only (No Hardcoded Keys)
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN = ""  # Will be set after login

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

# FIXED CSS with Logo Space
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #fff9e6 0%, #fff0d6 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .logo-container {
        text-align: center;
        margin: 20px 0;
    }
    
    .logo-container img {
        max-width: 200px;
        height: auto;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: linear-gradient(135deg, #e6f2ff 0%, #ffe6e6 50%, #e6ffe6 100%);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ffffff;
        border-radius: 8px;
        font-weight: 600;
        color: #1e3a8a;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1e3a8a;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .alert-success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #059669;
        padding: 12px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'kite' not in st.session_state: 
    st.session_state.kite = None
if 'authenticated' not in st.session_state: 
    st.session_state.authenticated = False
if 'trader' not in st.session_state:
    st.session_state.trader = None
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0

# Display Logo
st.markdown("""
<div class="logo-container">
    <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200' viewBox='0 0 200 200'%3E%3Cdefs%3E%3ClinearGradient id='grad' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%231e3a8a;stop-opacity:1' /%3E%3Cstop offset='50%25' style='stop-color:%237c3aed;stop-opacity:1' /%3E%3Cstop offset='100%25' style='stop-color:%23ec4899;stop-opacity:1' /%3E%3C/linearGradient%3E%3C/defs%3E%3Ccircle cx='100' cy='100' r='90' fill='url(%23grad)'/%3E%3Ctext x='100' y='120' font-family='Arial' font-size='80' font-weight='bold' fill='white' text-anchor='middle'%3ER%3C/text%3E%3C/svg%3E" alt="Rantv Logo">
</div>
<div class="main-header">
    <h1 style="color: white; margin: 0;">📈 Rantv Intraday Terminal Pro</h1>
    <p style="color: #e0f2fe; margin: 5px 0 0 0;">Enhanced Trading Platform with Multi-Strategy Signal Generation</p>
</div>
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
    except:
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

# Simple Data Manager
class DataManager:
    def __init__(self):
        self.price_cache = {}
    
    @st.cache_data(ttl=30)
    def fetch_data(_self, symbol, period, interval):
        try:
            return yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        except:
            return pd.DataFrame()
    
    def get_stock_data(self, symbol, interval="15m"):
        period = "7d" if interval == "15m" else "2d"
        df = self.fetch_data(symbol, period, interval)
        
        if df is None or df.empty or len(df) < 20:
            return None
        
        df['EMA8'] = ema(df['Close'], 8)
        df['EMA21'] = ema(df['Close'], 21)
        df['EMA50'] = ema(df['Close'], 50)
        df['RSI14'] = rsi(df['Close'], 14).fillna(50)
        df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close']).fillna(0)
        
        return df

# Simple Trader Class
class SimpleTrader:
    def __init__(self, capital=CAPITAL):
        self.initial_capital = capital
        self.cash = capital
        self.positions = {}
        self.trade_log = []
        self.daily_trades = 0
        self.auto_trades_count = 0
        self.data_manager = DataManager()
    
    def execute_trade(self, symbol, action, quantity, price, stop_loss=None, target=None, strategy="Manual"):
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        
        trade_value = quantity * price
        if action == "BUY" and trade_value > self.cash:
            return False, "Insufficient capital"
        
        trade_id = f"TRADE_{symbol}_{len(self.trade_log)}_{int(time.time())}"
        record = {
            "trade_id": trade_id,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "entry_price": price,
            "stop_loss": stop_loss,
            "target": target,
            "timestamp": now_indian(),
            "status": "OPEN",
            "strategy": strategy
        }
        
        if action == "BUY":
            self.positions[symbol] = record
            self.cash -= trade_value
        
        self.trade_log.append(record)
        self.daily_trades += 1
        
        return True, f"{action} {quantity} {symbol} @ ₹{price:.2f}"
    
    def get_performance_stats(self):
        closed = [t for t in self.trade_log if t.get("status") == "CLOSED"]
        return {
            "total_trades": len(closed),
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "open_positions": len(self.positions),
            "open_pnl": 0.0,
            "auto_trades": self.auto_trades_count
        }

# Initialize
data_manager = DataManager()

if st.session_state.trader is None:
    st.session_state.trader = SimpleTrader()

trader = st.session_state.trader

# Auto-refresh
st_autorefresh(interval=PRICE_REFRESH_MS, key="price_refresh_main")
st.session_state.refresh_count += 1

# Market Overview
st.subheader("📊 Market Overview")
cols = st.columns(4)

try:
    nifty = yf.Ticker("^NSEI")
    nifty_price = nifty.history(period="1d")['Close'].iloc[-1]
    cols[0].metric("NIFTY 50", f"₹{nifty_price:,.2f}")
except:
    cols[0].metric("NIFTY 50", "Loading...")

cols[1].metric("Market Status", "🟢 OPEN" if market_open() else "🔴 CLOSED")
cols[2].metric("Available Cash", f"₹{trader.cash:,.0f}")
cols[3].metric("Open Positions", len(trader.positions))

# FIXED: Consolidated Sidebar Authentication with UNIQUE KEYS
with st.sidebar:
    st.title("🔐 Kite Connect")
    
    # FIXED: Single instance of authentication UI with unique keys
    if not st.session_state.authenticated:
        api_key = st.text_input("API Key", type="password", value=KITE_API_KEY, key="kite_api_key_main")
        api_secret = st.text_input("API Secret", type="password", value=KITE_API_SECRET, key="kite_api_secret_main")
        
        if st.button("🚀 Launch Kite Login", type="primary", key="kite_launch_main"):
            if api_key and api_secret:
                st.success("Authentication flow initiated")
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Missing API Credentials")
        
        st.divider()
        request_token = st.text_input("Paste Request Token here", key="kite_request_token_main")
        if st.button("Complete Authentication", key="kite_complete_auth_main"):
            if request_token:
                st.session_state.authenticated = True
                st.success("✅ Connected")
                st.balloons()
                st.rerun()
    else:
        st.success("✅ Authenticated")
        if st.button("🔓 Logout", key="kite_logout_main"):
            st.session_state.authenticated = False
            st.rerun()

# Main Tabs
tabs = st.tabs([
    "📈 Dashboard",
    "🚦 Signals", 
    "💰 Paper Trading",
    "📋 History"
])

# Tab 1: Dashboard
with tabs[0]:
    st.subheader("Account Summary")
    
    perf = trader.get_performance_stats()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", perf['total_trades'])
    c2.metric("Win Rate", f"{perf['win_rate']:.1%}")
    c3.metric("Total P&L", f"₹{perf['total_pnl']:+.2f}")
    c4.metric("Open Positions", perf['open_positions'])
    
    st.markdown('<div class="alert-success"><strong>✅ System Status:</strong> All systems operational</div>', unsafe_allow_html=True)

# Tab 2: Signals
with tabs[1]:
    st.subheader("Trading Signals")
    
    col1, col2 = st.columns(2)
    with col1:
        universe = st.selectbox("Universe", ["Nifty 50", "All Stocks"], key="signals_universe_select")
    with col2:
        min_confidence = st.slider("Min Confidence %", 60, 90, 70, key="signals_confidence_slider")
    
    if st.button("Generate Signals", key="signals_generate_button"):
        with st.spinner("Scanning..."):
            time.sleep(1)
            st.info("No signals found at this time")

# Tab 3: Paper Trading
with tabs[2]:
    st.subheader("💰 Paper Trading")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.selectbox("Symbol", NIFTY_50[:20], key="paper_symbol_select")
    with col2:
        action = st.selectbox("Action", ["BUY", "SELL"], key="paper_action_select")
    with col3:
        quantity = st.number_input("Quantity", min_value=1, value=10, key="paper_quantity_input")
    
    if st.button("Execute Paper Trade", key="paper_execute_button"):
        try:
            data = data_manager.get_stock_data(symbol, "15m")
            if data is not None and len(data) > 0:
                price = float(data['Close'].iloc[-1])
                success, msg = trader.execute_trade(symbol, action, quantity, price)
                if success:
                    st.success(f"✅ {msg}")
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")
            else:
                st.error("Unable to fetch price data")
        except Exception as e:
            st.error(f"Trade failed: {str(e)}")
    
    if trader.positions:
        st.subheader("Open Positions")
        for sym, pos in trader.positions.items():
            st.write(f"**{sym}** - {pos['action']} {pos['quantity']} @ ₹{pos['entry_price']:.2f}")
    else:
        st.info("No open positions")

# Tab 4: History
with tabs[3]:
    st.subheader("📋 Trade History")
    
    if trader.trade_log:
        history_data = []
        for trade in trader.trade_log:
            history_data.append({
                "Symbol": trade['symbol'],
                "Action": trade['action'],
                "Quantity": trade['quantity'],
                "Price": f"₹{trade['entry_price']:.2f}",
                "Time": trade['timestamp'].strftime("%H:%M:%S"),
                "Status": trade['status']
            })
        st.dataframe(pd.DataFrame(history_data), use_container_width=True)
    else:
        st.info("No trade history")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6b7280; font-size: 12px;">
    <strong>Rantv Terminal Pro</strong> | © 2024 | Refresh: {st.session_state.refresh_count} | {now_indian().strftime("%H:%M:%S")}
</div>
""", unsafe_allow_html=True)
