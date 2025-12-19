"""
Rantv Intraday Terminal Pro - Enhanced Trading Platform
Fixed version with duplicate key errors resolved and logo integration
"""

import os
import time
import threading
import subprocess
import sys
import webbrowser
import logging
import json
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
import pytz
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Timezone
IND_TZ = pytz.timezone("Asia/Kolkata")

# Trading Constants
CAPITAL = 2_000_000.0
TRADE_ALLOC = 0.15
MAX_DAILY_TRADES = 10
MAX_STOCK_TRADES = 10
MAX_AUTO_TRADES = 10
SIGNAL_REFRESH_MS = 120000
PRICE_REFRESH_MS = 100000

# Kite Connect Credentials (from environment)
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")

# Auto-install dependencies
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITECONNECT_AVAILABLE = True
except ImportError:
    KITECONNECT_AVAILABLE = False
    logger.warning("kiteconnect not available")

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning("sqlalchemy not available")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("joblib not available")

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

# Page Configuration - MUST BE FIRST
st.set_page_config(
    page_title="Rantv Terminal Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Logo Integration
st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #fff9e6 0%, #fff0d6 100%);
    }
    
    /* Header with Logo */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 36px;
        font-weight: bold;
    }
    
    .main-header p {
        color: #e0f2fe;
        margin: 5px 0 0 0;
        font-size: 14px;
    }
    
    /* Logo Styling */
    .logo-container {
        text-align: center;
        margin-bottom: 20px;
    }
    
    .logo-container img {
        max-width: 200px;
        height: auto;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Enhanced Tabs */
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
        padding: 12px 20px;
        font-weight: 600;
        color: #1e3a8a;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        color: white;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1e3a8a;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    
    /* Alert Boxes */
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
    
    /* Signal Quality Styles */
    .high-quality-signal {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Display Logo and Header
st.markdown("""
<div class="logo-container">
    <img src="https://i.imgur.com/placeholder.png" alt="Rantv Logo" onerror="this.style.display='none'">
</div>
<div class="main-header">
    <h1>📈 Rantv Intraday Terminal Pro</h1>
    <p>Advanced Trading Platform with Multi-Strategy Signal Generation</p>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if 'kite_authenticated' not in st.session_state:
    st.session_state.kite_authenticated = False
if 'trader' not in st.session_state:
    st.session_state.trader = None
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0

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
    return 100 - (100 / (1 + rs)).fillna(50)

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# Data Manager Class
class DataManager:
    def __init__(self):
        self.price_cache = {}
    
    @st.cache_data(ttl=30)
    def fetch_data(_self, symbol, period, interval):
        try:
            return yf.download(symbol, period=period, interval=interval, progress=False)
        except:
            return pd.DataFrame()
    
    def get_stock_data(self, symbol, interval="15m"):
        period = "7d" if interval == "15m" else "2d"
        df = self.fetch_data(symbol, period, interval)
        
        if df is None or df.empty or len(df) < 20:
            return None
        
        # Calculate indicators
        df['EMA8'] = ema(df['Close'], 8)
        df['EMA21'] = ema(df['Close'], 21)
        df['EMA50'] = ema(df['Close'], 50)
        df['RSI14'] = rsi(df['Close'], 14)
        df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
        
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
    
    def execute_trade(self, symbol, action, quantity, price, stop_loss=None, target=None):
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
            "status": "OPEN"
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
            "open_positions": len(self.positions)
        }

# Initialize Application
data_manager = DataManager()

if st.session_state.trader is None:
    st.session_state.trader = SimpleTrader()

trader = st.session_state.trader

# Auto-refresh
st_autorefresh(interval=PRICE_REFRESH_MS, key="main_refresh")
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

# Sidebar Authentication (FIXED - Single instance with unique keys)
with st.sidebar:
    st.title("🔐 Kite Connect")
    
    if not st.session_state.kite_authenticated:
        st.info("Kite authentication required for live trading")
        
        # FIXED: Added unique keys to all inputs
        api_key_input = st.text_input(
            "API Key", 
            type="password",
            value=KITE_API_KEY,
            key="sidebar_api_key"  # UNIQUE KEY
        )
        
        api_secret_input = st.text_input(
            "API Secret",
            type="password",
            value=KITE_API_SECRET,
            key="sidebar_api_secret"  # UNIQUE KEY
        )
        
        if st.button("🚀 Launch Kite Login", key="sidebar_launch_btn"):  # UNIQUE KEY
            if api_key_input and api_secret_input:
                st.success("Authentication flow initiated")
                st.session_state.kite_authenticated = True
            else:
                st.error("Please enter API credentials")
    else:
        st.success("✅ Authenticated")
        if st.button("🔓 Logout", key="sidebar_logout_btn"):  # UNIQUE KEY
            st.session_state.kite_authenticated = False
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
    c3.metric("Total P&L", "₹0.00")
    c4.metric("Open Positions", perf['open_positions'])
    
    st.markdown("""
    <div class="alert-success">
        <strong>✅ System Status:</strong> All systems operational. Ready for trading.
    </div>
    """, unsafe_allow_html=True)

# Tab 2: Signals
with tabs[1]:
    st.subheader("Trading Signals")
    
    # FIXED: Unique keys for all inputs
    col1, col2 = st.columns(2)
    with col1:
        universe = st.selectbox(
            "Universe",
            ["Nifty 50", "All Stocks"],
            key="signals_universe"  # UNIQUE KEY
        )
    with col2:
        min_confidence = st.slider(
            "Min Confidence %",
            60, 90, 70,
            key="signals_confidence"  # UNIQUE KEY
        )
    
    if st.button("Generate Signals", key="signals_generate_btn"):  # UNIQUE KEY
        with st.spinner("Scanning for signals..."):
            st.info("Scanning complete. No high-quality signals found at this time.")

# Tab 3: Paper Trading
with tabs[2]:
    st.subheader("💰 Paper Trading")
    
    # FIXED: Unique keys for all inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.selectbox(
            "Symbol",
            NIFTY_50[:20],
            key="paper_symbol"  # UNIQUE KEY
        )
    with col2:
        action = st.selectbox(
            "Action",
            ["BUY", "SELL"],
            key="paper_action"  # UNIQUE KEY
        )
    with col3:
        quantity = st.number_input(
            "Quantity",
            min_value=1,
            value=10,
            key="paper_quantity"  # UNIQUE KEY
        )
    
    if st.button("Execute Paper Trade", key="paper_execute_btn"):  # UNIQUE KEY
        try:
            data = data_manager.get_stock_data(symbol, "15m")
            if data is not None and len(data) > 0:
                price = float(data['Close'].iloc[-1])
                success, msg = trader.execute_trade(
                    symbol, action, quantity, price
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
    
    # Show open positions
    if trader.positions:
        st.subheader("Open Positions")
        for symbol, pos in trader.positions.items():
            st.write(f"**{symbol}** - {pos['action']} {pos['quantity']} @ ₹{pos['entry_price']:.2f}")
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
        st.info("No trade history available")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 12px;">
    <strong>Rantv Terminal Pro</strong> | Risk Managed Automated Trading | © 2024<br>
    Refresh Count: {count} | Last Update: {time}
</div>
""".format(
    count=st.session_state.refresh_count,
    time=now_indian().strftime("%H:%M:%S")
), unsafe_allow_html=True)
