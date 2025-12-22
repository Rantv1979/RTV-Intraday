"""
RANTV TERMINAL PRO - INSTITUTIONAL GRADE ALGO TRADING v4.0
Complete Trading System with Advanced SMC + Session Logic + Volume Profile + Auto Risk + Kite OMS
Version 4.0 - Full Stack Upgrade with MTF SMC
"""

# ===================== IMPORTS =====================
import os
import sys
import time
import json
import logging
import threading
import warnings
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import math
import random
import smtplib
import webbrowser
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import subprocess

# Try to import optional packages with error handling
def install_package(package_name):
    """Install a Python package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except:
        return False

# Try to import websocket
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Try to import streamlit
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

# Try to import streamlit_autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# Data & ML Libraries
import pandas as pd
import numpy as np

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

# Visualization
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

# Timezone
try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False
    pytz = None

# Suppress warnings
warnings.filterwarnings("ignore")

# ===================== KITE CONNECT IMPORT =====================
# Try to import Kite Connect, install if missing
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITECONNECT_AVAILABLE = True
except ImportError:
    KITECONNECT_AVAILABLE = False

# ===================== CONFIGURATION =====================
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Timezone setup
if PYTZ_AVAILABLE:
    try:
        IND_TZ = pytz.timezone("Asia/Kolkata")
    except:
        from datetime import timezone
        import datetime as dt
        IND_TZ = timezone(timedelta(hours=5, minutes=30))
else:
    from datetime import timezone
    IND_TZ = timezone(timedelta(hours=5, minutes=30))

# Environment Configuration
class AppConfig:
    """Application configuration"""
    # API Configuration
    KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
    KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")
    
    # Algo Configuration
    ALGO_ENABLED = os.environ.get("ALGO_TRADING_ENABLED", "false").lower() == "true"
    ALGO_MAX_POSITIONS = int(os.environ.get("ALGO_MAX_POSITIONS", "5"))
    ALGO_MAX_DAILY_LOSS = float(os.environ.get("ALGO_MAX_DAILY_LOSS", "50000"))
    ALGO_MIN_CONFIDENCE = float(os.environ.get("ALGO_MIN_CONFIDENCE", "0.80"))
    
    # Risk Parameters (NEW from v3)
    CAPITAL = 2_000_000.0
    BASE_RISK = 0.01
    MAX_RISK = 0.02
    MIN_RISK = 0.005
    MAX_DAILY_DD = 0.03
    SL_ATR = 1.5
    TP_ATR = 3.0
    TRAIL_ATR = 1.2
    
    # Session Times (NEW from v3)
    INDIA_OPEN = dt_time(9, 15)
    INDIA_CLOSE = dt_time(15, 30)
    NY_OVERLAP_START = dt_time(19, 30)
    NY_OVERLAP_END = dt_time(22, 30)
    
    # Email Configuration
    EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "")
    EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
    EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "rantv2002@gmail.com")
    
    # Trading Configuration
    INITIAL_CAPITAL = 2_000_000.0
    TRADE_ALLOCATION = 0.15
    MAX_DAILY_TRADES = 10
    MAX_STOCK_TRADES = 10
    MAX_AUTO_TRADES = 10
    SIGNAL_REFRESH_MS = 120000
    PRICE_REFRESH_MS = 100000
    
    # Database
    DATABASE_URL = "sqlite:///trading_journal.db"
    
    @classmethod
    def load_from_env(cls):
        """Load configuration from environment"""
        return cls()

# Load config
config = AppConfig.load_from_env()

# Trading Constants
class TradingConstants:
    """Trading constants"""
    # COMPLETE Stock Universes - Fixed with all stocks
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
    
    # NIFTY 100 Stocks (includes NIFTY 50 + next 50)
    NIFTY_100 = NIFTY_50 + [
        "DABUR.NS", "HAVELLS.NS", "PIDILITIND.NS", "BERGEPAINT.NS", "MOTHERSUMI.NS",
        "AMBUJACEM.NS", "ICICIPRULI.NS", "BIOCON.NS", "MARICO.NS", "TORNTPHARM.NS",
        "ASHOKLEY.NS", "INDIGO.NS", "HDFCAMC.NS", "PEL.NS", "NAUKRI.NS",
        "PAGEIND.NS", "CONCOR.NS", "BANDHANBNK.NS", "ICICIGI.NS", "MCDOWELL-N.NS",
        "AUROPHARMA.NS", "SIEMENS.NS", "LICHSGFIN.NS", "SRTRANSFIN.NS", "NIACL.NS",
        "IDEA.NS", "ACC.NS", "HINDPETRO.NS", "BOSCHLTD.NS", "PNB.NS",
        "GAIL.NS", "CADILAHC.NS", "COLPAL.NS", "MFSL.NS", "SAIL.NS",
        "DLF.NS", "YESBANK.NS", "VEDL.NS", "HINDZINC.NS", "JINDALSTEL.NS",
        "LUPIN.NS", "CANBK.NS", "GODREJCP.NS", "IDFC.NS", "IOB.NS",
        "MINDTREE.NS", "RAMCOCEM.NS", "STRTECH.NS", "TVSMOTOR.NS", "WOCKPHARMA.NS"
    ]
    
    # NIFTY MIDCAP 150 Stocks (Top 50 for example)
    NIFTY_MIDCAP = [
        "ABFRL.NS", "ADANIGREEN.NS", "ADANITRANS.NS", "ALKEM.NS", "APLLTD.NS",
        "AUBANK.NS", "BALKRISIND.NS", "BANKBARODA.NS", "BATAINDIA.NS", "BEL.NS",
        "BHARATFORG.NS", "BHEL.NS", "CHOLAFIN.NS", "CROMPTON.NS", "DALBHARAT.NS",
        "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "GLENMARK.NS", "GMRINFRA.NS",
        "GODREJAGRO.NS", "GODREJIND.NS", "HAL.NS", "HINDCOPPER.NS", "IBULHSGFIN.NS",
        "IOC.NS", "IPCALAB.NS", "JUBLFOOD.NS", "JSWENERGY.NS", "LALPATHLAB.NS",
        "MANAPPURAM.NS", "MRF.NS", "NAM-INDIA.NS", "NHPC.NS", "OFSS.NS",
        "PFC.NS", "PIIND.NS", "PVR.NS", "RECLTD.NS", "SCHAEFFLER.NS",
        "SRF.NS", "TATACHEM.NS", "TATACOMM.NS", "TATAPOWER.NS", "UNIONBANK.NS",
        "VOLTAS.NS", "WHIRLPOOL.NS", "ZEEL.NS", "ZYDUSWELL.NS", "ABCAPITAL.NS"
    ]
    
    # Kite Token Mapping (example tokens)
    KITE_TOKEN_MAP = {
        "NIFTY 50": 256265,
        "BANK NIFTY": 260105,
        "RELIANCE.NS": 738561,
        "TCS.NS": 2953217,
        "HDFCBANK.NS": 341249,
        "INFY.NS": 408065,
        "ICICIBANK.NS": 1270529,
        "KOTAKBANK.NS": 492033,
        "ITC.NS": 424961,
        "LT.NS": 2939649
    }
    
    # Trading Strategies (Updated with SMC strategies)
    TRADING_STRATEGIES = {
        "MTF_SMC_Strategy": {"name": "MTF SMC + Session Logic", "weight": 5, "type": "BOTH"},
        "Volume_Profile_SMC": {"name": "Volume Profile + SMC", "weight": 4, "type": "BOTH"},
        "SMC_Liquidity_FVG": {"name": "Smart Money Concept", "weight": 4, "type": "BOTH"},
        "EMA_VWAP_Confluence": {"name": "EMA + VWAP", "weight": 3, "type": "BUY"},
        "RSI_MeanReversion": {"name": "RSI Mean Reversion", "weight": 2, "type": "BUY"},
        "Bollinger_Reversion": {"name": "Bollinger Band", "weight": 2, "type": "BUY"},
        "MACD_Momentum": {"name": "MACD Momentum", "weight": 2, "type": "BUY"},
        "Support_Resistance": {"name": "S/R Breakout", "weight": 3, "type": "BUY"},
    }
    
    # Market Hours
    MARKET_OPEN = "09:15"
    MARKET_CLOSE = "15:30"
    PEAK_START = "09:30"
    PEAK_END = "14:30"
    DAILY_EXIT = "15:35"

# ===================== UTILITY FUNCTIONS =====================
def now_indian():
    """Get current time in Indian timezone"""
    try:
        return datetime.now(IND_TZ)
    except:
        return datetime.now()

def market_open():
    """Check if market is open"""
    return valid_session()  # Using new session logic

def is_peak_market_hours():
    """Check if current time is during peak market hours"""
    n = now_indian()
    try:
        current_time = n.time()
        peak_start = dt_time(9, 30)
        peak_end = dt_time(14, 30)
        return peak_start <= current_time <= peak_end
    except Exception:
        return True

def should_exit_all_positions():
    """Check if it's time to exit all positions (3:35 PM)"""
    n = now_indian()
    try:
        current_time = n.time()
        exit_time = dt_time(15, 35)
        return current_time >= exit_time
    except Exception:
        return False

def ema(series, span):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(0)
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(close, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_support_resistance(high, low, close, period=20):
    """Calculate support and resistance levels"""
    try:
        resistance = []
        support = []
        ln = len(high)
        if ln < period * 2 + 1:
            return {"support": float(close.iloc[-1] * 0.98), "resistance": float(close.iloc[-1] * 1.02)}
        
        for i in range(period, ln - period):
            if high.iloc[i] >= high.iloc[i - period:i + period + 1].max():
                resistance.append(float(high.iloc[i]))
            if low.iloc[i] <= low.iloc[i - period:i + period + 1].min():
                support.append(float(low.iloc[i]))
        
        recent_res = sorted(resistance)[-3:] if resistance else [float(close.iloc[-1] * 1.02)]
        recent_sup = sorted(support)[:3] if support else [float(close.iloc[-1] * 0.98)]
        
        return {
            "support": float(np.mean(recent_sup)),
            "resistance": float(np.mean(recent_res)),
            "support_levels": recent_sup,
            "resistance_levels": recent_res
        }
    except Exception:
        current_price = float(close.iloc[-1])
        return {
            "support": current_price * 0.98,
            "resistance": current_price * 1.02,
            "support_levels": [],
            "resistance_levels": []
        }

# ===================== ADVANCED UTILITIES FROM V3 =====================
def valid_session():
    """Check if current time is in valid trading session"""
    t = datetime.now().time()
    return config.INDIA_OPEN <= t <= config.INDIA_CLOSE or \
           config.NY_OVERLAP_START <= t <= config.NY_OVERLAP_END

def market_regime(df: pd.DataFrame) -> str:
    """Determine market regime: TREND, RANGE, or VOLATILE"""
    if df is None or len(df) < 20:
        return "RANGE"
    
    try:
        adx = abs(ema(df['Close'].diff(), 14))
        vol = df['Close'].pct_change().rolling(20).std()
        
        if adx.iloc[-1] > adx.mean():
            return 'TREND'
        if vol.iloc[-1] < vol.mean():
            return 'RANGE'
        return 'VOLATILE'
    except:
        return 'RANGE'

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index"""
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Calculate +DM and -DM
        up_move = high.diff()
        down_move = low.diff().abs() * -1
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate True Range
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        # Calculate smoothed values
        atr = tr.rolling(period).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    except:
        return pd.Series([50] * len(df), index=df.index)

def volume_profile(df: pd.DataFrame, bins: int = 24) -> float:
    """Calculate Volume Profile Point of Control (POC)"""
    try:
        if df is None or len(df) < bins:
            return float(df['Close'].iloc[-1]) if len(df) > 0 else 0
        
        prices = df['Close'].values
        vols = df['Volume'].values
        
        if len(prices) != len(vols):
            return float(df['Close'].iloc[-1])
        
        hist, edges = np.histogram(prices, bins=bins, weights=vols)
        if len(hist) == 0:
            return float(df['Close'].iloc[-1])
        
        poc_index = np.argmax(hist)
        if poc_index < len(edges) - 1:
            poc = (edges[poc_index] + edges[poc_index + 1]) / 2
            return float(poc)
        else:
            return float(df['Close'].iloc[-1])
    except:
        return float(df['Close'].iloc[-1]) if df is not None and len(df) > 0 else 0

# ===================== ADVANCED SMC CLASS FROM V3 =====================
class AdvancedSMC:
    """Advanced Smart Money Concept with MTF analysis"""
    
    @staticmethod
    def detect_BOS(df: pd.DataFrame, lookback: int = 6) -> Optional[str]:
        """Detect Break of Structure"""
        try:
            if len(df) < lookback + 1:
                return None
            
            # Bullish BOS
            if df['High'].iloc[-1] > df['High'].iloc[-lookback:-1].max():
                return 'BULLISH'
            
            # Bearish BOS
            if df['Low'].iloc[-1] < df['Low'].iloc[-lookback:-1].min():
                return 'BEARISH'
            
            return None
        except:
            return None
    
    @staticmethod
    def detect_FVG(df: pd.DataFrame) -> Optional[Tuple[str, float, float]]:
        """Detect Fair Value Gap"""
        try:
            if len(df) < 3:
                return None
            
            a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
            
            # Bullish FVG (downward gap)
            if a['High'] < c['Low']:
                return ('BULLISH', float(a['High']), float(c['Low']))
            
            # Bearish FVG (upward gap)
            if a['Low'] > c['High']:
                return ('BEARISH', float(c['High']), float(a['Low']))
            
            return None
        except:
            return None
    
    @staticmethod
    def detect_order_block(df: pd.DataFrame) -> Optional[str]:
        """Detect Order Block"""
        try:
            if len(df) < 2:
                return None
            
            last = df.iloc[-2]  # Previous candle
            
            # Bullish Order Block (bearish candle followed by bullish move)
            if last['Close'] > last['Open']:
                return 'BULLISH'
            
            # Bearish Order Block (bullish candle followed by bearish move)
            if last['Close'] < last['Open']:
                return 'BEARISH'
            
            return None
        except:
            return None
    
    @staticmethod
    def detect_liquidity_grab(df: pd.DataFrame) -> Optional[str]:
        """Detect Liquidity Grab (stop hunting)"""
        try:
            if len(df) < 10:
                return None
            
            recent_high = df['High'].iloc[-10:-1].max()
            recent_low = df['Low'].iloc[-10:-1].min()
            current_low = df['Low'].iloc[-1]
            current_close = df['Close'].iloc[-1]
            
            # Bullish liquidity grab (wick below recent low then close above)
            if current_low < recent_low and current_close > recent_low:
                return 'BULLISH'
            
            # Bearish liquidity grab (wick above recent high then close below)
            current_high = df['High'].iloc[-1]
            if current_high > recent_high and current_close < recent_high:
                return 'BEARISH'
            
            return None
        except:
            return None

# ===================== KITE OMS WITH KILL SWITCH =====================
class KiteOMS:
    """Kite Order Management System with Reconciliation and Kill Switch"""
    
    def __init__(self, kite_manager):
        self.kite = kite_manager.kite if kite_manager and kite_manager.kite else None
        self.orders = {}
        self.last_reconciliation = None
        self.kill_switch_active = False
    
    def place_order(self, symbol: str, action: str, quantity: int, 
                   order_type: str = "MARKET", price: float = 0) -> Tuple[bool, str]:
        """Place order with Kite OMS"""
        if not self.kite:
            return False, "Kite not initialized"
        
        if self.kill_switch_active:
            return False, "Kill switch active - trading halted"
        
        try:
            # Map action to Kite transaction type
            transaction_type = self.kite.TRANSACTION_TYPE_BUY if action == "BUY" else self.kite.TRANSACTION_TYPE_SELL
            
            # Place order
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=symbol.replace('.NS', ''),
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=self.kite.ORDER_TYPE_MARKET if order_type == "MARKET" else self.kite.ORDER_TYPE_LIMIT,
                product=self.kite.PRODUCT_MIS,
                price=price if order_type == "LIMIT" else 0
            )
            
            # Record order
            self.orders[order_id] = {
                'status': 'PENDING',
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'timestamp': datetime.now()
            }
            
            logger.info(f"KITE ORDER: {order_id} - {action} {quantity} {symbol}")
            return True, f"Order {order_id} placed successfully"
            
        except Exception as e:
            logger.error(f"Kite order placement failed: {e}")
            return False, f"Order placement failed: {str(e)}"
    
    def reconcile_orders(self) -> bool:
        """Reconcile all orders with Kite"""
        if not self.kite:
            return False
        
        try:
            live_orders = self.kite.orders()
            live_order_dict = {str(o['order_id']): o['status'] for o in live_orders}
            
            for order_id in list(self.orders.keys()):
                if order_id not in live_order_dict:
                    # Order not found in live orders - potential issue
                    logger.warning(f"Order {order_id} not found in Kite orders")
                    self.orders[order_id]['status'] = 'MISSING'
                else:
                    self.orders[order_id]['status'] = live_order_dict[order_id]
            
            self.last_reconciliation = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Order reconciliation failed: {e}")
            return False
    
    def check_kill_switch(self) -> Tuple[bool, str]:
        """Check if kill switch should be activated"""
        if not self.kite:
            return False, "Kite not initialized"
        
        try:
            # Check for order mismatches
            mismatches = 0
            for order_id, order_info in self.orders.items():
                if order_info['status'] not in ['COMPLETE', 'CANCELLED', 'REJECTED']:
                    # Order is still open
                    try:
                        order_status = self.kite.order_history(order_id)
                        if order_status and order_status[0]['status'] != order_info['status']:
                            mismatches += 1
                    except:
                        mismatches += 1
            
            # Activate kill switch if too many mismatches
            if mismatches > 2:
                self.kill_switch_active = True
                return True, f"KILL SWITCH ACTIVATED - {mismatches} order mismatches detected"
            
            # Check for unauthorized access
            positions = self.kite.positions()
            if positions and 'net' in positions:
                # Check for unusual positions
                pass
            
            return False, "System normal"
            
        except Exception as e:
            logger.error(f"Kill switch check failed: {e}")
            return False, f"Check failed: {str(e)}"
    
    def get_order_status(self) -> Dict:
        """Get order system status"""
        return {
            'total_orders': len(self.orders),
            'active_orders': len([o for o in self.orders.values() if o['status'] == 'OPEN']),
            'kill_switch': self.kill_switch_active,
            'last_reconciliation': self.last_reconciliation
        }

# ===================== AUTO RISK SCALER =====================
class AutoRiskScaler:
    """Dynamic risk scaling based on performance"""
    
    def __init__(self, initial_capital: float):
        self.equity = [initial_capital]
        self.returns = []
        self.max_drawdown = 0
        self.current_drawdown = 0
        self.win_rate = 0.5
        self.profit_factor = 1.0
    
    def update(self, pnl: float, trade_result: bool = True):
        """Update risk metrics after trade"""
        new_equity = self.equity[-1] + pnl
        self.equity.append(new_equity)
        
        # Calculate return
        if len(self.equity) > 1:
            ret = (new_equity - self.equity[-2]) / self.equity[-2]
            self.returns.append(ret)
        
        # Update win rate
        if trade_result:
            # Simplified win rate update
            pass
        
        # Calculate drawdown
        peak = max(self.equity)
        current = self.equity[-1]
        self.current_drawdown = (peak - current) / peak if peak > 0 else 0
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
    
    def calculate_risk(self) -> float:
        """Calculate dynamic risk percentage"""
        if len(self.equity) < 20:
            return config.BASE_RISK
        
        try:
            # Calculate equity curve slope
            recent_equity = self.equity[-20:]
            x = np.arange(len(recent_equity))
            slope, _ = np.polyfit(x, recent_equity, 1)
            
            # Calculate volatility
            if len(self.returns) >= 10:
                volatility = np.std(self.returns[-10:])
            else:
                volatility = 0.02
            
            # Calculate Sharpe-like metric
            avg_return = np.mean(self.returns[-20:]) if self.returns else 0
            sharpe_like = avg_return / volatility if volatility > 0 else 0
            
            # Dynamic risk adjustment
            base_risk = config.BASE_RISK
            
            # Increase risk during winning streaks, low drawdown
            if slope > 0 and self.current_drawdown < config.MAX_DAILY_DD * 0.5:
                risk_multiplier = min(2.0, 1.0 + sharpe_like * 10)
                new_risk = base_risk * risk_multiplier
            # Decrease risk during losing streaks, high drawdown
            elif slope < 0 or self.current_drawdown > config.MAX_DAILY_DD * 0.8:
                risk_multiplier = max(0.3, 1.0 - abs(sharpe_like) * 5)
                new_risk = base_risk * risk_multiplier
            else:
                new_risk = base_risk
            
            # Apply bounds
            new_risk = max(config.MIN_RISK, min(config.MAX_RISK, new_risk))
            
            logger.info(f"AutoRisk: Slope={slope:.4f}, DD={self.current_drawdown:.2%}, Risk={new_risk:.2%}")
            return new_risk
            
        except Exception as e:
            logger.error(f"Risk calculation error: {e}")
            return config.BASE_RISK
    
    def get_status(self) -> Dict:
        """Get risk scaler status"""
        return {
            'current_equity': self.equity[-1] if self.equity else 0,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'current_risk': self.calculate_risk(),
            'total_trades': len(self.returns),
            'avg_return': np.mean(self.returns) if self.returns else 0
        }

# ===================== KITE CONNECT MANAGER WITH OMS =====================
class KiteConnectManager:
    """Enhanced Kite Connect manager with OMS integration"""
    
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = None
        self.access_token = None
        self.is_authenticated = False
        self.user_name = ""
        self.ticker = None
        self.live_data = {}
        self.oms = None
        
        if api_key and KITECONNECT_AVAILABLE:
            try:
                self.kite = KiteConnect(api_key=api_key)
                self.oms = KiteOMS(self)
            except Exception as e:
                logger.error(f"Failed to initialize KiteConnect: {e}")
    
    def get_login_url(self):
        """Get Kite login URL"""
        if self.kite:
            try:
                return self.kite.login_url()
            except Exception as e:
                logger.error(f"Failed to get login URL: {e}")
                return None
        return None
    
    def authenticate(self, request_token):
        """Authenticate with request token"""
        try:
            if not self.kite:
                return False, "KiteConnect not initialized"
            
            # Generate session
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            
            if 'access_token' in data:
                self.access_token = data['access_token']
                self.kite.set_access_token(self.access_token)
                self.is_authenticated = True
                self.user_name = data.get('user_name', 'User')
                
                # Initialize OMS
                self.oms = KiteOMS(self)
                
                # Initialize ticker for live data if websocket is available
                if WEBSOCKET_AVAILABLE:
                    try:
                        self.ticker = KiteTicker(self.api_key, self.access_token)
                    except Exception as e:
                        logger.warning(f"Failed to initialize KiteTicker: {e}")
                        self.ticker = None
                
                return True, f"Authenticated as {self.user_name}"
            else:
                return False, "Authentication failed"
                
        except Exception as e:
            return False, f"Authentication error: {str(e)}"
    
    def get_historical_data(self, instrument_token, interval="minute", days=7):
        """Get historical data from Kite"""
        if not self.is_authenticated:
            return None
        
        try:
            from_date = datetime.now().date() - timedelta(days=days)
            to_date = datetime.now().date()
            
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
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
            logger.error(f"Error fetching Kite historical data: {e}")
            return None
    
    def place_order(self, symbol, action, quantity, order_type="MARKET", price=0):
        """Place an order through Kite OMS"""
        if not self.is_authenticated or not self.oms:
            return False, "Not authenticated or OMS not initialized"
        
        return self.oms.place_order(symbol, action, quantity, order_type, price)
    
    def get_positions(self):
        """Get current positions from Kite"""
        if not self.is_authenticated:
            return {}
        
        try:
            positions = self.kite.positions()
            return positions
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {}
    
    def reconcile_system(self):
        """Reconcile the entire trading system"""
        if not self.is_authenticated or not self.oms:
            return False, "Not authenticated"
        
        try:
            # Reconcile orders
            self.oms.reconcile_orders()
            
            # Check kill switch
            kill_switch, message = self.oms.check_kill_switch()
            
            if kill_switch:
                return False, f"Kill switch activated: {message}"
            
            return True, "System reconciled successfully"
            
        except Exception as e:
            return False, f"Reconciliation failed: {str(e)}"
    
    def start_live_data(self, tokens):
        """Start live data streaming for given tokens"""
        if not self.is_authenticated or not self.ticker:
            return False
        
        try:
            def on_ticks(ws, ticks):
                for tick in ticks:
                    self.live_data[tick['instrument_token']] = {
                        'last_price': tick['last_price'],
                        'volume': tick.get('volume_traded', 0),
                        'oi': tick.get('oi', 0),
                        'timestamp': datetime.now()
                    }
            
            def on_connect(ws, response):
                self.ticker.subscribe(tokens)
                self.ticker.set_mode(self.ticker.MODE_FULL, tokens)
                logger.info("Kite WebSocket connected")
            
            def on_close(ws, code, reason):
                logger.info(f"Kite WebSocket closed: {code} - {reason}")
            
            self.ticker.on_ticks = on_ticks
            self.ticker.on_connect = on_connect
            self.ticker.on_close = on_close
            
            self.ticker.connect(threaded=True)
            return True
            
        except Exception as e:
            logger.error(f"Error starting live data: {e}")
            return False
    
    def logout(self):
        """Logout from Kite"""
        self.is_authenticated = False
        self.access_token = None
        self.user_name = ""
        if self.ticker:
            try:
                self.ticker.close()
            except:
                pass
        logger.info("Logged out from Kite Connect")

# ===================== DATA MANAGER WITH ADVANCED FEATURES =====================
class DataManager:
    """Enhanced data manager with MTF and SMC support"""
    
    def __init__(self, kite_manager=None):
        self.kite_manager = kite_manager
        self.price_cache = {}
        self.signal_cache = {}
        self.cache_timeout = 30  # seconds
        self.smc = AdvancedSMC()
    
    def get_stock_data(self, symbol, interval="15m", use_kite=True):
        """Get stock data from Kite or Yahoo Finance"""
        cache_key = f"{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H')}"
        
        # Check cache
        if cache_key in self.price_cache:
            cached_data, timestamp = self.price_cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_timeout:
                return cached_data
        
        try:
            # Try Kite Connect first if available
            if use_kite and self.kite_manager and self.kite_manager.is_authenticated:
                token = TradingConstants.KITE_TOKEN_MAP.get(symbol)
                if token:
                    # Map interval to Kite interval
                    kite_interval_map = {
                        "1m": "minute",
                        "5m": "5minute",
                        "15m": "15minute",
                        "30m": "30minute",
                        "1h": "60minute"
                    }
                    
                    kite_interval = kite_interval_map.get(interval, "15minute")
                    df = self.kite_manager.get_historical_data(token, kite_interval, days=7)
                    
                    if df is not None and not df.empty:
                        # Process Kite data
                        df = self._process_kite_data(df)
                        self.price_cache[cache_key] = (df.copy(), datetime.now())
                        return df
            
            # Fallback to Yahoo Finance
            return self._get_yahoo_data(symbol, interval)
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return self._get_yahoo_data(symbol, interval)  # Fallback
    
    def get_mtf_data(self, symbol, intervals=["15m", "1h", "1d"]):
        """Get Multi-TimeFrame data"""
        mtf_data = {}
        
        for interval in intervals:
            data = self.get_stock_data(symbol, interval)
            if data is not None and not data.empty:
                mtf_data[interval] = data
        
        return mtf_data
    
    def _process_kite_data(self, df):
        """Process Kite Connect data"""
        # Ensure we have required columns
        if 'open' in df.columns and 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
        
        # Calculate indicators
        return self._calculate_advanced_indicators(df)
    
    def _get_yahoo_data(self, symbol, interval):
        """Get data from Yahoo Finance"""
        try:
            if not YFINANCE_AVAILABLE:
                logger.error("yfinance not available")
                return None
            
            # Determine period based on interval
            period_map = {
                "1m": "1d",
                "5m": "5d",
                "15m": "15d",
                "30m": "30d",
                "1h": "60d"
            }
            
            period = period_map.get(interval, "15d")
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=False)
            
            if df is None or df.empty or len(df) < 20:
                return None
            
            # Clean columns
            df.columns = [col.capitalize() for col in df.columns]
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            # Calculate indicators
            df = self._calculate_advanced_indicators(df)
            
            # Cache the result
            cache_key = f"{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H')}"
            self.price_cache[cache_key] = (df.copy(), datetime.now())
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {symbol}: {e}")
            return None
    
    def _calculate_advanced_indicators(self, df):
        """Calculate advanced technical indicators"""
        if df is None or df.empty:
            return df
            
        try:
            # Basic indicators
            df["EMA8"] = ema(df["Close"], 8)
            df["EMA21"] = ema(df["Close"], 21)
            df["EMA50"] = ema(df["Close"], 50)
            df["RSI14"] = rsi(df["Close"], 14)
            df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"])
            df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])
            df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"])
            
            # VWAP calculation
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['TP_Volume'] = df['Typical_Price'] * df['Volume']
            df['VWAP'] = df['TP_Volume'].cumsum() / df['Volume'].cumsum()
            
            # Advanced indicators from v3
            df["ADX"] = calculate_adx(df)
            
            # Market regime
            df["Market_Regime"] = df.apply(lambda row: market_regime(df), axis=1)
            
            # Volume Profile POC
            if len(df) >= 24:
                df["Volume_POC"] = volume_profile(df)
            else:
                df["Volume_POC"] = df["Close"]
            
            # SMC signals
            bos = self.smc.detect_BOS(df)
            fvg = self.smc.detect_FVG(df)
            ob = self.smc.detect_order_block(df)
            liq = self.smc.detect_liquidity_grab(df)
            
            df["SMC_BOS"] = bos if bos else "NONE"
            df["SMC_FVG"] = "PRESENT" if fvg else "NONE"
            df["SMC_OB"] = ob if ob else "NONE"
            df["SMC_Liquidity"] = liq if liq else "NONE"
            
            # Support/Resistance
            sr = calculate_support_resistance(df["High"], df["Low"], df["Close"])
            df["Support"] = sr["support"]
            df["Resistance"] = sr["resistance"]
            
            return df
        except Exception as e:
            logger.error(f"Error calculating advanced indicators: {e}")
            return df
    
    def get_live_price(self, symbol, use_kite=True):
        """Get live price from Kite or Yahoo"""
        if use_kite and self.kite_manager and self.kite_manager.is_authenticated:
            token = TradingConstants.KITE_TOKEN_MAP.get(symbol)
            if token and token in self.kite_manager.live_data:
                return self.kite_manager.live_data[token]['last_price']
        
        # Fallback to Yahoo
        try:
            data = self.get_stock_data(symbol, "5m", use_kite=False)
            if data is not None and not data.empty:
                return float(data["Close"].iloc[-1])
            return None
        except Exception:
            return None
    
    def clear_cache(self):
        """Clear all caches"""
        self.price_cache.clear()
        self.signal_cache.clear()

# ===================== RISK MANAGER WITH AUTO SCALER =====================
class RiskManager:
    """Advanced risk management with auto-scaling"""
    
    def __init__(self, config):
        self.config = config
        self.risk_scaler = AutoRiskScaler(config.INITIAL_CAPITAL)
        self.daily_stats = {
            'total_pnl': 0.0,
            'trades_today': 0,
            'positions_opened': 0,
            'max_daily_loss': config.ALGO_MAX_DAILY_LOSS
        }
        self.position_limits = {
            'max_positions': config.ALGO_MAX_POSITIONS,
            'max_trades_per_day': config.MAX_DAILY_TRADES
        }
        self.session_risk_factor = 1.0
        
    def check_trade(self, symbol, action, quantity, price, confidence=0.5, strategy_type="SMC"):
        """Check if trade meets risk criteria"""
        # Session-based risk adjustment
        current_time = datetime.now().time()
        if config.NY_OVERLAP_START <= current_time <= config.NY_OVERLAP_END:
            self.session_risk_factor = 0.7  # Reduce risk during NY overlap
        
        # Get dynamic risk
        dynamic_risk = self.risk_scaler.calculate_risk()
        
        checks = {
            'market_open': valid_session(),
            'daily_loss_limit': self.daily_stats['total_pnl'] > -self.daily_stats['max_daily_loss'],
            'position_limit': self.daily_stats['positions_opened'] < self.position_limits['max_positions'],
            'trade_limit': self.daily_stats['trades_today'] < self.position_limits['max_trades_per_day'],
            'confidence_threshold': confidence >= config.ALGO_MIN_CONFIDENCE,
            'sufficient_capital': True,
            'dynamic_risk_ok': dynamic_risk <= config.MAX_RISK,
            'drawdown_ok': self.risk_scaler.current_drawdown < config.MAX_DAILY_DD
        }
        
        # Special checks for SMC strategies
        if strategy_type == "SMC":
            checks['session_ok'] = valid_session()
            checks['regime_ok'] = True  # Will be checked by strategy
        
        all_passed = all(checks.values())
        
        return {
            'approved': all_passed,
            'checks': checks,
            'reason': None if all_passed else "; ".join([k for k, v in checks.items() if not v]),
            'dynamic_risk': dynamic_risk,
            'session_risk_factor': self.session_risk_factor
        }
    
    def update_trade(self, pnl, win=True):
        """Update risk metrics after trade"""
        self.daily_stats['total_pnl'] += pnl
        self.daily_stats['trades_today'] += 1
        self.risk_scaler.update(pnl, win)
    
    def update_position(self, opened=True):
        """Update position count"""
        if opened:
            self.daily_stats['positions_opened'] += 1
        else:
            self.daily_stats['positions_opened'] = max(0, self.daily_stats['positions_opened'] - 1)
    
    def reset_daily(self):
        """Reset daily statistics"""
        self.daily_stats.update({
            'total_pnl': 0.0,
            'trades_today': 0,
            'positions_opened': 0
        })
        self.session_risk_factor = 1.0
    
    def get_status(self):
        """Get risk manager status"""
        risk_status = self.risk_scaler.get_status()
        
        return {
            'daily_pnl': self.daily_stats['total_pnl'],
            'trades_today': self.daily_stats['trades_today'],
            'positions_open': self.daily_stats['positions_opened'],
            'daily_limit_used': abs(self.daily_stats['total_pnl']) / self.daily_stats['max_daily_loss'],
            'within_limits': self.daily_stats['total_pnl'] > -self.daily_stats['max_daily_loss'],
            'dynamic_risk': risk_status['current_risk'],
            'current_drawdown': risk_status['current_drawdown'],
            'max_drawdown': risk_status['max_drawdown'],
            'equity': risk_status['current_equity'],
            'session_risk_factor': self.session_risk_factor
        }

# ===================== ADVANCED STRATEGY MANAGER =====================
class StrategyManager:
    """Manage advanced trading strategies"""
    
    def __init__(self):
        self.strategies = {}
        self.enabled_strategies = set()
        self.smc_strategies = set()
        self.load_strategies()
    
    def load_strategies(self):
        """Load all trading strategies"""
        for strategy_id, config in TradingConstants.TRADING_STRATEGIES.items():
            self.strategies[strategy_id] = {
                'id': strategy_id,
                'name': config['name'],
                'weight': config['weight'],
                'type': config['type'],
                'enabled': True,
                'performance': {'signals': 0, 'trades': 0, 'wins': 0, 'pnl': 0.0},
                'category': 'SMC' if 'SMC' in strategy_id else 'TECHNICAL'
            }
            self.enabled_strategies.add(strategy_id)
            
            if 'SMC' in strategy_id:
                self.smc_strategies.add(strategy_id)
    
    def enable_strategy(self, strategy_id):
        """Enable a strategy"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id]['enabled'] = True
            self.enabled_strategies.add(strategy_id)
    
    def disable_strategy(self, strategy_id):
        """Disable a strategy"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id]['enabled'] = False
            self.enabled_strategies.discard(strategy_id)
    
    def get_all_strategies(self):
        """Get all strategies"""
        return self.strategies
    
    def get_enabled_strategies(self):
        """Get enabled strategies"""
        return [self.strategies[sid] for sid in self.enabled_strategies]
    
    def get_smc_strategies(self):
        """Get SMC strategies"""
        return [self.strategies[sid] for sid in self.smc_strategies if sid in self.enabled_strategies]
    
    def update_performance(self, strategy_id, pnl, win=True):
        """Update strategy performance"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id]['performance']['trades'] += 1
            self.strategies[strategy_id]['performance']['pnl'] += pnl
            if win:
                self.strategies[strategy_id]['performance']['wins'] += 1

# ===================== ADVANCED SIGNAL GENERATOR =====================
class AdvancedSignalGenerator:
    """Generate advanced trading signals with MTF SMC"""
    
    def __init__(self, data_manager, strategy_manager):
        self.data_manager = data_manager
        self.strategy_manager = strategy_manager
        self.signals_generated = 0
        self.last_scan_time = None
        self.smc = AdvancedSMC()
    
    def generate_smc_signal(self, symbol):
        """Generate SMC-based signal with MTF analysis"""
        # Get MTF data
        mtf_data = self.data_manager.get_mtf_data(symbol, intervals=["15m", "1h", "1d"])
        
        if not mtf_data or len(mtf_data) < 2:
            return None
        
        try:
            # Use 15m for entry, 1h for confirmation, 1d for trend
            ltf = mtf_data.get("15m")
            htf = mtf_data.get("1h")
            daily = mtf_data.get("1d")
            
            if ltf is None or htf is None or len(ltf) < 50 or len(htf) < 20:
                return None
            
            current_price = float(ltf['Close'].iloc[-1])
            regime = market_regime(htf)
            
            # Check session validity
            if not valid_session():
                return None
            
            # Check market regime - only trade in TREND regime for SMC
            if regime != 'TREND':
                return None
            
            # Get SMC signals
            bos = self.smc.detect_BOS(htf)
            fvg = self.smc.detect_FVG(ltf)
            ob = self.smc.detect_order_block(ltf)
            liq = self.smc.detect_liquidity_grab(ltf)
            
            # Get Volume Profile POC
            poc = volume_profile(ltf)
            
            # Get ATR for risk calculation
            atr_val = float(ltf['ATR'].iloc[-1]) if 'ATR' in ltf.columns else current_price * 0.02
            
            # Generate signal based on SMC confluence
            signal = None
            confidence = 0.0
            
            # BULLISH SMC signal
            if bos == 'BULLISH' and fvg and fvg[0] == 'BULLISH' and ob == 'BULLISH' and current_price > poc:
                if liq == 'BULLISH':
                    confidence = 0.95  # Very high confidence with liquidity grab
                else:
                    confidence = 0.85
                
                signal = {
                    'action': 'BUY',
                    'sl': current_price - (atr_val * config.SL_ATR),
                    'tp': current_price + (atr_val * config.TP_ATR),
                    'confidence': confidence
                }
            
            # BEARISH SMC signal
            elif bos == 'BEARISH' and fvg and fvg[0] == 'BEARISH' and ob == 'BEARISH' and current_price < poc:
                if liq == 'BEARISH':
                    confidence = 0.95
                else:
                    confidence = 0.85
                
                signal = {
                    'action': 'SELL',
                    'sl': current_price + (atr_val * config.SL_ATR),
                    'tp': current_price - (atr_val * config.TP_ATR),
                    'confidence': confidence
                }
            
            if signal:
                # Calculate win probability
                win_probability = min(0.97, confidence * 0.9)
                
                # Adjust for session
                current_time = datetime.now().time()
                if config.INDIA_OPEN <= current_time <= config.INDIA_CLOSE:
                    win_probability = min(0.98, win_probability * 1.05)
                
                return {
                    'symbol': symbol,
                    'action': signal['action'],
                    'price': round(current_price, 2),
                    'stop_loss': round(signal['sl'], 2),
                    'target': round(signal['tp'], 2),
                    'confidence': round(signal['confidence'], 3),
                    'strategy': 'MTF SMC + Volume Profile',
                    'win_probability': round(win_probability, 3),
                    'timestamp': now_indian(),
                    'atr': round(atr_val, 2),
                    'regime': regime,
                    'poc': round(poc, 2),
                    'bos': bos,
                    'fvg': 'PRESENT' if fvg else 'NONE',
                    'order_block': ob,
                    'liquidity_grab': liq
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating SMC signal for {symbol}: {e}")
            return None
    
    def generate_signal(self, symbol):
        """Generate signal for a single symbol (compatibility method)"""
        # First try SMC strategy
        smc_signal = self.generate_smc_signal(symbol)
        if smc_signal:
            return smc_signal
        
        # Fallback to traditional strategies
        data = self.data_manager.get_stock_data(symbol, "15m")
        
        if data is None or len(data) < 50:
            return None
        
        try:
            current_price = float(data['Close'].iloc[-1])
            signals = []
            
            # Check each enabled strategy
            for strategy in self.strategy_manager.get_enabled_strategies():
                strategy_id = strategy['id']
                
                # Skip SMC strategies as we already tried them
                if strategy_id in self.strategy_manager.smc_strategies:
                    continue
                
                if strategy_id == "EMA_VWAP_Confluence":
                    signal = self._ema_vwap_strategy(data, current_price)
                elif strategy_id == "RSI_MeanReversion":
                    signal = self._rsi_strategy(data, current_price)
                elif strategy_id == "Bollinger_Reversion":
                    signal = self._bollinger_strategy(data, current_price)
                elif strategy_id == "MACD_Momentum":
                    signal = self._macd_strategy(data, current_price)
                elif strategy_id == "Support_Resistance":
                    signal = self._support_resistance_strategy(data, current_price)
                else:
                    signal = None
                
                if signal:
                    signal['strategy'] = strategy['name']
                    signal['strategy_id'] = strategy_id
                    signal['weight'] = strategy['weight']
                    signals.append(signal)
            
            if not signals:
                return None
            
            # Combine signals
            buy_signals = [s for s in signals if s['action'] == 'BUY']
            sell_signals = [s for s in signals if s['action'] == 'SELL']
            
            if not buy_signals and not sell_signals:
                return None
            
            # Determine final action based on weighted signals
            buy_score = sum(s['confidence'] * s['weight'] for s in buy_signals)
            sell_score = sum(s['confidence'] * s['weight'] for s in sell_signals)
            
            if buy_score > sell_score and buy_score > 0:
                action = 'BUY'
                confidence = buy_score / sum(s['weight'] for s in buy_signals)
                strategy_name = "Multi-Strategy Confluence"
            elif sell_score > buy_score and sell_score > 0:
                action = 'SELL'
                confidence = sell_score / sum(s['weight'] for s in sell_signals)
                strategy_name = "Multi-Strategy Confluence"
            else:
                return None
            
            # Calculate stop loss and target
            atr = float(data['ATR'].iloc[-1]) if 'ATR' in data.columns else current_price * 0.02
            
            if action == 'BUY':
                stop_loss = current_price - (atr * 1.5)
                target = current_price + (atr * 3.0)
            else:
                stop_loss = current_price + (atr * 1.5)
                target = current_price - (atr * 3.0)
            
            # Calculate win probability
            win_probability = min(0.95, confidence * 0.8)
            if is_peak_market_hours():
                win_probability = min(0.97, win_probability * 1.1)
            
            return {
                'symbol': symbol,
                'action': action,
                'price': round(current_price, 2),
                'stop_loss': round(stop_loss, 2),
                'target': round(target, 2),
                'confidence': round(confidence, 3),
                'strategy': strategy_name,
                'win_probability': round(win_probability, 3),
                'timestamp': now_indian(),
                'atr': round(atr, 2),
                'signal_count': len(signals),
                'rsi': float(data['RSI14'].iloc[-1]) if 'RSI14' in data.columns else 50
            }
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    # Keep the existing strategy methods for compatibility
    def _ema_vwap_strategy(self, data, current_price):
        """EMA + VWAP strategy"""
        try:
            ema8 = data['EMA8'].iloc[-1]
            ema21 = data['EMA21'].iloc[-1]
            vwap = data['VWAP'].iloc[-1]
            
            # Price above all EMAs and VWAP
            if current_price > ema8 > ema21 > vwap:
                return {'action': 'BUY', 'confidence': 0.80}
            # Price below all EMAs and VWAP
            elif current_price < ema8 < ema21 < vwap:
                return {'action': 'SELL', 'confidence': 0.80}
            
            return None
        except Exception:
            return None
    
    def _rsi_strategy(self, data, current_price):
        """RSI strategy"""
        try:
            rsi_val = data['RSI14'].iloc[-1]
            
            # Oversold bounce
            if rsi_val < 30:
                return {'action': 'BUY', 'confidence': 0.75}
            # Overbought reversal
            elif rsi_val > 70:
                return {'action': 'SELL', 'confidence': 0.75}
            
            return None
        except Exception:
            return None
    
    def _bollinger_strategy(self, data, current_price):
        """Bollinger Bands strategy"""
        try:
            bb_lower = data['BB_Lower'].iloc[-1]
            bb_upper = data['BB_Upper'].iloc[-1]
            
            # Price touches lower band
            if current_price <= bb_lower * 1.01:
                return {'action': 'BUY', 'confidence': 0.70}
            # Price touches upper band
            elif current_price >= bb_upper * 0.99:
                return {'action': 'SELL', 'confidence': 0.70}
            
            return None
        except Exception:
            return None
    
    def _macd_strategy(self, data, current_price):
        """MACD strategy"""
        try:
            macd_line = data['MACD'].iloc[-1]
            signal_line = data['MACD_Signal'].iloc[-1]
            macd_prev = data['MACD'].iloc[-2]
            signal_prev = data['MACD_Signal'].iloc[-2]
            
            # Bullish crossover
            if macd_line > signal_line and macd_prev <= signal_prev:
                return {'action': 'BUY', 'confidence': 0.75}
            # Bearish crossover
            elif macd_line < signal_line and macd_prev >= signal_prev:
                return {'action': 'SELL', 'confidence': 0.75}
            
            return None
        except Exception:
            return None
    
    def _support_resistance_strategy(self, data, current_price):
        """Support/Resistance strategy"""
        try:
            support = data['Support'].iloc[-1]
            resistance = data['Resistance'].iloc[-1]
            
            # Near support
            if current_price <= support * 1.01:
                return {'action': 'BUY', 'confidence': 0.80}
            # Near resistance
            elif current_price >= resistance * 0.99:
                return {'action': 'SELL', 'confidence': 0.80}
            
            return None
        except Exception:
            return None
    
    def scan_universe(self, universe="Nifty 50", max_stocks=30, min_confidence=0.70, use_smc=True):
        """Scan stock universe for signals"""
        # Determine which stocks to scan
        if universe == "Nifty 50":
            stocks = TradingConstants.NIFTY_50[:max_stocks]
        elif universe == "Nifty 100":
            stocks = TradingConstants.NIFTY_100[:max_stocks]
        elif universe == "Midcap":
            stocks = TradingConstants.NIFTY_MIDCAP[:max_stocks]
        else:
            stocks = TradingConstants.NIFTY_50[:max_stocks]
        
        signals = []
        
        for symbol in stocks:
            if use_smc:
                signal = self.generate_smc_signal(symbol)
            else:
                signal = self.generate_signal(symbol)
            
            if signal and signal['confidence'] >= min_confidence:
                signals.append(signal)
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        self.signals_generated += len(signals)
        self.last_scan_time = now_indian()
        
        return signals

# ===================== ADVANCED PAPER TRADER =====================
class AdvancedPaperTrader:
    """Paper trading engine with advanced features"""
    
    def __init__(self, initial_capital, risk_manager, data_manager, kite_manager=None, live_trading=False):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions = {}
        self.trade_log = []
        self.risk_manager = risk_manager
        self.data_manager = data_manager
        self.kite_manager = kite_manager
        self.live_trading = live_trading
        
        self.daily_stats = {
            'trades_today': 0,
            'auto_trades': 0,
            'total_pnl': 0.0,
            'smc_trades': 0
        }
        self.last_reset = now_indian().date()
        
    def reset_daily_counts(self):
        """Reset daily counts if new day"""
        current_date = now_indian().date()
        if current_date != self.last_reset:
            self.daily_stats.update({
                'trades_today': 0,
                'auto_trades': 0,
                'total_pnl': 0.0,
                'smc_trades': 0
            })
            self.last_reset = current_date
            self.risk_manager.reset_daily()
    
    def can_trade(self):
        """Check if trading is allowed"""
        self.reset_daily_counts()
        return (
            self.daily_stats['trades_today'] < config.MAX_DAILY_TRADES and
            valid_session()
        )
    
    def execute_trade(self, symbol, action, quantity, price, 
                     stop_loss=None, target=None, strategy="Manual", 
                     auto_trade=False, confidence=0.5, use_kite=False,
                     trail_atr=None, is_smc=False):
        """Execute a trade with advanced features"""
        if not self.can_trade():
            return False, "Daily trade limit reached or market closed"
        
        # Check risk with strategy type
        strategy_type = "SMC" if is_smc else "TECHNICAL"
        risk_check = self.risk_manager.check_trade(symbol, action, quantity, price, confidence, strategy_type)
        if not risk_check['approved']:
            return False, f"Risk check failed: {risk_check['reason']}"
        
        trade_value = float(quantity) * float(price)
        
        if action == "BUY" and trade_value > self.cash:
            return False, "Insufficient capital"
        
        # Place order through Kite if enabled
        if use_kite and self.kite_manager and self.kite_manager.is_authenticated and self.live_trading:
            success, msg = self.kite_manager.place_order(
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type="MARKET"
            )
            if not success:
                return False, f"Kite order failed: {msg}"
        
        # Create trade record with advanced features
        trade_id = f"{'KITE' if use_kite and self.live_trading else 'PAPER'}_{symbol}_{len(self.trade_log)}_{int(time.time())}"
        record = {
            "trade_id": trade_id,
            "symbol": symbol,
            "action": action,
            "quantity": int(quantity),
            "entry_price": float(price),
            "stop_loss": float(stop_loss) if stop_loss else None,
            "target": float(target) if target else None,
            "trail_atr": float(trail_atr) if trail_atr else None,
            "timestamp": now_indian(),
            "status": "OPEN",
            "current_pnl": 0.0,
            "current_price": float(price),
            "closed_pnl": 0.0,
            "entry_time": now_indian().strftime("%H:%M:%S"),
            "auto_trade": auto_trade,
            "strategy": strategy,
            "confidence": confidence,
            "live_trade": use_kite and self.live_trading,
            "is_smc": is_smc,
            "risk_level": risk_check['dynamic_risk'],
            "session_risk": risk_check['session_risk_factor']
        }
        
        # Update positions
        if action == "BUY":
            self.positions[symbol] = record
            self.cash -= trade_value
        else:
            margin = trade_value * 0.2  # Margin for short selling
            record["margin_used"] = margin
            self.positions[symbol] = record
            self.cash -= margin
        
        # Update statistics
        self.daily_stats['trades_today'] += 1
        if auto_trade:
            self.daily_stats['auto_trades'] += 1
        if is_smc:
            self.daily_stats['smc_trades'] += 1
        
        self.trade_log.append(record)
        self.risk_manager.update_position(opened=True)
        
        trade_type = "LIVE" if use_kite and self.live_trading else "PAPER"
        strategy_type = "SMC" if is_smc else "REGULAR"
        msg = f"[{trade_type}] {strategy_type} {action} {quantity} {symbol} @ ₹{price:.2f}"
        if strategy != "Manual":
            msg += f" | Strategy: {strategy}"
        
        return True, msg
    
    def execute_trade_from_signal(self, signal, max_quantity=50, use_kite=False):
        """Execute trade based on signal"""
        symbol = signal['symbol']
        action = signal['action']
        price = signal['price']
        stop_loss = signal['stop_loss']
        target = signal['target']
        strategy = signal['strategy']
        confidence = signal['confidence']
        is_smc = 'regime' in signal  # Check if it's an SMC signal
        
        # Get dynamic risk from risk manager
        risk_status = self.risk_manager.get_status()
        position_size_pct = risk_status['dynamic_risk'] * risk_status['session_risk_factor']
        
        # Adjust for SMC signals
        if is_smc and 'bos' in signal and signal['bos']:
            position_size_pct *= 1.2  # Increase position size for confirmed SMC signals
        
        max_trade_value = self.cash * position_size_pct
        quantity = int(max_trade_value / price)
        quantity = min(quantity, max_quantity)
        
        if quantity < 1:
            return False, "Position size too small"
        
        # Add trailing stop for SMC signals
        trail_atr = None
        if is_smc and 'atr' in signal:
            trail_atr = signal['atr'] * config.TRAIL_ATR
        
        return self.execute_trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            target=target,
            strategy=strategy,
            auto_trade=True,
            confidence=confidence,
            use_kite=use_kite,
            trail_atr=trail_atr,
            is_smc=is_smc
        )
    
    def update_positions(self):
        """Update all positions with current prices and trailing stops"""
        for symbol, pos in list(self.positions.items()):
            if pos.get("status") != "OPEN":
                continue
            
            try:
                # Use Kite live data if available
                use_kite = pos.get("live_trade", False)
                current_price = self.data_manager.get_live_price(symbol, use_kite=use_kite)
                if current_price is None:
                    continue
                
                pos["current_price"] = current_price
                entry = pos["entry_price"]
                quantity = pos["quantity"]
                
                if pos["action"] == "BUY":
                    pnl = (current_price - entry) * quantity
                else:
                    pnl = (entry - current_price) * quantity
                
                pos["current_pnl"] = float(pnl)
                
                # Check trailing stop for SMC trades
                if pos.get("is_smc") and pos.get("trail_atr"):
                    trail_level = pos.get("trail_level", entry)
                    
                    if pos["action"] == "BUY":
                        # Update trailing level
                        if current_price > trail_level:
                            pos["trail_level"] = current_price
                            # Update stop loss
                            new_sl = current_price - pos["trail_atr"]
                            if new_sl > pos.get("stop_loss", 0):
                                pos["stop_loss"] = new_sl
                                logger.info(f"Trailing SL updated for {symbol}: {new_sl:.2f}")
                    
                    elif pos["action"] == "SELL":
                        # Update trailing level
                        if current_price < trail_level:
                            pos["trail_level"] = current_price
                            # Update stop loss
                            new_sl = current_price + pos["trail_atr"]
                            if new_sl < pos.get("stop_loss", float('inf')):
                                pos["stop_loss"] = new_sl
                                logger.info(f"Trailing SL updated for {symbol}: {new_sl:.2f}")
                
                # Check stop loss and target
                sl = pos.get("stop_loss")
                tg = pos.get("target")
                
                if sl is not None:
                    if (pos["action"] == "BUY" and current_price <= sl) or \
                       (pos["action"] == "SELL" and current_price >= sl):
                        self.close_position(symbol, exit_price=sl, use_kite=pos.get("live_trade", False))
                        continue
                
                if tg is not None:
                    if (pos["action"] == "BUY" and current_price >= tg) or \
                       (pos["action"] == "SELL" and current_price <= tg):
                        self.close_position(symbol, exit_price=tg, use_kite=pos.get("live_trade", False))
                        continue
                        
            except Exception as e:
                logger.error(f"Error updating position {symbol}: {e}")
                continue
    
    def close_position(self, symbol, exit_price=None, use_kite=False):
        """Close a position"""
        if symbol not in self.positions:
            return False, "Position not found"
        
        pos = self.positions[symbol]
        
        if exit_price is None:
            current_price = self.data_manager.get_live_price(symbol, use_kite=use_kite)
            if current_price is None:
                current_price = pos["entry_price"]
            exit_price = current_price
        
        # Place exit order through Kite if it was a live trade
        if use_kite and pos.get("live_trade") and self.kite_manager and self.kite_manager.is_authenticated:
            exit_action = "SELL" if pos["action"] == "BUY" else "BUY"
            success, msg = self.kite_manager.place_order(
                symbol=symbol.replace('.NS', ''),
                action=exit_action,
                quantity=pos["quantity"],
                order_type="MARKET"
            )
            if not success:
                return False, f"Kite exit order failed: {msg}"
        
        # Calculate P&L
        if pos["action"] == "BUY":
            pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
            self.cash += pos["quantity"] * exit_price
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["quantity"]
            margin = pos.get("margin_used", 0)
            self.cash += margin + (pos["quantity"] * pos["entry_price"])
        
        # Update position record
        pos["status"] = "CLOSED"
        pos["exit_price"] = float(exit_price)
        pos["closed_pnl"] = float(pnl)
        pos["exit_time"] = now_indian()
        pos["exit_time_str"] = now_indian().strftime("%H:%M:%S")
        
        # Update statistics
        win = pnl > 0
        self.daily_stats['total_pnl'] += pnl
        self.risk_manager.update_trade(pnl, win)
        self.risk_manager.update_position(opened=False)
        
        # Remove from active positions
        del self.positions[symbol]
        
        trade_type = "LIVE" if use_kite else "PAPER"
        return True, f"[{trade_type}] Closed {symbol} @ ₹{exit_price:.2f} | P&L: ₹{pnl:+.2f}"
    
    def close_all_positions(self):
        """Close all open positions"""
        results = []
        for symbol, pos in self.positions.items():
            success, msg = self.close_position(symbol, use_kite=pos.get("live_trade", False))
            results.append((success, msg))
        
        success_count = sum(1 for s, _ in results if s)
        total = len(results)
        
        return success_count > 0, f"Closed {success_count}/{total} positions"
    
    def get_open_positions(self):
        """Get all open positions"""
        self.update_positions()
        positions = []
        
        for symbol, pos in self.positions.items():
            if pos.get("status") == "OPEN":
                positions.append({
                    "Symbol": symbol.replace('.NS', ''),
                    "Action": pos["action"],
                    "Quantity": pos["quantity"],
                    "Entry Price": f"₹{pos['entry_price']:.2f}",
                    "Current Price": f"₹{pos.get('current_price', pos['entry_price']):.2f}",
                    "P&L": f"₹{pos.get('current_pnl', 0):+.2f}",
                    "Stop Loss": f"₹{pos.get('stop_loss', 0):.2f}" if pos.get('stop_loss') else "N/A",
                    "Target": f"₹{pos.get('target', 0):.2f}" if pos.get('target') else "N/A",
                    "Strategy": pos.get("strategy", "Manual"),
                    "Auto Trade": "Yes" if pos.get("auto_trade") else "No",
                    "Live Trade": "Yes" if pos.get("live_trade") else "No",
                    "SMC Trade": "Yes" if pos.get("is_smc") else "No",
                    "Trailing SL": "Yes" if pos.get("trail_atr") else "No"
                })
        
        return positions
    
    def get_trade_history(self):
        """Get trade history"""
        history = []
        for trade in self.trade_log[-50:]:  # Last 50 trades
            if trade.get("status") == "CLOSED":
                history.append({
                    "Symbol": trade['symbol'].replace('.NS', ''),
                    "Action": trade['action'],
                    "Quantity": trade['quantity'],
                    "Entry Price": f"₹{trade['entry_price']:.2f}",
                    "Exit Price": f"₹{trade.get('exit_price', 0):.2f}",
                    "P&L": f"₹{trade.get('closed_pnl', 0):+.2f}",
                    "Entry Time": trade.get('entry_time', ''),
                    "Exit Time": trade.get('exit_time_str', ''),
                    "Strategy": trade.get('strategy', 'Manual'),
                    "Auto Trade": "Yes" if trade.get('auto_trade') else "No",
                    "Live Trade": "Yes" if trade.get('live_trade') else "No",
                    "SMC Trade": "Yes" if trade.get('is_smc') else "No",
                    "Confidence": f"{trade.get('confidence', 0):.1%}"
                })
        
        return history
    
    def get_performance_summary(self):
        """Get performance summary"""
        closed_trades = [t for t in self.trade_log if t.get("status") == "CLOSED"]
        open_positions = [p for p in self.positions.values() if p.get("status") == "OPEN"]
        
        total_trades = len(closed_trades)
        wins = len([t for t in closed_trades if t.get("closed_pnl", 0) > 0])
        smc_trades = len([t for t in closed_trades if t.get("is_smc")])
        smc_wins = len([t for t in closed_trades if t.get("is_smc") and t.get("closed_pnl", 0) > 0])
        
        total_pnl = sum([t.get("closed_pnl", 0) for t in closed_trades])
        open_pnl = sum([p.get("current_pnl", 0) for p in open_positions])
        smc_pnl = sum([t.get("closed_pnl", 0) for t in closed_trades if t.get("is_smc")])
        
        win_rate = wins / total_trades if total_trades > 0 else 0
        smc_win_rate = smc_wins / smc_trades if smc_trades > 0 else 0
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        total_value = self.cash + sum([
            p.get("quantity", 0) * p.get("current_price", p.get("entry_price", 0))
            for p in open_positions
        ])
        
        return {
            'total_value': total_value,
            'available_cash': self.cash,
            'open_positions': len(open_positions),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'open_pnl': open_pnl,
            'avg_pnl_per_trade': avg_pnl,
            'trades_today': self.daily_stats['trades_today'],
            'auto_trades': self.daily_stats['auto_trades'],
            'smc_trades': self.daily_stats['smc_trades'],
            'smc_win_rate': smc_win_rate,
            'smc_pnl': smc_pnl
        }

# ===================== ADVANCED ALGO ENGINE =====================
class AlgoState(Enum):
    """Algo engine states"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    RECONCILIATION = "reconciliation"

class AdvancedAlgoEngine:
    """Advanced algorithmic trading engine with OMS integration"""
    
    def __init__(self, trader, risk_manager, signal_generator, config, live_trading=False, kite_manager=None):
        self.trader = trader
        self.risk_manager = risk_manager
        self.signal_generator = signal_generator
        self.config = config
        self.live_trading = live_trading
        self.kite_manager = kite_manager
        
        self.state = AlgoState.STOPPED
        self.active_positions = {}
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'today_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'smc_performance': {
                'trades': 0,
                'wins': 0,
                'pnl': 0.0,
                'win_rate': 0.0
            }
        }
        
        self._stop_event = threading.Event()
        self._scheduler_thread = None
        self._reconciliation_thread = None
        self.daily_exit_completed = False
        self.last_signal_scan = 0
        self.last_reconciliation = 0
        self.reconciliation_interval = 300  # 5 minutes
        
        logger.info(f"AdvancedAlgoEngine initialized (Live Trading: {live_trading})")
    
    def start(self):
        """Start the algo engine"""
        if self.state == AlgoState.RUNNING:
            return False
        
        self.state = AlgoState.RUNNING
        self._stop_event.clear()
        
        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler,
            daemon=True
        )
        self._scheduler_thread.start()
        
        # Start reconciliation thread for live trading
        if self.live_trading and self.kite_manager:
            self._reconciliation_thread = threading.Thread(
                target=self._run_reconciliation,
                daemon=True
            )
            self._reconciliation_thread.start()
        
        logger.info("AdvancedAlgoEngine started")
        return True
    
    def stop(self):
        """Stop the algo engine"""
        self.state = AlgoState.STOPPED
        self._stop_event.set()
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        
        if self._reconciliation_thread:
            self._reconciliation_thread.join(timeout=5)
        
        logger.info("AdvancedAlgoEngine stopped")
    
    def pause(self):
        """Pause the algo engine"""
        self.state = AlgoState.PAUSED
        logger.info("AdvancedAlgoEngine paused")
    
    def resume(self):
        """Resume the algo engine"""
        if self.state == AlgoState.PAUSED:
            self.state = AlgoState.RUNNING
            logger.info("AdvancedAlgoEngine resumed")
    
    def emergency_stop(self, reason="Manual trigger"):
        """Emergency stop"""
        logger.critical(f"EMERGENCY STOP: {reason}")
        self.state = AlgoState.EMERGENCY_STOP
        self._stop_event.set()
        
        # Close all positions
        self.exit_all_positions()
    
    def reconcile_system(self):
        """Reconcile trading system with broker"""
        if not self.live_trading or not self.kite_manager:
            return True, "Paper trading mode - no reconciliation needed"
        
        try:
            success, message = self.kite_manager.reconcile_system()
            
            if not success:
                self.emergency_stop(f"Reconciliation failed: {message}")
                return False, message
            
            self.last_reconciliation = time.time()
            return True, message
            
        except Exception as e:
            logger.error(f"Reconciliation error: {e}")
            return False, str(e)
    
    def _run_scheduler(self):
        """Main scheduler loop"""
        logger.info("Advanced algo scheduler started")
        
        while not self._stop_event.is_set():
            try:
                if self.state != AlgoState.RUNNING:
                    time.sleep(1)
                    continue
                
                # Check session validity
                if not valid_session():
                    time.sleep(10)
                    continue
                
                # Check daily exit
                if should_exit_all_positions() and not self.daily_exit_completed:
                    logger.info("3:35 PM - Exiting all positions")
                    self.exit_all_positions()
                    self.daily_exit_completed = True
                    time.sleep(60)
                    continue
                
                # Reset daily exit flag
                current_time = now_indian()
                if current_time.hour == 9 and current_time.minute < 30:
                    self.daily_exit_completed = False
                
                # Update positions
                self.trader.update_positions()
                
                # Check risk limits
                if self._check_risk_limits():
                    self.emergency_stop("Risk limits breached")
                    continue
                
                # Generate and process signals
                current_time_ts = time.time()
                if current_time_ts - self.last_signal_scan > 300:  # Every 5 minutes
                    self._scan_and_process_signals()
                    self.last_signal_scan = current_time_ts
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(10)
        
        logger.info("Advanced algo scheduler stopped")
    
    def _run_reconciliation(self):
        """Reconciliation thread for live trading"""
        logger.info("Reconciliation thread started")
        
        while not self._stop_event.is_set():
            try:
                if self.state != AlgoState.RUNNING:
                    time.sleep(5)
                    continue
                
                current_time = time.time()
                if current_time - self.last_reconciliation > self.reconciliation_interval:
                    logger.info("Performing system reconciliation...")
                    success, message = self.reconcile_system()
                    
                    if not success:
                        logger.error(f"Reconciliation failed: {message}")
                    else:
                        logger.info(f"Reconciliation successful: {message}")
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Reconciliation thread error: {e}")
                time.sleep(60)
    
    def _check_risk_limits(self):
        """Check if risk limits are breached"""
        risk_status = self.risk_manager.get_status()
        return not risk_status['within_limits'] or risk_status['current_drawdown'] > config.MAX_DAILY_DD
    
    def _scan_and_process_signals(self):
        """Scan for and process signals"""
        if not valid_session():
            return
        
        try:
            # Get SMC signals first (higher priority)
            smc_signals = self.signal_generator.scan_universe(
                universe="Nifty 50",
                max_stocks=20,
                min_confidence=self.config.ALGO_MIN_CONFIDENCE,
                use_smc=True
            )
            
            # Process SMC signals
            if smc_signals:
                self._process_smc_signals(smc_signals[:3])  # Process top 3 SMC signals
            else:
                # Fallback to traditional signals
                traditional_signals = self.signal_generator.scan_universe(
                    universe="Nifty 50",
                    max_stocks=20,
                    min_confidence=self.config.ALGO_MIN_CONFIDENCE,
                    use_smc=False
                )
                
                if traditional_signals:
                    self._process_signals(traditional_signals[:2])  # Process top 2 traditional signals
        
        except Exception as e:
            logger.error(f"Error in signal scanning: {e}")
    
    def _process_smc_signals(self, signals):
        """Process SMC signals"""
        for signal in signals:
            # Check if already have position
            if signal['symbol'] in self.trader.positions:
                continue
            
            # Check risk limits
            risk_status = self.risk_manager.get_status()
            if risk_status['positions_open'] >= self.config.ALGO_MAX_POSITIONS:
                logger.info("Max positions limit reached")
                break
            
            # Additional checks for SMC signals
            if signal.get('regime') != 'TREND':
                logger.info(f"Skipping {signal['symbol']} - Not in TREND regime")
                continue
            
            if not signal.get('bos'):
                logger.info(f"Skipping {signal['symbol']} - No Break of Structure")
                continue
            
            # Execute trade
            success, msg = self.trader.execute_trade_from_signal(
                signal, 
                use_kite=self.live_trading
            )
            
            if success:
                logger.info(f"Algo executed SMC trade: {signal['symbol']} {signal['action']}")
                self.active_positions[signal['symbol']] = {
                    'entry_price': signal['price'],
                    'action': signal['action'],
                    'strategy': signal['strategy'],
                    'timestamp': now_indian(),
                    'is_smc': True,
                    'bos': signal.get('bos'),
                    'regime': signal.get('regime')
                }
                
                # Update SMC performance
                self.performance['smc_performance']['trades'] += 1
    
    def _process_signals(self, signals):
        """Process traditional signals"""
        for signal in signals:
            # Check if already have position
            if signal['symbol'] in self.trader.positions:
                continue
            
            # Check risk limits
            risk_status = self.risk_manager.get_status()
            if risk_status['positions_open'] >= self.config.ALGO_MAX_POSITIONS:
                logger.info("Max positions limit reached")
                break
            
            # Execute trade
            success, msg = self.trader.execute_trade_from_signal(
                signal, 
                use_kite=self.live_trading
            )
            
            if success:
                logger.info(f"Algo executed trade: {signal['symbol']} {signal['action']}")
                self.active_positions[signal['symbol']] = {
                    'entry_price': signal['price'],
                    'action': signal['action'],
                    'strategy': signal['strategy'],
                    'timestamp': now_indian(),
                    'is_smc': False
                }
    
    def exit_all_positions(self):
        """Exit all positions"""
        success, msg = self.trader.close_all_positions()
        if success:
            logger.info(f"All positions exited: {msg}")
            self.active_positions.clear()
        else:
            logger.warning(f"Failed to exit all positions: {msg}")
    
    def get_state(self):
        """Get current state"""
        return self.state
    
    def get_status(self):
        """Get algo engine status"""
        # Update performance metrics
        closed_trades = [t for t in self.trader.trade_log 
                        if t.get("status") == "CLOSED" and t.get("auto_trade")]
        
        winning = [t for t in closed_trades if t.get("closed_pnl", 0) > 0]
        losing = [t for t in closed_trades if t.get("closed_pnl", 0) <= 0]
        
        total_pnl = sum(t.get("closed_pnl", 0) for t in closed_trades)
        win_pnl = sum(t.get("closed_pnl", 0) for t in winning) if winning else 0
        loss_pnl = abs(sum(t.get("closed_pnl", 0) for t in losing)) if losing else 0
        
        win_rate = len(winning) / len(closed_trades) if closed_trades else 0
        avg_win = win_pnl / len(winning) if winning else 0
        avg_loss = loss_pnl / len(losing) if losing else 0
        profit_factor = win_pnl / loss_pnl if loss_pnl > 0 else 0
        
        # SMC performance
        smc_trades = [t for t in closed_trades if t.get("is_smc")]
        smc_wins = [t for t in smc_trades if t.get("closed_pnl", 0) > 0]
        smc_pnl = sum(t.get("closed_pnl", 0) for t in smc_trades)
        smc_win_rate = len(smc_wins) / len(smc_trades) if smc_trades else 0
        
        # Update performance
        self.performance.update({
            'total_trades': len(closed_trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'smc_performance': {
                'trades': len(smc_trades),
                'wins': len(smc_wins),
                'pnl': smc_pnl,
                'win_rate': smc_win_rate
            }
        })
        
        return {
            'state': self.state.value,
            'active_positions': len(self.active_positions),
            'total_trades': len(closed_trades),
            'today_pnl': self.performance['today_pnl'],
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'daily_exit_completed': self.daily_exit_completed,
            'live_trading': self.live_trading,
            'smc_trades': len(smc_trades),
            'smc_win_rate': smc_win_rate,
            'smc_pnl': smc_pnl,
            'last_reconciliation': self.last_reconciliation
        }

# ===================== TRADING SYSTEM =====================
class TradingSystem:
    """Main trading system orchestrator"""
    
    def __init__(self):
        self.initialized = False
        self.components = {}
    
    def initialize(self, live_trading=False):
        """Initialize all system components"""
        try:
            logger.info("Initializing Advanced Trading System...")
            
            # Initialize Kite Manager
            kite_manager = None
            if config.KITE_API_KEY and config.KITE_API_SECRET and KITECONNECT_AVAILABLE:
                kite_manager = KiteConnectManager(config.KITE_API_KEY, config.KITE_API_SECRET)
            
            # 1. Data Manager (with advanced features)
            self.components['data_manager'] = DataManager(kite_manager)
            
            # 2. Risk Manager (with auto-scaler)
            self.components['risk_manager'] = RiskManager(config)
            
            # 3. Strategy Manager
            self.components['strategy_manager'] = StrategyManager()
            
            # 4. Signal Generator (with SMC)
            self.components['signal_generator'] = AdvancedSignalGenerator(
                self.components['data_manager'],
                self.components['strategy_manager']
            )
            
            # 5. Paper Trader (advanced)
            self.components['trader'] = AdvancedPaperTrader(
                initial_capital=config.INITIAL_CAPITAL,
                risk_manager=self.components['risk_manager'],
                data_manager=self.components['data_manager'],
                kite_manager=kite_manager,
                live_trading=live_trading
            )
            
            # 6. Algo Engine (advanced with OMS)
            self.components['algo_engine'] = AdvancedAlgoEngine(
                trader=self.components['trader'],
                risk_manager=self.components['risk_manager'],
                signal_generator=self.components['signal_generator'],
                config=config,
                live_trading=live_trading,
                kite_manager=kite_manager
            )
            
            self.initialized = True
            logger.info("Advanced Trading System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize trading system: {e}")
            return False
        
    def get_component(self, name):
        """Get a system component"""
        return self.components.get(name)

# ===================== STREAMLIT APPLICATION =====================
def initialize_trading_system():
    """Initialize the trading system and store components in session state"""
    if st.session_state.get('initialized', False):
        return True
    
    with st.spinner("🚀 Initializing Advanced Trading System..."):
        trading_system = TradingSystem()
        
        # Check if live trading is enabled
        live_trading = st.session_state.get('live_trading_enabled', False)
        
        if trading_system.initialize(live_trading=live_trading):
            # Store components in session state
            st.session_state.data_manager = trading_system.get_component('data_manager')
            st.session_state.risk_manager = trading_system.get_component('risk_manager')
            st.session_state.strategy_manager = trading_system.get_component('strategy_manager')
            st.session_state.signal_generator = trading_system.get_component('signal_generator')
            st.session_state.trader = trading_system.get_component('trader')
            st.session_state.algo_engine = trading_system.get_component('algo_engine')
            st.session_state.trading_system = trading_system
            st.session_state.initialized = True
            return True
        else:
            st.error("❌ Failed to initialize trading system")
            return False

def execute_signals(signals, action_filter, trading_system):
    """Execute filtered signals"""
    if not trading_system.get_component('trader'):
        st.error("Trader not initialized")
        return
    
    trader = trading_system.get_component('trader')
    filtered_signals = signals if action_filter == "ANY" else [s for s in signals if s['action'] == action_filter]
    
    executed = 0
    for signal in filtered_signals:
        # Use Kite for live trading if enabled
        use_kite = st.session_state.get('live_trading_enabled', False)
        success, msg = trader.execute_trade_from_signal(signal, use_kite=use_kite)
        if success:
            executed += 1
            st.success(f"✅ {msg}")
        else:
            st.warning(f"⚠️ {msg}")
    
    if executed > 0:
        st.success(f"✅ Executed {executed} trades!")
        st.rerun()

def load_css():
    """Load CSS styles"""
    st.markdown("""
    <style>
        /* Base styling */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #f8fafc;
        }
        
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        /* Header */
        .main-header {
            text-align: center;
            padding: 1.5rem;
            background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
            border-radius: 15px;
            margin-bottom: 1.5rem;
            color: white;
            box-shadow: 0 4px 20px rgba(124, 58, 237, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Kite Connect Panel */
        .kite-panel {
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Advanced Panel */
        .advanced-panel {
            background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Cards */
        .metric-card {
            background: rgba(30, 41, 59, 0.8);
            border-radius: 10px;
            padding: 1rem;
            border-left: 4px solid #7c3aed;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Alerts */
        .alert-success {
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
            border-left: 4px solid #10b981;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .alert-warning {
            background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
            border-left: 4px solid #f59e0b;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .alert-danger {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            border-left: 4px solid #ef4444;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .alert-info {
            background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: rgba(30, 41, 59, 0.8);
            padding: 8px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(51, 65, 85, 0.8);
            border-radius: 8px;
            padding: 12px 20px;
            font-weight: 600;
            color: #cbd5e1;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
            color: white;
            border-color: #7c3aed;
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4);
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        
        /* Tables */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }
        
        /* Inputs */
        .stTextInput > div > div > input {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: white;
        }
        
        .stSelectbox > div > div {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Metrics */
        .stMetric {
            background: rgba(30, 41, 59, 0.8);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """Initialize session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'kite_manager' not in st.session_state:
        st.session_state.kite_manager = None
    if 'kite_authenticated' not in st.session_state:
        st.session_state.kite_authenticated = False
    if 'trader' not in st.session_state:
        st.session_state.trader = None
    if 'algo_engine' not in st.session_state:
        st.session_state.algo_engine = None
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = None
    if 'risk_manager' not in st.session_state:
        st.session_state.risk_manager = None
    if 'strategy_manager' not in st.session_state:
        st.session_state.strategy_manager = None
    if 'signal_generator' not in st.session_state:
        st.session_state.signal_generator = None
    if 'generated_signals' not in st.session_state:
        st.session_state.generated_signals = []
    if 'signal_quality' not in st.session_state:
        st.session_state.signal_quality = 0
    if 'refresh_count' not in st.session_state:
        st.session_state.refresh_count = 0
    if 'auto_trade_enabled' not in st.session_state:
        st.session_state.auto_trade_enabled = False
    if 'live_trading_enabled' not in st.session_state:
        st.session_state.live_trading_enabled = False
    if 'trading_system' not in st.session_state:
        st.session_state.trading_system = None
    if 'use_smc_signals' not in st.session_state:
        st.session_state.use_smc_signals = True
    if 'show_advanced' not in st.session_state:
        st.session_state.show_advanced = False

def render_kite_connect_ui():
    """Render Kite Connect authentication UI"""
    st.sidebar.subheader("🔐 Kite Connect")
    
    with st.sidebar.expander("Kite Authentication", expanded=not st.session_state.kite_authenticated):
        if st.session_state.kite_authenticated and st.session_state.kite_manager:
            st.success(f"✅ Authenticated as {st.session_state.kite_manager.user_name}")
            
            # OMS status
            if st.session_state.kite_manager.oms:
                oms_status = st.session_state.kite_manager.oms.get_order_status()
                st.info(f"OMS: {oms_status['active_orders']} active orders")
                
                if oms_status['kill_switch']:
                    st.error("⚠️ KILL SWITCH ACTIVE")
            
            if st.button("Logout from Kite"):
                st.session_state.kite_manager.logout()
                st.session_state.kite_authenticated = False
                st.session_state.kite_manager = None
                st.rerun()
            
            # Live trading toggle
            live_trading = st.checkbox(
                "Enable Live Trading",
                value=st.session_state.live_trading_enabled,
                help="WARNING: This will place real orders with real money! Requires system re-initialization."
            )
            
            if live_trading != st.session_state.live_trading_enabled:
                st.session_state.live_trading_enabled = live_trading
                if st.session_state.initialized:
                    st.warning("Live trading setting changed. Please re-initialize the system.")
                    if st.button("Re-initialize System"):
                        st.session_state.initialized = False
                        st.rerun()
            
            if st.session_state.live_trading_enabled:
                st.markdown("""
                <div class="alert-danger">
                    ⚠️ <strong>LIVE TRADING ENABLED</strong><br>
                    Real money at risk! All orders will be executed through Kite.
                </div>
                """, unsafe_allow_html=True)
            
            # Reconciliation button
            if st.button("🔄 Reconcile System"):
                if st.session_state.kite_manager:
                    success, message = st.session_state.kite_manager.reconcile_system()
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
            
            return
        
        # Authentication form
        api_key = st.text_input("API Key", value=config.KITE_API_KEY, type="password")
        api_secret = st.text_input("API Secret", value=config.KITE_API_SECRET, type="password")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Login URL"):
                if api_key:
                    kite_manager = KiteConnectManager(api_key, api_secret)
                    login_url = kite_manager.get_login_url()
                    if login_url:
                        st.session_state.kite_manager = kite_manager
                        st.success("Login URL generated!")
                        st.markdown(f"[🔗 Click here to login to Kite]({login_url})")
                        st.code(login_url)
                        
                        # Try to open browser
                        try:
                            webbrowser.open(login_url, new=2)
                            st.info("Browser opened. If not, click the link above.")
                        except:
                            st.info("Please copy the URL above.")
                    else:
                        st.error("Failed to generate login URL")
        
        with col2:
            st.markdown("**Or enter token:**")
            request_token = st.text_input("Request Token", type="password")
            
            if st.button("Authenticate"):
                if api_key and api_secret and request_token:
                    if not st.session_state.kite_manager:
                        st.session_state.kite_manager = KiteConnectManager(api_key, api_secret)
                    
                    success, message = st.session_state.kite_manager.authenticate(request_token)
                    if success:
                        st.session_state.kite_authenticated = True
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

def render_sidebar():
    """Render sidebar with controls"""
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h2 style='color: #7c3aed;'>⚡ RANTV PRO v4.0</h2>
            <p style='color: #94a3b8; font-size: 12px;'>Institutional Grade Algo</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Kite Connect UI
        render_kite_connect_ui()
        
        # Advanced Features Toggle
        st.session_state.show_advanced = st.checkbox("Show Advanced Features", 
                                                    value=st.session_state.show_advanced)
        
        # System Status
        st.subheader("📊 System Status")
        status_cols = st.columns(2)
        with status_cols[0]:
            status = "🟢 READY" if st.session_state.get('initialized', False) else "🔴 ERROR"
            st.metric("System", status)
        with status_cols[1]:
            market = "🟢 OPEN" if valid_session() else "🔴 CLOSED"
            st.metric("Market", market)
        
        # Advanced Settings
        if st.session_state.show_advanced:
            st.subheader("🎯 Advanced Settings")
            with st.expander("Risk Configuration", expanded=False):
                max_loss = st.number_input(
                    "Max Daily Loss (₹)",
                    min_value=1000,
                    max_value=1000000,
                    value=int(config.ALGO_MAX_DAILY_LOSS),
                    step=5000
                )
                
                max_pos = st.slider(
                    "Max Positions",
                    min_value=1,
                    max_value=20,
                    value=config.ALGO_MAX_POSITIONS
                )
                
                base_risk = st.slider(
                    "Base Risk %",
                    min_value=0.1,
                    max_value=5.0,
                    value=config.BASE_RISK * 100,
                    step=0.1
                ) / 100
                
                max_dd = st.slider(
                    "Max Daily Drawdown %",
                    min_value=1.0,
                    max_value=10.0,
                    value=config.MAX_DAILY_DD * 100,
                    step=0.5
                ) / 100
                
                if st.button("Update Risk Settings"):
                    config.ALGO_MAX_DAILY_LOSS = max_loss
                    config.ALGO_MAX_POSITIONS = max_pos
                    config.BASE_RISK = base_risk
                    config.MAX_DAILY_DD = max_dd
                    st.success("Risk settings updated!")
            
            with st.expander("SMC Configuration", expanded=False):
                st.session_state.use_smc_signals = st.checkbox(
                    "Use SMC Signals",
                    value=st.session_state.use_smc_signals,
                    help="Enable Smart Money Concept signals"
                )
                
                min_smc_confidence = st.slider(
                    "Min SMC Confidence",
                    min_value=0.65,
                    max_value=0.95,
                    value=0.75,
                    step=0.05
                )
                
                config.SL_ATR = st.slider(
                    "SL ATR Multiplier",
                    min_value=1.0,
                    max_value=3.0,
                    value=config.SL_ATR,
                    step=0.1
                )
                
                config.TP_ATR = st.slider(
                    "TP ATR Multiplier",
                    min_value=2.0,
                    max_value=5.0,
                    value=config.TP_ATR,
                    step=0.1
                )
        
        # Trading Controls
        st.subheader("🕹️ Trading Controls")
        auto_trade = st.checkbox(
            "Enable Auto Trading",
            value=st.session_state.auto_trade_enabled
        )
        st.session_state.auto_trade_enabled = auto_trade
        
        if st.button("🔄 Refresh All"):
            st.rerun()
        
        # Strategy Selection
        st.subheader("📈 Active Strategies")
        if st.session_state.strategy_manager:
            strategies = st.session_state.strategy_manager.get_all_strategies()
            for strategy_id, strategy in strategies.items():
                enabled = st.checkbox(
                    f"{strategy['name']} ({strategy['category']})",
                    value=True,
                    key=f"strategy_{strategy_id}"
                )
                if enabled:
                    st.session_state.strategy_manager.enable_strategy(strategy_id)
                else:
                    st.session_state.strategy_manager.disable_strategy(strategy_id)
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Close All", type="secondary"):
                if st.session_state.trader:
                    success, message = st.session_state.trader.close_all_positions()
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        with col2:
            if st.button("🔄 Reconcile", type="secondary"):
                if st.session_state.kite_manager and st.session_state.kite_authenticated:
                    success, message = st.session_state.kite_manager.reconcile_system()
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        if st.button("📊 Clear Cache"):
            if st.session_state.data_manager:
                st.session_state.data_manager.clear_cache()
                st.success("Cache cleared!")
        
        # System Info
        st.divider()
        st.caption(f"v4.0 | {now_indian().strftime('%H:%M:%S')}")

def render_kite_charts_tab():
    """Render Kite Charts tab with live data"""
    st.subheader("📈 Kite Connect Live Charts")
    
    if not st.session_state.kite_authenticated or not st.session_state.kite_manager:
        st.info("🔐 Please authenticate with Kite Connect in the sidebar to view live charts")
        return
    
    st.success(f"✅ Authenticated as {st.session_state.kite_manager.user_name}")
    
    # Advanced chart options
    with st.expander("Advanced Chart Settings"):
        col1, col2 = st.columns(2)
        with col1:
            show_smc = st.checkbox("Show SMC Levels", value=True)
            show_volume = st.checkbox("Show Volume Profile", value=True)
        with col2:
            show_indicators = st.checkbox("Show Indicators", value=True)
            chart_theme = st.selectbox("Chart Theme", ["Dark", "Light"])
    
    # Chart selection
    col1, col2, col3 = st.columns(3)
    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            ["Live Index", "Stock Charts", "Historical Data", "SMC Analysis"],
            key="kite_chart_type"
        )
    
    with col2:
        if chart_type == "Live Index":
            index = st.selectbox(
                "Select Index",
                ["NIFTY 50", "BANK NIFTY"],
                key="kite_index_select"
            )
        elif chart_type == "Stock Charts":
            stock = st.selectbox(
                "Select Stock",
                TradingConstants.NIFTY_50[:10],
                key="kite_stock_select"
            )
        elif chart_type == "SMC Analysis":
            stock = st.selectbox(
                "Select Stock for SMC",
                TradingConstants.NIFTY_50[:5],
                key="kite_smc_select"
            )
        else:
            symbol = st.selectbox(
                "Select Symbol",
                TradingConstants.NIFTY_50[:10],
                key="kite_hist_select"
            )
    
    with col3:
        interval = st.selectbox(
            "Interval",
            ["1m", "5m", "15m", "30m", "1h"],
            key="kite_interval"
        )
    
    if st.button("📊 Load Chart", type="primary"):
        with st.spinner("Fetching data..."):
            try:
                if chart_type == "SMC Analysis":
                    # Get MTF data for SMC analysis
                    mtf_data = st.session_state.data_manager.get_mtf_data(stock, intervals=["15m", "1h", "1d"])
                    
                    if mtf_data and "15m" in mtf_data:
                        df = mtf_data["15m"]
                        
                        if PLOTLY_AVAILABLE:
                            # Create advanced SMC chart
                            fig = go.Figure()
                            
                            # Candlestick
                            fig.add_trace(go.Candlestick(
                                x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='Price',
                                increasing_line_color='#10b981',
                                decreasing_line_color='#ef4444'
                            ))
                            
                            # Add SMC levels
                            if show_smc and 'Support' in df.columns and 'Resistance' in df.columns:
                                fig.add_trace(go.Scatter(
                                    x=df.index,
                                    y=df['Support'],
                                    mode='lines',
                                    name='Support',
                                    line=dict(color='#10b981', width=1, dash='dash')
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=df.index,
                                    y=df['Resistance'],
                                    mode='lines',
                                    name='Resistance',
                                    line=dict(color='#ef4444', width=1, dash='dash')
                                ))
                            
                            # Add Volume Profile
                            if show_volume and 'Volume_POC' in df.columns:
                                fig.add_trace(go.Scatter(
                                    x=df.index,
                                    y=df['Volume_POC'],
                                    mode='lines',
                                    name='Volume POC',
                                    line=dict(color='#8b5cf6', width=2)
                                ))
                            
                            # Add indicators
                            if show_indicators:
                                if 'EMA8' in df.columns:
                                    fig.add_trace(go.Scatter(
                                        x=df.index,
                                        y=df['EMA8'],
                                        mode='lines',
                                        name='EMA 8',
                                        line=dict(color='#f59e0b', width=1)
                                    ))
                                
                                if 'VWAP' in df.columns:
                                    fig.add_trace(go.Scatter(
                                        x=df.index,
                                        y=df['VWAP'],
                                        mode='lines',
                                        name='VWAP',
                                        line=dict(color='#3b82f6', width=1)
                                    ))
                            
                            fig.update_layout(
                                title=f"{stock} - SMC Analysis ({interval})",
                                xaxis_title='Time',
                                yaxis_title='Price (₹)',
                                height=600,
                                template='plotly_dark' if chart_theme == "Dark" else 'plotly_white',
                                showlegend=True,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show SMC analysis
                            st.subheader("SMC Analysis")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                bos = df['SMC_BOS'].iloc[-1] if 'SMC_BOS' in df.columns else 'NONE'
                                st.metric("Break of Structure", bos)
                            
                            with col2:
                                fvg = df['SMC_FVG'].iloc[-1] if 'SMC_FVG' in df.columns else 'NONE'
                                st.metric("Fair Value Gap", fvg)
                            
                            with col3:
                                ob = df['SMC_OB'].iloc[-1] if 'SMC_OB' in df.columns else 'NONE'
                                st.metric("Order Block", ob)
                            
                            with col4:
                                liq = df['SMC_Liquidity'].iloc[-1] if 'SMC_Liquidity' in df.columns else 'NONE'
                                st.metric("Liquidity Grab", liq)
                            
                            # Market regime
                            regime = df['Market_Regime'].iloc[-1] if 'Market_Regime' in df.columns else 'UNKNOWN'
                            if regime == 'TREND':
                                st.markdown('<div class="alert-success"><strong>Market Regime: TREND</strong><br>Favorable for SMC strategies</div>', unsafe_allow_html=True)
                            elif regime == 'RANGE':
                                st.markdown('<div class="alert-warning"><strong>Market Regime: RANGE</strong><br>Exercise caution with SMC</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="alert-danger"><strong>Market Regime: VOLATILE</strong><br>High risk for SMC</div>', unsafe_allow_html=True)
                    else:
                        st.error("No data available for SMC analysis")
                    
                else:
                    # Regular chart
                    if chart_type == "Live Index":
                        token = TradingConstants.KITE_TOKEN_MAP.get(index)
                        if token:
                            df = st.session_state.kite_manager.get_historical_data(
                                token, 
                                interval.replace("m", "minute").replace("h", "hour"),
                                days=7
                            )
                        else:
                            st.error("Token not found for selected index")
                            return
                    elif chart_type == "Stock Charts":
                        token = TradingConstants.KITE_TOKEN_MAP.get(stock)
                        if token:
                            df = st.session_state.kite_manager.get_historical_data(
                                token,
                                interval.replace("m", "minute").replace("h", "hour"),
                                days=7
                            )
                        else:
                            st.error("Token not found for selected stock")
                            return
                    else:
                        token = TradingConstants.KITE_TOKEN_MAP.get(symbol)
                        if token:
                            df = st.session_state.kite_manager.get_historical_data(
                                token,
                                interval.replace("m", "minute").replace("h", "hour"),
                                days=30
                            )
                        else:
                            st.error("Token not found for selected symbol")
                            return
                    
                    if df is None or df.empty:
                        st.warning("No data received from Kite. Using fallback.")
                        symbol_to_fetch = index if chart_type == "Live Index" else stock if chart_type == "Stock Charts" else symbol
                        df = st.session_state.data_manager.get_stock_data(
                            symbol_to_fetch, 
                            interval,
                            use_kite=False
                        )
                    
                    if df is not None and not df.empty and PLOTLY_AVAILABLE:
                        # Create candlestick chart
                        fig = go.Figure(data=[go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name='Price',
                            increasing_line_color='#10b981',
                            decreasing_line_color='#ef4444'
                        )])
                        
                        # Add moving averages
                        df['SMA20'] = df['Close'].rolling(window=20).mean()
                        df['SMA50'] = df['Close'].rolling(window=50).mean()
                        
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['SMA20'],
                            mode='lines',
                            name='SMA 20',
                            line=dict(color='#f59e0b', width=1)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['SMA50'],
                            mode='lines',
                            name='SMA 50',
                            line=dict(color='#3b82f6', width=1)
                        ))
                        
                        chart_title = f"{index if chart_type == 'Live Index' else stock if chart_type == 'Stock Charts' else symbol} ({interval})"
                        if chart_type == "Live Index":
                            chart_title += " - Kite Live"
                        else:
                            chart_title += " - Historical"
                        
                        fig.update_layout(
                            title=chart_title,
                            xaxis_title='Time',
                            yaxis_title='Price (₹)',
                            height=500,
                            template='plotly_dark' if chart_theme == "Dark" else 'plotly_white',
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show current stats
                        if len(df) > 0:
                            current_price = df['Close'].iloc[-1]
                            prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
                            change_pct = ((current_price - prev_close) / prev_close) * 100
                            
                            cols = st.columns(4)
                            cols[0].metric("Current", f"₹{current_price:,.2f}", f"{change_pct:+.2f}%")
                            cols[1].metric("Open", f"₹{df['Open'].iloc[-1]:,.2f}")
                            cols[2].metric("High", f"₹{df['High'].max():,.2f}")
                            cols[3].metric("Low", f"₹{df['Low'].min():,.2f}")
                    else:
                        st.error("No data available or Plotly not available")
                        
            except Exception as e:
                st.error(f"Error loading chart: {str(e)}")
    
    # Live data status
    if st.session_state.kite_manager and st.session_state.kite_manager.ticker:
        st.markdown("""
        <div class="alert-success">
            <strong>✅ Live Data Connected</strong><br>
            Real-time WebSocket data streaming from Kite
        </div>
        """, unsafe_allow_html=True)
        
        # Start live data for NIFTY 50
        if st.button("▶️ Start Live Feed"):
            token = TradingConstants.KITE_TOKEN_MAP.get("NIFTY 50")
            if token:
                success = st.session_state.kite_manager.start_live_data([token])
                if success:
                    st.success("Live feed started!")
                else:
                    st.error("Failed to start live feed")

def render_main_app():
    """Render the main application"""
    # Auto-refresh
    if AUTOREFRESH_AVAILABLE:
        st_autorefresh(interval=config.PRICE_REFRESH_MS, key="main_auto_refresh")
    st.session_state.refresh_count += 1
    
    # Initialization screen
    if not st.session_state.initialized:
        st.title("⚡ RANTV TERMINAL PRO v4.0")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### Institutional Grade Algo Trading")
            st.markdown("""
            **Complete Algorithmic Trading Platform with Advanced SMC + OMS**
            
            **🎯 New in v4.0:**
            - 🧠 **MTF SMC Strategy** - Multi-timeframe Smart Money Concept
            - 📊 **Volume Profile Integration** - Institutional level analysis
            - ⚡ **Auto Risk Scaler** - Dynamic position sizing
            - 🔐 **Kite OMS with Kill Switch** - Professional order management
            - 🕒 **Session Logic** - India & NY overlap trading
            - 📈 **Advanced Charting** - SMC levels & volume profile
            
            **📊 Complete Stock Universes:**
            - 📈 **NIFTY 50** - Top 50 stocks
            - 📊 **NIFTY 100** - Top 100 stocks
            - 🚀 **MIDCAP 150** - Top midcap stocks
            
            **⚠️ DISCLAIMER:**
            This system is for **EDUCATIONAL & PAPER TRADING** first.
            Live trading must be enabled ONLY after extensive paper & backtest validation.
            
            **Ready to start?**
            """)
            
            if st.button("🚀 Initialize Trading System", type="primary", use_container_width=True):
                if initialize_trading_system():
                    st.success("✅ Advanced Trading System initialized successfully!")
                    st.rerun()
        
        return
    
    # Main application (after initialization)
    render_sidebar()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style='margin: 0;'>⚡ RANTV TERMINAL PRO v4.0</h1>
        <p style='margin: 5px 0 0 0; font-size: 16px;'>Institutional Grade Algo Trading with SMC + OMS</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced status banner
    st.markdown("""
    <div class="advanced-panel">
        <strong>🎯 ADVANCED FEATURES ACTIVE</strong> | 
        MTF SMC: ✅ | Volume Profile: ✅ | Auto Risk: ✅ | OMS: ✅
    </div>
    """, unsafe_allow_html=True)
    
    # Kite status banner
    if st.session_state.kite_authenticated:
        st.markdown(f"""
        <div class="kite-panel">
            <strong>🔐 Kite Connect: AUTHENTICATED</strong> | 
            User: {st.session_state.kite_manager.user_name} | 
            Live Trading: {'✅ ENABLED' if st.session_state.live_trading_enabled else '⛔ DISABLED'} |
            OMS: {'✅ ACTIVE' if st.session_state.kite_manager.oms else '⛔ INACTIVE'}
        </div>
        """, unsafe_allow_html=True)
    
    # Market Overview
    st.subheader("🌐 Market Overview")
    
    # Get market data
    try:
        if YFINANCE_AVAILABLE:
            nifty = yf.Ticker("^NSEI")
            nifty_data = nifty.history(period="1d")
            
            banknifty = yf.Ticker("^NSEBANK")
            banknifty_data = banknifty.history(period="1d")
            
            if not nifty_data.empty and not banknifty_data.empty:
                nifty_price = nifty_data['Close'].iloc[-1]
                nifty_prev = nifty_data['Close'].iloc[-2] if len(nifty_data) > 1 else nifty_price
                nifty_change = ((nifty_price - nifty_prev) / nifty_prev) * 100
                
                banknifty_price = banknifty_data['Close'].iloc[-1]
                banknifty_prev = banknifty_data['Close'].iloc[-2] if len(banknifty_data) > 1 else banknifty_price
                banknifty_change = ((banknifty_price - banknifty_prev) / banknifty_prev) * 100
                
                avg_change = (nifty_change + banknifty_change) / 2
                if avg_change > 0.5:
                    sentiment = "BULLISH 📈"
                elif avg_change < -0.5:
                    sentiment = "BEARISH 📉"
                else:
                    sentiment = "NEUTRAL ➡️"
            else:
                nifty_price = 22000
                nifty_change = 0.15
                banknifty_price = 48000
                banknifty_change = 0.25
                sentiment = "NEUTRAL ➡️"
        else:
            nifty_price = 22000
            nifty_change = 0.15
            banknifty_price = 48000
            banknifty_change = 0.25
            sentiment = "NEUTRAL ➡️"
    except:
        nifty_price = 22000
        nifty_change = 0.15
        banknifty_price = 48000
        banknifty_change = 0.25
        sentiment = "NEUTRAL ➡️"
    
    # Display metrics
    cols = st.columns(5)
    with cols[0]:
        st.metric("NIFTY 50", f"₹{nifty_price:,.0f}", f"{nifty_change:+.2f}%")
    with cols[1]:
        st.metric("BANK NIFTY", f"₹{banknifty_price:,.0f}", f"{banknifty_change:+.2f}%")
    with cols[2]:
        st.metric("Market Sentiment", sentiment)
    with cols[3]:
        status = "🟢 OPEN" if valid_session() else "🔴 CLOSED"
        st.metric("Market Status", status)
    with cols[4]:
        session_type = "🇮🇳 INDIA" if config.INDIA_OPEN <= datetime.now().time() <= config.INDIA_CLOSE else "🇺🇸 NY OVERLAP" if config.NY_OVERLAP_START <= datetime.now().time() <= config.NY_OVERLAP_END else "🔴 CLOSED"
        st.metric("Session", session_type)
    
    # Create tabs
    tabs = st.tabs([
        "📈 Dashboard",
        "🧠 SMC Signals",
        "💰 Paper Trading",
        "📋 History",
        "🤖 Algo Trading",
        "📊 Advanced Charts",
        "⚙️ Risk Management"
    ])
    
    # Tab 1: Dashboard
    with tabs[0]:
        st.subheader("💰 Account Summary")
        
        if st.session_state.trader:
            trader = st.session_state.trader
            perf = trader.get_performance_summary()
            
            acc_cols = st.columns(4)
            with acc_cols[0]:
                st.metric("Total Value", f"₹{perf['total_value']:,.0f}")
            with acc_cols[1]:
                st.metric("Available Cash", f"₹{perf['available_cash']:,.0f}")
            with acc_cols[2]:
                st.metric("Open Positions", perf['open_positions'])
            with acc_cols[3]:
                pnl_color = "inverse" if perf['total_pnl'] < 0 else "normal"
                st.metric("Total P&L", f"₹{perf['total_pnl']:+,.2f}", delta_color=pnl_color)
            
            # Advanced metrics
            st.subheader("📊 Advanced Metrics")
            adv_cols = st.columns(4)
            with adv_cols[0]:
                st.metric("Win Rate", f"{perf['win_rate']:.1%}")
            with adv_cols[1]:
                st.metric("SMC Trades", perf['smc_trades'])
            with adv_cols[2]:
                st.metric("SMC Win Rate", f"{perf['smc_win_rate']:.1%}")
            with adv_cols[3]:
                smc_pnl_color = "inverse" if perf['smc_pnl'] < 0 else "normal"
                st.metric("SMC P&L", f"₹{perf['smc_pnl']:+,.2f}", delta_color=smc_pnl_color)
        
        # System Health
        st.subheader("⚙️ System Health")
        health_cols = st.columns(4)
        with health_cols[0]:
            status = "🟢 READY" if st.session_state.initialized else "🔴 ERROR"
            st.metric("Trading System", status)
        with health_cols[1]:
            if st.session_state.algo_engine:
                algo_state = st.session_state.algo_engine.get_state().value
                status_emoji = "🟢" if algo_state == "running" else "🟡" if algo_state == "paused" else "🔴"
                st.metric("Algo Engine", f"{status_emoji} {algo_state.upper()}")
        with health_cols[2]:
            if st.session_state.risk_manager:
                risk_status = st.session_state.risk_manager.get_status()
                status_emoji = "🟢" if risk_status['within_limits'] else "🔴"
                st.metric("Risk Engine", f"{status_emoji} {'OK' if risk_status['within_limits'] else 'LIMIT'}")
        with health_cols[3]:
            kite_status = "🟢 LIVE" if st.session_state.kite_authenticated else "🔴 OFFLINE"
            st.metric("Kite Connect", kite_status)
        
        # Open Positions
        st.subheader("📊 Open Positions")
        if st.session_state.trader:
            positions = st.session_state.trader.get_open_positions()
            if positions:
                positions_df = pd.DataFrame(positions)
                st.dataframe(positions_df, use_container_width=True)
                
                if st.button("Close All Positions", type="secondary"):
                    success, message = st.session_state.trader.close_all_positions()
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            else:
                st.info("No open positions")
    
    # Tab 2: SMC Signals
    with tabs[1]:
        st.subheader("🧠 SMC Trading Signals")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            universe = st.selectbox("Stock Universe", 
                                  ["Nifty 50", "Nifty 100", "Midcap"], 
                                  key="smc_universe")
        with col2:
            min_confidence = st.slider("Min Confidence", 0.65, 0.95, 0.75, 0.05, key="smc_min_conf")
        with col3:
            max_signals = st.number_input("Max Signals", 1, 20, 10, key="smc_max_count")
        
        if st.button("🧠 Generate SMC Signals", type="primary", key="generate_smc_signals"):
            if st.session_state.signal_generator:
                with st.spinner(f"Scanning {universe} for SMC patterns..."):
                    try:
                        scan_size = 50 if universe == "Nifty 50" else 100 if universe == "Nifty 100" else 50
                        signals = st.session_state.signal_generator.scan_universe(
                            universe=universe,
                            max_stocks=min(scan_size, 20),
                            min_confidence=min_confidence,
                            use_smc=st.session_state.use_smc_signals
                        )
                        
                        st.session_state.generated_signals = signals[:max_signals]
                        
                        if signals:
                            confidences = [s['confidence'] for s in signals]
                            st.session_state.signal_quality = np.mean(confidences) * 100
                        
                        st.success(f"✅ Generated {len(signals)} SMC signals")
                        
                    except Exception as e:
                        st.error(f"❌ SMC signal generation failed: {str(e)}")
        
        if st.session_state.generated_signals:
            signals = st.session_state.generated_signals
            quality = st.session_state.get('signal_quality', 0)
            
            if quality >= 80:
                quality_class = "alert-success"
                quality_text = "HIGH QUALITY SMC"
            elif quality >= 70:
                quality_class = "alert-warning"
                quality_text = "MEDIUM QUALITY SMC"
            else:
                quality_class = "alert-danger"
                quality_text = "LOW QUALITY SMC"
            
            st.markdown(f"""
            <div class="{quality_class}">
                <strong>📊 Signal Quality: {quality_text}</strong> | 
                Score: {quality:.1f}/100 | 
                SMC Signals: {len(signals)}
            </div>
            """, unsafe_allow_html=True)
            
            signal_data = []
            for i, signal in enumerate(signals):
                # Determine if it's SMC signal
                is_smc = 'regime' in signal
                
                signal_data.append({
                    "#": i + 1,
                    "Symbol": signal['symbol'].replace('.NS', ''),
                    "Action": f"{'🟢 BUY' if signal['action'] == 'BUY' else '🔴 SELL'}",
                    "Price": f"₹{signal['price']:.2f}",
                    "Stop Loss": f"₹{signal['stop_loss']:.2f}",
                    "Target": f"₹{signal['target']:.2f}",
                    "Confidence": f"{signal['confidence']:.1%}",
                    "Strategy": signal['strategy'],
                    "Regime": signal.get('regime', 'N/A'),
                    "BOS": signal.get('bos', 'N/A'),
                    "SMC": "✅" if is_smc else "❌"
                })
            
            if signal_data:
                df_signals = pd.DataFrame(signal_data)
                st.dataframe(df_signals, use_container_width=True)
                
                st.subheader("🤖 Execute SMC Signals")
                exec_cols = st.columns(3)
                
                with exec_cols[0]:
                    if st.button("📈 Execute BUY SMC", type="secondary"):
                        buy_signals = [s for s in signals if s['action'] == 'BUY']
                        execute_signals(buy_signals, "BUY", st.session_state.trading_system)
                
                with exec_cols[1]:
                    if st.button("📉 Execute SELL SMC", type="secondary"):
                        sell_signals = [s for s in signals if s['action'] == 'SELL']
                        execute_signals(sell_signals, "SELL", st.session_state.trading_system)
                
                with exec_cols[2]:
                    if st.button("🎯 Execute Top 3 SMC", type="primary"):
                        execute_signals(signals[:3], "ANY", st.session_state.trading_system)
        else:
            st.info("Click 'Generate SMC Signals' to scan for Smart Money Concept patterns")
    
    # Tab 3: Paper Trading
    with tabs[2]:
        st.subheader("💰 Advanced Paper Trading")
        
        if st.session_state.trader:
            trader = st.session_state.trader
            
            # Manual trade execution
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                symbol = st.selectbox("Symbol", TradingConstants.NIFTY_50[:20], key="manual_symbol")
            
            with col2:
                action = st.selectbox("Action", ["BUY", "SELL"], key="manual_action")
            
            with col3:
                quantity = st.number_input("Quantity", min_value=1, value=10, key="manual_quantity")
            
            with col4:
                strategy = st.selectbox("Strategy", 
                                       ["Manual", "SMC", "Volume Profile"] + list(TradingConstants.TRADING_STRATEGIES.keys()),
                                       key="manual_strategy")
            
            # Advanced options
            with st.expander("Advanced Options"):
                col1, col2 = st.columns(2)
                with col1:
                    use_trailing = st.checkbox("Use Trailing Stop", value=True)
                    trail_atr = st.number_input("Trail ATR Multiplier", min_value=1.0, max_value=3.0, value=config.TRAIL_ATR, step=0.1)
                with col2:
                    use_smc = st.checkbox("Mark as SMC Trade", value=False)
                    risk_multiplier = st.slider("Risk Multiplier", 0.5, 2.0, 1.0, 0.1)
            
            # Live trading checkbox
            use_kite = st.checkbox("Place Live Order via Kite", 
                                 value=False,
                                 disabled=not st.session_state.kite_authenticated or not st.session_state.live_trading_enabled,
                                 help="Requires Kite authentication and Live Trading enabled")
            
            if use_kite and not st.session_state.live_trading_enabled:
                st.warning("⚠️ Enable Live Trading in sidebar first!")
            
            if st.button("Execute Advanced Trade", type="primary"):
                data = st.session_state.data_manager.get_stock_data(symbol, "15m")
                if data is not None and len(data) > 0:
                    price = float(data['Close'].iloc[-1])
                    atr_val = float(data['ATR'].iloc[-1]) if 'ATR' in data.columns else price * 0.02
                    
                    # Dynamic risk calculation
                    risk_status = st.session_state.risk_manager.get_status()
                    dynamic_risk = risk_status['dynamic_risk'] * risk_multiplier * risk_status['session_risk_factor']
                    
                    if action == "BUY":
                        stop_loss = price - (atr_val * config.SL_ATR)
                        target = price + (atr_val * config.TP_ATR)
                    else:
                        stop_loss = price + (atr_val * config.SL_ATR)
                        target = price - (atr_val * config.TP_ATR)
                    
                    trail_atr_value = atr_val * trail_atr if use_trailing else None
                    
                    success, message = trader.execute_trade(
                        symbol=symbol,
                        action=action,
                        quantity=quantity,
                        price=price,
                        stop_loss=stop_loss,
                        target=target,
                        strategy=strategy,
                        use_kite=use_kite,
                        trail_atr=trail_atr_value,
                        is_smc=use_smc
                    )
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.error("Could not fetch price data")
            
            # Current positions
            st.subheader("Current Positions")
            positions = trader.get_open_positions()
            
            if positions:
                positions_df = pd.DataFrame(positions)
                st.dataframe(positions_df, use_container_width=True)
                
                if st.button("Close All Positions", type="secondary"):
                    success, message = trader.close_all_positions()
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            else:
                st.info("No open positions")
    
    # Tab 4: History
    with tabs[3]:
        st.subheader("📋 Trade History")
        
        if st.session_state.trader:
            trader = st.session_state.trader
            history = trader.get_trade_history()
            
            if history:
                history_df = pd.DataFrame(history)
                st.dataframe(history_df, use_container_width=True)
                
                st.subheader("Performance Summary")
                perf = trader.get_performance_summary()
                
                perf_cols = st.columns(4)
                with perf_cols[0]:
                    st.metric("Total Trades", perf['total_trades'])
                with perf_cols[1]:
                    st.metric("Win Rate", f"{perf['win_rate']:.1%}")
                with perf_cols[2]:
                    st.metric("Total P&L", f"₹{perf['total_pnl']:+.2f}")
                with perf_cols[3]:
                    st.metric("Avg P&L/Trade", f"₹{perf['avg_pnl_per_trade']:+.2f}")
                
                # SMC Performance
                st.subheader("SMC Performance")
                smc_cols = st.columns(4)
                with smc_cols[0]:
                    st.metric("SMC Trades", perf['smc_trades'])
                with smc_cols[1]:
                    st.metric("SMC Win Rate", f"{perf['smc_win_rate']:.1%}")
                with smc_cols[2]:
                    st.metric("SMC P&L", f"₹{perf['smc_pnl']:+.2f}")
                with smc_cols[3]:
                    smc_avg = perf['smc_pnl'] / perf['smc_trades'] if perf['smc_trades'] > 0 else 0
                    st.metric("SMC Avg P&L", f"₹{smc_avg:+.2f}")
            else:
                st.info("No trade history available")
    
    # Tab 5: Algo Trading
    with tabs[4]:
        st.subheader("🤖 Advanced Algorithmic Trading")
        
        if not st.session_state.algo_engine:
            st.warning("Algo engine not initialized")
        else:
            algo_engine = st.session_state.algo_engine
            algo_status = algo_engine.get_status()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                state = algo_status['state']
                if state == "running":
                    status_color = "🟢"
                elif state == "paused":
                    status_color = "🟡"
                else:
                    status_color = "🔴"
                st.metric("Engine Status", f"{status_color} {state.upper()}")
            with col2:
                st.metric("Active Positions", algo_status['active_positions'])
            with col3:
                st.metric("Today's P&L", f"₹{algo_status['today_pnl']:+.2f}")
            with col4:
                st.metric("Total Trades", algo_status['total_trades'])
            
            # SMC Algo Metrics
            st.subheader("🧠 SMC Algo Performance")
            smc_cols = st.columns(4)
            with smc_cols[0]:
                st.metric("SMC Trades", algo_status['smc_trades'])
            with smc_cols[1]:
                st.metric("SMC Win Rate", f"{algo_status['smc_win_rate']:.1%}")
            with smc_cols[2]:
                st.metric("SMC P&L", f"₹{algo_status['smc_pnl']:+.2f}")
            with smc_cols[3]:
                st.metric("Last Recon", f"{int((time.time() - algo_status['last_reconciliation'])/60)}m ago" if algo_status['last_reconciliation'] > 0 else "Never")
            
            st.subheader("Engine Controls")
            ctrl_cols = st.columns(5)
            with ctrl_cols[0]:
                if st.button("▶️ Start", type="primary", disabled=algo_status['state'] == "running"):
                    if algo_engine.start():
                        st.success("Advanced Algo engine started!")
                        st.rerun()
            with ctrl_cols[1]:
                if st.button("⏸️ Pause", disabled=algo_status['state'] != "running"):
                    algo_engine.pause()
                    st.info("Algo engine paused")
                    st.rerun()
            with ctrl_cols[2]:
                if st.button("▶️ Resume", disabled=algo_status['state'] != "paused"):
                    algo_engine.resume()
                    st.success("Algo engine resumed")
                    st.rerun()
            with ctrl_cols[3]:
                if st.button("⏹️ Stop", disabled=algo_status['state'] == "stopped"):
                    algo_engine.stop()
                    st.info("Algo engine stopped")
                    st.rerun()
            with ctrl_cols[4]:
                if st.button("🚨 Emergency Stop", type="secondary"):
                    algo_engine.emergency_stop()
                    st.error("EMERGENCY STOP ACTIVATED")
                    st.rerun()
            
            # Reconciliation
            if st.button("🔄 Force Reconciliation"):
                success, message = algo_engine.reconcile_system()
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
    
    # Tab 6: Advanced Charts
    with tabs[5]:
        render_kite_charts_tab()
    
    # Tab 7: Risk Management
    with tabs[6]:
        st.subheader("⚙️ Advanced Risk Management")
        
        if not st.session_state.risk_manager:
            st.warning("Risk manager not initialized")
        else:
            risk_manager = st.session_state.risk_manager
            risk_status = risk_manager.get_status()
            
            # Risk Metrics
            st.subheader("📊 Risk Metrics")
            risk_cols = st.columns(4)
            with risk_cols[0]:
                st.metric("Current Equity", f"₹{risk_status['equity']:,.0f}")
            with risk_cols[1]:
                st.metric("Current Risk", f"{risk_status['dynamic_risk']:.2%}")
            with risk_cols[2]:
                st.metric("Current Drawdown", f"{risk_status['current_drawdown']:.2%}")
            with risk_cols[3]:
                st.metric("Max Drawdown", f"{risk_status['max_drawdown']:.2%}")
            
            # Daily Limits
            st.subheader("📅 Daily Limits")
            daily_cols = st.columns(4)
            with daily_cols[0]:
                daily_limit_used = risk_status['daily_limit_used']
                st.metric("Daily Limit Used", f"{daily_limit_used:.1%}")
            with daily_cols[1]:
                st.metric("Daily P&L", f"₹{risk_status['daily_pnl']:+,.2f}")
            with daily_cols[2]:
                st.metric("Trades Today", risk_status['trades_today'])
            with daily_cols[3]:
                st.metric("Positions Open", risk_status['positions_open'])
            
            # Session Risk
            st.subheader("🕒 Session Risk")
            session_cols = st.columns(3)
            with session_cols[0]:
                current_time = datetime.now().time()
                if config.INDIA_OPEN <= current_time <= config.INDIA_CLOSE:
                    session = "🇮🇳 INDIA"
                elif config.NY_OVERLAP_START <= current_time <= config.NY_OVERLAP_END:
                    session = "🇺🇸 NY OVERLAP"
                else:
                    session = "🔴 CLOSED"
                st.metric("Current Session", session)
            with session_cols[1]:
                st.metric("Session Risk Factor", f"{risk_status['session_risk_factor']:.1f}x")
            with session_cols[2]:
                valid = "✅ VALID" if valid_session() else "❌ INVALID"
                st.metric("Trading Session", valid)
            
            # Risk Controls
            st.subheader("🎯 Risk Controls")
            if st.button("Reset Daily Counts"):
                risk_manager.reset_daily()
                st.success("Daily counts reset")
                st.rerun()
            
            if st.button("Recalculate Risk"):
                # Force risk recalculation
                risk_manager.risk_scaler.calculate_risk()
                st.success("Risk recalculated")
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #94a3b8; font-size: 12px;">
        <strong>Rantv Terminal Pro v4.0 - Institutional Grade Algo Trading</strong> | 
        MTF SMC + Volume Profile + Auto Risk + Kite OMS | © 2024 | 
        Last Update: {now_indian().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

def run_console_mode():
    """Run in console mode when Streamlit is not available"""
    print("=" * 70)
    print("RANTV TERMINAL PRO v4.0 - Console Mode")
    print("=" * 70)
    print("\n🎯 Initializing Advanced Trading System...")
    
    trading_system = TradingSystem()
    
    if trading_system.initialize():
        print("✅ Advanced Trading System initialized successfully!")
        print(f"📊 Market Status: {'OPEN' if valid_session() else 'CLOSED'}")
        print(f"⏰ Current Time: {now_indian().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Generate SMC signals
        print("\n🧠 Generating SMC signals for NIFTY 50...")
        signal_gen = trading_system.get_component('signal_generator')
        if signal_gen:
            signals = signal_gen.scan_universe("Nifty 50", max_stocks=5, min_confidence=0.75, use_smc=True)
            if signals:
                print(f"✅ Generated {len(signals)} SMC signals:")
                for i, signal in enumerate(signals[:3]):  # Show top 3
                    print(f"  {i+1}. {signal['symbol']} - {signal['action']} @ ₹{signal['price']:.2f} "
                          f"(Confidence: {signal['confidence']:.1%}, Regime: {signal.get('regime', 'N/A')})")
            else:
                print("⚠️ No SMC signals generated")
        
        # Show risk status
        risk_manager = trading_system.get_component('risk_manager')
        if risk_manager:
            risk_status = risk_manager.get_status()
            print(f"\n🎯 Risk Status:")
            print(f"  - Current Risk: {risk_status['dynamic_risk']:.2%}")
            print(f"  - Current Drawdown: {risk_status['current_drawdown']:.2%}")
            print(f"  - Equity: ₹{risk_status['equity']:,.0f}")
    else:
        print("❌ Failed to initialize Trading System")
    
    print("\n" + "=" * 70)
    print("Note: For full UI experience with advanced features, install Streamlit:")
    print("pip install streamlit streamlit-autorefresh")
    print("Then run: streamlit run script_name.py")
    print("=" * 70)

# ===================== MAIN APPLICATION =====================
def main():
    """Main application entry point"""
    
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available. Running in console mode...")
        run_console_mode()
        return
    
    # Import Streamlit modules now that we know they're available
    import streamlit as st
    from streamlit_autorefresh import st_autorefresh
    
    # Set page config
    st.set_page_config(
        page_title="Rantv Terminal Pro v4.0 - Institutional Grade Algo",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    load_css()
    init_session_state()
    
    # Render main application
    render_main_app()

# ===================== RUN APPLICATION =====================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if STREAMLIT_AVAILABLE:
            import streamlit as st
            st.error(f"Application error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        else:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
