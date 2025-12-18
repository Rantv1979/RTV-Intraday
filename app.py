# ======================================================================
# Rantv Intraday Trading Terminal Pro - OPTIMIZED VERSION
# Combined and optimized with Algo Trading Engine
# ======================================================================

import os
import sys
import time
import math
import json
import traceback
import subprocess
import threading
import warnings
import logging
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Tuple, Any
from collections import defaultdict
from enum import Enum
from functools import lru_cache

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ====================== CONDITIONAL IMPORTS ======================
KITECONNECT_AVAILABLE = SQLALCHEMY_AVAILABLE = JOBLIB_AVAILABLE = SKLEARN_AVAILABLE = False

# Try to import optional dependencies
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITECONNECT_AVAILABLE = True
except ImportError:
    pass

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    pass

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    pass

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

# ====================== CONFIGURATION ======================
@dataclass
class AppConfig:
    """Centralized configuration for the entire application"""
    
    # API Credentials
    KITE_API_KEY: str = os.environ.get("KITE_API_KEY", "np4vpl4wq4yez03u")
    KITE_API_SECRET: str = os.environ.get("KITE_API_SECRET", "hqorfq94c0qupc9gvjqps8tdsr0kfa86")
    
    # Trading Parameters
    CAPITAL: float = 2_000_000.0
    TRADE_ALLOC: float = 0.15
    MAX_DAILY_TRADES: int = 10
    MAX_STOCK_TRADES: int = 10
    MAX_AUTO_TRADES: int = 10
    
    # Risk Management
    MAX_DAILY_LOSS: float = 50000.0
    RISK_TOLERANCE: str = "MODERATE"
    MIN_CONFIDENCE: float = 0.70
    MIN_SCORE: int = 6
    MIN_RISK_REWARD: float = 2.5
    ADX_TREND_THRESHOLD: int = 25
    
    # Market Hours (Indian Market)
    MARKET_OPEN: dt_time = dt_time(9, 15)
    MARKET_CLOSE: dt_time = dt_time(15, 30)
    AUTO_CLOSE_TIME: dt_time = dt_time(15, 10)
    PEAK_START: dt_time = dt_time(10, 0)
    PEAK_END: dt_time = dt_time(14, 0)
    
    # Database
    DATABASE_URL: str = "sqlite:///data/trading_journal.db"
    
    # Refresh Intervals
    SIGNAL_REFRESH_MS: int = 120000
    PRICE_REFRESH_MS: int = 100000
    
    # ML Settings
    ENABLE_ML: bool = SKLEARN_AVAILABLE and JOBLIB_AVAILABLE
    ML_MODEL_PATH: str = "data/signal_quality_model.pkl"
    
    # Algo Trading
    ENABLE_ALGO_TRADING: bool = False
    ALGO_MIN_CONFIDENCE: float = 0.80
    ALGO_MAX_POSITIONS: int = 5
    ALGO_MAX_TDAILY_TRADES: int = 20
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables"""
        return cls()

# Global configuration
config = AppConfig.from_env()
IND_TZ = pytz.timezone("Asia/Kolkata")

# ====================== CACHING SYSTEM ======================
class CacheManager:
    """Optimized caching with TTL and size limits"""
    
    def __init__(self):
        self.caches = {
            'price': {},
            'data': {},
            'signal': {},
            'rsi': {}
        }
        self.ttl = {
            'price': 5,      # 5 seconds
            'data': 30,      # 30 seconds
            'signal': 120,   # 2 minutes
            'rsi': 300       # 5 minutes
        }
        self.max_size = 1000
    
    def get(self, cache_type: str, key: str) -> Any:
        """Get cached value if not expired"""
        cache = self.caches.get(cache_type)
        if not cache or key not in cache:
            return None
        
        entry = cache[key]
        if time.time() - entry['timestamp'] > self.ttl.get(cache_type, 30):
            del cache[key]
            return None
        
        return entry['value']
    
    def set(self, cache_type: str, key: str, value: Any):
        """Set value in cache with timestamp"""
        if cache_type not in self.caches:
            self.caches[cache_type] = {}
        
        cache = self.caches[cache_type]
        if len(cache) > self.max_size:
            # Remove oldest 20% of entries
            sorted_keys = sorted(cache.keys(), key=lambda k: cache[k]['timestamp'])[:self.max_size//5]
            for k in sorted_keys:
                del cache[k]
        
        cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
    
    def clear_expired(self):
        """Clear all expired cache entries"""
        current_time = time.time()
        for cache_type, cache in self.caches.items():
            expired_keys = [
                k for k, v in cache.items()
                if current_time - v['timestamp'] > self.ttl.get(cache_type, 30)
            ]
            for key in expired_keys:
                del cache[key]

# Global cache instance
cache = CacheManager()

# ====================== UTILITIES ======================
def now_indian() -> datetime:
    """Get current time in Indian timezone"""
    return datetime.now(IND_TZ)

def market_open() -> bool:
    """Check if market is open"""
    n = now_indian()
    try:
        open_time = IND_TZ.localize(datetime.combine(n.date(), config.MARKET_OPEN))
        close_time = IND_TZ.localize(datetime.combine(n.date(), config.MARKET_CLOSE))
        return open_time <= n <= close_time
    except Exception:
        return False

def is_peak_hours() -> bool:
    """Check if current time is during peak hours"""
    n = now_indian()
    try:
        peak_start = IND_TZ.localize(datetime.combine(n.date(), config.PEAK_START))
        peak_end = IND_TZ.localize(datetime.combine(n.date(), config.PEAK_END))
        return peak_start <= n <= peak_end
    except Exception:
        return False

def should_auto_close() -> bool:
    """Check if auto-close should trigger"""
    n = now_indian()
    try:
        auto_close_time = IND_TZ.localize(datetime.combine(n.date(), config.AUTO_CLOSE_TIME))
        return n >= auto_close_time
    except Exception:
        return False

# ====================== TECHNICAL INDICATORS ======================
@lru_cache(maxsize=128)
def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average (cached)"""
    return series.ewm(span=span, adjust=False).mean()

@lru_cache(maxsize=128)
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (cached)"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(0)
    return 100 - (100 / (1 + rs))

@lru_cache(maxsize=128)
def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (cached)"""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

@lru_cache(maxsize=128)
def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD indicator (cached)"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

@lru_cache(maxsize=128)
def bollinger_bands(close: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands (cached)"""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

# ====================== ENUMS AND DATA CLASSES ======================
class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    PLACED = "placed"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"

class AlgoState(Enum):
    """Algo engine state enumeration"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class RiskLimits:
    """Risk management limits"""
    max_positions: int = 5
    max_daily_loss: float = 50000.0
    max_position_size: float = 100000.0
    max_drawdown_pct: float = 5.0
    min_confidence: float = 0.80
    max_trades_per_day: int = 20
    max_trades_per_stock: int = 2
    cool_down_after_loss_seconds: int = 300

@dataclass
class AlgoStats:
    """Algo trading statistics"""
    total_orders: int = 0
    filled_orders: int = 0
    rejected_orders: int = 0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    daily_loss: float = 0.0
    max_drawdown: float = 0.0
    last_trade_time: Optional[datetime] = None
    trades_today: int = 0
    stock_trades: Dict[str, int] = field(default_factory=dict)

@dataclass
class AlgoOrder:
    """Algo trading order"""
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
    broker_order_id: Optional[str] = None
    placed_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    error_message: Optional[str] = None

# ====================== STOCK UNIVERSES ======================
class StockUniverses:
    """Centralized stock universe management"""
    
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
    
    MIDCAP_150 = [
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
    
    @classmethod
    def get_universe(cls, universe_name: str) -> List[str]:
        """Get stock list for given universe"""
        universes = {
            "Nifty 50": cls.NIFTY_50,
            "Nifty 100": cls.NIFTY_100,
            "Midcap 150": cls.MIDCAP_150,
            "All Stocks": list(dict.fromkeys(cls.NIFTY_50 + cls.NIFTY_100 + cls.MIDCAP_150))
        }
        return universes.get(universe_name, cls.NIFTY_50)

# ====================== TRADING STRATEGIES ======================
class TradingStrategies:
    """Centralized strategy definitions"""
    
    STANDARD_STRATEGIES = {
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
    
    @classmethod
    def get_all_strategies(cls) -> Dict[str, Dict]:
        """Get all strategies combined"""
        return {**cls.STANDARD_STRATEGIES, **cls.HIGH_ACCURACY_STRATEGIES}
    
    @classmethod
    def get_strategy_accuracy(cls, strategy: str) -> float:
        """Get historical accuracy for a strategy"""
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

# ====================== CORE COMPONENTS ======================
class EnhancedDataManager:
    """Optimized data manager with intelligent caching"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._batch_fetch_enabled = True
        
    @st.cache_data(ttl=30, show_spinner=False)
    def _fetch_yf_data(_self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data from yfinance with caching"""
        try:
            return yf.download(symbol, period=period, interval=interval, 
                              progress=False, auto_adjust=False, threads=True)
        except Exception as e:
            _self.logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def get_stock_data(self, symbol: str, interval: str = "15m") -> pd.DataFrame:
        """Get stock data with optimized caching and batching"""
        cache_key = f"{symbol}_{interval}"
        
        # Check cache first
        cached_data = cache.get('data', cache_key)
        if cached_data is not None:
            return cached_data
        
        # Determine period based on interval
        period_map = {
            "1m": "1d", "5m": "2d", "15m": "7d",
            "30m": "14d", "1h": "30d", "1d": "90d"
        }
        period = period_map.get(interval, "7d")
        
        # Fetch data
        df = self._fetch_yf_data(symbol, period, interval)
        
        if df.empty or len(df) < 20:
            df = self._create_demo_data(symbol)
        
        # Calculate indicators
        df = self._calculate_indicators(df)
        
        # Cache the result
        cache.set('data', cache_key, df)
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators efficiently"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Ensure we have required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            return df
        
        # Calculate basic indicators
        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI14"] = rsi(df["Close"], 14).fillna(50)
        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"]).fillna(method="ffill").fillna(0)
        
        # MACD
        macd_line, signal_line, hist = macd(df["Close"])
        df["MACD"] = macd_line
        df["MACD_Signal"] = signal_line
        df["MACD_Hist"] = hist
        
        # Bollinger Bands
        upper, middle, lower = bollinger_bands(df["Close"])
        df["BB_Upper"] = upper
        df["BB_Middle"] = middle
        df["BB_Lower"] = lower
        
        # VWAP
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()
        
        return df
    
    def _create_demo_data(self, symbol: str) -> pd.DataFrame:
        """Create demo data when real data is unavailable"""
        periods = 100
        end = now_indian()
        dates = pd.date_range(end=end, periods=periods, freq="15min")
        
        np.random.seed(hash(symbol) % 2**32)
        base_price = 1000 + (hash(symbol) % 5000) / 10
        returns = np.random.normal(0, 0.0009, periods)
        prices = base_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            "Open": prices * (1 + np.random.normal(0, 0.0012, periods)),
            "High": prices * (1 + abs(np.random.normal(0, 0.0045, periods))),
            "Low": prices * (1 - abs(np.random.normal(0, 0.0045, periods))),
            "Close": prices,
            "Volume": np.random.randint(1000, 200000, periods)
        }, index=dates)
        
        return self._calculate_indicators(df)

class RiskManager:
    """Advanced risk management with position sizing"""
    
    def __init__(self, max_daily_loss: float = config.MAX_DAILY_LOSS):
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
    
    def reset_daily_metrics(self):
        """Reset daily metrics if new day"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
    
    def calculate_kelly_position(self, win_probability: float, win_loss_ratio: float,
                                available_capital: float, price: float, atr: float) -> int:
        """Calculate position size using Kelly Criterion"""
        if win_loss_ratio <= 0:
            win_loss_ratio = 2.0
        
        # Kelly formula: f = p - (1-p)/b
        kelly_fraction = win_probability - (1 - win_probability) / win_loss_ratio
        
        # Conservative sizing: half-Kelly
        risk_capital = available_capital * 0.1
        position_value = risk_capital * (kelly_fraction / 2)
        
        if price <= 0:
            return 1
        
        quantity = int(position_value / price)
        
        # Limits: min 1, max 20% of capital
        return max(1, min(quantity, int(available_capital * 0.2 / price)))
    
    def check_trade_viability(self, symbol: str, action: str, quantity: int,
                             price: float, current_positions: Dict) -> Tuple[bool, str]:
        """Check if trade is viable within risk limits"""
        self.reset_daily_metrics()
        
        if price <= 0:
            return False, "Invalid price"
        
        # Calculate portfolio value
        portfolio_value = sum([
            pos.get("quantity", 0) * pos.get("entry_price", 0)
            for pos in current_positions.values()
            if pos.get("entry_price", 0) > 0
        ])
        
        if portfolio_value <= 0:
            portfolio_value = price * max(quantity, 1)
        
        trade_value = quantity * price
        
        # Concentration limit (20%)
        max_allowed = max(portfolio_value * 0.20, 1)
        
        if trade_value > max_allowed:
            adjusted_qty = int(max_allowed // price)
            if adjusted_qty < 1:
                adjusted_qty = 1
            quantity = adjusted_qty
        
        # Hard cap (50%)
        hard_cap = portfolio_value * 0.50
        if trade_value > hard_cap:
            adjusted_qty = int(hard_cap // price)
            quantity = max(1, adjusted_qty)
        
        # Daily loss check
        if self.daily_pnl < -self.max_daily_loss:
            return False, "Daily loss limit exceeded"
        
        return True, f"Trade viable (quantity: {quantity})"

class MLSignalEnhancer:
    """ML-based signal confidence enhancer"""
    
    def __init__(self):
        self.enabled = config.ENABLE_ML
        self.model = None
        self.scaler = None
        
        if self.enabled:
            self._load_model()
    
    def _load_model(self):
        """Load pre-trained ML model"""
        try:
            if os.path.exists(config.ML_MODEL_PATH):
                self.model = joblib.load(config.ML_MODEL_PATH)
        except Exception as e:
            logging.warning(f"Could not load ML model: {e}")
    
    def enhance_confidence(self, data: pd.DataFrame, signal_type: str = "BUY") -> float:
        """Enhance signal confidence using ML or rule-based fallback"""
        if not self.enabled or self.model is None:
            return self._rule_based_confidence(data, signal_type)
        
        try:
            features = self._extract_features(data)
            if features.empty:
                return 0.7
            
            # Predict using ML model
            confidence = self.model.predict_proba(features)[0][1]
            return max(0.3, min(0.95, confidence))
            
        except Exception as e:
            logging.error(f"ML prediction error: {e}")
            return self._rule_based_confidence(data, signal_type)
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for ML model"""
        features = {}
        
        try:
            # Basic features
            features['rsi'] = data['RSI14'].iloc[-1] if 'RSI14' in data.columns else 50
            features['volume_ratio'] = self._calculate_volume_ratio(data)
            features['atr_ratio'] = data['ATR'].iloc[-1] / data['Close'].iloc[-1] if data['Close'].iloc[-1] > 0 else 0.01
            
            # Trend features
            features['ema_alignment'] = self._calculate_ema_alignment(data)
            features['trend_strength'] = self._calculate_trend_strength(data)
            
            return pd.DataFrame([features])
        except Exception:
            return pd.DataFrame()
    
    def _rule_based_confidence(self, data: pd.DataFrame, signal_type: str) -> float:
        """Fallback rule-based confidence calculation"""
        confidence = 0.5
        
        # RSI contribution
        rsi_val = data['RSI14'].iloc[-1] if 'RSI14' in data.columns else 50
        if 35 <= rsi_val <= 65:
            confidence += 0.1
        
        # Volume contribution
        vol_ratio = self._calculate_volume_ratio(data)
        if vol_ratio >= 1.5:
            confidence += 0.15
        
        # Trend contribution
        trend_strength = self._calculate_trend_strength(data)
        if trend_strength > 0.7:
            confidence += 0.1
        
        return max(0.3, min(0.9, confidence))
    
    def _calculate_volume_ratio(self, data: pd.DataFrame) -> float:
        """Calculate volume ratio vs average"""
        if 'Volume' not in data.columns or len(data) < 20:
            return 1.0
        
        current_vol = data['Volume'].iloc[-1]
        avg_vol = data['Volume'].rolling(20).mean().iloc[-1]
        
        return current_vol / avg_vol if avg_vol > 0 else 1.0
    
    def _calculate_ema_alignment(self, data: pd.DataFrame) -> float:
        """Calculate EMA alignment score (0-1)"""
        if all(col in data.columns for col in ['EMA8', 'EMA21', 'EMA50']):
            ema8 = data['EMA8'].iloc[-1]
            ema21 = data['EMA21'].iloc[-1]
            ema50 = data['EMA50'].iloc[-1]
            
            if ema8 > ema21 > ema50:
                return 1.0  # Strong uptrend
            elif ema8 < ema21 < ema50:
                return 0.0  # Strong downtrend
            else:
                return 0.5  # Mixed
        return 0.5
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength (0-1)"""
        if 'ADX' in data.columns:
            adx = data['ADX'].iloc[-1]
            return min(adx / 50, 1.0)  # Normalize ADX to 0-1
        return 0.5

# ====================== MAIN TRADING ENGINE ======================
class MultiStrategyIntradayTrader:
    """Main trading engine combining all strategies"""
    
    def __init__(self, capital: float = config.CAPITAL):
        self.initial_capital = float(capital)
        self.cash = float(capital)
        self.positions = {}
        self.trade_log = []
        self.daily_trades = 0
        self.auto_trades_count = 0
        self.last_reset = datetime.now().date()
        
        # Components
        self.data_manager = EnhancedDataManager()
        self.risk_manager = RiskManager()
        self.ml_enhancer = MLSignalEnhancer() if config.ENABLE_ML else None
        
        # Performance tracking
        self.strategy_performance = {}
        all_strategies = TradingStrategies.get_all_strategies()
        for strategy in all_strategies.keys():
            self.strategy_performance[strategy] = {
                "signals": 0, "trades": 0, "wins": 0, "pnl": 0.0
            }
    
    def reset_daily_counts(self):
        """Reset daily trading counts"""
        current_date = datetime.now().date()
        if current_date != self.last_reset:
            self.daily_trades = 0
            self.auto_trades_count = 0
            self.last_reset = current_date
    
    def can_auto_trade(self) -> bool:
        """Check if auto trading is allowed"""
        return (
            self.auto_trades_count < config.MAX_AUTO_TRADES and
            self.daily_trades < config.MAX_DAILY_TRADES and
            market_open()
        )
    
    def generate_signals(self, universe: str, max_scan: Optional[int] = None,
                        min_confidence: float = config.MIN_CONFIDENCE,
                        min_score: int = config.MIN_SCORE,
                        use_high_accuracy: bool = True) -> List[Dict]:
        """Generate trading signals for given universe"""
        
        # Get stocks to scan
        stocks = StockUniverses.get_universe(universe)
        if max_scan and max_scan < len(stocks):
            stocks = stocks[:max_scan]
        
        signals = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, symbol in enumerate(stocks):
            try:
                # Update progress
                status_text.text(f"Scanning {symbol} ({idx+1}/{len(stocks)})")
                progress_bar.progress((idx + 1) / len(stocks))
                
                # Get data and generate signals
                data = self.data_manager.get_stock_data(symbol, "15m")
                if data is None or len(data) < 30:
                    continue
                
                # Generate high accuracy signals if enabled
                if use_high_accuracy:
                    high_acc_signals = self._generate_high_accuracy_signals(symbol, data)
                    signals.extend(high_acc_signals)
                
                # Generate standard signals
                standard_signals = self._generate_standard_signals(symbol, data)
                signals.extend(standard_signals)
                
            except Exception as e:
                logging.error(f"Error scanning {symbol}: {e}")
                continue
        
        # Clean up progress display
        progress_bar.empty()
        status_text.empty()
        
        # Filter and sort signals
        signals = [s for s in signals if s.get("confidence", 0) >= min_confidence 
                  and s.get("score", 0) >= min_score]
        
        # Apply quality filters
        signals = self._filter_high_quality(signals)
        
        # Sort by score and confidence
        signals.sort(key=lambda x: (x.get("score", 0), x.get("confidence", 0)), reverse=True)
        
        return signals[:20]  # Return top 20 signals
    
    def _generate_standard_signals(self, symbol: str, data: pd.DataFrame) -> List[Dict]:
        """Generate standard trading signals"""
        signals = []
        
        if data.empty or len(data) < 30:
            return signals
        
        try:
            current_price = float(data["Close"].iloc[-1])
            ema8 = float(data["EMA8"].iloc[-1])
            ema21 = float(data["EMA21"].iloc[-1])
            rsi_val = float(data["RSI14"].iloc[-1])
            volume = float(data["Volume"].iloc[-1])
            vol_avg = float(data["Volume"].rolling(20).mean().iloc[-1]) if len(data) >= 20 else volume
            
            # BUY Signals
            if ema8 > ema21 and rsi_val < 65 and volume > vol_avg * 1.3:
                signals.append(self._create_signal(
                    symbol=symbol,
                    action="BUY",
                    price=current_price,
                    data=data,
                    strategy="EMA_VWAP_Confluence",
                    confidence=0.82
                ))
            
            # SELL Signals
            if ema8 < ema21 and rsi_val > 35 and volume > vol_avg * 1.3:
                signals.append(self._create_signal(
                    symbol=symbol,
                    action="SELL",
                    price=current_price,
                    data=data,
                    strategy="EMA_VWAP_Downtrend",
                    confidence=0.82
                ))
            
            # Update strategy counts
            for signal in signals:
                strategy = signal.get("strategy")
                if strategy in self.strategy_performance:
                    self.strategy_performance[strategy]["signals"] += 1
            
            return [s for s in signals if s is not None]
            
        except Exception as e:
            logging.error(f"Error generating signals for {symbol}: {e}")
            return []
    
    def _generate_high_accuracy_signals(self, symbol: str, data: pd.DataFrame) -> List[Dict]:
        """Generate high accuracy signals"""
        signals = []
        
        if data.empty or len(data) < 50:
            return signals
        
        try:
            current_price = float(data["Close"].iloc[-1])
            ema8 = float(data["EMA8"].iloc[-1])
            ema21 = float(data["EMA21"].iloc[-1])
            ema50 = float(data["EMA50"].iloc[-1])
            rsi_val = float(data["RSI14"].iloc[-1])
            volume = float(data["Volume"].iloc[-1])
            vol_avg = float(data["Volume"].rolling(20).mean().iloc[-1]) if len(data) >= 20 else volume
            vwap = float(data["VWAP"].iloc[-1])
            
            # Multi-Confirmation Strategy
            if (ema8 > ema21 > ema50 and current_price > vwap and 
                rsi_val > 50 and rsi_val < 70 and volume > vol_avg * 1.5):
                
                signal = self._create_signal(
                    symbol=symbol,
                    action="BUY",
                    price=current_price,
                    data=data,
                    strategy="Multi_Confirmation",
                    confidence=0.88
                )
                if signal:
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logging.error(f"Error generating high accuracy signals for {symbol}: {e}")
            return []
    
    def _create_signal(self, symbol: str, action: str, price: float, 
                      data: pd.DataFrame, strategy: str, confidence: float) -> Optional[Dict]:
        """Create a standardized signal dictionary"""
        try:
            atr = float(data["ATR"].iloc[-1]) if "ATR" in data.columns else price * 0.01
            
            # Calculate stop loss and target
            if action == "BUY":
                stop_loss = price - (atr * 1.2)
                target = price + (atr * 2.5)
            else:
                stop_loss = price + (atr * 1.2)
                target = price - (atr * 2.5)
            
            risk_reward = abs(target - price) / max(abs(price - stop_loss), 1e-6)
            
            if risk_reward < config.MIN_RISK_REWARD:
                return None
            
            # Enhance confidence with ML if available
            if self.ml_enhancer:
                ml_confidence = self.ml_enhancer.enhance_confidence(data, action)
                confidence = (confidence + ml_confidence) / 2
            
            return {
                "symbol": symbol,
                "action": action,
                "entry": price,
                "current_price": price,
                "target": target,
                "stop_loss": stop_loss,
                "confidence": confidence,
                "risk_reward": risk_reward,
                "score": self._calculate_signal_score(confidence, risk_reward),
                "strategy": strategy,
                "strategy_name": TradingStrategies.get_all_strategies().get(strategy, {}).get("name", strategy),
                "rsi": float(data["RSI14"].iloc[-1]) if "RSI14" in data.columns else 50
            }
        except Exception as e:
            logging.error(f"Error creating signal: {e}")
            return None
    
    def _calculate_signal_score(self, confidence: float, risk_reward: float) -> int:
        """Calculate signal score (0-10)"""
        score = int(confidence * 5)  # 0-5 from confidence
        score += min(int(risk_reward), 5)  # 0-5 from risk reward
        return min(score, 10)
    
    def _filter_high_quality(self, signals: List[Dict]) -> List[Dict]:
        """Filter only high-quality signals"""
        filtered = []
        
        for signal in signals:
            try:
                # Skip if already filtered by basic criteria
                if signal.get("confidence", 0) < config.MIN_CONFIDENCE:
                    continue
                
                if signal.get("risk_reward", 0) < config.MIN_RISK_REWARD:
                    continue
                
                # Get data for advanced filtering
                data = self.data_manager.get_stock_data(signal["symbol"], "15m")
                if data.empty:
                    continue
                
                # Volume check
                volume = data["Volume"].iloc[-1]
                avg_volume = data["Volume"].rolling(20).mean().iloc[-1] if len(data) >= 20 else volume
                if volume / avg_volume < 1.3:
                    continue
                
                # ADX trend check
                if 'ADX' in data.columns and data['ADX'].iloc[-1] < config.ADX_TREND_THRESHOLD:
                    continue
                
                filtered.append(signal)
                
            except Exception as e:
                logging.error(f"Error filtering signal: {e}")
                continue
        
        return filtered
    
    def execute_trade(self, symbol: str, action: str, quantity: int, price: float,
                     stop_loss: Optional[float] = None, target: Optional[float] = None,
                     win_probability: float = 0.75, auto_trade: bool = False,
                     strategy: Optional[str] = None) -> Tuple[bool, str]:
        """Execute a trade"""
        self.reset_daily_counts()
        
        # Check limits
        if self.daily_trades >= config.MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        
        if auto_trade and self.auto_trades_count >= config.MAX_AUTO_TRADES:
            return False, "Auto trade limit reached"
        
        # Check risk viability
        risk_ok, risk_msg = self.risk_manager.check_trade_viability(
            symbol, action, quantity, price, self.positions
        )
        if not risk_ok:
            return False, f"Risk check failed: {risk_msg}"
        
        # Create trade record
        trade_id = f"TRADE_{symbol}_{len(self.trade_log)}_{int(time.time())}"
        trade_record = {
            "trade_id": trade_id,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "entry_price": price,
            "stop_loss": stop_loss,
            "target": target,
            "timestamp": now_indian(),
            "status": "OPEN",
            "current_pnl": 0.0,
            "win_probability": win_probability,
            "auto_trade": auto_trade,
            "strategy": strategy
        }
        
        # Execute trade
        trade_value = quantity * price
        if action == "BUY":
            if trade_value > self.cash:
                return False, "Insufficient capital"
            self.positions[symbol] = trade_record
            self.cash -= trade_value
        else:
            margin = trade_value * 0.2
            trade_record["margin_used"] = margin
            self.positions[symbol] = trade_record
            self.cash -= margin
        
        # Update counts
        self.trade_log.append(trade_record)
        self.daily_trades += 1
        if auto_trade:
            self.auto_trades_count += 1
        
        # Update strategy performance
        if strategy and strategy in self.strategy_performance:
            self.strategy_performance[strategy]["trades"] += 1
        
        return True, f"{'[AUTO] ' if auto_trade else ''}{action} {quantity} {symbol} @ ₹{price:.2f}"
    
    def update_positions_pnl(self):
        """Update P&L for all open positions"""
        if should_auto_close():
            self._auto_close_all_positions()
            return
        
        for symbol, pos in list(self.positions.items()):
            if pos.get("status") != "OPEN":
                continue
            
            try:
                data = self.data_manager.get_stock_data(symbol, "5m")
                if data.empty:
                    continue
                
                current_price = float(data["Close"].iloc[-1])
                entry_price = pos["entry_price"]
                quantity = pos["quantity"]
                
                if pos["action"] == "BUY":
                    pnl = (current_price - entry_price) * quantity
                else:
                    pnl = (entry_price - current_price) * quantity
                
                pos["current_price"] = current_price
                pos["current_pnl"] = pnl
                
                # Check stop loss and target
                stop_loss = pos.get("stop_loss")
                target = pos.get("target")
                
                if stop_loss is not None:
                    if (pos["action"] == "BUY" and current_price <= stop_loss) or \
                       (pos["action"] == "SELL" and current_price >= stop_loss):
                        self.close_position(symbol, stop_loss)
                        continue
                
                if target is not None:
                    if (pos["action"] == "BUY" and current_price >= target) or \
                       (pos["action"] == "SELL" and current_price <= target):
                        self.close_position(symbol, target)
                        continue
                        
            except Exception as e:
                logging.error(f"Error updating position {symbol}: {e}")
                continue
    
    def _auto_close_all_positions(self):
        """Auto-close all positions at market close"""
        for symbol in list(self.positions.keys()):
            self.close_position(symbol)
    
    def close_position(self, symbol: str, exit_price: Optional[float] = None) -> Tuple[bool, str]:
        """Close a position"""
        if symbol not in self.positions:
            return False, "Position not found"
        
        pos = self.positions[symbol]
        
        if exit_price is None:
            try:
                data = self.data_manager.get_stock_data(symbol, "5m")
                exit_price = float(data["Close"].iloc[-1]) if not data.empty else pos["entry_price"]
            except Exception:
                exit_price = pos["entry_price"]
        
        # Calculate P&L
        if pos["action"] == "BUY":
            pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
            self.cash += pos["quantity"] * exit_price
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["quantity"]
            self.cash += pos.get("margin_used", 0) + (pos["quantity"] * pos["entry_price"])
        
        # Update position
        pos["status"] = "CLOSED"
        pos["exit_price"] = exit_price
        pos["closed_pnl"] = pnl
        pos["exit_time"] = now_indian()
        
        # Update strategy performance
        strategy = pos.get("strategy")
        if strategy and strategy in self.strategy_performance:
            if pnl > 0:
                self.strategy_performance[strategy]["wins"] += 1
            self.strategy_performance[strategy]["pnl"] += pnl
        
        # Remove from active positions
        del self.positions[symbol]
        
        return True, f"Closed {symbol} @ ₹{exit_price:.2f} | P&L: ₹{pnl:+.2f}"
    
    def get_open_positions(self) -> List[Dict]:
        """Get formatted open positions"""
        self.update_positions_pnl()
        positions = []
        
        for symbol, pos in self.positions.items():
            if pos.get("status") != "OPEN":
                continue
            
            try:
                current_price = pos.get("current_price", pos["entry_price"])
                entry_price = pos["entry_price"]
                quantity = pos["quantity"]
                
                if pos["action"] == "BUY":
                    pnl = (current_price - entry_price) * quantity
                else:
                    pnl = (entry_price - current_price) * quantity
                
                positions.append({
                    "Symbol": symbol.replace(".NS", ""),
                    "Action": pos["action"],
                    "Quantity": quantity,
                    "Entry": f"₹{entry_price:.2f}",
                    "Current": f"₹{current_price:.2f}",
                    "P&L": f"₹{pnl:+.2f}",
                    "Stop Loss": f"₹{pos.get('stop_loss', 0):.2f}",
                    "Target": f"₹{pos.get('target', 0):.2f}"
                })
            except Exception:
                continue
        
        return positions
    
    def get_performance_stats(self) -> Dict:
        """Get trading performance statistics"""
        self.update_positions_pnl()
        
        closed_trades = [t for t in self.trade_log if t.get("status") == "CLOSED"]
        total_trades = len(closed_trades)
        
        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "open_positions": len(self.positions),
                "open_pnl": sum([p.get("current_pnl", 0) for p in self.positions.values()]),
                "auto_trades": self.auto_trades_count
            }
        
        wins = len([t for t in closed_trades if t.get("closed_pnl", 0) > 0])
        total_pnl = sum([t.get("closed_pnl", 0) for t in closed_trades])
        
        return {
            "total_trades": total_trades,
            "win_rate": wins / total_trades,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / total_trades,
            "open_positions": len(self.positions),
            "open_pnl": sum([p.get("current_pnl", 0) for p in self.positions.values()]),
            "auto_trades": self.auto_trades_count
        }

# ====================== ALGO TRADING ENGINE ======================
class AlgoEngine:
    """Automated algorithmic trading engine"""
    
    def __init__(self, trader: MultiStrategyIntradayTrader):
        self.trader = trader
        self.state = AlgoState.STOPPED
        self.risk_limits = RiskLimits()
        self.stats = AlgoStats()
        
        self.orders: Dict[str, AlgoOrder] = {}
        self.active_positions: Dict[str, AlgoOrder] = {}
        self.order_history: List[AlgoOrder] = []
        
        self._stop_event = threading.Event()
        self._scheduler_thread = None
        self._lock = threading.Lock()
        
        self.last_scan = datetime.now(IND_TZ)
        self.scan_interval = 60  # seconds
    
    def start(self) -> bool:
        """Start the algo engine"""
        if self.state == AlgoState.RUNNING:
            return False
        
        self.state = AlgoState.RUNNING
        self._stop_event.clear()
        
        self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        
        logging.info("AlgoEngine started")
        return True
    
    def stop(self):
        """Stop the algo engine"""
        self.state = AlgoState.STOPPED
        self._stop_event.set()
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        
        logging.info("AlgoEngine stopped")
    
    def _run_scheduler(self):
        """Main scheduler loop"""
        logging.info("Algo scheduler started")
        
        while not self._stop_event.is_set():
            try:
                if self.state != AlgoState.RUNNING:
                    time.sleep(1)
                    continue
                
                if not market_open():
                    time.sleep(10)
                    continue
                
                now = datetime.now(IND_TZ)
                if (now - self.last_scan).total_seconds() >= self.scan_interval:
                    self._scan_and_execute()
                    self.last_scan = now
                
                self._check_positions()
                self._check_risk_limits()
                
                time.sleep(5)
                
            except Exception as e:
                logging.error(f"Scheduler error: {e}")
                time.sleep(10)
        
        logging.info("Algo scheduler stopped")
    
    def _scan_and_execute(self):
        """Scan for signals and execute trades"""
        if self.state != AlgoState.RUNNING:
            return
        
        logging.info("Scanning for algo signals...")
        
        try:
            signals = self.trader.generate_signals(
                universe="All Stocks",
                max_scan=50,
                min_confidence=self.risk_limits.min_confidence,
                min_score=7,
                use_high_accuracy=True
            )
            
            if not signals:
                return
            
            for signal in signals[:3]:  # Limit to top 3 signals
                if self._can_execute_signal(signal):
                    self._execute_signal(signal)
                    
        except Exception as e:
            logging.error(f"Signal scan error: {e}")
    
    def _can_execute_signal(self, signal: Dict) -> bool:
        """Check if signal can be executed"""
        with self._lock:
            # Position limits
            if len(self.active_positions) >= self.risk_limits.max_positions:
                return False
            
            # Daily loss limit
            if self.stats.daily_loss >= self.risk_limits.max_daily_loss:
                return False
            
            # Daily trade limit
            if self.stats.trades_today >= self.risk_limits.max_trades_per_day:
                return False
            
            # Stock trade limit
            symbol = signal.get("symbol", "")
            stock_trades = self.stats.stock_trades.get(symbol, 0)
            if stock_trades >= self.risk_limits.max_trades_per_stock:
                return False
            
            # Avoid duplicate positions
            if symbol in self.active_positions:
                return False
            
            # Confidence check
            if signal.get("confidence", 0) < self.risk_limits.min_confidence:
                return False
            
            # Cooldown after loss
            if self.stats.last_trade_time:
                time_since_last = (datetime.now(IND_TZ) - self.stats.last_trade_time).total_seconds()
                if self.stats.daily_loss > 0 and time_since_last < self.risk_limits.cool_down_after_loss_seconds:
                    return False
            
            return True
    
    def _execute_signal(self, signal: Dict) -> Optional[AlgoOrder]:
        """Execute a trading signal"""
        try:
            order_id = f"ALGO_{int(time.time() * 1000)}"
            
            order = AlgoOrder(
                order_id=order_id,
                symbol=signal["symbol"],
                action=signal["action"],
                quantity=self._calculate_position_size(signal),
                price=signal["entry"],
                stop_loss=signal["stop_loss"],
                target=signal["target"],
                strategy=signal.get("strategy", "ALGO"),
                confidence=signal.get("confidence", 0.75)
            )
            
            # Execute as paper trade
            success, msg = self.trader.execute_trade(
                symbol=order.symbol,
                action=order.action,
                quantity=order.quantity,
                price=order.price,
                stop_loss=order.stop_loss,
                target=order.target,
                win_probability=order.confidence,
                auto_trade=True,
                strategy=order.strategy
            )
            
            if success:
                with self._lock:
                    order.status = OrderStatus.FILLED
                    order.filled_at = datetime.now(IND_TZ)
                    order.filled_price = order.price
                    
                    self.active_positions[order.symbol] = order
                    self.orders[order_id] = order
                    
                    self.stats.total_orders += 1
                    self.stats.filled_orders += 1
                    self.stats.trades_today += 1
                    self.stats.stock_trades[order.symbol] = self.stats.stock_trades.get(order.symbol, 0) + 1
                    self.stats.last_trade_time = datetime.now(IND_TZ)
                
                logging.info(f"Algo order filled: {order.action} {order.quantity} {order.symbol}")
                return order
            
            return None
            
        except Exception as e:
            logging.error(f"Algo execution error: {e}")
            return None
    
    def _calculate_position_size(self, signal: Dict) -> int:
        """Calculate position size for algo trading"""
        position_value = min(
            self.risk_limits.max_position_size,
            self.trader.cash * 0.15
        )
        quantity = int(position_value / signal["entry"])
        return max(1, quantity)
    
    def _check_positions(self):
        """Check and update algo positions"""
        self.trader.update_positions_pnl()
        
        # Update algo stats from trader stats
        perf = self.trader.get_performance_stats()
        self.stats.realized_pnl = perf.get('total_pnl', 0)
        
        if self.stats.realized_pnl < 0:
            self.stats.daily_loss = abs(self.stats.realized_pnl)
        
        # Update unrealized P&L
        unrealized = 0
        for symbol, order in self.active_positions.items():
            if symbol in self.trader.positions:
                pos = self.trader.positions[symbol]
                if pos.get("status") == "OPEN":
                    unrealized += pos.get("current_pnl", 0)
        
        self.stats.unrealized_pnl = unrealized
    
    def _check_risk_limits(self):
        """Check if risk limits are breached"""
        total_loss = self.stats.realized_pnl + self.stats.unrealized_pnl
        
        if total_loss < -self.risk_limits.max_daily_loss:
            self.emergency_stop(f"Daily loss limit exceeded: {total_loss:.2f}")
            return
        
        if self.trader.initial_capital > 0:
            drawdown_pct = abs(total_loss) / self.trader.initial_capital * 100
            if drawdown_pct > self.risk_limits.max_drawdown_pct:
                self.emergency_stop(f"Max drawdown exceeded: {drawdown_pct:.2f}%")
                return
    
    def emergency_stop(self, reason: str = "Manual trigger"):
        """Emergency stop all algo trading"""
        logging.critical(f"ALGO EMERGENCY STOP: {reason}")
        self.state = AlgoState.EMERGENCY_STOP
        self.stop()
    
    def get_status(self) -> Dict:
        """Get algo engine status"""
        return {
            "state": self.state.value,
            "active_positions": len(self.active_positions),
            "total_orders": self.stats.total_orders,
            "filled_orders": self.stats.filled_orders,
            "trades_today": self.stats.trades_today,
            "realized_pnl": self.stats.realized_pnl,
            "unrealized_pnl": self.stats.unrealized_pnl,
            "daily_loss": self.stats.daily_loss,
            "market_open": market_open(),
            "peak_hours": is_peak_hours()
        }

# ====================== STREAMLIT UI COMPONENTS ======================
class StreamlitUI:
    """Streamlit UI components and layout"""
    
    def __init__(self):
        self.setup_styling()
    
    def setup_styling(self):
        """Setup CSS styling for the app"""
        st.markdown("""
        <style>
            /* Main container */
            .stApp {
                background: linear-gradient(135deg, #fff9e6 0%, #fff0d6 100%);
            }
            
            /* Cards */
            .metric-card {
                background: white;
                padding: 15px;
                border-radius: 10px;
                border-left: 4px solid #1e3a8a;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            /* Signal quality */
            .high-quality-signal {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                padding: 12px;
                border-radius: 8px;
                margin: 8px 0;
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 4px;
                background: linear-gradient(135deg, #e6f2ff 0%, #ffe6e6 50%, #e6ffe6 100%);
                padding: 8px;
                border-radius: 12px;
                margin-bottom: 1rem;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def create_market_gauge(self, name: str, value: float, change: float, sentiment: int) -> str:
        """Create circular market gauge HTML"""
        sentiment = max(0, min(100, sentiment))
        
        if sentiment >= 70:
            color = "#059669"
            text = "BULLISH"
            emoji = "📈"
        elif sentiment <= 30:
            color = "#dc2626"
            text = "BEARISH"
            emoji = "📉"
        else:
            color = "#d97706"
            text = "NEUTRAL"
            emoji = "➡️"
        
        return f"""
        <div style="background: white; border-radius: 50%; padding: 25px; margin: 10px auto; 
                    border: 4px solid #e0f2fe; box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                    width: 200px; height: 200px; display: flex; flex-direction: column;
                    align-items: center; justify-content: center; text-align: center;">
            <div style="font-size: 14px; font-weight: bold; margin-bottom: 8px; color: #1e3a8a;">
                {emoji} {name}
            </div>
            <div style="width: 100px; height: 100px; border-radius: 50%; 
                        background: conic-gradient({color} 0% {sentiment}%, #e5e7eb {sentiment}% 100%);
                        display: flex; align-items: center; justify-content: center; margin: 8px 0;">
                <div style="width: 70px; height: 70px; border-radius: 50%; background: white;
                            display: flex; align-items: center; justify-content: center;
                            font-weight: bold; font-size: 14px;">
                    {sentiment}%
                </div>
            </div>
            <div style="font-size: 16px; font-weight: bold; margin: 3px 0;">₹{value:,.0f}</div>
            <div style="color: {'#059669' if change >= 0 else '#dc2626'}; font-size: 12px; margin-top: 3px;">
                {change:+.2f}%
            </div>
        </div>
        """
    
    def render_dashboard(self, trader: MultiStrategyIntradayTrader):
        """Render main dashboard"""
        st.markdown("<h1 style='text-align:center; color: #1e3a8a;'>Rantv Intraday Terminal Pro</h1>", 
                   unsafe_allow_html=True)
        
        # Market overview
        cols = st.columns(5)
        with cols[0]:
            st.metric("Market Status", "LIVE" if market_open() else "CLOSED")
        with cols[1]:
            st.metric("Peak Hours", "ACTIVE" if is_peak_hours() else "INACTIVE")
        with cols[2]:
            st.metric("Available Cash", f"₹{trader.cash:,.0f}")
        with cols[3]:
            st.metric("Open Positions", len(trader.positions))
        with cols[4]:
            pnl = sum([p.get("current_pnl", 0) for p in trader.positions.values()])
            st.metric("Open P&L", f"₹{pnl:+.2f}")
        
        # Market gauges
        st.subheader("📊 Market Mood Gauges")
        gauge_cols = st.columns(4)
        
        # Sample gauge data (in real app, fetch from market)
        with gauge_cols[0]:
            st.markdown(self.create_market_gauge("NIFTY 50", 22000, 0.5, 65), unsafe_allow_html=True)
        with gauge_cols[1]:
            st.markdown(self.create_market_gauge("BANK NIFTY", 48000, 0.3, 70), unsafe_allow_html=True)
        with gauge_cols[2]:
            status = 80 if market_open() else 20
            st.markdown(self.create_market_gauge("MARKET", 0, 0, status), unsafe_allow_html=True)
        with gauge_cols[3]:
            peak_status = 80 if is_peak_hours() else 30
            st.markdown(self.create_market_gauge("PEAK HOURS", 0, 0, peak_status), unsafe_allow_html=True)
    
    def render_signals_tab(self, trader: MultiStrategyIntradayTrader):
        """Render signals tab"""
        st.subheader("🚦 Trading Signals")
        
        # Signal filters
        col1, col2, col3 = st.columns(3)
        with col1:
            universe = st.selectbox("Universe", ["All Stocks", "Nifty 50", "Nifty 100", "Midcap 150"])
        with col2:
            min_conf = st.slider("Min Confidence %", 60, 85, config.MIN_CONFIDENCE * 100, 5)
        with col3:
            enable_high_acc = st.checkbox("High Accuracy", value=True)
        
        if st.button("Generate Signals", type="primary"):
            with st.spinner(f"Scanning {universe}..."):
                signals = trader.generate_signals(
                    universe=universe,
                    min_confidence=min_conf/100,
                    use_high_accuracy=enable_high_acc
                )
            
            if signals:
                st.success(f"Found {len(signals)} signals")
                
                # Display signals
                for signal in signals[:10]:
                    action_color = "🟢" if signal["action"] == "BUY" else "🔴"
                    quality = signal.get("quality_score", 0)
                    
                    with st.container():
                        st.markdown(f"""
                        <div style="background: {'#d1fae5' if signal['action'] == 'BUY' else '#fee2e2'}; 
                                    padding: 12px; border-radius: 8px; margin: 5px 0;
                                    border-left: 4px solid {'#059669' if signal['action'] == 'BUY' else '#dc2626'}">
                            <strong>{action_color} {signal['symbol'].replace('.NS', '')}</strong> | 
                            {signal['action']} @ ₹{signal['entry']:.2f}<br>
                            Strategy: {signal['strategy_name']} | 
                            Confidence: {signal['confidence']:.1%} | 
                            R:R: {signal['risk_reward']:.2f}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No signals found. Try adjusting filters.")
    
    def render_algo_tab(self, algo_engine: AlgoEngine):
        """Render algo trading tab"""
        st.subheader("🤖 Algo Trading")
        
        status = algo_engine.get_status()
        
        # Status display
        cols = st.columns(4)
        with cols[0]:
            state = status["state"]
            color = {"running": "🟢", "stopped": "🔴", "paused": "🟡", "emergency_stop": "⛔"}.get(state, "⚪")
            st.metric("Engine", f"{color} {state.upper()}")
        with cols[1]:
            st.metric("Positions", f"{status['active_positions']}/{algo_engine.risk_limits.max_positions}")
        with cols[2]:
            st.metric("Trades Today", f"{status['trades_today']}/{algo_engine.risk_limits.max_trades_per_day}")
        with cols[3]:
            st.metric("Daily P&L", f"₹{status['realized_pnl']:+.2f}")
        
        # Controls
        control_cols = st.columns(4)
        with control_cols[0]:
            if st.button("▶️ Start", type="primary", disabled=(status["state"] == "running")):
                algo_engine.start()
                st.success("Started")
                st.rerun()
        with control_cols[1]:
            if st.button("⏸️ Pause", disabled=(status["state"] != "running")):
                algo_engine.state = AlgoState.PAUSED
                st.info("Paused")
                st.rerun()
        with control_cols[2]:
            if st.button("▶️ Resume", disabled=(status["state"] != "paused")):
                algo_engine.state = AlgoState.RUNNING
                st.success("Resumed")
                st.rerun()
        with control_cols[3]:
            if st.button("⏹️ Stop", type="secondary", disabled=(status["state"] == "stopped")):
                algo_engine.stop()
                st.info("Stopped")
                st.rerun()
        
        # Emergency stop
        if st.button("🚨 EMERGENCY STOP", type="primary"):
            algo_engine.emergency_stop("Manual trigger")
            st.error("EMERGENCY STOP ACTIVATED")
            st.rerun()
        
        # Risk settings
        with st.expander("Risk Settings"):
            col1, col2 = st.columns(2)
            with col1:
                new_max_pos = st.number_input("Max Positions", 1, 20, algo_engine.risk_limits.max_positions)
                new_max_loss = st.number_input("Max Daily Loss", 1000, 500000, int(algo_engine.risk_limits.max_daily_loss))
            with col2:
                new_min_conf = st.slider("Min Confidence", 0.5, 0.99, algo_engine.risk_limits.min_confidence, 0.05)
                new_max_trades = st.number_input("Max Trades/Day", 1, 50, algo_engine.risk_limits.max_trades_per_day)
            
            if st.button("Update Settings"):
                algo_engine.risk_limits.max_positions = new_max_pos
                algo_engine.risk_limits.max_daily_loss = float(new_max_loss)
                algo_engine.risk_limits.min_confidence = new_min_conf
                algo_engine.risk_limits.max_trades_per_day = new_max_trades
                st.success("Settings updated")

# ====================== MAIN APPLICATION ======================
def main():
    """Main application entry point"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Setup page config
    st.set_page_config(
        page_title="Rantv Intraday Terminal Pro",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize UI
    ui = StreamlitUI()
    
    # Initialize trader in session state
    if "trader" not in st.session_state:
        st.session_state.trader = MultiStrategyIntradayTrader()
    
    trader = st.session_state.trader
    
    # Initialize algo engine
    if "algo_engine" not in st.session_state:
        st.session_state.algo_engine = AlgoEngine(trader)
    
    algo_engine = st.session_state.algo_engine
    
    # Auto-refresh
    st_autorefresh(interval=config.PRICE_REFRESH_MS, key="price_refresh")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Trading settings
        trader.auto_execution = st.checkbox("Auto Execution", value=False)
        
        # Universe selection
        selected_universe = st.selectbox(
            "Trading Universe",
            ["All Stocks", "Nifty 50", "Nifty 100", "Midcap 150"]
        )
        
        # Risk settings
        st.subheader("🎯 Risk Settings")
        config.MIN_CONFIDENCE = st.slider("Min Confidence %", 60, 85, 70, 5) / 100
        config.MIN_SCORE = st.slider("Min Score", 5, 9, 6, 1)
        
        # ML settings
        if config.ENABLE_ML:
            st.subheader("🤖 ML Settings")
            enable_ml = st.checkbox("Enable ML Enhancement", value=True)
            config.ENABLE_ML = enable_ml
    
    # Main content - Tabs
    tabs = st.tabs([
        "📈 Dashboard", "🚦 Signals", "💰 Trading", 
        "📊 Analytics", "🤖 Algo Trading"
    ])
    
    # Tab 1: Dashboard
    with tabs[0]:
        ui.render_dashboard(trader)
        
        # Performance metrics
        perf = trader.get_performance_stats()
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Trades", perf["total_trades"])
        with cols[1]:
            st.metric("Win Rate", f"{perf['win_rate']:.1%}")
        with cols[2]:
            st.metric("Total P&L", f"₹{perf['total_pnl']:+.2f}")
        with cols[3]:
            st.metric("Auto Trades", perf["auto_trades"])
        
        # Open positions
        st.subheader("📊 Open Positions")
        positions = trader.get_open_positions()
        if positions:
            st.dataframe(pd.DataFrame(positions), use_container_width=True)
        else:
            st.info("No open positions")
    
    # Tab 2: Signals
    with tabs[1]:
        ui.render_signals_tab(trader)
    
    # Tab 3: Trading
    with tabs[2]:
        st.subheader("💰 Manual Trading")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.selectbox("Symbol", StockUniverses.NIFTY_50[:20])
        with col2:
            action = st.selectbox("Action", ["BUY", "SELL"])
        with col3:
            quantity = st.number_input("Quantity", 1, 1000, 10)
        
        if st.button("Execute Trade", type="primary"):
            # Get current price
            data = trader.data_manager.get_stock_data(symbol, "5m")
            if not data.empty:
                price = float(data["Close"].iloc[-1])
                
                success, msg = trader.execute_trade(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    price=price,
                    strategy="Manual"
                )
                
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.error("Could not fetch price data")
    
    # Tab 4: Analytics
    with tabs[3]:
        st.subheader("📊 Trading Analytics")
        
        # Strategy performance
        strategy_data = []
        for strategy, perf in trader.strategy_performance.items():
            if perf["trades"] > 0:
                win_rate = perf["wins"] / perf["trades"] if perf["trades"] > 0 else 0
                strategy_data.append({
                    "Strategy": TradingStrategies.get_all_strategies().get(strategy, {}).get("name", strategy),
                    "Signals": perf["signals"],
                    "Trades": perf["trades"],
                    "Wins": perf["wins"],
                    "Win Rate": f"{win_rate:.1%}",
                    "P&L": f"₹{perf['pnl']:+.2f}"
                })
        
        if strategy_data:
            st.dataframe(pd.DataFrame(strategy_data), use_container_width=True)
        else:
            st.info("No strategy performance data yet")
    
    # Tab 5: Algo Trading
    with tabs[4]:
        ui.render_algo_tab(algo_engine)

if __name__ == "__main__":
    # Handle missing dependencies
    if not all([SQLALCHEMY_AVAILABLE, JOBLIB_AVAILABLE]):
        st.warning("Some optional dependencies missing. Some features may be limited.")
    
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.code(traceback.format_exc())
