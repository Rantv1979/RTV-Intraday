# ======================================================================
# Rantv Intraday Trading Terminal Pro - PRODUCTION READY
# Complete Integration: Manual Trading + Algo Engine + Kite Connect
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
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict
from enum import Enum

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ====================== AUTO-INSTALL DEPENDENCIES ======================
def install_package(package):
    """Auto-install missing packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except:
        return False

# Try importing and auto-install if needed
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITECONNECT_AVAILABLE = True
except ImportError:
    if install_package("kiteconnect"):
        from kiteconnect import KiteConnect, KiteTicker
        KITECONNECT_AVAILABLE = True
    else:
        KITECONNECT_AVAILABLE = False

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    if install_package("sqlalchemy"):
        import sqlalchemy
        from sqlalchemy import create_engine, text
        SQLALCHEMY_AVAILABLE = True
    else:
        SQLALCHEMY_AVAILABLE = False

# ====================== CONFIGURATION ======================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RantvTerminal")
warnings.filterwarnings('ignore')

IND_TZ = pytz.timezone("Asia/Kolkata")

# ===== KITE CONNECT CREDENTIALS =====
KITE_API_KEY = "np4vpl4wq4yez03u"
KITE_API_SECRET = "hqorfq94c0qupc9gvjqps8tdsr0kfa86"

# Trading Parameters
CAPITAL = 2_000_000.0
TRADE_ALLOC = 0.15
MAX_DAILY_TRADES = 10
MAX_AUTO_TRADES = 10
PRICE_REFRESH_MS = 100000

# Market Hours
MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)
PEAK_START = dt_time(10, 0)
PEAK_END = dt_time(14, 0)

# Risk Parameters
MIN_CONFIDENCE = 0.70
MIN_SCORE = 6
MIN_RISK_REWARD = 2.5
MAX_DAILY_LOSS = 50000.0

# ====================== STOCK UNIVERSES ======================
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

# ====================== ENUMS & DATA CLASSES ======================

class AlgoState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"

class OrderStatus(Enum):
    PENDING = "pending"
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
    placed_at: datetime = field(default_factory=lambda: datetime.now(IND_TZ))
    filled_at: Optional[datetime] = None

# ====================== UTILITIES ======================

def now_indian() -> datetime:
    return datetime.now(IND_TZ)

def market_open() -> bool:
    n = now_indian()
    if n.weekday() >= 5:
        return False
    try:
        open_time = IND_TZ.localize(datetime.combine(n.date(), MARKET_OPEN))
        close_time = IND_TZ.localize(datetime.combine(n.date(), MARKET_CLOSE))
        return open_time <= n <= close_time
    except:
        return False

def is_peak_hours() -> bool:
    n = now_indian()
    try:
        peak_start = IND_TZ.localize(datetime.combine(n.date(), PEAK_START))
        peak_end = IND_TZ.localize(datetime.combine(n.date(), PEAK_END))
        return peak_start <= n <= peak_end
    except:
        return False

# ====================== TECHNICAL INDICATORS ======================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs.fillna(0)))

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ====================== KITE CONNECT MANAGER ======================

class KiteConnectManager:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = None
        self.access_token = None
        self.is_authenticated = False
        
        if KITECONNECT_AVAILABLE and self.api_key:
            self.kite = KiteConnect(api_key=self.api_key)
    
    def get_login_url(self) -> str:
        """Get Kite Connect login URL"""
        if self.kite:
            return self.kite.login_url()
        return ""
    
    def authenticate(self, request_token: str) -> bool:
        """Authenticate with request token"""
        try:
            if not self.kite:
                return False
            
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            self.is_authenticated = True
            
            # Save to session
            st.session_state.kite_access_token = self.access_token
            st.session_state.kite_user = data.get("user_name", "")
            
            return True
        except Exception as e:
            logger.error(f"Kite authentication error: {e}")
            return False
    
    def check_session(self) -> bool:
        """Check if existing session is valid"""
        try:
            if "kite_access_token" in st.session_state:
                self.access_token = st.session_state.kite_access_token
                if not self.kite:
                    self.kite = KiteConnect(api_key=self.api_key)
                self.kite.set_access_token(self.access_token)
                profile = self.kite.profile()
                self.is_authenticated = True
                return True
        except:
            pass
        return False
    
    def logout(self):
        """Logout from Kite"""
        self.access_token = None
        self.is_authenticated = False
        if "kite_access_token" in st.session_state:
            del st.session_state.kite_access_token
        if "kite_user" in st.session_state:
            del st.session_state.kite_user

# ====================== DATA MANAGER ======================

class EnhancedDataManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.cache_ttl = 30
        self.kite_manager = KiteConnectManager(KITE_API_KEY, KITE_API_SECRET)
    
    @st.cache_data(ttl=30, show_spinner=False)
    def _fetch_yf(_self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        try:
            return yf.download(symbol, period=period, interval=interval, 
                              progress=False, auto_adjust=False, threads=True)
        except Exception as e:
            _self.logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def get_stock_data(self, symbol: str, interval: str = "15m") -> pd.DataFrame:
        cache_key = f"{symbol}_{interval}"
        
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
        
        period_map = {"1m": "1d", "5m": "2d", "15m": "7d", "30m": "14d", "1h": "30d"}
        period = period_map.get(interval, "7d")
        
        df = self._fetch_yf(symbol, period, interval)
        
        if df.empty or len(df) < 20:
            df = self._create_demo_data(symbol)
        
        df = self._calculate_indicators(df)
        self.cache[cache_key] = (time.time(), df)
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        df = df.copy()
        
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            return df
        
        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI14"] = rsi(df["Close"], 14).fillna(50)
        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"]).ffill().fillna(0)
        
        macd_line, signal_line, hist = macd(df["Close"])
        df["MACD"] = macd_line
        df["MACD_Signal"] = signal_line
        df["MACD_Hist"] = hist
        
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()
        
        return df
    
    def _create_demo_data(self, symbol: str) -> pd.DataFrame:
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

# ====================== RISK MANAGER ======================

class RiskManager:
    def __init__(self, max_daily_loss: float = MAX_DAILY_LOSS):
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
    
    def reset_daily_metrics(self):
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
    
    def calculate_position_size(self, price: float, capital: float, allocation: float = TRADE_ALLOC) -> int:
        position_value = capital * allocation
        quantity = int(position_value / price)
        return max(1, quantity)
    
    def check_trade_viability(self, price: float, quantity: int) -> Tuple[bool, str]:
        self.reset_daily_metrics()
        
        if price <= 0:
            return False, "Invalid price"
        
        if self.daily_pnl < -self.max_daily_loss:
            return False, "Daily loss limit exceeded"
        
        return True, f"Trade viable (quantity: {quantity})"

# ====================== TRADING ENGINE ======================

class MultiStrategyIntradayTrader:
    def __init__(self, capital: float = CAPITAL):
        self.initial_capital = float(capital)
        self.cash = float(capital)
        self.positions = {}
        self.trade_log = []
        self.daily_trades = 0
        self.auto_trades_count = 0
        self.last_reset = datetime.now().date()
        
        self.data_manager = EnhancedDataManager()
        self.risk_manager = RiskManager()
        
        self.strategy_performance = {}
    
    def reset_daily_counts(self):
        current_date = datetime.now().date()
        if current_date != self.last_reset:
            self.daily_trades = 0
            self.auto_trades_count = 0
            self.last_reset = current_date
    
    def can_auto_trade(self) -> bool:
        return (
            self.auto_trades_count < MAX_AUTO_TRADES and
            self.daily_trades < MAX_DAILY_TRADES and
            market_open()
        )
    
    def generate_quality_signals(self, universe: str = "Nifty 50",
                                 max_scan: Optional[int] = None,
                                 min_confidence: float = MIN_CONFIDENCE,
                                 use_high_accuracy: bool = True) -> List[Dict]:
        
        stocks = NIFTY_50 if universe == "Nifty 50" else NIFTY_50
        
        if max_scan and max_scan < len(stocks):
            stocks = stocks[:max_scan]
        
        signals = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, symbol in enumerate(stocks):
            try:
                status_text.text(f"Scanning {symbol} ({idx+1}/{len(stocks)})")
                progress_bar.progress((idx + 1) / len(stocks))
                
                data = self.data_manager.get_stock_data(symbol, "15m")
                if data is None or len(data) < 30:
                    continue
                
                signal_list = self._generate_signals_for_stock(symbol, data)
                signals.extend(signal_list)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        signals = [s for s in signals if s.get("confidence", 0) >= min_confidence]
        signals.sort(key=lambda x: (x.get("score", 0), x.get("confidence", 0)), reverse=True)
        
        return signals[:20]
    
    def _generate_signals_for_stock(self, symbol: str, data: pd.DataFrame) -> List[Dict]:
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
            atr = float(data["ATR"].iloc[-1]) if "ATR" in data.columns else current_price * 0.01
            
            # BUY Signal
            if ema8 > ema21 and rsi_val < 65 and volume > vol_avg * 1.3:
                stop_loss = current_price - (atr * 1.2)
                target = current_price + (atr * 2.5)
                risk_reward = abs(target - current_price) / max(abs(current_price - stop_loss), 1e-6)
                
                if risk_reward >= MIN_RISK_REWARD:
                    signals.append({
                        "symbol": symbol,
                        "action": "BUY",
                        "entry": current_price,
                        "current_price": current_price,
                        "target": target,
                        "stop_loss": stop_loss,
                        "confidence": 0.82,
                        "risk_reward": risk_reward,
                        "score": self._calculate_signal_score(0.82, risk_reward),
                        "strategy": "EMA_Crossover",
                        "rsi": rsi_val
                    })
            
            # SELL Signal
            if ema8 < ema21 and rsi_val > 35 and volume > vol_avg * 1.3:
                stop_loss = current_price + (atr * 1.2)
                target = current_price - (atr * 2.5)
                risk_reward = abs(target - current_price) / max(abs(current_price - stop_loss), 1e-6)
                
                if risk_reward >= MIN_RISK_REWARD:
                    signals.append({
                        "symbol": symbol,
                        "action": "SELL",
                        "entry": current_price,
                        "current_price": current_price,
                        "target": target,
                        "stop_loss": stop_loss,
                        "confidence": 0.82,
                        "risk_reward": risk_reward,
                        "score": self._calculate_signal_score(0.82, risk_reward),
                        "strategy": "EMA_Crossunder",
                        "rsi": rsi_val
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return []
    
    def _calculate_signal_score(self, confidence: float, risk_reward: float) -> int:
        score = int(confidence * 5)
        score += min(int(risk_reward), 5)
        return min(score, 10)
    
    def execute_trade(self, symbol: str, action: str, quantity: int, price: float,
                     stop_loss: Optional[float] = None, target: Optional[float] = None,
                     win_probability: float = 0.75, auto_trade: bool = False,
                     strategy: Optional[str] = None) -> Tuple[bool, str]:
        
        self.reset_daily_counts()
        
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        
        if auto_trade and self.auto_trades_count >= MAX_AUTO_TRADES:
            return False, "Auto trade limit reached"
        
        risk_ok, risk_msg = self.risk_manager.check_trade_viability(price, quantity)
        if not risk_ok:
            return False, f"Risk check failed: {risk_msg}"
        
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
            "auto_trade": auto_trade,
            "strategy": strategy
        }
        
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
        
        self.trade_log.append(trade_record)
        self.daily_trades += 1
        if auto_trade:
            self.auto_trades_count += 1
        
        return True, f"{'[AUTO] ' if auto_trade else ''}{action} {quantity} {symbol} @ ₹{price:.2f}"
    
    def update_positions_pnl(self):
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
                sl = pos.get("stop_loss")
                tg = pos.get("target")
                
                if sl is not None:
                    if (pos["action"] == "BUY" and current_price <= sl) or \
                       (pos["action"] == "SELL" and current_price >= sl):
                        self.close_position(symbol, sl)
                        continue
                
                if tg is not None:
                    if (pos["action"] == "BUY" and current_price >= tg) or \
                       (pos["action"] == "SELL" and current_price <= tg):
                        self.close_position(symbol, tg)
                        continue
                        
            except Exception as e:
                logger.error(f"Error updating position {symbol}: {e}")
                continue
    
    def close_position(self, symbol: str, exit_price: Optional[float] = None) -> Tuple[bool, str]:
        if symbol not in self.positions:
            return False, "Position not found"
        
        pos = self.positions[symbol]
        
        if exit_price is None:
            try:
                data = self.data_manager.get_stock_data(symbol, "5m")
                exit_price = float(data["Close"].iloc[-1]) if not data.empty else pos["entry_price"]
            except:
                exit_price = pos["entry_price"]
        
        if pos["action"] == "BUY":
            pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
            self.cash += pos["quantity"] * exit_price
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["quantity"]
            self.cash += pos.get("margin_used", 0) + (pos["quantity"] * pos["entry_price"])
        
        pos["status"] = "CLOSED"
        pos["exit_price"] = exit_price
        pos["closed_pnl"] = pnl
        pos["exit_time"] = now_indian()
        
        del self.positions[symbol]
        
        return True, f"Closed {symbol} @ ₹{exit_price:.2f} | P&L: ₹{pnl:+.2f}"
    
    def get_performance_stats(self) -> Dict:
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

# ====================== ALGO ENGINE ======================

class AlgoEngine:
    def __init__(self, trader: MultiStrategyIntradayTrader):
        self.state = AlgoState.STOPPED
        self.trader = trader
        self.max_positions = 5
        self.min_confidence = 0.82
        self.daily_loss_limit = 50000.0
        self.active_positions: Dict[str, AlgoOrder] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._scheduler_thread = None
        
        logger.info("AlgoEngine initialized")
    
    def start(self):
        with self._lock:
            if self.state != AlgoState.RUNNING:
                self.state = AlgoState.RUNNING
                self._stop_event.clear()
                self._scheduler_thread = threading.Thread(target=self._run_engine_loop, daemon=True)
                self._scheduler_thread.start()
                logger.info("AlgoEngine started")
                return True
        return False
    
    def stop(self):
        with self._lock:
            self.state = AlgoState.STOPPED
            self._stop_event.set()
            logger.info("AlgoEngine stopped")
    
    def pause(self):
        with self._lock:
            self.state = AlgoState.PAUSED
            logger.info("AlgoEngine paused")
    
    def resume(self):
        with self._lock:
            if self.state == AlgoState.PAUSED:
                self.state = AlgoState.RUNNING
                logger.info("AlgoEngine resumed")
    
    def _run_engine_loop(self):
        last_scan_time = 0
        
        while not self._stop_event.is_set():
            try:
                if self.state == AlgoState.RUNNING:
                    now = time.time()
                    
                    if market_open():
                        if now - last_scan_time > 60:
                            self._scan_and_execute()
                            last_scan_time = now
                        
                        self._check_risk_breaches()
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Engine loop error: {e}")
                time.sleep(10)
    
    def _scan_and_execute(self):
        if len(self.active_positions) >= self.max_positions:
            return
        
        logger.info("AlgoEngine: Scanning for signals...")
        
        try:
            signals = self.trader.generate_quality_signals(
                universe="Nifty 50",
                max_
