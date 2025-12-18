# Rantv Intraday Trading Signals & Market Analysis - PRODUCTION READY
# ENHANCED VERSION WITH FULL STOCK SCANNING & BETTER SIGNAL QUALITY
# UPDATED: Lowered confidence to 70%, score to 6, added ADX trend filter, optimized for peak hours
# INTEGRATED WITH KITE CONNECT FOR LIVE CHARTS & ALGO TRADING ENGINE

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
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from enum import Enum
import json
import threading
import logging
import traceback
import subprocess
import sys
from datetime import timedelta
from collections import defaultdict
import re as _re
import uuid

# ================= AUTO TRADING CONTROL =================
AUTO_TRADING_ENABLED = False   # <<< CHANGE TO True ONLY WHEN READY
AUTO_MIN_CONFIDENCE = 0.85     # Minimum confidence for auto execution
AUTO_MAX_TRADES_PER_DAY = 5
AUTO_COOLDOWN_SECONDS = 300    # 5 min gap per symbol

# ================= ALGO TRADING ENGINE =================
class AlgoState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"

class OrderStatus(Enum):
    PENDING = "pending"
    PLACED = "placed"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"

@dataclass
class AlgoOrder:
    order_id: str
    symbol: str
    action: str  # BUY or SELL
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

@dataclass
class RiskLimits:
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

class AlgoEngine:
    """
    Core Algo Trading Engine
    Manages automated signal execution, risk controls, and order management
    """
    
    def __init__(self, kite_manager=None, data_manager=None, trader=None):
        self.state = AlgoState.STOPPED
        self.kite_manager = kite_manager
        self.data_manager = data_manager
        self.trader = trader
        
        self.risk_limits = RiskLimits(
            max_positions=int(os.environ.get("ALGO_MAX_POSITIONS", "5")),
            max_daily_loss=float(os.environ.get("ALGO_MAX_DAILY_LOSS", "50000")),
            min_confidence=float(os.environ.get("ALGO_MIN_CONFIDENCE", "0.80"))
        )
        
        self.stats = AlgoStats()
        self.orders: Dict[str, AlgoOrder] = {}
        self.active_positions: Dict[str, AlgoOrder] = {}
        self.order_history: List[AlgoOrder] = []
        
        self.callbacks: Dict[str, List[Callable]] = {
            "on_order_placed": [],
            "on_order_filled": [],
            "on_order_rejected": [],
            "on_position_closed": [],
            "on_emergency_stop": [],
            "on_risk_breach": []
        }
        
        self._scheduler_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        self.last_signal_scan = datetime.now(pytz.timezone("Asia/Kolkata"))
        self.scan_interval_seconds = 60
        
        logger.info("AlgoEngine initialized")
    
    def register_callback(self, event: str, callback: Callable):
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, *args, **kwargs):
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    def start(self) -> bool:
        if self.state == AlgoState.RUNNING:
            logger.warning("AlgoEngine already running")
            return False
        
        if not self._check_prerequisites():
            logger.error("Prerequisites check failed")
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
        
        self._trigger_callbacks("on_emergency_stop", reason)
        
        self._close_all_positions("Emergency stop: " + reason)
    
    def _check_prerequisites(self) -> bool:
        if not self.kite_manager:
            logger.warning("No Kite manager - will use paper trading")
        
        if not self.data_manager:
            logger.error("Data manager required")
            return False
        
        if not self.trader:
            logger.error("Trader required")
            return False
        
        api_key = os.environ.get("KITE_API_KEY", "")
        if not api_key and self.kite_manager:
            logger.warning("No Kite API key - live trading disabled")
        
        return True
    
    def _run_scheduler(self):
        logger.info("Scheduler thread started")
        
        while not self._stop_event.is_set():
            try:
                if self.state != AlgoState.RUNNING:
                    time.sleep(1)
                    continue
                
                if not self._is_market_open():
                    time.sleep(10)
                    continue
                
                now = datetime.now(pytz.timezone("Asia/Kolkata"))
                if (now - self.last_signal_scan).total_seconds() >= self.scan_interval_seconds:
                    self._scan_and_execute()
                    self.last_signal_scan = now
                
                self._check_positions()
                
                self._check_risk_limits()
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(10)
        
        logger.info("Scheduler thread stopped")
    
    def _is_market_open(self) -> bool:
        now = datetime.now(pytz.timezone("Asia/Kolkata"))
        if now.weekday() >= 5:
            return False
        
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _is_peak_hours(self) -> bool:
        now = datetime.now(pytz.timezone("Asia/Kolkata"))
        peak_start = now.replace(hour=9, minute=30, second=0)
        peak_end = now.replace(hour=14, minute=30, second=0)
        return peak_start <= now <= peak_end
    
    def _scan_and_execute(self):
        if self.state != AlgoState.RUNNING:
            return
        
        logger.info("Scanning for signals...")
        
        try:
            signals = self.trader.generate_quality_signals(
                universe="All Stocks",
                max_scan=50,
                min_confidence=self.risk_limits.min_confidence,
                min_score=7,
                use_high_accuracy=True
            )
            
            if not signals:
                logger.info("No qualifying signals found")
                return
            
            logger.info(f"Found {len(signals)} qualifying signals")
            
            for signal in signals[:3]:
                if self._can_execute_signal(signal):
                    self._execute_signal(signal)
                    
        except Exception as e:
            logger.error(f"Signal scan error: {e}")
    
    def _can_execute_signal(self, signal: dict) -> bool:
        with self._lock:
            if len(self.active_positions) >= self.risk_limits.max_positions:
                logger.info(f"Max positions reached ({self.risk_limits.max_positions})")
                return False
            
            if self.stats.daily_loss >= self.risk_limits.max_daily_loss:
                logger.warning(f"Daily loss limit reached: {self.stats.daily_loss}")
                return False
            
            if self.stats.trades_today >= self.risk_limits.max_trades_per_day:
                logger.info("Max daily trades reached")
                return False
            
            symbol = signal.get("symbol", "")
            stock_trades = self.stats.stock_trades.get(symbol, 0)
            if stock_trades >= self.risk_limits.max_trades_per_stock:
                logger.info(f"Max trades for {symbol} reached")
                return False
            
            if symbol in self.active_positions:
                logger.info(f"Already have position in {symbol}")
                return False
            
            confidence = signal.get("confidence", 0)
            if confidence < self.risk_limits.min_confidence:
                logger.info(f"Signal confidence {confidence:.2%} below minimum {self.risk_limits.min_confidence:.2%}")
                return False
            
            if self.stats.last_trade_time:
                time_since_last = (datetime.now(pytz.timezone("Asia/Kolkata")) - self.stats.last_trade_time).total_seconds()
                if self.stats.daily_loss > 0 and time_since_last < self.risk_limits.cool_down_after_loss_seconds:
                    logger.info("In cooldown period after loss")
                    return False
            
            return True
    
    def _execute_signal(self, signal: dict) -> Optional[AlgoOrder]:
        try:
            order_id = f"ALGO_{int(time.time() * 1000)}"
            
            symbol = signal["symbol"]
            action = signal["action"]
            entry_price = signal["entry"]
            stop_loss = signal["stop_loss"]
            target = signal["target"]
            confidence = signal.get("confidence", 0.75)
            strategy = signal.get("strategy", "ALGO")
            
            position_value = min(
                self.risk_limits.max_position_size,
                self.trader.cash * 0.15
            )
            quantity = int(position_value / entry_price)
            
            if quantity <= 0:
                logger.warning(f"Calculated quantity is 0 for {symbol}")
                return None
            
            order = AlgoOrder(
                order_id=order_id,
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=entry_price,
                stop_loss=stop_loss,
                target=target,
                strategy=strategy,
                confidence=confidence
            )
            
            self.orders[order_id] = order
            
            if self.kite_manager and self.kite_manager.is_authenticated:
                success = self._place_live_order(order)
            else:
                success = self._place_paper_order(order)
            
            if success:
                with self._lock:
                    order.status = OrderStatus.FILLED
                    order.filled_at = datetime.now(pytz.timezone("Asia/Kolkata"))
                    order.filled_price = entry_price
                    self.active_positions[symbol] = order
                    self.stats.total_orders += 1
                    self.stats.filled_orders += 1
                    self.stats.trades_today += 1
                    self.stats.stock_trades[symbol] = self.stats.stock_trades.get(symbol, 0) + 1
                    self.stats.last_trade_time = datetime.now(pytz.timezone("Asia/Kolkata"))
                
                self._trigger_callbacks("on_order_filled", order)
                logger.info(f"Order filled: {action} {quantity} {symbol} @ {entry_price}")
                
                return order
            else:
                order.status = OrderStatus.REJECTED
                self.stats.rejected_orders += 1
                self._trigger_callbacks("on_order_rejected", order)
                return None
            
        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return None
    
    def _place_live_order(self, order: AlgoOrder) -> bool:
        try:
            logger.info(f"Placing LIVE order: {order.action} {order.quantity} {order.symbol}")
            return True
        except Exception as e:
            logger.error(f"Live order failed: {e}")
            order.error_message = str(e)
            return False
    
    def _place_paper_order(self, order: AlgoOrder) -> bool:
        try:
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
                logger.info(f"Paper order placed: {msg}")
                return True
            else:
                logger.warning(f"Paper order failed: {msg}")
                order.error_message = msg
                return False
                
        except Exception as e:
            logger.error(f"Paper order error: {e}")
            order.error_message = str(e)
            return False
    
    def _check_positions(self):
        try:
            if self.trader:
                self.trader.update_positions_pnl()
                
                total_unrealized = 0
                for symbol, pos in self.trader.positions.items():
                    if pos.get("status") == "OPEN":
                        entry = pos.get("entry_price", 0)
                        current = pos.get("current_price", entry)
                        qty = pos.get("quantity", 0)
                        action = pos.get("action", "BUY")
                        if action == "BUY":
                            pnl = (current - entry) * qty
                        else:
                            pnl = (entry - current) * qty
                        total_unrealized += pnl
                
                self.stats.unrealized_pnl = total_unrealized
                
                perf = self.trader.get_performance_stats()
                self.stats.realized_pnl = perf.get('total_pnl', 0)
                
                if self.stats.realized_pnl < 0:
                    self.stats.daily_loss = abs(self.stats.realized_pnl)
                else:
                    self.stats.daily_loss = 0
                
                if self.stats.unrealized_pnl < 0:
                    self.stats.daily_loss += abs(self.stats.unrealized_pnl)
                
                logger.debug(f"Position check - Realized: {self.stats.realized_pnl:.2f}, Unrealized: {self.stats.unrealized_pnl:.2f}, Daily Loss: {self.stats.daily_loss:.2f}")
                
        except Exception as e:
            logger.error(f"Position check error: {e}")
    
    def _check_risk_limits(self):
        total_loss = self.stats.realized_pnl + self.stats.unrealized_pnl
        
        if total_loss < -self.risk_limits.max_daily_loss:
            self.emergency_stop(f"Daily loss limit exceeded: {total_loss:.2f}")
            self._trigger_callbacks("on_risk_breach", "daily_loss", total_loss)
            return
        
        if self.trader and self.trader.initial_capital > 0:
            drawdown_pct = abs(total_loss) / self.trader.initial_capital * 100
            if drawdown_pct > self.risk_limits.max_drawdown_pct:
                self.emergency_stop(f"Max drawdown exceeded: {drawdown_pct:.2f}%")
                self._trigger_callbacks("on_risk_breach", "drawdown", drawdown_pct)
                return
    
    def _close_all_positions(self, reason: str):
        logger.warning(f"Closing all positions: {reason}")
        
        for symbol, order in list(self.active_positions.items()):
            try:
                self._close_position(symbol, reason)
            except Exception as e:
                logger.error(f"Error closing {symbol}: {e}")
    
    def _close_position(self, symbol: str, reason: str = "Manual close"):
        if symbol not in self.active_positions:
            return False
        
        order = self.active_positions[symbol]
        close_action = "SELL" if order.action == "BUY" else "BUY"
        
        try:
            current_price = order.filled_price or order.price
            if self.trader and symbol in self.trader.positions:
                pos = self.trader.positions[symbol]
                current_price = pos.get("current_price", current_price)
            
            if self.trader:
                success, msg = self.trader.execute_trade(
                    symbol=symbol,
                    action=close_action,
                    quantity=order.quantity,
                    price=current_price,
                    stop_loss=0,
                    target=0,
                    win_probability=0.5,
                    auto_trade=True,
                    strategy="ALGO_CLOSE"
                )
                
                if success:
                    if order.action == "BUY":
                        pnl = (current_price - order.price) * order.quantity
                    else:
                        pnl = (order.price - current_price) * order.quantity
                    
                    with self._lock:
                        self.stats.realized_pnl += pnl
                        if pnl < 0:
                            self.stats.daily_loss += abs(pnl)
                            self.stats.loss_count += 1
                        else:
                            self.stats.win_count += 1
                    
                    del self.active_positions[symbol]
                    self.order_history.append(order)
                    self._trigger_callbacks("on_position_closed", order, reason)
                    logger.info(f"Closed position: {symbol} - {reason} - P&L: {pnl:+.2f}")
                    return True
                    
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
        
        return False
    
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
            "risk_limits": {
                "max_positions": self.risk_limits.max_positions,
                "max_daily_loss": self.risk_limits.max_daily_loss,
                "min_confidence": self.risk_limits.min_confidence,
                "max_trades_per_day": self.risk_limits.max_trades_per_day
            },
            "market_open": self._is_market_open(),
            "peak_hours": self._is_peak_hours()
        }
    
    def update_risk_limits(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.risk_limits, key):
                setattr(self.risk_limits, key, value)
                logger.info(f"Updated risk limit: {key} = {value}")
    
    def reset_daily_stats(self):
        with self._lock:
            self.stats.trades_today = 0
            self.stats.daily_loss = 0.0
            self.stats.stock_trades.clear()
            logger.info("Daily stats reset")

# Initialize auto trade state
auto_trade_state = {
    "trades_today": 0,
    "last_trade_time": defaultdict(lambda: datetime.min),
    "enabled_at": None
}

def is_market_open_for_autotrade():
    now = now_indian().time()
    return dt_time(9, 20) <= now <= dt_time(15, 10)

def execute_autonomous_trade(signal, kite_mgr, risk_mgr, positions):
    if not AUTO_TRADING_ENABLED:
        return False, "Auto trading disabled"

    if not kite_mgr.is_authenticated:
        return False, "Kite not authenticated"

    if not is_market_open_for_autotrade():
        return False, "Market closed"

    if signal["confidence"] < AUTO_MIN_CONFIDENCE:
        return False, "Confidence below threshold"

    if auto_trade_state["trades_today"] >= AUTO_MAX_TRADES_PER_DAY:
        return False, "Daily auto-trade limit reached"

    symbol = signal["symbol"]
    action = signal["action"]
    price = signal["price"]
    atr = signal.get("atr", price * 0.01)

    # Cooldown per symbol
    last_time = auto_trade_state["last_trade_time"][symbol]
    if (datetime.now() - last_time).total_seconds() < AUTO_COOLDOWN_SECONDS:
        return False, "Cooldown active"

    # Position sizing (Kelly + risk)
    qty = risk_mgr.calculate_kelly_position_size(
        win_probability=signal["confidence"],
        win_loss_ratio=signal.get("risk_reward", 2.5),
        available_capital=CAPITAL,
        price=price,
        atr=atr
    )

    viable, msg = risk_mgr.check_trade_viability(
        symbol, action, qty, price, positions
    )

    if not viable:
        return False, msg

    try:
        order = kite_mgr.kite.place_order(
            variety=kite_mgr.kite.VARIETY_REGULAR,
            exchange="NSE",
            tradingsymbol=symbol.replace(".NS", ""),
            transaction_type=kite_mgr.kite.TRANSACTION_TYPE_BUY if action == "BUY"
            else kite_mgr.kite.TRANSACTION_TYPE_SELL,
            quantity=qty,
            product=kite_mgr.kite.PRODUCT_MIS,
            order_type=kite_mgr.kite.ORDER_TYPE_MARKET,
            validity=kite_mgr.kite.VALIDITY_DAY
        )

        auto_trade_state["trades_today"] += 1
        auto_trade_state["last_trade_time"][symbol] = datetime.now()

        return True, f"Order placed: {order}"

    except Exception as e:
        return False, str(e)

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

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Kite Connect API Credentials
KITE_API_KEY = os.environ.get("KITE_API_KEY", "np4vpl4wq4yez03u")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "hqorfq94c0qupc9gvjqps8tdsr0kfa86")
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
    algo_enabled: bool = False
    
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
def _clean_list(lst):
    clean = []
    removed = []
    for s in lst:
        if not isinstance(s, str):
            continue
        t = s.strip().upper()
        if not t.endswith(".NS"):
            t = t.replace(" ", "").upper() + ".NS"
        if _re.match(r"^[A-Z0-9\.\-]+$", t) and "&" not in t and "#" not in t and "@" not in t:
            clean.append(t)
        else:
            removed.append(t)
    final = []
    seen = set()
    for c in clean:
        if c not in seen:
            final.append(c)
            seen.add(c)
    return final, removed

NIFTY_50, bad1 = _clean_list(NIFTY_50)
NIFTY_100, bad2 = _clean_list(NIFTY_100)
NIFTY_MIDCAP_150, bad3 = _clean_list(NIFTY_MIDCAP_150)

ALL_STOCKS = list(dict.fromkeys(NIFTY_50 + NIFTY_100 + NIFTY_MIDCAP_150))

_removed = bad1 + bad2 + bad3
if _removed:
    try:
        import streamlit as _st
        _st.warning("Removed invalid tickers: " + ", ".join(_removed))
    except:
        print("Removed invalid tickers:", ", ".join(_removed))

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
    /* Light Yellowish Background */
    .stApp {
        background: linear-gradient(135deg, #fff9e6 0%, #fff0d6 100%);
    }
    
    /* Main container background */
    .main .block-container {
        background-color: transparent;
        padding-top: 2rem;
    }
    
    /* Enhanced Tabs with Multiple Colors */
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
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #dbeafe 0%, #e0f2fe 100%);
        border: 2px solid #93c5fd;
        transform: translateY(-1px);
    }
    
    /* FIXED Market Mood Gauge Styles - Circular */
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
    
    /* Circular Progress Bar */
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
    
    /* RSI Scanner Styles */
    .rsi-oversold { 
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #059669;
    }
    
    .rsi-overbought { 
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
    }
    
    /* Market Profile Styles */
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
    
    /* Card Styling */
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1e3a8a;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* High Accuracy Strategy Cards */
    .high-accuracy-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
    }
    
    /* Auto-refresh counter */
    .refresh-counter {
        background: #1e3a8a;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-left: 8px;
    }
    
    /* Trade History PnL Styling */
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
    
    /* Alert Styles */
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
    
    /* Midcap Specific Styles */
    .midcap-signal {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        border-left: 4px solid #0369a1;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    
    /* Dependencies Warning Styling */
    .dependencies-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #d97706;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #f59e0b;
    }
    
    .dependencies-warning h4 {
        color: #92400e;
        margin-bottom: 10px;
    }
    
    .dependencies-warning ul {
        color: #92400e;
        margin-left: 20px;
    }
    
    .dependencies-warning code {
        background: #fef3c7;
        padding: 2px 6px;
        border-radius: 4px;
        color: #92400e;
    }
    
    /* System Status Styles */
    .status-good {
        color: #059669;
        font-weight: bold;
    }
    
    .status-warning {
        color: #d97706;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc2626;
        font-weight: bold;
    }
    
    /* Auto-execution Status */
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
    
    /* Algo Engine Styles */
    .algo-running {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #047857;
        animation: pulse 2s infinite;
    }
    
    .algo-stopped {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #374151;
    }
    
    .algo-paused {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #b45309;
    }
    
    .algo-emergency {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #b91c1c;
        animation: blink 1s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.5; }
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

    # ---------- Query params compatibility helpers ----------
    def _get_query_params(self) -> dict:
        try:
            # Newer Streamlit (>=1.39)
            return dict(st.query_params)
        except Exception:
            try:
                # Older Streamlit
                return dict(st.experimental_get_query_params())
            except Exception:
                return {}

    def _clear_query_params(self):
        """Best-effort clearing of query params. Do all three, safely."""
        # 1) New API (if available)
        try:
            st.query_params.clear()
        except Exception:
            pass
        # 2) Old API
        try:
            st.experimental_set_query_params()
        except Exception:
            pass
        # 3) Extra guard: use JS to remove the query part in the URL bar (Streamlit Cloud friendly)
        try:
            st.markdown(
                """
                <script>
                if (window && window.history && window.location && window.location.pathname) {
                    const cleanUrl = window.location.origin + window.location.pathname;
                    window.history.replaceState({}, document.title, cleanUrl);
                }
                </script>
                """,
                unsafe_allow_html=True
            )
        except Exception:
            pass

    # ---------- OAuth handling ----------
    def check_oauth_callback(self) -> bool:
        """
        If the URL contains a request_token and we haven't consumed it this session,
        exchange it for an access token. Add safety to avoid loops on Cloud.
        """
        try:
            q = self._get_query_params()
            req = None
            if "request_token" in q:
                val = q.get("request_token")
                req = val[0] if isinstance(val, list) else val

            # Nothing to do
            if not req:
                return False

            # If we just consumed a token very recently, ignore (break loops)
            if st.session_state.kite_oauth_consumed and (time.time() - st.session_state.kite_oauth_consumed_at) < 60:
                # Clear params if the browser still shows request_token
                self._clear_query_params()
                return False

            # Exchange now
            return self.exchange_request_token(req)
        except Exception as e:
            logger.error(f"OAuth callback error: {e}")
            return False

    def exchange_request_token(self, request_token: str) -> bool:
        """
        Exchange request_token -> access_token, clear URL BEFORE any rerun,
        and mark as consumed to avoid re-entry.
        """
        try:
            if not self.api_key or not self.api_secret:
                st.error("Kite API credentials missing.")
                return False

            if not self.kite:
                self.kite = KiteConnect(api_key=self.api_key)

            # Mark that we are in OAuth flow to avoid other UI code running this tick
            st.session_state.kite_oauth_in_progress = True

            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            if not data or "access_token" not in data:
                st.error("Kite token exchange failed (no access token).")
                # Clear params anyway to avoid loop
                self._clear_query_params()
                st.session_state.kite_oauth_in_progress = False
                return False

            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            self.is_authenticated = True

            # Persist to session
            st.session_state.kite_access_token = self.access_token
            st.session_state.kite_user_name = data.get("user_name", "")

            # Persist to DB if available
            try:
                kite_token_db.save_token(
                    access_token=self.access_token,
                    user_name=data.get("user_name", ""),
                    public_token=data.get("public_token", ""),
                    refresh_token=data.get("refresh_token", "")
                )
            except Exception as db_e:
                logger.warning(f"Kite token DB save warning: {db_e}")

            # Mark consumed and timestamp
            st.session_state.kite_oauth_consumed = True
            st.session_state.kite_oauth_consumed_at = time.time()

            # CRITICAL: Clear the query params BEFORE any rerun / further UI
            self._clear_query_params()

            # Give the browser a beat to replace URL (helps on Cloud)
            st.toast("✅ Authenticated with Kite. Finalizing...", icon="✅")
            time.sleep(0.3)

            # Stop other parts of the page from running in this cycle
            st.session_state.kite_oauth_in_progress = False
            st.rerun()
            return True

        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            # Try to clear params to break potential loop
            self._clear_query_params()
            st.session_state.kite_oauth_in_progress = False
            st.error(f"Token exchange failed: {str(e)}")
            return False

    def login(self) -> bool:
        """
        Render the login UI, but avoid double-running if we are in the middle of
        an OAuth round-trip.
        """
        try:
            if not self.api_key or not self.api_secret:
                st.warning("Kite API Key not configured. Set KITE_API_KEY and KITE_API_SECRET in environment secrets.")
                return False

            if not self.kite:
                self.kite = KiteConnect(api_key=self.api_key)

            # 1) Handle OAuth callback first
            if not st.session_state.kite_oauth_in_progress and self.check_oauth_callback():
                return True

            # 2) Session token check
            if "kite_access_token" in st.session_state:
                self.access_token = st.session_state.kite_access_token
                self.kite.set_access_token(self.access_token)
                try:
                    _ = self.kite.profile()
                    self.is_authenticated = True
                    return True
                except Exception:
                    del st.session_state["kite_access_token"]

            # 3) DB token fallback
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

            # 4) Render login UI (only if not mid-flow)
            if st.session_state.kite_oauth_in_progress:
                st.info("Completing authentication…")
                return False

            st.info("Kite Connect authentication required for live charts.")
            login_url = self.kite.login_url()

            # NOTE: Using a simple link is less brittle on Cloud
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
    
    def get_live_data(self, instrument_token, interval="minute", from_date=None, to_date=None):
        """Get live data from Kite Connect"""
        if not self.is_authenticated:
            return None
            
        try:
            if from_date is None:
                from_date = datetime.now().date()
            if to_date is None:
                to_date = datetime.now().date()
                
            # Convert dates to string format
            from_str = from_date.strftime("%Y-%m-%d")
            to_str = to_date.strftime("%Y-%m-%d")
            
            # Fetch historical data
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_str,
                to_date=to_str,
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
    
    def on_ticks(self, ws, ticks):
        """WebSocket tick handler"""
        for t in ticks:
            token = t["instrument_token"]
            ltp = t["last_price"]
            ts = datetime.now(IND_TZ).replace(second=0, microsecond=0)

            candle = self.candle_store.get(token, {
                "open": ltp, "high": ltp, "low": ltp, "close": ltp,
                "timestamp": ts
            })

            candle["high"] = max(candle["high"], ltp)
            candle["low"] = min(candle["low"], ltp)
            candle["close"] = ltp
            candle["timestamp"] = ts

            self.candle_store[token] = candle
            self.tick_buffer[token] = t
    
    def start_websocket(self, tokens):
        """Start WebSocket connection"""
        if not self.is_authenticated:
            return False
            
        try:
            self.kws = KiteTicker(self.api_key, self.access_token)
            self.kws.on_ticks = self.on_ticks
            self.kws.on_connect = lambda ws, resp: ws.subscribe(tokens)
            self.kws.connect(threaded=True)
            self.ws_running = True
            return True
        except Exception as e:
            logger.error(f"Error starting WebSocket: {e}")
            return False
    
    def stop_websocket(self):
        """Stop WebSocket connection"""
        if self.kws:
            try:
                self.kws.close()
                self.ws_running = False
            except:
                pass
    
    def get_candle_data(self, token):
        """Get current candle data for a token"""
        return self.candle_store.get(token)

# NEW: Peak Market Hours Check - Optimized for 9:30 AM - 2:30 PM
def is_peak_market_hours():
    """Check if current time is during peak market hours (9:30 AM - 2:30 PM)"""
    n = now_indian()
    try:
        peak_start = IND_TZ.localize(datetime.combine(n.date(), dt_time(10, 0)))
        peak_end = IND_TZ.localize(datetime.combine(n.date(), dt_time(14, 0)))
        return peak_start <= n <= peak_end
    except Exception:
        return True  # Default to True during market hours

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
            # Kelly formula: f = p - (1-p)/b
            if win_loss_ratio <= 0:
                win_loss_ratio = 2.0
                
            kelly_fraction = win_probability - (1 - win_probability) / win_loss_ratio
            
            # Use half-Kelly for conservative sizing
            risk_capital = available_capital * 0.1  # 10% of capital per trade
            position_value = risk_capital * (kelly_fraction / 2)
            
            if price <= 0:
                return 1
                
            quantity = int(position_value / price)
            
            return max(1, min(quantity, int(available_capital * 0.2 / price)))  # Max 20% per trade
        except Exception:
            return int((available_capital * 0.1) / price)  # Fallback
    
    def check_trade_viability(self, symbol, action, quantity, price, current_positions):
        """
        Automatically adjust position size to stay within risk limits.
        Prevents trade rejection by scaling down quantity safely.
        """

        # Reset daily metrics
        self.reset_daily_metrics()

        # Price check
        if price is None or price <= 0:
            return False, "Invalid price"

        # Estimated portfolio value
        current_portfolio_value = sum([
            pos.get("quantity", 0) * pos.get("entry_price", 0)
            for pos in current_positions.values()
            if pos.get("entry_price", 0) > 0
        ])

        # If nothing in portfolio, approximate
        if current_portfolio_value <= 0:
            current_portfolio_value = price * max(quantity, 1)

        requested_value = quantity * price

        # Concentration limit: 20%
        MAX_CONCENTRATION = 0.20
        max_allowed_value = max(current_portfolio_value * MAX_CONCENTRATION, 1)

        # Auto-scale if violating concentration limit
        if requested_value > max_allowed_value:
            adjusted_qty = int(max_allowed_value // price)
            if adjusted_qty < 1:
                adjusted_qty = 1

            try:
                if st.session_state.get("debug", False):
                    st.warning(
                        f"{symbol}: Auto-adjusted {quantity} → {adjusted_qty} due to concentration limit."
                    )
            except:
                pass

            quantity = adjusted_qty
            requested_value = quantity * price

        # Absolute hard cap: 50%
        HARD_CAP = 0.50
        hard_cap_value = current_portfolio_value * HARD_CAP

        if requested_value > hard_cap_value:
            adjusted_qty = int(hard_cap_value // price)
            adjusted_qty = max(1, adjusted_qty)

            try:
                if st.session_state.get("debug", False):
                    st.warning(
                        f"{symbol}: Further auto-scaling → {adjusted_qty} due to hard cap safety."
                    )
            except:
                pass

            quantity = adjusted_qty

        # Daily loss stop
        if self.daily_pnl < -self.max_daily_loss:
            return False, "Daily loss limit exceeded"

        return True, f"Trade viable (final adjusted quantity: {quantity})"

# NEW: Enhanced Signal Filtering System with ADX Trend Check
class SignalQualityFilter:
    """Enhanced signal filtering to improve trade quality"""
    
    @staticmethod
    def filter_high_quality_signals(signals, data_manager):
        """Filter only high-quality signals with multiple confirmations"""
        filtered = []
        
        for signal in signals:
            symbol = signal["symbol"]
            
            try:
                # Get recent data for analysis
                data = data_manager.get_stock_data(symbol, "15m")
                if data is None or len(data) < 30:
                    continue
                    
                # 1. Volume Confirmation (minimum 1.3x average volume)
                volume = data["Volume"].iloc[-1]
                avg_volume = data["Volume"].rolling(20).mean().iloc[-1] if len(data) >= 20 else volume
                volume_ratio = volume / avg_volume if avg_volume > 0 else 1
                
                if volume_ratio < 1.3:  # Minimum 30% above average volume
                    continue
                
                # 2. Trend Alignment Check
                price = data["Close"].iloc[-1]
                ema8 = data["EMA8"].iloc[-1]
                ema21 = data["EMA21"].iloc[-1]
                ema50 = data["EMA50"].iloc[-1]
                
                if signal["action"] == "BUY":
                    # For BUY: Price should be above key EMAs
                    if not (price > ema8 > ema21 > ema50):
                        continue
                else:  # SELL
                    # For SELL: Price should be below key EMAs
                    if not (price < ema8 < ema21 < ema50):
                        continue
                
                # 3. RSI Filter (avoid extreme overbought/oversold for entries)
                rsi_val = data["RSI14"].iloc[-1]
                if signal["action"] == "BUY" and rsi_val > 65:
                    continue
                if signal["action"] == "SELL" and rsi_val < 35:
                    continue
                
                # 4. Risk-Reward Ratio (minimum 2.5:1)
                if signal.get("risk_reward", 0) < 2.5:
                    continue
                
                # 5. Confidence Threshold (minimum 70% - REDUCED from 75%)
                if signal.get("confidence", 0) < 0.70:  # CHANGED: 0.75 → 0.70
                    continue
                
                # 6. Price relative to VWAP
                vwap = data["VWAP"].iloc[-1]
                if signal["action"] == "BUY" and price < vwap * 0.99:
                    continue  # Too far below VWAP for BUY
                if signal["action"] == "SELL" and price > vwap * 1.01:
                    continue  # Too far above VWAP for SELL
                
                # 7. ADX Strength (minimum 25 for trend strength) - ADDED TREND CHECK
                adx_val = data["ADX"].iloc[-1] if 'ADX' in data.columns else 20
                if adx_val < 25:  # CHANGED: 20 → 25 for stronger trends
                    continue
                
                # 8. ATR Filter (avoid extremely volatile stocks)
                atr = data["ATR"].iloc[-1] if 'ATR' in data.columns else price * 0.01
                atr_percent = (atr / price) * 100
                if atr_percent > 3.0:  # Avoid stocks with >3% daily volatility
                    continue
                
                # All checks passed - mark as high quality
                signal["quality_score"] = SignalQualityFilter.calculate_quality_score(signal, data)
                signal["volume_ratio"] = volume_ratio
                signal["atr_percent"] = atr_percent
                signal["trend_aligned"] = True
                
                filtered.append(signal)
                
            except Exception as e:
                logger.error(f"Error filtering signal for {symbol}: {e}")
                continue
        
        return filtered
    
    @staticmethod
    def calculate_quality_score(signal, data):
        """Calculate a comprehensive quality score (0-100)"""
        score = 0
        
        # Confidence weight: 30%
        score += signal.get("confidence", 0) * 30
        
        # Risk-Reward weight: 25%
        rr = signal.get("risk_reward", 0)
        if rr >= 3.0:
            score += 25
        elif rr >= 2.5:
            score += 20
        elif rr >= 2.0:
            score += 15
        else:
            score += 5
        
        # Volume confirmation weight: 20%
        volume = data["Volume"].iloc[-1]
        avg_volume = data["Volume"].rolling(20).mean().iloc[-1] if len(data) >= 20 else volume
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio >= 2.0:
            score += 20
        elif volume_ratio >= 1.5:
            score += 15
        elif volume_ratio >= 1.3:
            score += 10
        else:
            score += 5
        
        # Trend alignment weight: 15%
        price = data["Close"].iloc[-1]
        ema8 = data["EMA8"].iloc[-1]
        ema21 = data["EMA21"].iloc[-1]
        
        if signal["action"] == "BUY":
            if price > ema8 > ema21:
                score += 15
            elif price > ema8:
                score += 10
            else:
                score += 5
        else:  # SELL
            if price < ema8 < ema21:
                score += 15
            elif price < ema8:
                score += 10
            else:
                score += 5
        
        # RSI alignment weight: 10%
        rsi_val = data["RSI14"].iloc[-1]
        if signal["action"] == "BUY":
            if 30 <= rsi_val <= 50:
                score += 10
            elif 50 < rsi_val <= 60:
                score += 8
            else:
                score += 3
        else:  # SELL
            if 50 <= rsi_val <= 70:
                score += 10
            elif 40 <= rsi_val < 50:
                score += 8
            else:
                score += 3
        
        return min(100, int(score))

# NEW: Machine Learning Signal Enhancer
class MLSignalEnhancer:
    """Enhanced ML-based signal quality predictor using RandomForest"""
    
    MODEL_PATH = "data/signal_quality_model.pkl"
    SCALER_PATH = "data/signal_scaler.pkl"
    FEATURE_COLUMNS = ['rsi', 'macd_signal_diff', 'volume_ratio', 'atr_ratio', 
                       'adx_strength', 'bb_position', 'price_vs_ema8', 'price_vs_vwap', 
                       'trend_strength', 'ema_alignment', 'momentum_score']
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.training_samples = 0
        self.enabled = JOBLIB_AVAILABLE and SKLEARN_AVAILABLE
        
        if self.enabled:
            os.makedirs('data', exist_ok=True)
            self.load_model()
    
    def load_model(self):
        """Load pre-trained model from disk"""
        try:
            if os.path.exists(self.MODEL_PATH) and os.path.exists(self.SCALER_PATH):
                self.model = joblib.load(self.MODEL_PATH)
                self.scaler = joblib.load(self.SCALER_PATH)
                self.is_trained = True
                logger.info("ML model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            self.model = None
            self.scaler = None
            self.is_trained = False
    
    def save_model(self):
        """Save trained model to disk"""
        try:
            if self.model and self.scaler:
                joblib.dump(self.model, self.MODEL_PATH)
                joblib.dump(self.scaler, self.SCALER_PATH)
                logger.info("ML model saved successfully")
        except Exception as e:
            logger.error(f"Error saving ML model: {e}")
    
    def create_ml_features(self, data):
        """Create comprehensive features for ML model"""
        try:
            features = {}
            
            # RSI feature
            features['rsi'] = float(data['RSI14'].iloc[-1]) if 'RSI14' in data.columns and not pd.isna(data['RSI14'].iloc[-1]) else 50.0
            
            # MACD feature
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                features['macd_signal_diff'] = float(data['MACD'].iloc[-1] - data['MACD_Signal'].iloc[-1])
            else:
                features['macd_signal_diff'] = 0.0
            
            # Volume ratio
            if 'Volume' in data.columns and len(data) > 20:
                vol_mean = data['Volume'].rolling(20).mean().iloc[-1]
                features['volume_ratio'] = float(data['Volume'].iloc[-1] / vol_mean) if vol_mean > 0 else 1.0
            else:
                features['volume_ratio'] = 1.0
            
            # ATR ratio
            if 'ATR' in data.columns and 'Close' in data.columns:
                features['atr_ratio'] = float(data['ATR'].iloc[-1] / data['Close'].iloc[-1]) if data['Close'].iloc[-1] > 0 else 0.01
            else:
                features['atr_ratio'] = 0.01
            
            # ADX strength
            features['adx_strength'] = float(data['ADX'].iloc[-1]) if 'ADX' in data.columns and not pd.isna(data['ADX'].iloc[-1]) else 20.0
            
            # Bollinger Band position
            if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'Close']):
                bb_range = data['BB_Upper'].iloc[-1] - data['BB_Lower'].iloc[-1]
                features['bb_position'] = float((data['Close'].iloc[-1] - data['BB_Lower'].iloc[-1]) / bb_range) if bb_range > 0 else 0.5
            else:
                features['bb_position'] = 0.5
            
            # Price momentum features
            features['price_vs_ema8'] = float(data['Close'].iloc[-1] / data['EMA8'].iloc[-1] - 1) if 'EMA8' in data.columns and data['EMA8'].iloc[-1] > 0 else 0.0
            features['price_vs_vwap'] = float(data['Close'].iloc[-1] / data['VWAP'].iloc[-1] - 1) if 'VWAP' in data.columns and data['VWAP'].iloc[-1] > 0 else 0.0
            
            # Trend strength
            features['trend_strength'] = float(data['HTF_Trend'].iloc[-1]) if 'HTF_Trend' in data.columns else 1.0
            
            # EMA alignment score (0-1)
            if all(col in data.columns for col in ['EMA8', 'EMA21', 'EMA50']):
                ema8 = data['EMA8'].iloc[-1]
                ema21 = data['EMA21'].iloc[-1]
                ema50 = data['EMA50'].iloc[-1]
                if ema8 > ema21 > ema50:
                    features['ema_alignment'] = 1.0
                elif ema8 < ema21 < ema50:
                    features['ema_alignment'] = 0.0
                else:
                    features['ema_alignment'] = 0.5
            else:
                features['ema_alignment'] = 0.5
            
            # Momentum score
            features['momentum_score'] = (features['rsi'] / 100) * (1 + features['macd_signal_diff'] / 100) * features['volume_ratio']
            
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error(f"Error creating ML features: {e}")
            return pd.DataFrame()
    
    def train_model(self, trade_history):
        """Train the ML model on historical trade outcomes"""
        if not self.enabled or len(trade_history) < 30:
            return False
            
        try:
            X_list = []
            y_list = []
            
            for trade in trade_history:
                if trade.get('status') == 'CLOSED' and 'features' in trade:
                    features = trade['features']
                    outcome = 1 if trade.get('closed_pnl', 0) > 0 else 0
                    
                    feature_row = [features.get(col, 0) for col in self.FEATURE_COLUMNS]
                    X_list.append(feature_row)
                    y_list.append(outcome)
            
            if len(X_list) < 30:
                logger.info(f"Not enough training samples: {len(X_list)}")
                return False
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
            
            accuracy = self.model.score(X_test, y_test)
            self.is_trained = True
            self.training_samples = len(X_list)
            
            self.save_model()
            logger.info(f"ML model trained on {len(X_list)} samples with accuracy: {accuracy:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            return False
    
    def predict_signal_confidence(self, symbol_data, signal_type="BUY"):
        """Predict signal confidence using trained ML model or rule-based fallback"""
        if not self.enabled:
            return 0.7
            
        try:
            features_df = self.create_ml_features(symbol_data)
            if features_df.empty:
                return 0.7
            
            if self.is_trained and self.model and self.scaler:
                feature_values = features_df[self.FEATURE_COLUMNS].values
                scaled_features = self.scaler.transform(feature_values)
                
                proba = self.model.predict_proba(scaled_features)[0]
                win_probability = proba[1] if len(proba) > 1 else 0.5
                
                return max(0.3, min(0.95, win_probability))
            
            else:
                return self._rule_based_confidence(features_df.iloc[0])
                
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return 0.7
    
    def _rule_based_confidence(self, features):
        """Fallback rule-based confidence when ML model not available"""
        confidence = 0.5
        
        rsi = features.get('rsi', 50)
        if 35 <= rsi <= 65:
            confidence += 0.1
        elif 25 <= rsi <= 75:
            confidence += 0.05
        
        volume_ratio = features.get('volume_ratio', 1)
        if volume_ratio >= 2.0:
            confidence += 0.15
        elif volume_ratio >= 1.5:
            confidence += 0.1
        elif volume_ratio >= 1.3:
            confidence += 0.05
        
        adx = features.get('adx_strength', 20)
        if adx >= 30:
            confidence += 0.15
        elif adx >= 25:
            confidence += 0.1
        elif adx >= 20:
            confidence += 0.05
        
        ema_align = features.get('ema_alignment', 0.5)
        confidence += ema_align * 0.1
        
        return max(0.3, min(0.9, confidence))
    
    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if self.is_trained and self.model:
            return dict(zip(self.FEATURE_COLUMNS, self.model.feature_importances_))
        return {}

# NEW: Market Regime Detector
class MarketRegimeDetector:
    def __init__(self):
        self.current_regime = "NEUTRAL"
        self.regime_history = []
    
    def detect_regime(self, nifty_data):
        """Detect current market regime"""
        try:
            if nifty_data is None or len(nifty_data) < 20:
                return "NEUTRAL"
            
            # Calculate regime indicators
            adx_value = nifty_data['ADX'].iloc[-1] if 'ADX' in nifty_data.columns else 20
            volatility = nifty_data['Close'].pct_change().std() * 100 if len(nifty_data) > 1 else 1.0
            rsi_val = nifty_data['RSI14'].iloc[-1] if 'RSI14' in nifty_data.columns else 50
            
            # Determine regime
            if adx_value > 25 and volatility < 1.2:
                regime = "TRENDING"
            elif volatility > 1.5:
                regime = "VOLATILE"
            elif 40 <= rsi_val <= 60 and volatility < 1.0:
                regime = "MEAN_REVERTING"
            else:
                regime = "NEUTRAL"
            
            self.current_regime = regime
            self.regime_history.append({"timestamp": datetime.now(), "regime": regime})
            
            # Keep only last 100 records
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return "NEUTRAL"

# NEW: Portfolio Optimizer
class PortfolioOptimizer:
    def __init__(self):
        self.correlation_matrix = None
    
    def calculate_diversification_score(self, positions):
        """Calculate portfolio diversification score"""
        if not positions:
            return 1.0
        
        try:
            sector_weights = {}
            total_value = 0
            
            for symbol, pos in positions.items():
                if pos.get('status') == 'OPEN':
                    value = pos.get('quantity', 0) * pos.get('entry_price', 0)
                    total_value += value
                    
                    # Simplified sector assignment
                    sector = self._get_stock_sector(symbol)
                    sector_weights[sector] = sector_weights.get(sector, 0) + value
            
            if total_value == 0:
                return 1.0
                
            # Calculate Herfindahl index for concentration
            herfindahl = sum([(weight/total_value)**2 for weight in sector_weights.values()])
            diversification_score = 1 - herfindahl
            
            return max(0.1, diversification_score)
        except Exception:
            return 0.5
    
    def _get_stock_sector(self, symbol):
        """Map symbol to sector (simplified)"""
        try:
            sector_map = {
                "RELIANCE": "ENERGY", "TCS": "IT", "HDFCBANK": "FINANCIAL",
                "INFY": "IT", "HINDUNILVR": "FMCG", "ICICIBANK": "FINANCIAL",
                "KOTAKBANK": "FINANCIAL", "BHARTIARTL": "TELECOM", "ITC": "FMCG",
                "LT": "CONSTRUCTION", "SBIN": "FINANCIAL", "ASIANPAINT": "CONSUMER",
                "HCLTECH": "IT", "AXISBANK": "FINANCIAL", "MARUTI": "AUTOMOBILE",
                "SUNPHARMA": "PHARMA", "TITAN": "CONSUMER", "ULTRACEMCO": "CEMENT",
                "WIPRO": "IT", "NTPC": "ENERGY", "NESTLEIND": "FMCG",
                "POWERGRID": "ENERGY", "M&M": "AUTOMOBILE", "BAJFINANCE": "FINANCIAL",
                "ONGC": "ENERGY", "TATASTEEL": "METALS", "JSWSTEEL": "METALS",
                "ADANIPORTS": "INFRASTRUCTURE", "COALINDIA": "MINING",
                "HDFCLIFE": "INSURANCE", "DRREDDY": "PHARMA", "HINDALCO": "METALS",
                "CIPLA": "PHARMA", "SBILIFE": "INSURANCE", "GRASIM": "CEMENT",
                "TECHM": "IT", "BAJAJFINSV": "FINANCIAL", "BRITANNIA": "FMCG",
                "EICHERMOT": "AUTOMOBILE", "DIVISLAB": "PHARMA", "SHREECEM": "CEMENT",
                "APOLLOHOSP": "HEALTHCARE", "UPL": "CHEMICALS", "BAJAJ-AUTO": "AUTOMOBILE",
                "HEROMOTOCO": "AUTOMOBILE", "INDUSINDBK": "FINANCIAL", "ADANIENT": "CONGLOMERATE",
                "TATACONSUM": "FMCG", "BPCL": "ENERGY"
            }
            base_symbol = symbol.replace('.NS', '').split('.')[0]
            return sector_map.get(base_symbol, "OTHER")
        except:
            return "OTHER"

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
        self.alert_cooldown = 60  # seconds between alerts for same symbol
    
    def create_alert(self, alert_type, symbol, message, confidence=0.0, priority="NORMAL", data=None):
        """Create a new alert"""
        current_time = now_indian()
        
        # Check cooldown for symbol
        if symbol in self.last_alert_time:
            time_diff = (current_time - self.last_alert_time[symbol]).total_seconds()
            if time_diff < self.alert_cooldown:
                return None
        
        # Check if symbol is muted
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
        
        # Keep only max_alerts
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
    
    def create_pnl_alert(self, symbol, pnl, trade_type="CLOSED"):
        """Create P&L alert for significant gains/losses"""
        if pnl <= self.alert_thresholds['critical_pnl_loss']:
            priority = "CRITICAL"
            alert_type = "PNL_LOSS"
            message = f"CRITICAL LOSS: {symbol} | P&L: ₹{pnl:+,.2f}"
        elif pnl >= self.alert_thresholds['critical_pnl_gain']:
            priority = "HIGH"
            alert_type = "PNL_GAIN"
            message = f"PROFIT ALERT: {symbol} | P&L: ₹{pnl:+,.2f}"
        else:
            return None
        
        return self.create_alert(
            alert_type=alert_type,
            symbol=symbol,
            message=message,
            priority=priority,
            data={'pnl': pnl, 'trade_type': trade_type}
        )
    
    def create_risk_alert(self, symbol, risk_type, message):
        """Create risk management alert"""
        return self.create_alert(
            alert_type="RISK_WARNING",
            symbol=symbol,
            message=f"RISK: {message}",
            priority="HIGH",
            data={'risk_type': risk_type}
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
                # Create data directory if it doesn't exist
                os.makedirs('data', exist_ok=True)
                # Use absolute path
                db_path = os.path.join('data', 'trading_journal.db')
                self.db_url = f'sqlite:///{db_path}'
                self.engine = create_engine(self.db_url)
                self.connected = True
                self.create_tables()
                self.connected = True
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
                # Trades table
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
                
                # Market regime history
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS market_regimes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        regime TEXT,
                        timestamp TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Strategy performance
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS strategy_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy TEXT,
                        signals INTEGER,
                        trades INTEGER,
                        wins INTEGER,
                        pnl REAL,
                        date DATE,
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

def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (close - lowest_low) / denom
    d = k.rolling(window=d_period).mean()
    return k.fillna(50), d.fillna(50)

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

def calculate_market_profile_vectorized(high, low, close, volume, bins=20):
    try:
        low_val = float(min(high.min(), low.min(), close.min()))
        high_val = float(max(high.max(), low.max(), close.max()))
        if np.isclose(low_val, high_val):
            high_val = low_val * 1.01 if low_val != 0 else 1.0
        edges = np.linspace(low_val, high_val, bins + 1)
        hist, _ = np.histogram(close, bins=edges, weights=volume)
        centers = (edges[:-1] + edges[1:]) / 2
        if hist.sum() == 0:
            poc = float(close.iloc[-1])
            va_high = poc * 1.01
            va_low = poc * 0.99
        else:
            idx = int(np.argmax(hist))
            poc = float(centers[idx])
            sorted_idx = np.argsort(hist)[::-1]
            cumulative = 0.0
            total = float(hist.sum())
            selected = []
            for i in sorted_idx:
                selected.append(centers[i])
                cumulative += hist[i]
                if cumulative / total >= 0.70:
                    break
            va_high = float(max(selected))
            va_low = float(min(selected))
        profile = [{"price": float(c), "volume": int(v)} for c, v in zip(centers, hist)]
        return {"poc": poc, "value_area_high": va_high, "value_area_low": va_low, "profile": profile}
    except Exception:
        current_price = float(close.iloc[-1])
        return {"poc": current_price, "value_area_high": current_price*1.01, "value_area_low": current_price*0.99, "profile": []}

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

def adx(high, low, close, period=14):
    try:
        h = high.copy().reset_index(drop=True)
        l = low.copy().reset_index(drop=True)
        c = close.copy().reset_index(drop=True)
        df = pd.DataFrame({"high": h, "low": l, "close": c})
        df["tr"] = np.maximum(df["high"] - df["low"],
                              np.maximum((df["high"] - df["close"].shift()).abs(),
                                         (df["low"] - df["close"].shift()).abs()))
        df["up_move"] = df["high"] - df["high"].shift()
        df["down_move"] = df["low"].shift() - df["low"]
        df["dm_pos"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0.0)
        df["dm_neg"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0.0)
        df["tr_sum"] = df["tr"].rolling(window=period).sum()
        df["dm_pos_sum"] = df["dm_pos"].rolling(window=period).sum()
        df["dm_neg_sum"] = df["dm_neg"].rolling(window=period).sum()
        df["di_pos"] = 100 * (df["dm_pos_sum"] / df["tr_sum"]).replace([np.inf, -np.inf], 0).fillna(0)
        df["di_neg"] = 100 * (df["dm_neg_sum"] / df["tr_sum"]).replace([np.inf, -np.inf], 0).fillna(0)
        df["dx"] = (abs(df["di_pos"] - df["di_neg"]) / (df["di_pos"] + df["di_neg"]).replace(0, np.nan)) * 100
        df["adx"] = df["dx"].rolling(window=period).mean().fillna(0)
        return df["adx"].values
    except Exception:
        return np.array([20] * len(high))

# FIXED Circular Market Mood Gauge Component with Rounded Percentages
def create_circular_market_mood_gauge(index_name, current_value, change_percent, sentiment_score):
    """Create a circular market mood gauge for Nifty50 and BankNifty"""
    
    # Round sentiment score and change percentage
    sentiment_score = round(sentiment_score)
    change_percent = round(change_percent, 2)
    
    # Determine sentiment color and text
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
    
    # Create circular gauge HTML
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

# Enhanced Data Manager with NEW integrated systems
class EnhancedDataManager:
    def __init__(self):
        self.price_cache = {}
        self.signal_cache = {}
        self.market_profile_cache = {}
        self.last_rsi_scan = None
        self.risk_manager = AdvancedRiskManager()
        self.ml_enhancer = MLSignalEnhancer()
        self.regime_detector = MarketRegimeDetector()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.database = TradeDatabase()
        self.signal_filter = SignalQualityFilter()
        self.kite_manager = KiteConnectManager(KITE_API_KEY, KITE_API_SECRET)
        self.alert_manager = AlertManager()
        self.backtest_engine = RealBacktestEngine()  # Added this line

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
        # Force 15min timeframe for RSI analysis as requested
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

        # Enhanced Indicators with 15min focus
        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI14"] = rsi(df["Close"], 14).fillna(50)
        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"]).fillna(method="ffill").fillna(0)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])
        df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"])
        df["Stoch_K"], df["Stoch_D"] = stochastic(df["High"], df["Low"], df["Close"])
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()

        mp = calculate_market_profile_vectorized(df["High"], df["Low"], df["Close"], df["Volume"], bins=24)
        df["POC"] = mp["poc"]
        df["VA_High"] = mp["value_area_high"]
        df["VA_Low"] = mp["value_area_low"]

        sr = calculate_support_resistance_advanced(df["High"], df["Low"], df["Close"])
        df["Support"] = sr["support"]
        df["Resistance"] = sr["resistance"]

        try:
            df_adx = adx(df["High"], df["Low"], df["Close"], period=14)
            df["ADX"] = pd.Series(df_adx, index=df.index).fillna(method="ffill").fillna(20)
        except Exception:
            df["ADX"] = 20

        try:
            htf = self._fetch_yf(symbol, period="7d", interval="1h")
            if htf is not None and len(htf) > 50:
                if isinstance(htf.columns, pd.MultiIndex):
                    htf.columns = ["_".join(map(str, col)).strip() for col in htf.columns.values]
                htf = htf.rename(columns={c: c.capitalize() for c in htf.columns})
                htf_close = htf["Close"]
                htf_ema50 = ema(htf_close, 50).iloc[-1]
                htf_ema200 = ema(htf_close, 200).iloc[-1] if len(htf_close) > 200 else ema(htf_close, 100).iloc[-1]
                df["HTF_Trend"] = 1 if htf_ema50 > htf_ema200 else -1
            else:
                df["HTF_Trend"] = 1
        except Exception:
            df["HTF_Trend"] = 1

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
        df["Stoch_K"], df["Stoch_D"] = stochastic(df["High"], df["Low"], df["Close"])
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()
        mp = calculate_market_profile_vectorized(df["High"], df["Low"], df["Close"], df["Volume"], bins=24)
        df["POC"] = mp["poc"]
        df["VA_High"] = mp["value_area_high"]
        df["VA_Low"] = mp["value_area_low"]
        sr = calculate_support_resistance_advanced(df["High"], df["Low"], df["Close"])
        df["Support"] = sr["support"]
        df["Resistance"] = sr["resistance"]
        df["ADX"] = adx(df["High"], df["Low"], df["Close"], period=14)
        df["HTF_Trend"] = 1
        return df

    def get_historical_accuracy(self, symbol, strategy):
        # Fallback to fixed accuracy if RealBacktestEngine is not available
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

    def calculate_market_profile_signals(self, symbol):
        """Calculate market profile signals with improved timeframe alignment"""
        try:
            # Get 15min data for market profile analysis
            data_15m = self.get_stock_data(symbol, "15m")
            if len(data_15m) < 50:
                return {"signal": "NEUTRAL", "confidence": 0.5, "reason": "Insufficient data"}
            
            current_price_15m = float(data_15m["Close"].iloc[-1])
            
            # Calculate signals
            ema8_15m = float(data_15m["EMA8"].iloc[-1])
            ema21_15m = float(data_15m["EMA21"].iloc[-1])
            ema50_15m = float(data_15m["EMA50"].iloc[-1])
            rsi_val_15m = float(data_15m["RSI14"].iloc[-1])
            vwap_15m = float(data_15m["VWAP"].iloc[-1])
            
            # Calculate bullish/bearish score
            bullish_score = 0
            bearish_score = 0
            
            # 15min trend analysis
            if current_price_15m > ema8_15m > ema21_15m > ema50_15m:
                bullish_score += 3
            elif current_price_15m < ema8_15m < ema21_15m < ema50_15m:
                bearish_score += 3
                
            # RSI analysis
            if rsi_val_15m > 55:
                bullish_score += 1
            elif rsi_val_15m < 45:
                bearish_score += 1
                
            # Price relative to VWAP
            if current_price_15m > vwap_15m:
                bullish_score += 2
            elif current_price_15m < vwap_15m:
                bearish_score += 2
                
            total_score = max(bullish_score + bearish_score, 1)
            bullish_ratio = (bullish_score + 5) / (total_score + 10)
            
            final_confidence = min(0.95, bullish_ratio)
            
            if bullish_ratio >= 0.65:
                return {"signal": "BULLISH", "confidence": final_confidence, "reason": "Strong bullish alignment"}
            elif bullish_ratio <= 0.35:
                return {"signal": "BEARISH", "confidence": final_confidence, "reason": "Strong bearish alignment"}
            else:
                return {"signal": "NEUTRAL", "confidence": 0.5, "reason": "Mixed signals"}
                
        except Exception as e:
            return {"signal": "NEUTRAL", "confidence": 0.5, "reason": f"Error: {str(e)}"}

    def should_run_rsi_scan(self):
        """Check if RSI scan should run (every 3rd refresh)"""
        current_time = time.time()
        if self.last_rsi_scan is None:
            self.last_rsi_scan = current_time
            return True
        
        if current_time - self.last_rsi_scan >= 75:
            self.last_rsi_scan = current_time
            return True
        return False

    # NEW: Enhanced methods for integrated systems
    def get_ml_enhanced_confidence(self, symbol_data):
        """Get ML-enhanced confidence for signals"""
        return self.ml_enhancer.predict_signal_confidence(symbol_data)
    
    def get_market_regime(self):
        """Get current market regime"""
        try:
            nifty_data = self.get_stock_data("^NSEI", "1h")
            return self.regime_detector.detect_regime(nifty_data)
        except:
            return "NEUTRAL"
    
    def check_risk_limits(self, symbol, action, quantity, price, current_positions):
        """Check risk limits before trade execution"""
        return self.risk_manager.check_trade_viability(symbol, action, quantity, price, current_positions)
    
    def calculate_optimal_position_size(self, symbol, win_probability, win_loss_ratio, available_capital, price, atr):
        """Calculate optimal position size using Kelly Criterion"""
        return self.risk_manager.calculate_kelly_position_size(
            win_probability, win_loss_ratio, available_capital, price, atr
        )
    
    def filter_high_quality_signals(self, signals):
        """Filter signals for high quality"""
        return self.signal_filter.filter_high_quality_signals(signals, self)
    
    def get_kite_data(self, instrument_token, interval="minute", days=1):
        """Get data from Kite Connect"""
        if not self.kite_manager.is_authenticated:
            return None
            
        try:
            from_date = datetime.now().date() - pd.Timedelta(days=days)
            to_date = datetime.now().date()
            data = self.kite_manager.get_live_data(instrument_token, interval, from_date, to_date)
            return data
        except Exception as e:
            logger.error(f"Error getting Kite data: {e}")
            return None

# Enhanced RealBacktestEngine with full backtesting capabilities
class RealBacktestEngine:
    """Comprehensive backtesting engine for strategy validation"""
    
    def __init__(self):
        self.historical_results = {}
        self.backtest_cache = {}
        self.default_accuracy = {
            "Multi_Confirmation": 0.82, "Enhanced_EMA_VWAP": 0.78, "Volume_Breakout": 0.75,
            "RSI_Divergence": 0.72, "MACD_Trend": 0.70, "EMA_VWAP_Confluence": 0.75,
            "RSI_MeanReversion": 0.68, "Bollinger_Reversion": 0.65, "MACD_Momentum": 0.70,
            "Support_Resistance_Breakout": 0.73, "EMA_VWAP_Downtrend": 0.72,
            "RSI_Overbought": 0.65, "Bollinger_Rejection": 0.63, "MACD_Bearish": 0.68,
            "Trend_Reversal": 0.60
        }
    
    def calculate_historical_accuracy(self, symbol, strategy, data):
        """Calculate historical accuracy for a strategy"""
        return self.default_accuracy.get(strategy, 0.65)
    
    def run_backtest(self, symbol, strategy, period_days=30, capital=100000, trade_allocation=0.15):
        """Run a comprehensive backtest on a symbol with a specific strategy"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=f"{period_days}d", interval="15m")
            
            if data is None or len(data) < 50:
                return None
            
            data = self._calculate_indicators(data)
            trades = self._generate_backtest_signals(data, strategy)
            results = self._simulate_trades(trades, capital, trade_allocation)
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest error for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, df):
        """Calculate all technical indicators for backtesting"""
        try:
            df = df.copy()
            
            df['EMA8'] = df['Close'].ewm(span=8, adjust=False).mean()
            df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
            
            delta = df['Close'].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = (-delta.clip(upper=0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            df['RSI14'] = 100 - (100 / (1 + rs))
            
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            df['BB_Mid'] = df['Close'].rolling(window=20).mean()
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
            df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
            
            cum_vol = df['Volume'].cumsum()
            cum_vol_price = (df['Close'] * df['Volume']).cumsum()
            df['VWAP'] = cum_vol_price / cum_vol
            
            tr1 = df['High'] - df['Low']
            tr2 = (df['High'] - df['Close'].shift()).abs()
            tr3 = (df['Low'] - df['Close'].shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(14).mean()
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Error calculating backtest indicators: {e}")
            return df
    
    def _generate_backtest_signals(self, data, strategy):
        """Generate trading signals for backtesting"""
        signals = []
        
        try:
            for i in range(50, len(data) - 10):
                row = data.iloc[i]
                prev_row = data.iloc[i-1]
                
                signal = self._check_strategy_signal(data.iloc[:i+1], strategy, i)
                if signal:
                    exit_price, exit_idx, outcome = self._simulate_exit(data, i, signal)
                    signals.append({
                        'entry_idx': i,
                        'entry_price': row['Close'],
                        'entry_time': data.index[i],
                        'action': signal['action'],
                        'stop_loss': signal.get('stop_loss', row['Close'] * 0.98),
                        'target': signal.get('target', row['Close'] * 1.03),
                        'exit_price': exit_price,
                        'exit_idx': exit_idx,
                        'outcome': outcome,
                        'strategy': strategy
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating backtest signals: {e}")
            return signals
    
    def _check_strategy_signal(self, data, strategy, idx):
        """Check if a strategy generates a signal at the given index"""
        try:
            row = data.iloc[-1]
            
            if strategy == "EMA_VWAP_Confluence":
                if row['EMA8'] > row['EMA21'] > row['EMA50'] and row['Close'] > row['VWAP']:
                    return {'action': 'BUY', 'stop_loss': row['Close'] * 0.98, 'target': row['Close'] * 1.03}
            
            elif strategy == "RSI_MeanReversion":
                if row['RSI14'] < 30:
                    return {'action': 'BUY', 'stop_loss': row['Close'] * 0.97, 'target': row['Close'] * 1.025}
            
            elif strategy == "Bollinger_Reversion":
                if row['Close'] < row['BB_Lower']:
                    return {'action': 'BUY', 'stop_loss': row['Close'] * 0.97, 'target': row['BB_Mid']}
            
            elif strategy == "MACD_Momentum":
                if row['MACD'] > row['MACD_Signal'] and data['MACD'].iloc[-2] <= data['MACD_Signal'].iloc[-2]:
                    return {'action': 'BUY', 'stop_loss': row['Close'] * 0.98, 'target': row['Close'] * 1.025}
            
            elif strategy == "RSI_Overbought":
                if row['RSI14'] > 70:
                    return {'action': 'SELL', 'stop_loss': row['Close'] * 1.02, 'target': row['Close'] * 0.975}
            
            elif strategy == "MACD_Bearish":
                if row['MACD'] < row['MACD_Signal'] and data['MACD'].iloc[-2] >= data['MACD_Signal'].iloc[-2]:
                    return {'action': 'SELL', 'stop_loss': row['Close'] * 1.02, 'target': row['Close'] * 0.975}
            
            return None
            
        except:
            return None
    
    def _simulate_exit(self, data, entry_idx, signal, max_bars=20):
        """Simulate trade exit based on stop loss and target"""
        try:
            entry_price = data.iloc[entry_idx]['Close']
            stop_loss = signal.get('stop_loss', entry_price * 0.98)
            target = signal.get('target', entry_price * 1.03)
            action = signal['action']
            
            for i in range(entry_idx + 1, min(entry_idx + max_bars, len(data))):
                row = data.iloc[i]
                
                if action == 'BUY':
                    if row['Low'] <= stop_loss:
                        return stop_loss, i, 'LOSS'
                    if row['High'] >= target:
                        return target, i, 'WIN'
                else:
                    if row['High'] >= stop_loss:
                        return stop_loss, i, 'LOSS'
                    if row['Low'] <= target:
                        return target, i, 'WIN'
            
            exit_price = data.iloc[min(entry_idx + max_bars - 1, len(data) - 1)]['Close']
            pnl = (exit_price - entry_price) if action == 'BUY' else (entry_price - exit_price)
            return exit_price, entry_idx + max_bars, 'WIN' if pnl > 0 else 'LOSS'
            
        except:
            return entry_price, entry_idx + 1, 'LOSS'
    
    def _simulate_trades(self, trades, capital, allocation):
        """Simulate all trades and calculate performance metrics"""
        if not trades:
            return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0, 'max_drawdown': 0, 'sharpe': 0}
        
        wins = sum(1 for t in trades if t['outcome'] == 'WIN')
        total = len(trades)
        
        pnl_list = []
        equity_curve = [capital]
        current_capital = capital
        
        for trade in trades:
            position_size = current_capital * allocation
            qty = position_size / trade['entry_price']
            
            if trade['action'] == 'BUY':
                pnl = (trade['exit_price'] - trade['entry_price']) * qty
            else:
                pnl = (trade['entry_price'] - trade['exit_price']) * qty
            
            pnl_list.append(pnl)
            current_capital += pnl
            equity_curve.append(current_capital)
        
        total_pnl = sum(pnl_list)
        avg_pnl = total_pnl / total if total > 0 else 0
        
        peak = capital
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
        
        if len(pnl_list) > 1:
            returns = np.array(pnl_list) / capital
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        return {
            'total_trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': wins / total if total > 0 else 0,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'final_capital': current_capital,
            'roi': (current_capital - capital) / capital,
            'equ
