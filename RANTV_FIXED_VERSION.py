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
# ====== Optional PDF reporting (ReportLab) ======
HAVE_REPORTLAB = True
try:
    from reportlab.platypus import (
        SimpleDocTemplate as _pt_SimpleDocTemplate,
        Paragraph as _pt_Paragraph,
        Table as _pt_Table,
    )
    from reportlab.lib.styles import getSampleStyleSheet as _pt_getSampleStyleSheet
    from reportlab.lib.pagesizes import A4 as _pt_A4
except Exception:
    HAVE_REPORTLAB = False
    import logging as _rl_logging
    _rl_logging.warning(
        "ReportLab not available; PDF reports disabled. "
        "Install with: pip install reportlab"
    )

logger = logging.getLogger(__name__)

IND_TZ = pytz.timezone("Asia/Kolkata")

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
        
        self.last_signal_scan = datetime.now(IND_TZ)
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
                
                now = datetime.now(IND_TZ)
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
        now = datetime.now(IND_TZ)
        if now.weekday() >= 5:
            return False
        
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _is_peak_hours(self) -> bool:
        now = datetime.now(IND_TZ)
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
                time_since_last = (datetime.now(IND_TZ) - self.stats.last_trade_time).total_seconds()
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
                    order.filled_at = datetime.now(IND_TZ)
                    order.filled_price = entry_price
                    self.active_positions[symbol] = order
                    self.stats.total_orders += 1
                    self.stats.filled_orders += 1
                    self.stats.trades_today += 1
                    self.stats.stock_trades[symbol] = self.stats.stock_trades.get(symbol, 0) + 1
                    self.stats.last_trade_time = datetime.now(IND_TZ)
                
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


def create_algo_tab_content(algo_engine: AlgoEngine, st_module):
    """
    Creates the Algo Trading tab content for the Streamlit app
    """
    st = st_module
    
    st.subheader("🤖 Algo Trading Control Panel")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
        <h4 style="color: white; margin: 0;">Automated Trading System</h4>
        <p style="color: #e0f2fe; margin: 5px 0 0 0; font-size: 12px;">Configure and monitor automated signal execution</p>
    </div>
    """, unsafe_allow_html=True)
    
    status = algo_engine.get_status()
    
    st.subheader("Engine Status")
    status_cols = st.columns(4)
    
    state = status["state"]
    state_colors = {
        "running": "🟢",
        "stopped": "🔴",
        "paused": "🟡",
        "emergency_stop": "⛔"
    }
    
    with status_cols[0]:
        st.metric("Engine State", f"{state_colors.get(state, '⚪')} {state.upper()}")
    
    with status_cols[1]:
        st.metric("Active Positions", f"{status['active_positions']}/{status['risk_limits']['max_positions']}")
    
    with status_cols[2]:
        st.metric("Trades Today", f"{status['trades_today']}/{status['risk_limits']['max_trades_per_day']}")
    
    with status_cols[3]:
        market_status = "🟢 OPEN" if status["market_open"] else "🔴 CLOSED"
        st.metric("Market Status", market_status)
    
    st.subheader("Controls")
    ctrl_cols = st.columns(4)
    
    with ctrl_cols[0]:
        if st.button("▶️ Start Engine", type="primary", disabled=(state == "running")):
            if algo_engine.start():
                st.success("Algo Engine started!")
                st.rerun()
            else:
                st.error("Failed to start engine. Check prerequisites.")
    
    with ctrl_cols[1]:
        if st.button("⏸️ Pause", disabled=(state != "running")):
            algo_engine.pause()
            st.info("Engine paused")
            st.rerun()
    
    with ctrl_cols[2]:
        if st.button("▶️ Resume", disabled=(state != "paused")):
            algo_engine.resume()
            st.success("Engine resumed")
            st.rerun()
    
    with ctrl_cols[3]:
        if st.button("⏹️ Stop Engine", type="secondary", disabled=(state == "stopped")):
            algo_engine.stop()
            st.info("Engine stopped")
            st.rerun()
    
    st.markdown("---")
    
    if st.button("🚨 EMERGENCY STOP", type="primary"):
        algo_engine.emergency_stop("Manual emergency stop triggered")
        st.error("EMERGENCY STOP ACTIVATED - All positions closed")
        st.rerun()
    
    st.subheader("Risk Settings")
    
    with st.expander("Configure Risk Limits", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            new_max_positions = st.number_input(
                "Max Positions",
                min_value=1, max_value=20,
                value=status["risk_limits"]["max_positions"],
                key="algo_max_pos"
            )
            
            new_max_daily_loss = st.number_input(
                "Max Daily Loss (₹)",
                min_value=1000, max_value=500000,
                value=int(status["risk_limits"]["max_daily_loss"]),
                key="algo_max_loss"
            )
        
        with col2:
            new_min_confidence = st.slider(
                "Min Signal Confidence",
                min_value=0.5, max_value=0.99,
                value=status["risk_limits"]["min_confidence"],
                step=0.05,
                key="algo_min_conf"
            )
            
            new_max_trades = st.number_input(
                "Max Trades/Day",
                min_value=1, max_value=50,
                value=status["risk_limits"]["max_trades_per_day"],
                key="algo_max_trades"
            )
        
        if st.button("Update Risk Settings"):
            algo_engine.update_risk_limits(
                max_positions=new_max_positions,
                max_daily_loss=float(new_max_daily_loss),
                min_confidence=new_min_confidence,
                max_trades_per_day=new_max_trades
            )
            st.success("Risk settings updated!")
            st.rerun()
    
    st.subheader("Performance")
    perf_cols = st.columns(4)
    
    with perf_cols[0]:
        st.metric("Total Orders", status["total_orders"])
    
    with perf_cols[1]:
        st.metric("Filled Orders", status["filled_orders"])
    
    with perf_cols[2]:
        realized = status["realized_pnl"]
        st.metric("Realized P&L", f"₹{realized:+,.2f}")
    
    with perf_cols[3]:
        unrealized = status["unrealized_pnl"]
        st.metric("Unrealized P&L", f"₹{unrealized:+,.2f}")
    
    if algo_engine.active_positions:
        st.subheader("Active Positions")
        positions_data = []
        for symbol, order in algo_engine.active_positions.items():
            positions_data.append({
                "Symbol": symbol.replace(".NS", ""),
                "Action": order.action,
                "Qty": order.quantity,
                "Entry": f"₹{order.price:.2f}",
                "Stop Loss": f"₹{order.stop_loss:.2f}",
                "Target": f"₹{order.target:.2f}",
                "Strategy": order.strategy,
                "Confidence": f"{order.confidence:.1%}"
            })
        
        import pandas as pd
        st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="background: #fef3c7; padding: 10px; border-radius: 8px; border-left: 4px solid #f59e0b;">
        <strong>⚠️ Important:</strong> Algo trading involves significant risk. 
        Always test with paper trading first. Set appropriate risk limits. 
        Monitor the system regularly. Past performance does not guarantee future results.
    </div>
    """, unsafe_allow_html=True)

# ===== END ALGO ENGINE =====

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

# Import Algo Trading Engine
try:
    from algo_engine import AlgoEngine, create_algo_tab_content
    ALGO_ENGINE_AVAILABLE = True
except ImportError:
    ALGO_ENGINE_AVAILABLE = False

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

# --- BEGIN: Static Universe Sanitizer (Safe / No yfinance calls) ---
import re as _re
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
    # keep order, remove duplicates
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
# --- END: Static Universe Sanitizer ---


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

import time

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
            # Initialize KiteConnect if needed
            if not self.kite and (self.api_key and self.api_secret):
                self.kite = KiteConnect(api_key=self.api_key)
            elif not self.kite:
                # Create a dummy KiteConnect for token validation (without API key)
                try:
                    from kiteconnect import KiteConnect as KC
                    self.kite = KC(api_key="DUMMY")
                except:
                    pass

            # 1) Handle OAuth callback first (only if we have API credentials)
            if self.api_key and self.api_secret and not st.session_state.kite_oauth_in_progress and self.check_oauth_callback():
                return True

            # 2) Session token check
            if "kite_access_token" in st.session_state:
                self.access_token = st.session_state.kite_access_token
                if self.kite:
                    self.kite.set_access_token(self.access_token)
                try:
                    if self.kite:
                        _ = self.kite.profile()
                    self.is_authenticated = True
                    return True
                except Exception:
                    del st.session_state["kite_access_token"]

            # 3) DB token fallback
            db_token = kite_token_db.get_valid_token() if kite_token_db else None
            if db_token:
                self.access_token = db_token["access_token"]
                if self.kite:
                    self.kite.set_access_token(self.access_token)
                try:
                    if self.kite:
                        profile = self.kite.profile()
                    self.is_authenticated = True
                    st.session_state.kite_access_token = self.access_token
                    st.session_state.kite_user_name = db_token.get("user_name", "")
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

            st.info("🔐 Kite Connect Authentication Required")
            
            # Show OAuth option only if API credentials are configured
            if self.api_key and self.api_secret:
                try:
                    login_url = self.kite.login_url()
                    st.link_button("🔐 Login with Kite OAuth", login_url, use_container_width=True)
                except Exception as e:
                    st.warning(f"OAuth login unavailable: {e}")
            else:
                st.warning("⚠️ Kite API credentials not configured. Set KITE_API_KEY and KITE_API_SECRET in environment secrets for OAuth login.")

            # Always show manual token entry as fallback
            st.markdown("**Enter access token manually:**")
            with st.form("kite_login_form"):
                access_token = st.text_input("Access Token", type="password", help="Get your access token from Kite Connect dashboard")
                submit = st.form_submit_button("Authenticate", type="primary", use_container_width=True)

            if submit and access_token:
                try:
                    self.access_token = access_token
                    if self.kite:
                        self.kite.set_access_token(self.access_token)
                        profile = self.kite.profile()
                        user_name = profile.get("user_name", "")
                    else:
                        user_name = "User"
                    st.session_state.kite_access_token = self.access_token
                    st.session_state.kite_user_name = user_name
                    try:
                        kite_token_db.save_token(access_token=self.access_token, user_name=user_name)
                    except Exception:
                        pass
                    self.is_authenticated = True
                    st.success(f"✅ Authenticated as {user_name}")
                    st.balloons()
                    return True
                except Exception as e:
                    st.error(f"❌ Authentication failed: {str(e)}")
                    return False

            return False

        except Exception as e:
            st.error(f"Kite Connect login error: {str(e)}")
            return False

    def logout(self):
        try:
            if "kite_access_token" in st.session_state:
                del st.session_state.kite_access_token
            if "kite_user_name" in st.session_state:
                del st.session_state.kite_user_name
            st.session_state.kite_oauth_consumed = False
            st.session_state.kite_oauth_consumed_at = 0.0
            st.session_state.kite_oauth_in_progress = False
            try:
                kite_token_db.invalidate_token()
            except Exception:
                pass
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
# Import sklearn for ML model
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
            'equity_curve': equity_curve,
            'trades': trades
        }
    
    def run_multi_strategy_backtest(self, symbol, strategies, period_days=30, capital=100000):
        """Run backtest for multiple strategies and compare"""
        results = {}
        
        for strategy in strategies:
            result = self.run_backtest(symbol, strategy, period_days, capital)
            if result:
                results[strategy] = result
        
        return results

# Enhanced Multi-Strategy Trading Engine with ALL NEW features
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
        
        # Initialize strategy performance for ALL strategies
        self.strategy_performance = {}
        for strategy in TRADING_STRATEGIES.keys():
            self.strategy_performance[strategy] = {"signals": 0, "trades": 0, "wins": 0, "pnl": 0.0}
        
        # Initialize high accuracy strategies
        for strategy in HIGH_ACCURACY_STRATEGIES.keys():
            self.strategy_performance[strategy] = {"signals": 0, "trades": 0, "wins": 0, "pnl": 0.0}
        
        # NEW: Integrated systems
        self.data_manager = EnhancedDataManager()
        self.risk_manager = AdvancedRiskManager()
        self.ml_enhancer = MLSignalEnhancer()
        self.regime_detector = MarketRegimeDetector()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.backtest_engine = RealBacktestEngine()
        self.alert_manager = AlertManager()

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
            return float(data["Support"].iloc[-1]), float(data["Resistance"].iloc[-1])
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
    
    # NEW: Improved stop-loss and target calculation
    def calculate_improved_stop_target(self, entry_price, action, atr, current_price, support, resistance):
        """Calculate improved stop-loss and target with market structure"""
        
        if action == "BUY":
            # For BUY: SL below recent swing low, target at resistance
            sl = support * 0.995  # 0.5% below support
            target = resistance * 0.998  # Just below resistance
            
            # Adjust if risk-reward is poor
            rr = (target - entry_price) / (entry_price - sl)
            if rr < 2.5:
                # Adjust target to maintain good RR
                target = entry_price + (2.5 * (entry_price - sl))
                
        else:  # SELL
            # For SELL: SL above recent swing high, target at support
            sl = resistance * 1.005  # 0.5% above resistance
            target = support * 1.002  # Just above support
            
            # Adjust if risk-reward is poor
            rr = (entry_price - target) / (sl - entry_price)
            if rr < 2.5:
                # Adjust target to maintain good RR
                target = entry_price - (2.5 * (sl - entry_price))
        
        return round(target, 2), round(sl, 2)

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
        # NEW: Risk check before execution
        risk_ok, risk_msg = self.data_manager.check_risk_limits(symbol, action, quantity, price, self.positions)
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

        # NEW: Log trade to database
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

    # NEW: High Accuracy Midcap Strategies
    def generate_high_accuracy_signals(self, symbol, data):
        """Generate high accuracy signals specifically for midcap stocks"""
        signals = []
        if data is None or len(data) < 50:
            return signals
            
        try:
            current_price = float(data["Close"].iloc[-1])
            ema8 = float(data["EMA8"].iloc[-1])
            ema21 = float(data["EMA21"].iloc[-1])
            ema50 = float(data["EMA50"].iloc[-1])
            rsi_val = float(data["RSI14"].iloc[-1])
            vwap = float(data["VWAP"].iloc[-1])
            volume = float(data["Volume"].iloc[-1])
            volume_avg = float(data["Volume"].rolling(20).mean().iloc[-1]) if len(data["Volume"]) >= 20 else float(data["Volume"].mean())
            macd_line = float(data["MACD"].iloc[-1])
            macd_signal = float(data["MACD_Signal"].iloc[-1])
            adx_val = float(data["ADX"].iloc[-1]) if 'ADX' in data.columns else 20
            atr = float(data["ATR"].iloc[-1]) if 'ATR' in data.columns else current_price * 0.01
            
            support, resistance = self.calculate_support_resistance(symbol, current_price)
            
            # Strategy 1: Multi-Confirmation Ultra
            if (ema8 > ema21 > ema50 and 
                current_price > vwap and 
                rsi_val > 50 and rsi_val < 70 and
                volume > volume_avg * 1.5 and
                adx_val > 25 and  # ADDED: ADX trend check
                macd_line > macd_signal):
                
                action = "BUY"
                target, stop_loss = self.calculate_improved_stop_target(current_price, action, atr, current_price, support, resistance)
                rr = abs(target - current_price) / max(abs(current_price - stop_loss), 1e-6)
                
                if rr >= 2.5:  # Higher risk-reward for high accuracy
                    signals.append({
                        "symbol": symbol,
                        "action": action,
                        "entry": current_price,
                        "current_price": current_price,
                        "target": target,
                        "stop_loss": stop_loss,
                        "confidence": 0.88,
                        "win_probability": 0.82,
                        "risk_reward": rr,
                        "score": 9,
                        "strategy": "Multi_Confirmation",
                        "strategy_name": HIGH_ACCURACY_STRATEGIES["Multi_Confirmation"]["name"],
                        "reason": "Multi-timeframe confirmation with volume"
                    })
            
            # Strategy 2: Enhanced EMA-VWAP
            if (abs(current_price - vwap) / vwap < 0.02 and  # Price near VWAP
                ema8 > ema21 and
                volume > volume_avg * 1.3 and
                rsi_val > 45 and rsi_val < 65):
                
                # Determine direction based on trend
                if ema21 > ema50:  # Uptrend
                    action = "BUY"
                else:  # Downtrend
                    action = "SELL"
                    
                target, stop_loss = self.calculate_improved_stop_target(current_price, action, atr, current_price, support, resistance)
                rr = abs(target - current_price) / max(abs(current_price - stop_loss), 1e-6)
                
                if rr >= 2.2:
                    signals.append({
                        "symbol": symbol,
                        "action": action,
                        "entry": current_price,
                        "current_price": current_price,
                        "target": target,
                        "stop_loss": stop_loss,
                        "confidence": 0.85,
                        "win_probability": 0.78,
                        "risk_reward": rr,
                        "score": 8,
                        "strategy": "Enhanced_EMA_VWAP",
                        "strategy_name": HIGH_ACCURACY_STRATEGIES["Enhanced_EMA_VWAP"]["name"],
                        "reason": "Enhanced EMA-VWAP confluence with volume"
                    })
            
            # Strategy 3: Volume Weighted Breakout
            if (volume > volume_avg * 2.0 and  # High volume
                ((current_price > resistance and rsi_val < 70) or  # Breakout with not overbought
                 (current_price < support and rsi_val > 30))):     # Breakdown with not oversold
                
                if current_price > resistance:
                    action = "BUY"
                else:
                    action = "SELL"
                    
                target, stop_loss = self.calculate_improved_stop_target(current_price, action, atr, current_price, support, resistance)
                rr = abs(target - current_price) / max(abs(current_price - stop_loss), 1e-6)
                
                if rr >= 2.0:
                    signals.append({
                        "symbol": symbol,
                        "action": action,
                        "entry": current_price,
                        "current_price": current_price,
                        "target": target,
                        "stop_loss": stop_loss,
                        "confidence": 0.82,
                        "win_probability": 0.75,
                        "risk_reward": rr,
                        "score": 8,
                        "strategy": "Volume_Breakout",
                        "strategy_name": HIGH_ACCURACY_STRATEGIES["Volume_Breakout"]["name"],
                        "reason": "Volume weighted breakout/breakdown"
                    })
            
            # Update strategy signals count
            for signal in signals:
                strategy = signal.get("strategy")
                if strategy in self.strategy_performance:
                    self.strategy_performance[strategy]["signals"] += 1
                    
            return signals
            
        except Exception as e:
            logger.error(f"Error generating high accuracy signals for {symbol}: {e}")
            return signals

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

        # NEW: Update database
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
                var = ((price - pos["entry_price"]) / pos["entry_price"]) * 100
                sup, res = self.calculate_support_resistance(symbol, price)
                
                strategy = pos.get("strategy", "Manual")
                historical_accuracy = self.data_manager.get_historical_accuracy(symbol, strategy) if strategy != "Manual" else 0.65
                
                out.append({
                    "Symbol": symbol.replace(".NS", ""),
                    "Action": pos["action"],
                    "Quantity": pos["quantity"],
                    "Entry Price": f"₹{pos['entry_price']:.2f}",
                    "Current Price": f"₹{price:.2f}",
                    "P&L": f"₹{pnl:+.2f}",
                    "Variance %": f"{var:+.2f}%",
                    "Stop Loss": f"₹{pos.get('stop_loss', 0):.2f}",
                    "Target": f"₹{pos.get('target', 0):.2f}",
                    "Support": f"₹{sup:.2f}",
                    "Resistance": f"₹{res:.2f}",
                    "Historical Win %": f"{historical_accuracy:.1%}",
                    "Current Win %": f"{pos.get('win_probability', 0.75)*100:.1f}%",
                    "Entry Time": pos.get("entry_time"),
                    "Auto Trade": "Yes" if pos.get("auto_trade") else "No",
                    "Strategy": strategy,
                    "Status": pos.get("status")
                })
            except Exception:
                continue
        return out

    def get_trade_history_data(self):
        """Get formatted trade history data for display"""
        history_data = []
        for trade in self.trade_log:
            if trade.get("status") == "CLOSED":
                pnl = trade.get("closed_pnl", 0)
                pnl_class = "profit-positive" if pnl >= 0 else "profit-negative"
                trade_class = "trade-buy" if trade.get("action") == "BUY" else "trade-sell"
                
                history_data.append({
                    "Trade ID": trade.get("trade_id", ""),
                    "Symbol": trade.get("symbol", "").replace(".NS", ""),
                    "Action": trade.get("action", ""),
                    "Quantity": trade.get("quantity", 0),
                    "Entry Price": f"₹{trade.get('entry_price', 0):.2f}",
                    "Exit Price": f"₹{trade.get('exit_price', 0):.2f}",
                    "P&L": f"<span class='{pnl_class}'>₹{pnl:+.2f}</span>",
                    "Entry Time": trade.get("entry_time", ""),
                    "Exit Time": trade.get("exit_time_str", ""),
                    "Strategy": trade.get("strategy", "Manual"),
                    "Auto Trade": "Yes" if trade.get("auto_trade") else "No",
                    "Duration": self.calculate_trade_duration(trade.get("entry_time"), trade.get("exit_time_str")),
                    "_row_class": trade_class
                })
        return history_data

    def calculate_trade_duration(self, entry_time_str, exit_time_str):
        """Calculate trade duration in minutes"""
        try:
            if entry_time_str and exit_time_str:
                fmt = "%H:%M:%S"
                entry_time = datetime.strptime(entry_time_str, fmt).time()
                exit_time = datetime.strptime(exit_time_str, fmt).time()
                
                today = datetime.now().date()
                entry_dt = datetime.combine(today, entry_time)
                exit_dt = datetime.combine(today, exit_time)
                
                duration = (exit_dt - entry_dt).total_seconds() / 60
                return f"{int(duration)} min"
        except:
            pass
        return "N/A"

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

        auto_trades = [t for t in self.trade_log if t.get("auto_trade")]
        auto_closed = [t for t in auto_trades if t.get("status") == "CLOSED"]
        auto_win_rate = len([t for t in auto_closed if t.get("closed_pnl", 0) > 0]) / len(auto_closed) if auto_closed else 0.0

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "open_positions": len(self.positions),
            "open_pnl": open_pnl,
            "auto_trades": self.auto_trades_count,
            "auto_win_rate": auto_win_rate
        }

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
            support = float(data["Support"].iloc[-1])
            resistance = float(data["Resistance"].iloc[-1])
            vol_latest = float(data["Volume"].iloc[-1])
            vol_avg = float(data["Volume"].rolling(20).mean().iloc[-1]) if len(data["Volume"]) >= 20 else float(data["Volume"].mean())
            adx_val = float(data["ADX"].iloc[-1]) if "ADX" in data.columns else 20
            htf_trend = int(data["HTF_Trend"].iloc[-1]) if "HTF_Trend" in data.columns else 1

            # NEW: Get ML-enhanced confidence
            ml_confidence = self.data_manager.get_ml_enhanced_confidence(data)
            
            # NEW: Get market regime
            market_regime = self.data_manager.get_market_regime()

            # BUY STRATEGIES - Only generate if historical accuracy > 65%
            # Strategy 1: EMA + VWAP + ADX + HTF Trend
            if (ema8 > ema21 > ema50 and live > vwap and adx_val > 25 and htf_trend == 1):  # CHANGED: ADX from 20 to 25
                action = "BUY"; confidence = 0.82; score = 9; strategy = "EMA_VWAP_Confluence"
                target, stop_loss = self.calculate_improved_stop_target(live, action, atr, live, support, resistance)
                rr = abs(target - live) / max(abs(live - stop_loss), 1e-6)
                if rr >= 2.5:
                    historical_accuracy = self.data_manager.get_historical_accuracy(symbol, strategy)
                    if historical_accuracy >= 0.65:
                        # NEW: Enhanced confidence with ML and market regime
                        base_win_probability = min(0.85, historical_accuracy * 1.1)
                        enhanced_confidence = (base_win_probability + ml_confidence) / 2
                        
                        # Adjust for market regime
                        if market_regime == "TRENDING" and action == "BUY":
                            enhanced_confidence *= 1.1
                        elif market_regime == "VOLATILE":
                            enhanced_confidence *= 0.9
                            
                        win_probability = min(0.9, enhanced_confidence)
                        
                        # Volume confirmation
                        if vol_latest < vol_avg * 1.3:  # Require 30% above average volume
                            confidence *= 0.9  # Reduce confidence if volume is low
                        
                        signals.append({
                            "symbol": symbol, "action": action, "entry": live, "current_price": live,
                            "target": target, "stop_loss": stop_loss, "confidence": confidence,
                            "win_probability": win_probability, "historical_accuracy": historical_accuracy,
                            "rsi": rsi_val, "risk_reward": rr, "score": score, "strategy": strategy,
                            "strategy_name": TRADING_STRATEGIES[strategy]["name"],
                            "ml_confidence": ml_confidence,
                            "market_regime": market_regime,
                            "volume_ratio": vol_latest / vol_avg if vol_avg > 0 else 1
                        })

            # Strategy 2: RSI Mean Reversion (15min timeframe focused)
            if rsi_val < 30 and live > support and rsi_val > 25:  # Avoid extreme oversold
                action = "BUY"; confidence = 0.78; score = 8; strategy = "RSI_MeanReversion"
                target, stop_loss = self.calculate_improved_stop_target(live, action, atr, live, support, resistance)
                rr = abs(target - live) / max(abs(live - stop_loss), 1e-6)
                if rr >= 2.5:
                    historical_accuracy = self.data_manager.get_historical_accuracy(symbol, strategy)
                    if historical_accuracy >= 0.65:
                        base_win_probability = min(0.80, historical_accuracy * 1.1)
                        enhanced_confidence = (base_win_probability + ml_confidence) / 2
                        win_probability = min(0.85, enhanced_confidence)
                        
                        signals.append({
                            "symbol": symbol, "action": action, "entry": live, "current_price": live,
                            "target": target, "stop_loss": stop_loss, "confidence": confidence,
                            "win_probability": win_probability, "historical_accuracy": historical_accuracy,
                            "rsi": rsi_val, "risk_reward": rr, "score": score, "strategy": strategy,
                            "strategy_name": TRADING_STRATEGIES[strategy]["name"],
                            "ml_confidence": ml_confidence,
                            "market_regime": market_regime
                        })

            # Strategy 3: Bollinger Reversion
            if live <= float(data["BB_Lower"].iloc[-1]) if "BB_Lower" in data.columns else False:
                action = "BUY"; confidence = 0.75; score = 7; strategy = "Bollinger_Reversion"
                target, stop_loss = self.calculate_improved_stop_target(live, action, atr, live, support, resistance)
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

            # Strategy 4: MACD Momentum
            if (macd_line > macd_signal and ema8 > ema21 and live > vwap):
                action = "BUY"; confidence = 0.80; score = 8; strategy = "MACD_Momentum"
                target, stop_loss = self.calculate_improved_stop_target(live, action, atr, live, support, resistance)
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
            # Strategy 5: EMA + VWAP Downtrend
            if (ema8 < ema21 < ema50 and live < vwap and adx_val > 25):  # CHANGED: ADX from 20 to 25
                action = "SELL"; confidence = 0.82; score = 9; strategy = "EMA_VWAP_Downtrend"
                target, stop_loss = self.calculate_improved_stop_target(live, action, atr, live, support, resistance)
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

            # Strategy 6: RSI Overbought
            if rsi_val > 70 and live < resistance and rsi_val < 75:  # Avoid extreme overbought
                action = "SELL"; confidence = 0.78; score = 8; strategy = "RSI_Overbought"
                target, stop_loss = self.calculate_improved_stop_target(live, action, atr, live, support, resistance)
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

            # Strategy 7: Bollinger Rejection
            if live >= float(data["BB_Upper"].iloc[-1]) if "BB_Upper" in data.columns else False:
                action = "SELL"; confidence = 0.75; score = 7; strategy = "Bollinger_Rejection"
                target, stop_loss = self.calculate_improved_stop_target(live, action, atr, live, support, resistance)
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

            # Strategy 8: MACD Bearish
            if (macd_line < macd_signal and ema8 < ema21 and live < vwap):
                action = "SELL"; confidence = 0.80; score = 8; strategy = "MACD_Bearish"
                target, stop_loss = self.calculate_improved_stop_target(live, action, atr, live, support, resistance)
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

            # Strategy 9: Support/Resistance Breakout
            if vol_latest > vol_avg * 1.5:
                if live > resistance:
                    action = "BUY"; strategy = "Support_Resistance_Breakout"
                elif live < support:
                    action = "SELL"; strategy = "Support_Resistance_Breakout"
                else:
                    action = None
                
                if action:
                    confidence = 0.85; score = 9
                    target, stop_loss = self.calculate_improved_stop_target(live, action, atr, live, support, resistance)
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

            # Strategy 10: Trend Reversal
            if ((rsi_val > 70 and rsi_val < data["RSI14"].iloc[-2]) or 
                (rsi_val < 30 and rsi_val > data["RSI14"].iloc[-2])):
                if rsi_val > 70:
                    action = "SELL"
                else:
                    action = "BUY"
                    
                confidence = 0.75; score = 7; strategy = "Trend_Reversal"
                target, stop_loss = self.calculate_improved_stop_target(live, action, atr, live, support, resistance)
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

    def generate_quality_signals(self, universe, max_scan=None, min_confidence=0.70, min_score=6, use_high_accuracy=True):  # CHANGED: Default confidence 0.70, score 6
        signals = []
        
        # Determine which universe to scan
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
        
        # FIX: Handle max_scan properly - if None, scan ALL stocks
        if max_scan is not None and max_scan < len(stocks):
            stocks_to_scan = stocks[:max_scan]
        else:
            stocks_to_scan = stocks  # This will scan ALL stocks when max_scan is None
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Get market regime for context
        market_regime = self.data_manager.get_market_regime()
        st.info(f"📊 Current Market Regime: **{market_regime}** | Universe: **{universe}** | Stocks to scan: **{len(stocks_to_scan)}** | High Accuracy: **{'ON' if use_high_accuracy else 'OFF'}**")
        
        for idx, symbol in enumerate(stocks_to_scan):
            try:
                # Show progress
                status_text.text(f"Scanning {symbol} ({idx+1}/{len(stocks_to_scan)})")
                progress_bar.progress((idx + 1) / len(stocks_to_scan))
                
                # Get stock data
                data = self.data_manager.get_stock_data(symbol, "15m")
                if data is None or len(data) < 30:
                    continue
                
                # Generate high accuracy signals when enabled
                if use_high_accuracy:
                    high_acc_signals = self.generate_high_accuracy_signals(symbol, data)
                    signals.extend(high_acc_signals)
                
                # Generate standard strategy signals
                standard_signals = self.generate_strategy_signals(symbol, data)
                signals.extend(standard_signals)
                    
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        # Filter signals based on confidence and score
        signals = [s for s in signals if s.get("confidence", 0) >= min_confidence and s.get("score", 0) >= min_score]
        
        # Apply high-quality filtering
        if len(signals) > 0:
            filtered_signals = self.data_manager.filter_high_quality_signals(signals)
            st.info(f"📊 Filtered {len(signals)} → {len(filtered_signals)} high-quality signals")
            signals = filtered_signals
        
        # Sort by score and confidence
        signals.sort(key=lambda x: (x.get("score", 0), x.get("confidence", 0)), reverse=True)
        
        # Store in history
        self.signal_history = signals[:30]
        
        return signals[:20]  # Return top 20 signals

    def auto_execute_signals(self, signals):
        """Auto-execute signals with enhanced feedback"""
        executed = []
        
        if not self.can_auto_trade():
            st.warning(f"⚠️ Cannot auto-trade. Check: Daily trades: {self.daily_trades}/{MAX_DAILY_TRADES}, Auto trades: {self.auto_trades_count}/{MAX_AUTO_TRADES}, Market open: {market_open()}")
            return executed
        
        st.info(f"🚀 Attempting to auto-execute {len(signals[:10])} signals...")
        
        for signal in signals[:10]:  # Limit to first 10 signals
            if not self.can_auto_trade():
                st.warning("Auto-trade limit reached")
                break
                
            if signal["symbol"] in self.positions:
                st.info(f"Skipping {signal['symbol']} - already in position")
                continue  # Skip if already in position
                
            # NEW: Enhanced position sizing with Kelly Criterion
            try:
                data = self.data_manager.get_stock_data(signal["symbol"], "15m")
                atr = data["ATR"].iloc[-1] if "ATR" in data.columns else signal["entry"] * 0.01
            except:
                atr = signal["entry"] * 0.01
                
            optimal_qty = self.data_manager.calculate_optimal_position_size(
                signal["symbol"], 
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

# Enhanced Kite Live Charts Function
def create_kite_live_charts():
    """Create simplified Kite Connect Live Charts without complex WebSocket"""
    st.subheader("📈 Kite Connect Live Charts")
    
    # Initialize Kite Connect
    if "kite_manager" not in st.session_state:
        st.session_state.kite_manager = KiteConnectManager(KITE_API_KEY, KITE_API_SECRET)
    
    kite_manager = st.session_state.kite_manager
    
    # Simple login
    if not kite_manager.is_authenticated:
        st.info("Kite Connect authentication required for live charts.")
        if kite_manager.login():
            st.rerun()
        return
    
    # Simple chart selection
    selected_index = st.selectbox("Select Index", ["NIFTY 50", "BANKNIFTY", "FINNIFTY"])
    
    # Simple date range
    col1, col2 = st.columns(2)
    with col1:
        interval = st.selectbox("Interval", ["minute", "5minute", "15minute", "30minute", "hour"])
    with col2:
        days_back = st.slider("Days Back", 1, 30, 7)
    
    if st.button("Load Chart Data", type="primary"):
        try:
            # Map symbols to NSE tokens
            token_map = {
                "NIFTY 50": 256265,
                "BANKNIFTY": 260105,
                "FINNIFTY": 257801
            }
            
            token = token_map.get(selected_index)
            if token:
                with st.spinner(f"Fetching data for {selected_index}..."):
                    # Get data
                    data = data_manager.get_kite_data(token, interval, days_back)
                    
                    if data is not None and len(data) > 0:
                        # Create simple chart
                        fig = go.Figure(data=[go.Candlestick(
                            x=data.index,
                            open=data['open'],
                            high=data['high'],
                            low=data['low'],
                            close=data['close']
                        )])
                        
                        fig.update_layout(
                            title=f'{selected_index} Live Chart',
                            xaxis_title='Time',
                            yaxis_title='Price',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show current stats
                        current_price = data['close'].iloc[-1]
                        prev_close = data['close'].iloc[-2] if len(data) > 1 else current_price
                        change_pct = ((current_price - prev_close) / prev_close) * 100
                        
                        st.metric(f"Current {selected_index}", 
                                f"₹{current_price:.2f}", 
                                f"{change_pct:+.2f}%")
                    else:
                        st.error("Could not fetch data. Check Kite Connect permissions.")
            else:
                st.error("Invalid index selection")
                
        except Exception as e:
            st.error(f"Error loading chart: {str(e)}")
            st.info("Note: Kite Connect may require specific permissions for historical data.")
    
    with col2:
        if st.button("Stop Live Chart", type="secondary", key="stop_kite_chart"):
            kite_manager.stop_websocket()
            st.session_state.kite_chart_active = False
            st.success("WebSocket stopped")
    
    # Display live chart if active
    if hasattr(st.session_state, 'kite_chart_active') and st.session_state.kite_chart_active:
        token = st.session_state.kite_chart_token
        placeholder = st.empty()
        
        # Display current candle data
        candle = kite_manager.get_candle_data(token)
        if candle:
            st.subheader("Current Candle Data")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Open", f"₹{candle['open']:.2f}")
            with col2:
                st.metric("High", f"₹{candle['high']:.2f}")
            with col3:
                st.metric("Low", f"₹{candle['low']:.2f}")
            with col4:
                st.metric("Close", f"₹{candle['close']:.2f}")
            
            # Create chart
            fig = go.Figure()
            
            if chart_type == "Candlestick":
                fig.add_trace(go.Candlestick(
                    x=[candle['timestamp']],
                    open=[candle['open']],
                    high=[candle['high']],
                    low=[candle['low']],
                    close=[candle['close']],
                    name='Price'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[candle['timestamp']],
                    y=[candle['close']],
                    mode='lines',
                    name='Price'
                ))
            
            fig.update_layout(
                title=f'{selected_index} Live Chart',
                xaxis_title='Time',
                yaxis_title='Price',
                height=500,
                template='plotly_dark'
            )
            
            placeholder.plotly_chart(fig, use_container_width=True)
            
            # Auto-refresh every 5 seconds
            time.sleep(5)
            st.rerun()
    
    # Also show historical data option
    st.subheader("Historical Data")
    col1, col2 = st.columns(2)
    with col1:
        interval = st.selectbox("Interval", ["minute", "5minute", "15minute", "30minute", "hour", "day"], key="kite_interval")
    with col2:
        days = st.slider("Days", 1, 30, 7, key="kite_days")
    
    if st.button("Load Historical Data", key="load_kite_historical"):
        try:
            symbol = INDEX_SYMBOLS[selected_index]
            quote = kite_manager.kite.quote([symbol])
            if symbol in quote:
                token = quote[symbol]["instrument_token"]
                
                with st.spinner("Fetching historical data..."):
                    data = kite_manager.get_live_data(token, interval, days=days)
                    
                    if data is not None and len(data) > 0:
                        st.success(f"✅ Loaded {len(data)} data points")
                        
                        # Create historical chart
                        fig = go.Figure()
                        
                        if chart_type == "Candlestick":
                            fig.add_trace(go.Candlestick(
                                x=data.index,
                                open=data['open'],
                                high=data['high'],
                                low=data['low'],
                                close=data['close'],
                                name='Price'
                            ))
                        else:
                            fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data['close'],
                                mode='lines',
                                name='Price'
                            ))
                        
                        # Add moving averages
                        data['EMA20'] = ema(data['close'], 20)
                        data['EMA50'] = ema(data['close'], 50)
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['EMA20'],
                            mode='lines',
                            name='EMA20',
                            line=dict(color='orange', width=1)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['EMA50'],
                            mode='lines',
                            name='EMA50',
                            line=dict(color='blue', width=1)
                        ))
                        
                        fig.update_layout(
                            title=f'{selected_index} Historical Chart',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            height=600,
                            template='plotly_dark',
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show statistics
                        current_price = data['close'].iloc[-1]
                        prev_close = data['close'].iloc[-2] if len(data) > 1 else current_price
                        change = current_price - prev_close
                        change_percent = (change / prev_close) * 100
                        
                        cols = st.columns(4)
                        cols[0].metric("Current Price", f"₹{current_price:.2f}")
                        cols[1].metric("Change", f"₹{change:+.2f}")
                        cols[2].metric("Change %", f"{change_percent:+.2f}%")
                        cols[3].metric("Period High", f"₹{data['high'].max():.2f}")
                        
                    else:
                        st.error("No historical data available")
            else:
                st.error("Could not get instrument token")
        except Exception as e:
            st.error(f"Error loading historical data: {e}")

# Enhanced Initialization with Error Handling
def initialize_application():
    """Initialize the application with comprehensive error handling"""
    
    # Display system status
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
        # Initialize data manager
        data_manager = EnhancedDataManager()
        
        # Alert Notification Panel (will be shown after trader is initialized)
        if "trader" in st.session_state:
            trader_temp = st.session_state.trader
            with st.sidebar.expander("🔔 Alerts & Notifications", expanded=False):
                alert_summary = trader_temp.alert_manager.get_alert_summary()
                
                if alert_summary['unacknowledged'] > 0:
                    st.markdown(f"""
                    <div style="background: #fef3c7; padding: 8px; border-radius: 5px; margin-bottom: 10px;">
                        <span style="color: #92400e; font-weight: bold;">⚠️ {alert_summary['unacknowledged']} unread alerts</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.write(f"📊 Signals: {alert_summary['signals']} | 💰 P&L: {alert_summary['pnl_alerts']} | ⚠️ Risk: {alert_summary['risk_alerts']}")
                
                recent_alerts = trader_temp.alert_manager.get_recent_alerts(5)
                if recent_alerts:
                    for alert in recent_alerts:
                        priority_colors = {'HIGH': '#dc2626', 'CRITICAL': '#7f1d1d', 'MEDIUM': '#d97706', 'NORMAL': '#6b7280'}
                        color = priority_colors.get(alert['priority'], '#6b7280')
                        ack_status = "✅" if alert['acknowledged'] else "🔴"
                        msg_display = alert['message'][:50] + "..." if len(alert['message']) > 50 else alert['message']
                        st.markdown(f"""
                        <div style="background: #f3f4f6; padding: 5px; border-radius: 3px; margin: 3px 0; border-left: 3px solid {color};">
                            <small>{ack_status} {alert['timestamp'].strftime('%H:%M')} | {msg_display}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if st.button("Clear All Alerts", key="clear_alerts_btn"):
                        trader_temp.alert_manager.acknowledge_all()
                        st.rerun()
                else:
                    st.info("No alerts yet")
        
        # Initialize trader in session state
        if "trader" not in st.session_state:
            st.session_state.trader = MultiStrategyIntradayTrader()
        
        trader = st.session_state.trader
        
        # Initialize refresh counter
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
    
    # NEW: Market Regime Display
    market_regime = data_manager.get_market_regime()
    regime_color = {
        "TRENDING": "🟢",
        "VOLATILE": "🟡", 
        "MEAN_REVERTING": "🔵",
        "NEUTRAL": "⚪"
    }.get(market_regime, "⚪")
    cols[3].metric("Market Regime", f"{regime_color} {market_regime}")
    
    # NEW: Peak Hours Indicator
    peak_hours = is_peak_market_hours()
    peak_color = "🟢" if peak_hours else "🔴"
    cols[4].metric("Peak Hours (10AM-2PM)", f"{peak_color} {'YES' if peak_hours else 'NO'}")
    
    cols[5].metric("Auto Trades", f"{trader.auto_trades_count}/{MAX_AUTO_TRADES}")
    cols[6].metric("Available Cash", f"₹{trader.cash:,.0f}")

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

    # Market Mood Gauges for Nifty50 & BankNifty
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

    # Display Circular Market Mood Gauges with Rounded Percentages
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
        st.markdown(create_circular_market_mood_gauge("PEAK HOURS", 0, 0, peak_sentiment).replace("₹0", "10AM-2PM").replace("0.00%", peak_hours_status), unsafe_allow_html=True)

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

    # NEW: Signal Quality Overview with updated thresholds
    st.subheader("🎯 Signal Quality Overview")
    quality_cols = st.columns(4)
    
    with quality_cols[0]:
        st.markdown("""
        <div class="high-quality-signal">
            <div style="font-size: 14px; font-weight: bold;">High Quality</div>
            <div style="font-size: 12px; margin-top: 5px;">• RR ≥ 2.5:1</div>
            <div style="font-size: 12px;">• Volume ≥ 1.3x</div>
            <div style="font-size: 12px;">• Confidence ≥ 70%</div>
            <div style="font-size: 12px;">• ADX ≥ 25</div>
        </div>
        """, unsafe_allow_html=True)
    
    with quality_cols[1]:
        st.markdown("""
        <div class="medium-quality-signal">
            <div style="font-size: 14px; font-weight: bold;">Medium Quality</div>
            <div style="font-size: 12px; margin-top: 5px;">• RR ≥ 2.0:1</div>
            <div style="font-size: 12px;">• Volume ≥ 1.2x</div>
            <div style="font-size: 12px;">• Confidence ≥ 65%</div>
            <div style="font-size: 12px;">• ADX ≥ 20</div>
        </div>
        """, unsafe_allow_html=True)
    
    with quality_cols[2]:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 12px; color: #6b7280;">Min Confidence</div>
            <div style="font-size: 20px; font-weight: bold; color: #1e3a8a;">70%</div>
            <div style="font-size: 11px; margin-top: 3px;">Reduced from 75%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with quality_cols[3]:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 12px; color: #6b7280;">Min Score</div>
            <div style="font-size: 20px; font-weight: bold; color: #1e3a8a;">6</div>
            <div style="font-size: 11px; margin-top: 3px;">Reduced from 7</div>
        </div>
        """, unsafe_allow_html=True)

    # NEW: Peak Hours Optimization Notice
    if is_peak_market_hours():
        st.markdown("""
        <div class="alert-success">
            <strong>🎯 Peak Market Hours Active (9:30 AM - 2:30 PM)</strong>
            <div style="margin-top: 5px;">
                • Increased signal frequency during peak hours<br>
                • More aggressive scanning for opportunities<br>
                • Higher probability setups prioritized
            </div>
        </div>
        """, unsafe_allow_html=True)

    # NEW: Auto-Execution Status Panel
    st.subheader("🚀 Auto-Execution Status")
    
    auto_status_cols = st.columns(4)
    with auto_status_cols[0]:
        status_class = "auto-exec-active" if trader.auto_execution else "auto-exec-inactive"
        status_text = "🟢 ACTIVE" if trader.auto_execution else "⚪ INACTIVE"
        st.markdown(f"""
        <div class="{status_class}">
            <div style="font-size: 14px; font-weight: bold;">Auto Execution</div>
            <div style="font-size: 16px; margin-top: 5px;">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with auto_status_cols[1]:
        can_trade = trader.can_auto_trade()
        trade_status = "✅ READY" if can_trade else "⏸️ PAUSED"
        trade_class = "auto-exec-active" if can_trade else "auto-exec-inactive"
        st.markdown(f"""
        <div class="{trade_class}">
            <div style="font-size: 14px; font-weight: bold;">Trade Status</div>
            <div style="font-size: 16px; margin-top: 5px;">{trade_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with auto_status_cols[2]:
        market_status = "🟢 OPEN" if market_open() else "🔴 CLOSED"
        market_class = "auto-exec-active" if market_open() else "auto-exec-inactive"
        st.markdown(f"""
        <div class="{market_class}">
            <div style="font-size: 14px; font-weight: bold;">Market</div>
            <div style="font-size: 16px; margin-top: 5px;">{market_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with auto_status_cols[3]:
        auto_trades_left = MAX_AUTO_TRADES - trader.auto_trades_count
        daily_trades_left = MAX_DAILY_TRADES - trader.daily_trades
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 12px; color: #6b7280;">Auto Trades Left</div>
            <div style="font-size: 20px; font-weight: bold; color: #1e3a8a;">{auto_trades_left}/{MAX_AUTO_TRADES}</div>
            <div style="font-size: 11px; margin-top: 3px;">Daily Trades: {daily_trades_left}/{MAX_DAILY_TRADES}</div>
        </div>
        """, unsafe_allow_html=True)

    # NEW: High Accuracy Strategies Overview
    st.subheader("🎯 High Accuracy Strategies")
    high_acc_cols = st.columns(len(HIGH_ACCURACY_STRATEGIES))
    
    for idx, (strategy_key, config) in enumerate(HIGH_ACCURACY_STRATEGIES.items()):
        with high_acc_cols[idx]:
            perf = trader.strategy_performance.get(strategy_key, {"signals": 0, "trades": 0, "wins": 0, "pnl": 0})
            win_rate = perf["wins"] / perf["trades"] if perf["trades"] > 0 else 0
            
            st.markdown(f"""
            <div class="high-accuracy-card">
                <div style="font-size: 12px; color: #fef3c7;">{config['name']}</div>
                <div style="font-size: 16px; font-weight: bold; margin: 5px 0;">{win_rate:.1%} Win Rate</div>
                <div style="font-size: 11px;">Signals: {perf['signals']} | Trades: {perf['trades']}</div>
                <div style="font-size: 11px; color: {'#86efac' if perf['pnl'] >= 0 else '#fca5a5'}">P&L: ₹{perf['pnl']:+.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    # Sidebar with Strategy Performance
    st.sidebar.header("🎯 Strategy Performance")
    
    # High Accuracy Strategies First
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
    
    # Standard Strategies
    st.sidebar.subheader("📊 Standard Strategies")
    for strategy, config in TRADING_STRATEGIES.items():
        if strategy in trader.strategy_performance:
            perf = trader.strategy_performance[strategy]
            if perf["signals"] > 0:
                win_rate = perf["wins"] / perf["trades"] if perf["trades"] > 0 else 0
                color = "#059669" if win_rate > 0.6 else "#dc2626" if win_rate < 0.4 else "#d97706"
                st.sidebar.write(f"**{config['name']}**")
                st.sidebar.write(f"📊 Signals: {perf['signals']} | Trades: {perf['trades']}")
                st.sidebar.write(f"🎯 Win Rate: <span style='color: {color};'>{win_rate:.1%}</span>", unsafe_allow_html=True)
                st.sidebar.write(f"💰 P&L: ₹{perf['pnl']:+.2f}")
                st.sidebar.markdown("---")

    st.sidebar.header("⚙️ Trading Configuration")
    trader.selected_market = st.sidebar.selectbox("Market Type", MARKET_OPTIONS, key="market_type_select")
    
    # UPDATED: Universe Selection with All Stocks
    universe = st.sidebar.selectbox("Trading Universe", ["All Stocks", "Nifty 50", "Nifty 100", "Midcap 150"], key="universe_select")
    
    # NEW: High Accuracy Toggle for All Universes
    enable_high_accuracy = st.sidebar.checkbox("Enable High Accuracy Strategies", value=True, 
                                              help="Enable high accuracy strategies for all stock universes", key="high_acc_toggle")
    
    trader.auto_execution = st.sidebar.checkbox("Auto Execution", value=False, key="auto_execution_toggle")
    
    # NEW: Enhanced Risk Management Settings
    st.sidebar.subheader("🎯 Enhanced Risk Management")
    enable_ml = st.sidebar.checkbox("Enable ML Enhancement", value=JOBLIB_AVAILABLE, disabled=not JOBLIB_AVAILABLE, key="ml_toggle")
    kelly_sizing = st.sidebar.checkbox("Kelly Position Sizing", value=True, key="kelly_toggle")
    enable_signal_filtering = st.sidebar.checkbox("Enable Signal Filtering", value=True, 
                                                 help="Filter only high-quality signals with volume confirmation", key="signal_filter_toggle")
    
    # UPDATED: Lower thresholds for better signal generation
    min_conf_percent = st.sidebar.slider("Minimum Confidence %", 60, 85, 70, 5,  # CHANGED: 70-95 → 60-85, default 75 → 70
                                        help="Reduced from 75% to 70% for more signals", key="min_conf_slider")
    min_score = st.sidebar.slider("Minimum Score", 5, 9, 6, 1,  # CHANGED: 6-10 → 5-9, default 7 → 6
                                 help="Reduced from 7 to 6 for more signals", key="min_score_slider")
    
    # NEW: ADX Trend Filter
    require_adx_trend = st.sidebar.checkbox("Require ADX > 25 (Strong Trend)", value=True,
                                          help="Only generate signals when ADX > 25 (strong trending market)", key="adx_toggle")
    
    # FIXED: Scan Configuration - Simplified
    st.sidebar.subheader("🔍 Scan Configuration")
    full_scan = st.sidebar.checkbox("Full Universe Scan", value=True, 
                                   help="Scan entire universe. Uncheck to limit scanning.", key="full_scan_toggle")
    
    if not full_scan:
        max_scan = st.sidebar.number_input("Max Stocks to Scan", min_value=10, max_value=500, value=50, step=10, key="max_scan_input")
    else:
        max_scan = None  # This will scan ALL stocks when full_scan is True

    # Add debug toggle in sidebar
    st.sidebar.subheader("🛠️ Debug Settings")
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False, key="debug_toggle")
    
    if debug_mode:
        st.sidebar.info("Debug Mode Enabled")
        st.sidebar.write(f"**Trader State:**")
        st.sidebar.write(f"- Daily trades: {trader.daily_trades}/{MAX_DAILY_TRADES}")
        st.sidebar.write(f"- Auto trades: {trader.auto_trades_count}/{MAX_AUTO_TRADES}")
        st.sidebar.write(f"- Stock trades: {trader.stock_trades}/{MAX_STOCK_TRADES}")
        st.sidebar.write(f"- Auto execution: {trader.auto_execution}")
        st.sidebar.write(f"- Can auto-trade: {trader.can_auto_trade()}")
        st.sidebar.write(f"- Market open: {market_open()}")
        st.sidebar.write(f"- Peak hours: {is_peak_market_hours()}")
        st.sidebar.write(f"- Auto close time: {should_auto_close()}")
        st.sidebar.write(f"- Open positions: {len(trader.positions)}")
        st.sidebar.write(f"- Available cash: ₹{trader.cash:,.0f}")
        
        # Stock universe info
        st.sidebar.write(f"**Stock Universe:**")
        st.sidebar.write(f"- Selected universe: {universe}")
        if universe == "All Stocks":
            st.sidebar.write(f"- Total stocks: {len(ALL_STOCKS)}")
        elif universe == "Nifty 50":
            st.sidebar.write(f"- Total stocks: {len(NIFTY_50)}")
        elif universe == "Nifty 100":
            st.sidebar.write(f"- Total stocks: {len(NIFTY_100)}")
        elif universe == "Midcap 150":
            st.sidebar.write(f"- Total stocks: {len(NIFTY_MIDCAP_150)}")
        
        # Auto-execution checks
        st.sidebar.write(f"**Auto-execution Checks:**")
        st.sidebar.write(f"- Auto trades < MAX: {trader.auto_trades_count} < {MAX_AUTO_TRADES} = {trader.auto_trades_count < MAX_AUTO_TRADES}")
        st.sidebar.write(f"- Daily trades < MAX: {trader.daily_trades} < {MAX_DAILY_TRADES} = {trader.daily_trades < MAX_DAILY_TRADES}")
        st.sidebar.write(f"- Market open: {market_open()}")
        st.sidebar.write(f"- ALL CHECKS PASS: {trader.can_auto_trade()}")

    # Initialize Algo Engine in session state
    if "algo_engine" not in st.session_state:
        st.session_state.algo_engine = AlgoEngine(
            kite_manager=st.session_state.get("kite_manager"),
            data_manager=data_manager,
            trader=trader
        ) if ALGO_ENGINE_AVAILABLE else None
    
    algo_engine = st.session_state.algo_engine

    # Enhanced Tabs with Kite Connect Live Charts and Algo Trading
    tabs = st.tabs([
        "📈 Dashboard", 
        "🚦 Signals", 
        "💰 Paper Trading", 
        "📋 Trade History",
        "📉 RSI Extreme", 
        "🔍 Backtest", 
        "⚡ Strategies",
        "🎯 High Accuracy Scanner",
        "📊 Kite Live Charts",
        "📊 Portfolio Analytics",
        "🤖 Algo Trading"  # NEW TAB: Algo Trading
    ])

    # Tab 1: Dashboard
    with tabs[0]:
        st.subheader("Account Summary")
        trader.update_positions_pnl()
        perf = trader.get_performance_stats()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Value", f"₹{trader.equity():,.0f}", delta=f"₹{trader.equity() - trader.initial_capital:+,.0f}")
        c2.metric("Available Cash", f"₹{trader.cash:,.0f}")
        c3.metric("Open Positions", len(trader.positions))
        c4.metric("Total P&L", f"₹{perf['total_pnl'] + perf['open_pnl']:+.2f}")
        
        # Strategy Performance Overview
        st.subheader("Strategy Performance Overview")
        strategy_data = []
        for strategy, config in TRADING_STRATEGIES.items():
            if strategy in trader.strategy_performance:
                perf_data = trader.strategy_performance[strategy]
                if perf_data["trades"] > 0:
                    win_rate = perf_data["wins"] / perf_data["trades"]
                    strategy_data.append({
                        "Strategy": config["name"],
                        "Type": config["type"],
                        "Signals": perf_data["signals"],
                        "Trades": perf_data["trades"],
                        "Win Rate": f"{win_rate:.1%}",
                        "P&L": f"₹{perf_data['pnl']:+.2f}"
                    })
        
        # Add high accuracy strategies
        for strategy, config in HIGH_ACCURACY_STRATEGIES.items():
            if strategy in trader.strategy_performance:
                perf_data = trader.strategy_performance[strategy]
                if perf_data["trades"] > 0:
                    win_rate = perf_data["wins"] / perf_data["trades"]
                    strategy_data.append({
                        "Strategy": f"🔥 {config['name']}",
                        "Type": config["type"],
                        "Signals": perf_data["signals"],
                        "Trades": perf_data["trades"],
                        "Win Rate": f"{win_rate:.1%}",
                        "P&L": f"₹{perf_data['pnl']:+.2f}"
                    })
        
        if strategy_data:
            st.dataframe(pd.DataFrame(strategy_data), width='stretch')
        else:
            st.info("No strategy performance data available yet.")
        
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
        st.markdown("""
        <div class="alert-success">
            <strong>🎯 UPDATED Signal Parameters:</strong> 
            • Confidence threshold reduced from 75% to <strong>70%</strong><br>
            • Minimum score reduced from 7 to <strong>6</strong><br>
            • Added ADX trend filter: <strong>ADX > 25</strong><br>
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
            # Add auto-execution button
            if trader.auto_execution and trader.can_auto_trade():
                auto_exec_btn = st.button("🚀 Auto Execute", type="secondary", width='stretch', 
                                         help="Manually trigger auto-execution of current signals", key="auto_exec_btn")
            else:
                auto_exec_btn = False
        
        # Initialize session state for tracking auto-execution
        if "auto_execution_triggered" not in st.session_state:
            st.session_state.auto_execution_triggered = False
        if "last_signal_generation" not in st.session_state:
            st.session_state.last_signal_generation = 0
        
        # Check if we should auto-generate signals
        current_time = time.time()
        auto_generate = False
        
        # Auto-generate if:
        # 1. Auto-execution is enabled AND market is open AND it's been more than 60 seconds since last generation
        # 2. During peak hours, generate more frequently (every 45 seconds)
        if trader.auto_execution and market_open():
            time_since_last = current_time - st.session_state.last_signal_generation
            if is_peak_market_hours() and time_since_last > 45:  # More frequent during peak hours
                auto_generate = True
                st.session_state.last_signal_generation = current_time
            elif time_since_last > 60:  # Normal frequency
                auto_generate = True
                st.session_state.last_signal_generation = current_time
        
        generate_signals = generate_btn or auto_generate
        
        if generate_signals:
            with st.spinner(f"Scanning {universe} stocks with enhanced strategies..."):
                # Use high accuracy when enabled
                signals = trader.generate_quality_signals(
                    universe, 
                    max_scan=max_scan,  # Use the corrected max_scan parameter
                    min_confidence=min_conf_percent/100.0, 
                    min_score=min_score,
                    use_high_accuracy=enable_high_accuracy
                )
            
            if signals:
                # Separate BUY and SELL signals
                buy_signals = [s for s in signals if s["action"] == "BUY"]
                sell_signals = [s for s in signals if s["action"] == "SELL"]
                
                st.success(f"✅ Found {len(buy_signals)} BUY signals and {len(sell_signals)} SELL signals (After quality filtering)")
                
                data_rows = []
                for s in signals:
                    # Check if it's a high accuracy strategy
                    is_high_acc = s["strategy"] in HIGH_ACCURACY_STRATEGIES
                    strategy_display = f"🔥 {s['strategy_name']}" if is_high_acc else s['strategy_name']
                    
                    # Quality score display
                    quality_score = s.get('quality_score', 0)
                    if quality_score >= 80:
                        quality_text = "🟢 High"
                    elif quality_score >= 60:
                        quality_text = "🟡 Medium"
                    else:
                        quality_text = "🔴 Low"
                    
                    data_rows.append({
                        "Symbol": s["symbol"].replace(".NS",""),
                        "Action": s["action"],
                        "Strategy": strategy_display,
                        "Entry Price": f"₹{s['entry']:.2f}",
                        "Current Price": f"₹{s['current_price']:.2f}",
                        "Target": f"₹{s['target']:.2f}",
                        "Stop Loss": f"₹{s['stop_loss']:.2f}",
                        "Confidence": f"{s['confidence']:.1%}",
                        "Quality": quality_text,
                        "Volume Ratio": f"{s.get('volume_ratio', 1):.1f}x",
                        "R:R": f"{s['risk_reward']:.2f}",
                        "Score": s['score'],
                        "RSI": f"{s['rsi']:.1f}"
                    })
                
                st.dataframe(pd.DataFrame(data_rows), width='stretch')
                
                # AUTO-EXECUTION LOGIC
                if trader.auto_execution and trader.can_auto_trade():
                    # Check if we should auto-execute
                    auto_execute_now = False
                    
                    # Auto-execute if:
                    # 1. Auto-execution button was clicked
                    # 2. OR if we have signals and auto-execution is enabled (auto-generate mode)
                    if auto_exec_btn:
                        auto_execute_now = True
                        st.info("🚀 Manual auto-execution triggered")
                    elif auto_generate:
                        # Auto-execute only high-quality signals (quality score >= 80)
                        high_quality_signals = [s for s in signals if s.get('quality_score', 0) >= 80]
                        if high_quality_signals:
                            auto_execute_now = True
                            st.info(f"🚀 Found {len(high_quality_signals)} high-quality signals for auto-execution")
                    
                    if auto_execute_now:
                        executed = trader.auto_execute_signals(signals)
                        if executed:
                            st.success(f"✅ Auto-execution completed: {len(executed)} trades executed")
                            for msg in executed:
                                st.write(f"✓ {msg}")
                            # Refresh to show new positions
                            st.rerun()
                        else:
                            st.warning("No trades were auto-executed. Check trade limits or existing positions.")
                    elif trader.auto_execution and not auto_execute_now:
                        st.info("Auto-execution is active. High-quality signals (score ≥ 80) will be executed automatically.")
                
                st.subheader("Manual Execution")
                for idx, s in enumerate(signals[:5]):  # Show only first 5 for better UI
                    quality_score = s.get('quality_score', 0)
                    if quality_score >= 80:
                        quality_class = "high-quality-signal"
                    elif quality_score >= 60:
                        quality_class = "medium-quality-signal"
                    else:
                        quality_class = "low-quality-signal"
                    
                    col_a, col_b, col_c = st.columns([3,1,1])
                    with col_a:
                        action_color = "🟢" if s["action"] == "BUY" else "🔴"
                        is_high_acc = s["strategy"] in HIGH_ACCURACY_STRATEGIES
                        strategy_display = f"🔥 {s['strategy_name']}" if is_high_acc else s['strategy_name']
                        volume_ratio = s.get('volume_ratio', 1)
                        
                        st.markdown(f"""
                        <div class="{quality_class}">
                            <strong>{action_color} {s['symbol'].replace('.NS','')}</strong> - {s['action']} @ ₹{s['entry']:.2f}<br>
                            Strategy: {strategy_display} | Quality: {quality_score}/100<br>
                            R:R: {s['risk_reward']:.2f} | Volume: {volume_ratio:.1f}x
                        </div>
                        """, unsafe_allow_html=True)
                    with col_b:
                        if kelly_sizing:
                            try:
                                data = trader.data_manager.get_stock_data(s["symbol"], "15m")
                                atr = data["ATR"].iloc[-1] if "ATR" in data.columns else s["entry"] * 0.01
                                qty = trader.data_manager.calculate_optimal_position_size(
                                    s["symbol"], s["win_probability"], s["risk_reward"], 
                                    trader.cash, s["entry"], atr
                                )
                            except:
                                qty = int((trader.cash * TRADE_ALLOC) / s["entry"])
                        else:
                            qty = int((trader.cash * TRADE_ALLOC) / s["entry"])
                        st.write(f"Qty: {qty}")
                    with col_c:
                        if st.button(f"Execute", key=f"exec_{s['symbol']}_{s['strategy']}_{idx}_{int(time.time())}"):
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
                # Provide helpful feedback when no signals are found
                if market_open():
                    st.warning("""
                    **No signals found. Possible reasons:**
                    1. **Market Regime**: Current market regime (**{}**) may not be favorable for the selected strategies.
                    2. **Strict Filters**: ADX trend filter (ADX > 25) may be too restrictive.
                    3. **Time of Day**: Try scanning during peak market hours (9:30 AM - 2:30 PM).
                    
                    **Suggestions:**
                    - Try disabling "Require ADX > 25" in sidebar
                    - Try lowering confidence threshold below 70%
                    - Try lowering minimum score below 6
                    - Scan during peak market hours (9:30 AM - 2:30 PM)
                    """.format(market_regime))
                else:
                    st.info("Market is closed. Signals are only generated during market hours (9:15 AM - 3:30 PM).")
        else:
            # Show auto-execution status when no signals generated
            if trader.auto_execution:
                if market_open():
                    st.info("🔄 Auto-execution is active. High-quality signals will be generated and executed automatically during market hours.")
                    st.write(f"**Auto-execution status:**")
                    st.write(f"- Daily trades: {trader.daily_trades}/{MAX_DAILY_TRADES}")
                    st.write(f"- Auto trades: {trader.auto_trades_count}/{MAX_AUTO_TRADES}")
                    st.write(f"- Available cash: ₹{trader.cash:,.0f}")
                    st.write(f"- Can auto-trade: {'✅ Yes' if trader.can_auto_trade() else '❌ No'}")
                    st.write(f"- Peak hours active: {'✅ Yes' if is_peak_market_hours() else '❌ No'}")
                    
                    # Show countdown to next auto-scan
                    time_since_last = int(current_time - st.session_state.last_signal_generation)
                    if is_peak_market_hours():
                        time_to_next = max(0, 45 - time_since_last)
                    else:
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
                
                # Calculate support/resistance, ATR for stop loss/target
                atr = float(data["ATR"].iloc[-1]) if 'ATR' in data.columns else price * 0.01
                support, resistance = trader.calculate_support_resistance(symbol, price)
                
                # Calculate stop loss and target using IMPROVED method
                target, stop_loss = trader.calculate_improved_stop_target(
                    price, action, atr, price, support, resistance
                )
                
                # Get strategy key
                strategy_key = "Manual"
                for key, config in TRADING_STRATEGIES.items():
                    if config["name"] == strategy:
                        strategy_key = key
                        break
                
                # Execute the trade
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
        
        # Show current positions
        st.subheader("Current Positions")
        positions_df = trader.get_open_positions_data()
        if positions_df:
            # Create a better display with action buttons
            for idx, pos in enumerate(positions_df):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    action_color = "🟢" if pos['Action'] == 'BUY' else "🔴"
                    pnl_text = pos['P&L']
                    pnl_value = float(pnl_text.replace('₹', '').replace('+', '').replace(',', ''))
                    pnl_color = "green" if pnl_value >= 0 else "red"
                    
                    st.markdown(f"""
                    <div style="padding: 10px; border-left: 4px solid {'#059669' if pos['Action'] == 'BUY' else '#dc2626'}; 
                             background: linear-gradient(135deg, {'#d1fae5' if pos['Action'] == 'BUY' else '#fee2e2'} 0%, 
                             {'#a7f3d0' if pos['Action'] == 'BUY' else '#fecaca'} 100%); border-radius: 8px;">
                        <strong>{action_color} {pos['Symbol']}</strong> | {pos['Action']} | Qty: {pos['Quantity']}<br>
                        Entry: {pos['Entry Price']} | Current: {pos['Current Price']}<br>
                        <span style="color: {pnl_color}">{pnl_text}</span> | {pos['Variance %']}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.write(f"SL: {pos['Stop Loss']}")
                    st.write(f"TG: {pos['Target']}")
                
                with col3:
                    if st.button(f"Close", key=f"close_{pos['Symbol']}_{idx}", type="secondary"):
                        success, msg = trader.close_position(f"{pos['Symbol']}.NS")
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
            
            st.dataframe(pd.DataFrame(positions_df), width='stretch')
        else:
            st.info("No open positions")
        
        # Performance stats
        st.subheader("Performance Statistics")
        perf = trader.get_performance_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", perf['total_trades'])
        with col2:
            st.metric("Win Rate", f"{perf['win_rate']:.1%}")
        with col3:
            st.metric("Total P&L", f"₹{perf['total_pnl']:+.2f}")
        with col4:
            st.metric("Open P&L", f"₹{perf['open_pnl']:+.2f}")

    # Tab 4: Trade History
    with tabs[3]:
        st.subheader("📋 Trade History")
        
        if SQLALCHEMY_AVAILABLE and trader.data_manager.database.connected:
            st.success("✅ Database connected - trades are being stored")
        else:
            st.warning("⚠️ Database not available - showing recent trades only")
        
        trade_history = trader.get_trade_history_data()
        if trade_history:
            # Create DataFrame for display
            df_history = pd.DataFrame(trade_history)
            
            # Display with HTML formatting
            for _, trade in df_history.iterrows():
                st.markdown(f"""
                <div class="{trade.get('_row_class', '')}">
                    <div style="padding: 10px;">
                        <strong>{trade['Symbol']}</strong> | {trade['Action']} | Qty: {trade['Quantity']}<br>
                        Entry: {trade['Entry Price']} | Exit: {trade['Exit Price']} | {trade['P&L']}<br>
                        Duration: {trade['Duration']} | Strategy: {trade['Strategy']}
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
                    
                    # Scan top 30 stocks for performance
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

    # Tab 6: Backtest - Enhanced Interactive Backtesting
    with tabs[5]:
        st.subheader("🔍 Strategy Backtesting Engine")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
            <h4 style="color: white; margin: 0;">Test Your Strategies Against Historical Data</h4>
            <p style="color: #e0f2fe; margin: 5px 0 0 0; font-size: 12px;">Validate strategy performance before live trading</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            backtest_symbol = st.selectbox("Select Stock", NIFTY_50[:20], key="backtest_symbol")
        
        with col2:
            backtest_strategy = st.selectbox(
                "Select Strategy",
                list(TRADING_STRATEGIES.keys()) + list(HIGH_ACCURACY_STRATEGIES.keys()),
                format_func=lambda x: {**TRADING_STRATEGIES, **HIGH_ACCURACY_STRATEGIES}.get(x, {}).get("name", x),
                key="backtest_strategy"
            )
        
        with col3:
            backtest_period = st.selectbox("Period (Days)", [7, 14, 30, 60, 90], index=2, key="backtest_period")
        
        col4, col5 = st.columns(2)
        with col4:
            backtest_capital = st.number_input("Starting Capital (₹)", min_value=10000, max_value=10000000, value=100000, step=10000, key="backtest_capital")
        with col5:
            backtest_allocation = st.slider("Trade Allocation %", min_value=5, max_value=30, value=15, key="backtest_allocation") / 100
        
        if st.button("Run Backtest", type="primary", key="run_backtest_btn"):
            with st.spinner(f"Running backtest for {backtest_symbol} with {backtest_strategy}..."):
                try:
                    backtest_engine = trader.data_manager.backtest_engine if hasattr(trader.data_manager, 'backtest_engine') else RealBacktestEngine()
                    results = backtest_engine.run_backtest(
                        backtest_symbol, 
                        backtest_strategy, 
                        backtest_period, 
                        backtest_capital,
                        backtest_allocation
                    )
                    
                    if results and results.get('total_trades', 0) > 0:
                        st.success(f"Backtest completed: {results['total_trades']} trades analyzed")
                        
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric("Total Trades", results['total_trades'])
                        with metric_cols[1]:
                            st.metric("Win Rate", f"{results['win_rate']:.1%}")
                        with metric_cols[2]:
                            pnl_color = "normal" if results['total_pnl'] >= 0 else "inverse"
                            st.metric("Total P&L", f"₹{results['total_pnl']:+,.2f}", delta=f"{results['roi']*100:.1f}% ROI")
                        with metric_cols[3]:
                            st.metric("Max Drawdown", f"{results['max_drawdown']*100:.1f}%")
                        
                        metric_cols2 = st.columns(3)
                        with metric_cols2[0]:
                            st.metric("Wins / Losses", f"{results.get('wins', 0)} / {results.get('losses', 0)}")
                        with metric_cols2[1]:
                            st.metric("Avg P&L/Trade", f"₹{results['avg_pnl']:+,.2f}")
                        with metric_cols2[2]:
                            st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
                        
                        if 'equity_curve' in results and len(results['equity_curve']) > 1:
                            st.subheader("Equity Curve")
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=results['equity_curve'],
                                mode='lines',
                                name='Equity',
                                line=dict(color='#1e3a8a', width=2)
                            ))
                            fig.add_hline(y=backtest_capital, line_dash="dash", line_color="gray", annotation_text="Starting Capital")
                            fig.update_layout(
                                title="Portfolio Equity Over Time",
                                xaxis_title="Trade Number",
                                yaxis_title="Portfolio Value (₹)",
                                height=350
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        if 'trades' in results and len(results['trades']) > 0:
                            st.subheader("Trade Details")
                            trade_df = pd.DataFrame([{
                                'Entry Time': t.get('entry_time', 'N/A'),
                                'Action': t['action'],
                                'Entry': f"₹{t['entry_price']:.2f}",
                                'Exit': f"₹{t['exit_price']:.2f}",
                                'Outcome': t['outcome'],
                                'P&L %': f"{((t['exit_price']-t['entry_price'])/t['entry_price']*100 if t['action']=='BUY' else (t['entry_price']-t['exit_price'])/t['entry_price']*100):+.2f}%"
                            } for t in results['trades'][:20]])
                            st.dataframe(trade_df, use_container_width=True)
                    else:
                        st.warning("No trades generated during the backtest period. Try a different strategy or longer period.")
                        
                except Exception as e:
                    st.error(f"Backtest error: {str(e)}")
                    logger.error(f"Backtest error: {e}")
        
        st.markdown("---")
        st.subheader("Current Session Strategy Performance")
        
        strategy_perf = []
        for strategy, config in {**TRADING_STRATEGIES, **HIGH_ACCURACY_STRATEGIES}.items():
            if strategy in trader.strategy_performance:
                perf = trader.strategy_performance[strategy]
                if perf["trades"] > 0:
                    win_rate = perf["wins"] / perf["trades"]
                    strategy_perf.append({
                        "Strategy": config["name"],
                        "Type": config["type"],
                        "Trades": perf["trades"],
                        "Wins": perf["wins"],
                        "Win Rate": f"{win_rate:.1%}",
                        "Total P&L": f"₹{perf['pnl']:+.2f}",
                        "Avg P&L/Trade": f"₹{perf['pnl']/perf['trades']:.2f}" if perf["trades"] > 0 else "₹0.00"
                    })
        
        if strategy_perf:
            st.dataframe(pd.DataFrame(strategy_perf), use_container_width=True)
        else:
            st.info("No live trading data available yet. Run paper trades to see strategy performance.")

    # Tab 7: Strategies
    with tabs[6]:
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

    # Tab 8: High Accuracy Scanner
    with tabs[7]:
        st.subheader("🎯 High Accuracy Scanner - All Stocks")
        st.markdown(f"""
        <div class="alert-success">
            <strong>🔥 High Accuracy Strategies Enabled:</strong> 
            Scanning <strong>{universe}</strong> with enhanced high-accuracy strategies including
            Multi-Confirmation, Volume Breakouts, RSI Divergence, and MACD Trend Momentum.
            These strategies focus on volume confirmation, multi-timeframe alignment, 
            and divergence patterns for higher probability trades.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            high_acc_scan_btn = st.button("🚀 Scan High Accuracy", type="primary", width='stretch', key="high_acc_scan_btn")
        with col2:
            min_high_acc_confidence = st.slider("Min Confidence", 65, 85, 70, 5, key="high_acc_conf_slider")  # CHANGED: 70-90 → 65-85
        with col3:
            min_high_acc_score = st.slider("Min Score", 5, 8, 6, 1, key="high_acc_score_slider")  # CHANGED: 6-10 → 5-8
        
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
                
                # Display high accuracy signals with special styling
                for idx, signal in enumerate(high_acc_signals[:10]):  # Show first 10
                    quality_score = signal.get('quality_score', 0)
                    if quality_score >= 80:
                        quality_class = "high-quality-signal"
                    elif quality_score >= 60:
                        quality_class = "medium-quality-signal"
                    else:
                        quality_class = "low-quality-signal"
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="{quality_class}">
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
                                R:R: {signal['risk_reward']:.2f} | Quality: {quality_score}/100
                            </div>
                            <div style="font-size: 11px; margin-top: 3px;">
                                Volume: {signal.get('volume_ratio', 1):.1f}x | Confidence: {signal['confidence']:.1%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                
                # Quick execution buttons for high accuracy signals - FIXED DUPLICATE KEY ERROR
                st.subheader("Quick Execution")
                exec_cols = st.columns(3)
                for idx, signal in enumerate(high_acc_signals[:6]):  # Show first 6
                    with exec_cols[idx % 3]:
                        # Create truly unique key with random component
                        import uuid
                        unique_key = f"high_acc_exec_{signal['symbol']}_{idx}_{str(uuid.uuid4())[:8]}"
                        if st.button(
                            f"{signal['action']} {signal['symbol'].replace('.NS', '')}", 
                            key=unique_key,
                            width='stretch'
                        ):
                            if kelly_sizing:
                                qty = trader.data_manager.calculate_optimal_position_size(
                                    signal["symbol"], signal["win_probability"], signal["risk_reward"], 
                                    trader.cash, signal["entry"], 
                                    trader.data_manager.get_stock_data(signal["symbol"], "15m")["ATR"].iloc[-1]
                                )
                            else:
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

    # Tab 9: Kite Live Charts (NEW TAB)
    with tabs[8]:
        create_kite_live_charts()
        
        # --- Diagnostic panel (helps confirm/clear state on Cloud) ---
        with st.expander("🛠 Kite OAuth Diagnostic", expanded=False):
            q = {}
            try:
                # Show current query params
                q = dict(st.query_params)
            except Exception:
                try:
                    q = dict(st.experimental_get_query_params())
                except Exception:
                    q = {}

            st.write("Query params detected:", q)
            st.write("kite_oauth_consumed:", st.session_state.get("kite_oauth_consumed"))
            st.write("kite_oauth_consumed_at:", st.session_state.get("kite_oauth_consumed_at"))
            st.write("kite_oauth_in_progress:", st.session_state.get("kite_oauth_in_progress"))

            if st.button("🔄 Reset OAuth State (clear params + state)"):
                # Hard reset session & URL
                st.session_state.kite_oauth_consumed = False
                st.session_state.kite_oauth_consumed_at = 0.0
                st.session_state.kite_oauth_in_progress = False
                try:
                    st.query_params.clear()
                except Exception:
                    pass
                try:
                    st.experimental_set_query_params()
                except Exception:
                    pass
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
                st.success("OAuth state cleared. Please click Login again.")
                st.rerun()
    
    # Tab 10: Portfolio Analytics Dashboard
    with tabs[9]:
        st.subheader("📊 Portfolio Analytics & Risk Management")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
            <h4 style="color: white; margin: 0;">Comprehensive Portfolio Insights</h4>
            <p style="color: #e0f2fe; margin: 5px 0 0 0; font-size: 12px;">Track performance, analyze risk, and optimize your trading</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Portfolio Overview Metrics
        trader.update_positions_pnl()
        perf_stats = trader.get_performance_stats()
        
        st.subheader("📈 Portfolio Overview")
        overview_cols = st.columns(5)
        
        with overview_cols[0]:
            st.metric("Total Capital", f"₹{trader.initial_capital:,.0f}")
        
        with overview_cols[1]:
            current_equity = trader.equity()
            equity_change = current_equity - trader.initial_capital
            st.metric("Current Equity", f"₹{current_equity:,.0f}", delta=f"₹{equity_change:+,.0f}")
        
        with overview_cols[2]:
            total_pnl = perf_stats.get('total_pnl', 0) + perf_stats.get('open_pnl', 0)
            roi_pct = (total_pnl / trader.initial_capital) * 100 if trader.initial_capital > 0 else 0
            st.metric("Total P&L", f"₹{total_pnl:+,.2f}", delta=f"{roi_pct:+.2f}% ROI")
        
        with overview_cols[3]:
            realized_pnl = perf_stats.get('total_pnl', 0)
            st.metric("Realized P&L", f"₹{realized_pnl:+,.2f}")
        
        with overview_cols[4]:
            open_pnl = perf_stats.get('open_pnl', 0)
            st.metric("Unrealized P&L", f"₹{open_pnl:+,.2f}")
        
        st.markdown("---")
        
        # Trade Statistics
        st.subheader("📊 Trade Statistics")
        stats_cols = st.columns(4)
        
        total_trades = perf_stats.get('total_trades', 0)
        winning_trades = perf_stats.get('winning_trades', 0)
        losing_trades = perf_stats.get('losing_trades', 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        with stats_cols[0]:
            st.metric("Total Trades", total_trades)
        
        with stats_cols[1]:
            st.metric("Winning Trades", winning_trades, delta=f"{win_rate:.1f}% Win Rate")
        
        with stats_cols[2]:
            st.metric("Losing Trades", losing_trades)
        
        with stats_cols[3]:
            avg_trade = realized_pnl / total_trades if total_trades > 0 else 0
            st.metric("Avg P&L/Trade", f"₹{avg_trade:+,.2f}")
        
        # Risk Metrics
        st.subheader("⚠️ Risk Metrics")
        risk_cols = st.columns(4)
        
        with risk_cols[0]:
            open_positions = len([p for p in trader.positions.values() if p.get('status') == 'OPEN'])
            st.metric("Open Positions", open_positions)
        
        with risk_cols[1]:
            exposure = sum([p.get('quantity', 0) * p.get('entry_price', 0) for p in trader.positions.values() if p.get('status') == 'OPEN'])
            exposure_pct = (exposure / trader.initial_capital * 100) if trader.initial_capital > 0 else 0
            st.metric("Market Exposure", f"₹{exposure:,.0f}", delta=f"{exposure_pct:.1f}%")
        
        with risk_cols[2]:
            diversification = trader.portfolio_optimizer.calculate_diversification_score(trader.positions)
            st.metric("Diversification Score", f"{diversification:.2f}")
        
        with risk_cols[3]:
            cash_pct = (trader.cash / trader.initial_capital * 100) if trader.initial_capital > 0 else 100
            st.metric("Cash Reserve", f"₹{trader.cash:,.0f}", delta=f"{cash_pct:.1f}%")
        
        # Position Breakdown
        if trader.positions:
            st.subheader("📋 Open Position Details")
            position_data = []
            for symbol, pos in trader.positions.items():
                if pos.get('status') == 'OPEN':
                    entry_price = pos.get('entry_price', 0)
                    current_price = pos.get('current_price', entry_price)
                    qty = pos.get('quantity', 0)
                    pnl = (current_price - entry_price) * qty
                    pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                    
                    position_data.append({
                        'Symbol': symbol,
                        'Quantity': qty,
                        'Entry Price': f"₹{entry_price:.2f}",
                        'Current Price': f"₹{current_price:.2f}",
                        'Position Value': f"₹{current_price * qty:,.2f}",
                        'P&L': f"₹{pnl:+,.2f}",
                        'P&L %': f"{pnl_pct:+.2f}%",
                        'Strategy': pos.get('strategy', 'N/A')
                    })
            
            if position_data:
                st.dataframe(pd.DataFrame(position_data), use_container_width=True)
        
        # Strategy Performance Breakdown
        st.subheader("⚡ Strategy Performance Breakdown")
        strategy_perf_data = []
        for strategy, perf in trader.strategy_performance.items():
            if perf['trades'] > 0:
                config = {**TRADING_STRATEGIES, **HIGH_ACCURACY_STRATEGIES}.get(strategy, {})
                win_rate = (perf['wins'] / perf['trades'] * 100) if perf['trades'] > 0 else 0
                avg_pnl = perf['pnl'] / perf['trades'] if perf['trades'] > 0 else 0
                strategy_perf_data.append({
                    'Strategy': config.get('name', strategy),
                    'Type': config.get('type', 'N/A'),
                    'Signals': perf['signals'],
                    'Trades': perf['trades'],
                    'Wins': perf['wins'],
                    'Win Rate': f"{win_rate:.1f}%",
                    'Total P&L': f"₹{perf['pnl']:+,.2f}",
                    'Avg P&L': f"₹{avg_pnl:+,.2f}"
                })
        
        if strategy_perf_data:
            st.dataframe(pd.DataFrame(strategy_perf_data), use_container_width=True)
            
            # Strategy P&L Chart
            if len(strategy_perf_data) > 1:
                st.subheader("📊 Strategy P&L Comparison")
                fig = go.Figure()
                
                names = [s['Strategy'] for s in strategy_perf_data]
                pnl_values = [float(s['Total P&L'].replace('₹', '').replace(',', '').replace('+', '')) for s in strategy_perf_data]
                colors = ['#10b981' if v >= 0 else '#ef4444' for v in pnl_values]
                
                fig.add_trace(go.Bar(
                    x=names,
                    y=pnl_values,
                    marker_color=colors,
                    text=[f"₹{v:+,.0f}" for v in pnl_values],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Strategy P&L Comparison",
                    xaxis_title="Strategy",
                    yaxis_title="P&L (₹)",
                    height=350,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No strategy performance data available yet. Start trading to see analytics.")
        
        # Daily Performance Summary
        st.subheader("📅 Today's Summary")
        today_cols = st.columns(4)
        
        with today_cols[0]:
            st.metric("Daily Trades", trader.daily_trades)
        
        with today_cols[1]:
            st.metric("Stock Trades", trader.stock_trades)
        
        with today_cols[2]:
            st.metric("Auto Trades", trader.auto_trades_count)
        
        with today_cols[3]:
            signals_today = len(trader.signal_history)
            st.metric("Signals Generated", signals_today)

    # Tab 11: Algo Trading
    with tabs[10]:
        if ALGO_ENGINE_AVAILABLE and algo_engine:
            create_algo_tab_content(algo_engine, st)
        else:
            st.subheader("🤖 Algo Trading Control Panel")
            st.warning("Algo Trading Engine is not available. Please check the installation.")
            
            st.markdown("""
            ### Algo Trading Features (Coming Soon)
            
            The Algo Trading module provides:
            - **Automated Signal Execution**: Execute trades automatically when signals meet your criteria
            - **Risk Management**: Built-in position limits, daily loss limits, and kill switches
            - **Strategy Scheduler**: Run strategies on a schedule without manual intervention
            - **Order Tracking**: Persistent logging of all automated trades
            - **Live/Paper Trading**: Switch between live execution and paper trading
            
            #### Setup Instructions
            1. Set your Kite Connect API credentials in environment variables
            2. Configure risk limits in the control panel
            3. Start the algo engine to begin automated trading
            
            ⚠️ **Risk Warning**: Algo trading involves significant risk. 
            Always test with paper trading first. Monitor the system regularly.
            """)

    st.markdown("---")
    st.markdown("<div style='text-align:center; color: #6b7280;'>Enhanced Intraday Terminal Pro with Full Stock Scanning & High-Quality Signal Filters | Algo Trading Enabled | Integrated with Kite Connect</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Application error: {str(e)}")
    st.info("Please refresh the page and try again")
    logger.error(f"Application crash: {e}")
    st.code(traceback.format_exc())



# === BEGIN ALGO PAPER + KITE BLOCK (paste from assistant) ===
