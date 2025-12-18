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
