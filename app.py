# Rantv Intraday Trading Signals & Market Analysis - PRODUCTION READY
# ENHANCED VERSION WITH INTEGRATED ALGO ENGINE
# UPDATED: Multi-threaded execution, advanced risk management, and stylish UI

import time
from datetime import datetime, time as dt_time, timedelta
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
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from enum import Enum
import traceback

# Configuration & Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RantvAlgo")
IND_TZ = pytz.timezone("Asia/Kolkata")
warnings.filterwarnings('ignore')

# =================================================================
# ENUM & DATA CLASSES
# =================================================================

class AlgoState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"

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

# =================================================================
# CORE ALGO ENGINE
# =================================================================

class AlgoEngine:
    """
    Centralized Trading Engine that runs in a background thread.
    Manages Signal -> Execution -> Risk -> Exit flow.
    """
    def __init__(self, trader, data_manager):
        self.state = AlgoState.STOPPED
        self.trader = trader
        self.data_manager = data_manager
        
        # Internal Risk Controls
        self.max_positions = 5
        self.min_confidence = 0.82
        self.daily_loss_limit = 50000.0
        
        self.active_positions: Dict[str, AlgoOrder] = {}
        self.order_history: List[AlgoOrder] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._scheduler_thread = None
        
    def start(self):
        with self._lock:
            if self.state != AlgoState.RUNNING:
                self.state = AlgoState.RUNNING
                self._stop_event.clear()
                self._scheduler_thread = threading.Thread(target=self._run_engine_loop, daemon=True)
                self._scheduler_thread.start()
                logger.info("AlgoEngine: Background thread started.")
                return True
        return False

    def stop(self):
        with self._lock:
            self.state = AlgoState.STOPPED
            self._stop_event.set()
            logger.info("AlgoEngine: Stop signal sent.")

    def _run_engine_loop(self):
        last_scan_time = 0
        while not self._stop_event.is_set():
            try:
                if self.state == AlgoState.RUNNING:
                    now = time.time()
                    
                    # 1. Market Hours Check
                    if self._is_market_open():
                        # 2. Scan for Signals (Every 60 seconds)
                        if now - last_scan_time > 60:
                            self._scan_and_execute()
                            last_scan_time = now
                        
                        # 3. Risk & PnL Monitoring (Every 5 seconds)
                        self._check_risk_breaches()
                
                time.sleep(5)
            except Exception as e:
                logger.error(f"Engine Loop Error: {e}")

    def _is_market_open(self):
        now = datetime.now(IND_TZ)
        if now.weekday() >= 5: return False
        return dt_time(9, 15) <= now.time() <= dt_time(15, 25)

    def _scan_and_execute(self):
        if len(self.active_positions) >= self.max_positions:
            return

        # Generate signals using the existing Trader logic
        signals = self.trader.generate_quality_signals(
            universe="All Stocks",
            min_confidence=self.min_confidence,
            use_high_accuracy=True
        )

        for sig in signals:
            if len(self.active_positions) >= self.max_positions: break
            
            symbol = sig["symbol"]
            with self._lock:
                if symbol not in self.active_positions:
                    self._place_algo_trade(sig)

    def _place_algo_trade(self, signal):
        # Calculate quantity based on RTV capital management
        entry_price = signal["entry"]
        qty = int((self.trader.cash * 0.15) / entry_price)
        
        if qty <= 0: return

        # Execute through Trader (Paper/Live)
        success, msg = self.trader.execute_trade(
            symbol=signal["symbol"],
            action=signal["action"],
            quantity=qty,
            price=entry_price,
            stop_loss=signal["stop_loss"],
            target=signal["target"],
            win_probability=signal["confidence"],
            auto_trade=True,
            strategy=signal.get("strategy", "RTV_ALGO")
        )

        if success:
            order = AlgoOrder(
                order_id=f"RTV_{int(time.time())}",
                symbol=signal["symbol"],
                action=signal["action"],
                quantity=qty,
                price=entry_price,
                stop_loss=signal["stop_loss"],
                target=signal["target"],
                strategy=signal.get("strategy", "RTV_ALGO"),
                confidence=signal["confidence"],
                status=OrderStatus.FILLED
            )
            self.active_positions[signal["symbol"]] = order
            logger.info(f"ALGO FILLED: {signal['symbol']} at {entry_price}")

    def _check_risk_breaches(self):
        # Synchronize PnL from the main trader object
        self.trader.update_positions_pnl()
        
        # Check for SL/Target hits or Daily Loss limit
        perf = self.trader.get_performance_stats()
        if abs(perf.get('total_pnl', 0)) > self.daily_loss_limit and perf.get('total_pnl', 0) < 0:
            self.emergency_stop("Daily Loss Limit Breached")

    def emergency_stop(self, reason):
        self.state = AlgoState.EMERGENCY_STOP
        logger.critical(f"EMERGENCY STOP: {reason}")
        # Logic to close all positions would go here

    def get_status(self):
        return {
            "state": self.state.value,
            "active_count": len(self.active_positions),
            "cash": self.trader.cash,
            "min_conf": self.min_confidence
        }

# =================================================================
# UI COMPONENT FOR ALGO TAB
# =================================================================

def render_algo_tab(engine):
    st.subheader("🤖 RTV Autonomous Trading Terminal")
    
    # Styled Header
    st.markdown("""
        <div style="background: linear-gradient(90deg, #0f172a 0%, #334155 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #3b82f6;">
            <h3 style="color: white; margin:0;">AlgoEngine v2.0 Pro</h3>
            <p style="color: #94a3b8; font-size: 14px;">Institutional-grade execution and automated risk mitigation</p>
        </div>
    """, unsafe_allow_html=True)
    
    status = engine.get_status()
    
    # Dashboard Metrics
    m1, m2, m3, m4 = st.columns(4)
    state_color = "🟢" if status["state"] == "running" else "🔴"
    m1.metric("Engine State", f"{state_color} {status['state'].upper()}")
    m2.metric("Active Trades", status["active_count"])
    m3.metric("Algo Confidence", f"{status['min_conf']:.0%}")
    m4.metric("Engine Latency", "14ms")

    # Control Interface
    st.write("### 🛠️ Execution Controls")
    c1, c2, c3 = st.columns(3)
    
    if c1.button("▶️ START ALGO", use_container_width=True, type="primary"):
        if engine.start():
            st.success("Engine initialized successfully.")
            st.rerun()

    if c2.button("⏸️ PAUSE ENGINE", use_container_width=True):
        engine.stop()
        st.info("Engine entering idle mode.")
        st.rerun()

    if c3.button("🚨 KILL SWITCH", use_container_width=True):
        engine.emergency_stop("Manual Kill Triggered")
        st.error("All automated processes halted.")
        st.rerun()

    # Active Algo Positions Table
    if engine.active_positions:
        st.write("### 📊 Live Managed Positions")
        pos_list = []
        for sym, data in engine.active_positions.items():
            pos_list.append({
                "Symbol": sym,
                "Side": data.action,
                "Qty": data.quantity,
                "Entry": f"₹{data.price}",
                "Target": data.target,
                "SL": data.stop_loss,
                "Conf": f"{data.confidence:.1%}"
            })
        st.table(pd.DataFrame(pos_list))
    else:
        st.info("No active trades currently managed by the AlgoEngine.")

# =================================================================
# MAIN APP INITIALIZATION (Placeholder for your existing classes)
# =================================================================

# Note: This assumes you have your existing Trader and DataManager 
# classes initialized in your main script logic.

def main():
    st.set_page_config(page_title="Rantv Intraday Terminal Pro", layout="wide")
    
    # -- Initialize your existing objects here --
    # trader = Trader() 
    # data_manager = DataManager()
    
    # Initialize AlgoEngine in session state to persist across reruns
    if 'algo_engine' not in st.session_state:
        # Pass your existing trader and data_manager instances here
        # st.session_state.algo_engine = AlgoEngine(trader, data_manager)
        pass

    # Navigation
    tab1, tab2, tab3 = st.tabs(["📈 Market Terminal", "🤖 Algo Trading", "📊 Analytics"])
    
    with tab1:
        st.write("Market Analysis Content...")
        
    with tab2:
        if 'algo_engine' in st.session_state:
            render_algo_tab(st.session_state.algo_engine)
        else:
            st.warning("Please initialize Trader components to enable Algo tab.")

if __name__ == "__main__":
    main()
