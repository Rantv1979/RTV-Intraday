# =================================================================
# ENHANCED ALGO ENGINE FOR RANTV - INTEGRATED & OPTIMIZED
# =================================================================

import os
import time
import threading
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum
import pytz

logger = logging.getLogger(__name__)
IND_TZ = pytz.timezone("Asia/Kolkata")

class AlgoState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"

class OrderStatus(Enum):
    PENDING, PLACED, FILLED, REJECTED, FAILED = "pending", "placed", "filled", "rejected", "failed"

@dataclass
class AlgoOrder:
    order_id: str; symbol: str; action: str; quantity: int; price: float
    stop_loss: float; target: float; strategy: str; confidence: float
    status: OrderStatus = OrderStatus.PENDING
    placed_at: datetime = field(default_factory=lambda: datetime.now(IND_TZ))

class AlgoEngine:
    """
    Optimized Algo Engine for Rantv. 
    Reduces latency by offloading signal scanning to a background thread.
    """
    def __init__(self, trader, data_manager, kite_manager=None):
        self.state = AlgoState.STOPPED
        self.trader = trader
        self.data_manager = data_manager
        self.kite_manager = kite_manager
        
        # Risk Limits (Synchronized with App Constants)
        self.max_positions = 5
        self.min_confidence = 0.85 # Higher threshold for Auto-Execution
        self.daily_trade_limit = 10
        
        self.active_positions: Dict[str, AlgoOrder] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._scheduler_thread = None

    def start(self):
        if self.state != AlgoState.RUNNING:
            self.state = AlgoState.RUNNING
            self._stop_event.clear()
            self._scheduler_thread = threading.Thread(target=self._run_loop, daemon=True)
            self._scheduler_thread.start()
            logger.info("Rantv Algo Engine Started")

    def stop(self):
        self.state = AlgoState.STOPPED
        self._stop_event.set()

    def _run_loop(self):
        """Background process: Scans every 60s, checks positions every 5s."""
        last_scan = 0
        while not self._stop_event.is_set():
            if self.state == AlgoState.RUNNING:
                now = time.time()
                # 1. Signal Scanning (Throttled to optimize API usage)
                if now - last_scan > 60: 
                    self._scan_and_execute()
                    last_scan = now
                
                # 2. Position Monitoring
                self._monitor_pnl()
            
            time.sleep(5)

    def _scan_and_execute(self):
        """Uses existing Rantv Trader logic to find high-quality signals."""
        if len(self.active_positions) >= self.max_positions:
            return

        # Integration: Use the trader's existing signal generator
        signals = self.trader.generate_quality_signals(
            universe="All Stocks",
            min_confidence=self.min_confidence,
            use_high_accuracy=True
        )

        for sig in signals[:2]: # Execute top 2 signals per scan
            symbol = sig["symbol"]
            if symbol not in self.active_positions:
                self._execute_order(sig)

    def _execute_order(self, sig):
        """Executes trade through Kite (Live) or Paper Trading."""
        with self._lock:
            # Calculate Qty based on Rantv Risk Management
            price = sig["entry"]
            qty = int((self.trader.cash * 0.15) / price)
            
            if qty <= 0: return

            # Attempt Execution
            success, msg = self.trader.execute_trade(
                symbol=sig["symbol"], action=sig["action"], quantity=qty,
                price=price, stop_loss=sig["stop_loss"], target=sig["target"],
                win_probability=sig["confidence"], auto_trade=True, strategy=sig["strategy"]
            )

            if success:
                self.active_positions[sig["symbol"]] = AlgoOrder(
                    order_id=f"RTV_{int(time.time())}", **sig, quantity=qty, status=OrderStatus.FILLED
                )
                logger.info(f"Algo Executed: {sig['symbol']} @ {price}")

    def _monitor_pnl(self):
        """Updates Unrealized P&L without blocking the UI."""
        if self.trader:
            self.trader.update_positions_pnl()

    def get_status(self):
        return {
            "state": self.state.value,
            "positions": len(self.active_positions),
            "cash": self.trader.cash,
            "market_open": self.trader.can_auto_trade() # Uses existing check
        }
