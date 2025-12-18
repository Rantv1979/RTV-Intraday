"""
STANDALONE ALGO TRADING APPLICATION
----------------------------------
• Fully autonomous
• Market-hours controlled
• Risk-managed
• Paper trading by default
• Cloud deployable

Author: Final Production Version
"""

import time
import threading
import logging
from datetime import datetime
from enum import Enum
import pytz
import random

# =======================
# CONFIGURATION
# =======================
IND_TZ = pytz.timezone("Asia/Kolkata")

MAX_POSITIONS = 3
MAX_DAILY_LOSS = 20000
MIN_CONFIDENCE = 0.80
RISK_PER_TRADE = 0.10
AUTO_SQUARE_OFF_TIME = (15, 25)

CAPITAL = 500000  # Paper capital

# =======================
# LOGGING
# =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ALGO")

# =======================
# MARKET SESSION
# =======================
class MarketSession:
    @staticmethod
    def is_open():
        now = datetime.now(IND_TZ)
        if now.weekday() >= 5:
            return False
        open_t = now.replace(hour=9, minute=15, second=0)
        close_t = now.replace(hour=15, minute=30, second=0)
        return open_t <= now <= close_t

    @staticmethod
    def auto_square_off():
        now = datetime.now(IND_TZ)
        exit_t = now.replace(
            hour=AUTO_SQUARE_OFF_TIME[0],
            minute=AUTO_SQUARE_OFF_TIME[1],
            second=0
        )
        return now >= exit_t

# =======================
# ALGO STATE
# =======================
class AlgoState(Enum):
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    EMERGENCY = "EMERGENCY"

# =======================
# PAPER TRADER
# =======================
class Trader:
    def __init__(self, capital):
        self.capital = capital
        self.positions = {}
        self.daily_loss = 0

    def execute_trade(self, signal):
        symbol = signal["symbol"]
        entry = signal["entry"]
        action = signal["action"]

        qty = int((self.capital * RISK_PER_TRADE) / entry)
        if qty <= 0:
            return False

        self.positions[symbol] = {
            "entry": entry,
            "qty": qty,
            "action": action,
            "time": datetime.now(IND_TZ)
        }

        logger.info(f"PAPER TRADE → {action} {symbol} | Qty: {qty} | Price: {entry}")
        return True

    def update_pnl(self):
        # Simulated MTM (random for paper trading)
        for sym in list(self.positions.keys()):
            pnl = random.randint(-2000, 3000)
            if pnl < 0:
                self.daily_loss += abs(pnl)

    def close_all_positions(self, reason):
        if self.positions:
            logger.warning(f"SQUARE-OFF ALL POSITIONS → {reason}")
            self.positions.clear()

# =======================
# SIGNAL ENGINE (DEMO)
# =======================
class SignalEngine:
    def generate_signals(self):
        sample_signals = [
            {
                "symbol": "RELIANCE",
                "action": "BUY",
                "entry": 2500,
                "confidence": 0.85
            },
            {
                "symbol": "INFY",
                "action": "SELL",
                "entry": 1500,
                "confidence": 0.78
            }
        ]
        return [s for s in sample_signals if s["confidence"] >= MIN_CONFIDENCE]

# =======================
# ALGO ENGINE
# =======================
class AlgoEngine:
    def __init__(self):
        self.state = AlgoState.STOPPED
        self.trader = Trader(CAPITAL)
        self.signals = SignalEngine()
        self.thread = None
        self.stop_event = threading.Event()

    def start(self):
        if not MarketSession.is_open():
            logger.info("Market closed → Algo not started")
            return

        if self.state == AlgoState.RUNNING:
            return

        self.state = AlgoState.RUNNING
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        logger.info("ALGO STARTED")

    def stop(self):
        self.state = AlgoState.STOPPED
        self.stop_event.set()
        logger.info("ALGO STOPPED")

    def emergency_stop(self, reason):
        logger.critical(f"EMERGENCY STOP → {reason}")
        self.state = AlgoState.EMERGENCY
        self.stop_event.set()
        self.trader.close_all_positions(reason)

    def run(self):
        while not self.stop_event.is_set():

            if MarketSession.auto_square_off():
                self.trader.close_all_positions("AUTO SQUARE-OFF 3:25 PM")
                self.stop()
                break

            if not MarketSession.is_open():
                time.sleep(10)
                continue

            self.scan_and_trade()
            self.trader.update_pnl()

            if self.trader.daily_loss >= MAX_DAILY_LOSS:
                self.emergency_stop("MAX DAILY LOSS HIT")
                break

            time.sleep(30)

    def scan_and_trade(self):
        signals = self.signals.generate_signals()

        for sig in signals:
            if len(self.trader.positions) >= MAX_POSITIONS:
                return
            if sig["symbol"] in self.trader.positions:
                continue
            self.trader.execute_trade(sig)

    def status(self):
        return {
            "state": self.state.value,
            "positions": len(self.trader.positions),
            "daily_loss": self.trader.daily_loss
        }

# =======================
# MAIN RUNNER
# =======================
def main():
    algo = AlgoEngine()
    logger.info("STANDALONE ALGO SERVICE STARTED")

    while True:
        if MarketSession.is_open():
            algo.start()
        else:
            if algo.state == AlgoState.RUNNING:
                algo.stop()
        time.sleep(60)

# =======================
# ENTRY POINT
# =======================
if __name__ == "__main__":
    main()
