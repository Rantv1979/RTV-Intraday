"""
RANTV TERMINAL PRO - PROFESSIONAL EDITION
Enhanced with Institutional-Grade Features for Professional Trading
- High-Frequency Strategies
- Large Capital Deployment
- Consistent Alpha Generation
- Advanced Risk Management
"""

# ===================== PART 1: IMPORTS AND CONFIGURATION =====================
import os
import time
import threading
import subprocess
import sys
import webbrowser
import logging
import json
import pytz
import traceback
import smtplib
import random
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import websocket

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import warnings
import re
from scipy import stats
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import warnings
warnings.filterwarnings('ignore')

# Auto-install missing dependencies
def install_package(package_name):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except:
        return False

# Check and install required packages
KITECONNECT_AVAILABLE = False
SQLALCHEMY_AVAILABLE = False
JOBLIB_AVAILABLE = False
TALIB_AVAILABLE = False
WEBSOCKET_AVAILABLE = False
ARCH_AVAILABLE = False

try:
    from kiteconnect import KiteConnect, KiteTicker
    KITECONNECT_AVAILABLE = True
except ImportError:
    if install_package("kiteconnect"):
        from kiteconnect import KiteConnect, KiteTicker
        KITECONNECT_AVAILABLE = True

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    if install_package("sqlalchemy"):
        import sqlalchemy
        from sqlalchemy import create_engine, text
        SQLALCHEMY_AVAILABLE = True

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    if install_package("joblib"):
        import joblib
        JOBLIB_AVAILABLE = True

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    if install_package("TA-Lib"):
        try:
            import talib
            TALIB_AVAILABLE = True
        except:
            TALIB_AVAILABLE = False
            st.warning("TA-Lib not available. Using custom indicators.")
    else:
        TALIB_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    if install_package("arch"):
        try:
            from arch import arch_model
            ARCH_AVAILABLE = True
        except:
            ARCH_AVAILABLE = False
            st.warning("ARCH not available. Basic volatility modeling.")

# Check for websocket-client
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    if install_package("websocket-client"):
        import websocket
        WEBSOCKET_AVAILABLE = True
    else:
        WEBSOCKET_AVAILABLE = False
        st.warning("websocket-client not available. Real-time features disabled.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ===================== PROFESSIONAL CONFIGURATION =====================
class ProfessionalConfig:
    # Capital Management
    TOTAL_CAPITAL = 10_000_000.0  # ₹1 Crore
    MAX_CAPITAL_UTILIZATION = 0.60  # Max 60% deployed
    MIN_POSITION_SIZE = 50000.0  # Minimum ₹50k per position
    MAX_POSITION_SIZE = 2000000.0  # Maximum ₹20L per position
    
    # Risk Management
    RISK_PER_TRADE = 0.01  # 1% risk per trade
    MAX_DAILY_LOSS = 200000.0  # ₹2L daily loss limit
    MAX_DRAWDOWN = 0.10  # 10% max drawdown
    
    # Execution
    EXECUTION_ALGORITHMS = ["TWAP", "VWAP", "POV", "Iceberg", "Sniper"]
    DEFAULT_SLIPPAGE = 0.0005  # 5 basis points
    COMMISSION_RATE = 0.0003  # 3 basis points
    
    # Alpha Strategies
    ENABLED_STRATEGIES = [
        "statistical_arbitrage",
        "pairs_trading", 
        "market_neutral",
        "volatility_arbitrage",
        "event_driven",
        "momentum_reversal",
        "mean_reversion"
    ]
    
    # High-Frequency Settings
    HFT_ENABLED = True
    HFT_LATENCY_TARGET = 50  # ms
    HFT_MIN_PROFIT = 0.0005  # 5 basis points minimum profit
    HFT_MAX_POSITIONS = 20
    
    # Backtesting
    BACKTEST_PERIOD = 365  # days
    WALK_FORWARD_WINDOWS = 5
    MIN_SAMPLE_SIZE = 100
    MIN_SHARPE_RATIO = 1.5
    MIN_PROFIT_FACTOR = 1.8
    
    # Market Microstructure
    ORDER_BOOK_DEPTH = 10
    VOLUME_PROFILE_BINS = 20
    LARGE_ORDER_THRESHOLD = 0.95  # 95th percentile
    
    # Multi-Timeframe Analysis
    TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    TIMEFRAME_WEIGHTS = {
        "1m": 0.25, "5m": 0.20, "15m": 0.15, 
        "30m": 0.12, "1h": 0.10, "4h": 0.10, "1d": 0.08
    }

config = ProfessionalConfig()

# Update existing configuration
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN = ""

ALGO_ENABLED = os.environ.get("ALGO_TRADING_ENABLED", "true").lower() == "true"
ALGO_MAX_POSITIONS = int(os.environ.get("ALGO_MAX_POSITIONS", "10"))
ALGO_MAX_DAILY_LOSS = float(os.environ.get("ALGO_MAX_DAILY_LOSS", "200000"))
ALGO_MIN_CONFIDENCE = float(os.environ.get("ALGO_MIN_CONFIDENCE", "0.80"))

# Email configuration for professional reports
EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "your_email@gmail.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = "rantv2002@gmail.com"
EMAIL_CC = ["professional@rantv.com"]  # Add professional team emails

@dataclass
class AppConfig:
    database_url: str = 'sqlite:///trading_journal.db'
    risk_tolerance: str = 'PROFESSIONAL'
    max_daily_loss: float = config.MAX_DAILY_LOSS
    enable_ml: bool = True
    kite_api_key: str = KITE_API_KEY
    kite_api_secret: str = KITE_API_SECRET
    algo_enabled: bool = ALGO_ENABLED
    algo_max_positions: int = ALGO_MAX_POSITIONS
    algo_max_daily_loss: float = ALGO_MAX_DAILY_LOSS
    algo_min_confidence: float = ALGO_MIN_CONFIDENCE
    total_capital: float = config.TOTAL_CAPITAL
    max_capital_utilization: float = config.MAX_CAPITAL_UTILIZATION
    hft_enabled: bool = config.HFT_ENABLED
    
    @classmethod
    def from_env(cls):
        return cls()

config_app = AppConfig.from_env()
st.set_page_config(page_title="Rantv Terminal Pro - Professional Edition", layout="wide", initial_sidebar_state="expanded")
IND_TZ = pytz.timezone("Asia/Kolkata")

# Trading Constants - UPDATED FOR PROFESSIONAL TRADING
CAPITAL = config.TOTAL_CAPITAL
TRADE_ALLOC = 0.10  # 10% per trade maximum
MAX_DAILY_TRADES = 50  # Increased for HFT
MAX_STOCK_TRADES = 5
MAX_AUTO_TRADES = 100
SIGNAL_REFRESH_MS = 30000  # 30 seconds for HFT
PRICE_REFRESH_MS = 5000  # 5 seconds for real-time
MARKET_OPTIONS = ["CASH", "FUTURES", "OPTIONS"]

# ===================== PROFESSIONAL TRADING MODULES =====================

class DynamicCapitalAllocator:
    """Professional capital allocation using Kelly Criterion"""
    
    def __init__(self, total_capital):
        self.total_capital = total_capital
        self.deployed_capital = 0.0
        self.position_sizing_mode = "half_kelly"  # Options: kelly, half_kelly, quarter_kelly, fixed_fractional
        self.max_position_size_pct = 0.10  # Maximum 10% per position
        self.min_position_size_pct = 0.01  # Minimum 1% per position
        self.position_history = []
        
    def calculate_kelly_fraction(self, win_prob, win_loss_ratio):
        """Calculate Kelly Criterion fraction"""
        if win_loss_ratio <= 0:
            return 0.0
        
        kelly_f = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        
        # Apply conservative approach
        if self.position_sizing_mode == "half_kelly":
            return kelly_f * 0.5
        elif self.position_sizing_mode == "quarter_kelly":
            return kelly_f * 0.25
        elif self.position_sizing_mode == "fixed_fractional":
            return 0.02  # Fixed 2%
        else:
            return max(0.01, min(kelly_f, 0.10))
    
    def calculate_position_size(self, symbol, entry_price, stop_loss, confidence, strategy_stats):
        """Calculate optimal position size"""
        
        # Calculate win probability from historical stats
        win_prob = strategy_stats.get("win_rate", confidence)
        avg_win = strategy_stats.get("avg_win", 2.0)
        avg_loss = strategy_stats.get("avg_loss", 1.0)
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 2.0
        
        # Calculate Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(win_prob, win_loss_ratio)
        
        # Calculate risk per trade
        risk_per_trade = self.total_capital * config.RISK_PER_TRADE
        
        # Calculate position size based on stop loss
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            risk_per_share = entry_price * 0.02  # Default 2% risk
        
        # Maximum shares based on risk
        max_shares_risk = int(risk_per_trade / risk_per_share)
        
        # Maximum shares based on capital allocation
        max_shares_capital = int((self.total_capital * kelly_fraction) / entry_price)
        
        # Choose minimum of the two
        shares = min(max_shares_risk, max_shares_capital)
        
        # Apply position size limits
        min_shares = int((self.total_capital * self.min_position_size_pct) / entry_price)
        max_shares = int((self.total_capital * self.max_position_size_pct) / entry_price)
        
        shares = max(min_shares, min(shares, max_shares))
        
        # Check available capital
        available_capital = self.total_capital - self.deployed_capital
        max_shares_available = int(available_capital / entry_price)
        shares = min(shares, max_shares_available)
        
        position_value = shares * entry_price
        
        return {
            "shares": shares,
            "position_value": position_value,
            "risk_amount": shares * risk_per_share,
            "kelly_fraction": kelly_fraction,
            "capital_utilization": position_value / self.total_capital
        }
    
    def update_deployed_capital(self, position_value, action="add"):
        """Track deployed capital"""
        if action == "add":
            self.deployed_capital += position_value
        else:
            self.deployed_capital -= position_value
        
        self.deployed_capital = max(0.0, self.deployed_capital)
    
    def get_capital_utilization(self):
        """Get current capital utilization"""
        return self.deployed_capital / self.total_capital

class HighFrequencyEngine:
    """High-Frequency Trading Engine with microsecond precision"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.active_strategies = []
        self.tick_data = {}
        self.order_book = {}
        self.latency_target = config.HFT_LATENCY_TARGET
        self.min_profit = config.HFT_MIN_PROFIT
        self.position_counts = {}
        self.pnl_tracker = {}
        
    def add_strategy(self, strategy_name, parameters):
        """Add HFT strategy"""
        self.active_strategies.append({
            "name": strategy_name,
            "params": parameters,
            "active": True,
            "performance": {"trades": 0, "pnl": 0.0}
        })
    
    def process_tick(self, symbol, tick_data):
        """Process tick-by-tick data for HFT"""
        # Store tick data
        if symbol not in self.tick_data:
            self.tick_data[symbol] = []
        
        self.tick_data[symbol].append(tick_data)
        
        # Keep only recent ticks
        if len(self.tick_data[symbol]) > 1000:
            self.tick_data[symbol] = self.tick_data[symbol][-1000:]
        
        # Run HFT strategies
        signals = []
        
        # 1. Tick Momentum Strategy
        momentum_signal = self.detect_tick_momentum(symbol)
        if momentum_signal:
            signals.append(momentum_signal)
        
        # 2. Order Flow Imbalance
        flow_signal = self.analyze_order_flow(symbol, tick_data)
        if flow_signal:
            signals.append(flow_signal)
        
        # 3. Statistical Arbitrage (if multiple symbols)
        if len(self.tick_data) >= 2:
            arb_signal = self.statistical_arbitrage_hft()
            if arb_signal:
                signals.append(arb_signal)
        
        # 4. Volatility Scalping
        vol_signal = self.volatility_scalping(symbol)
        if vol_signal:
            signals.append(vol_signal)
        
        return signals
    
    def detect_tick_momentum(self, symbol):
        """Detect momentum from tick data"""
        if symbol not in self.tick_data or len(self.tick_data[symbol]) < 10:
            return None
        
        recent_ticks = self.tick_data[symbol][-10:]
        prices = [tick.get('price', 0) for tick in recent_ticks]
        volumes = [tick.get('volume', 0) for tick in recent_ticks]
        
        if len(prices) < 5:
            return None
        
        # Calculate price change and volume acceleration
        price_change = (prices[-1] - prices[0]) / prices[0]
        volume_acceleration = sum(volumes[-5:]) / sum(volumes[-10:-5]) if sum(volumes[-10:-5]) > 0 else 1
        
        # Momentum conditions
        if price_change > 0.001 and volume_acceleration > 1.5:  # 0.1% price move with 50% volume increase
            return {
                "symbol": symbol,
                "action": "BUY",
                "price": prices[-1],
                "confidence": min(0.95, 0.7 + abs(price_change) * 10),
                "strategy": "HFT_Tick_Momentum",
                "type": "HFT"
            }
        elif price_change < -0.001 and volume_acceleration > 1.5:
            return {
                "symbol": symbol,
                "action": "SELL",
                "price": prices[-1],
                "confidence": min(0.95, 0.7 + abs(price_change) * 10),
                "strategy": "HFT_Tick_Momentum",
                "type": "HFT"
            }
        
        return None
    
    def analyze_order_flow(self, symbol, current_tick):
        """Analyze order book flow for imbalances"""
        # Simulated order book analysis
        bid_volume = current_tick.get('bid_volume', random.randint(1000, 10000))
        ask_volume = current_tick.get('ask_volume', random.randint(1000, 10000))
        
        if bid_volume == 0 or ask_volume == 0:
            return None
        
        volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        # Large order detection
        large_order_threshold = np.percentile([bid_volume, ask_volume], 90)
        large_bid = bid_volume > large_order_threshold
        large_ask = ask_volume > large_order_threshold
        
        if volume_imbalance > 0.3 and large_bid:
            return {
                "symbol": symbol,
                "action": "BUY",
                "price": current_tick.get('price', 0),
                "confidence": 0.75,
                "strategy": "HFT_Order_Flow",
                "type": "HFT"
            }
        elif volume_imbalance < -0.3 and large_ask:
            return {
                "symbol": symbol,
                "action": "SELL",
                "price": current_tick.get('price', 0),
                "confidence": 0.75,
                "strategy": "HFT_Order_Flow",
                "type": "HFT"
            }
        
        return None
    
    def statistical_arbitrage_hft(self):
        """High-frequency statistical arbitrage between correlated stocks"""
        # Get top Nifty stocks
        symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
        
        # Get recent prices
        prices = {}
        for sym in symbols:
            if sym in self.tick_data and len(self.tick_data[sym]) > 0:
                prices[sym] = self.tick_data[sym][-1].get('price', 0)
        
        if len(prices) < 2:
            return None
        
        # Calculate z-scores
        price_df = pd.DataFrame([prices])
        z_scores = zscore(price_df, axis=1)[0]
        
        # Find extreme values
        extreme_buy = []
        extreme_sell = []
        
        for i, (sym, z) in enumerate(zip(prices.keys(), z_scores)):
            if z < -1.5:
                extreme_buy.append(sym)
            elif z > 1.5:
                extreme_sell.append(sym)
        
        if extreme_buy and extreme_sell:
            # Create market-neutral portfolio
            return {
                "symbol": f"ARB_{extreme_buy[0]}_{extreme_sell[0]}",
                "action": "ARBITRAGE",
                "long": extreme_buy[0],
                "short": extreme_sell[0],
                "confidence": 0.8,
                "strategy": "HFT_Statistical_Arb",
                "type": "HFT_ARB"
            }
        
        return None
    
    def volatility_scalping(self, symbol):
        """Scalp volatility expansions"""
        if symbol not in self.tick_data or len(self.tick_data[symbol]) < 50:
            return None
        
        recent_prices = [tick.get('price', 0) for tick in self.tick_data[symbol][-50:]]
        
        if len(recent_prices) < 20:
            return None
        
        # Calculate rolling volatility
        returns = np.diff(np.log(recent_prices))
        if len(returns) < 10:
            return None
        
        recent_vol = np.std(returns[-10:])
        historical_vol = np.std(returns)
        
        if historical_vol == 0:
            return None
        
        vol_ratio = recent_vol / historical_vol
        
        # Volatility breakout
        if vol_ratio > 2.0 and returns[-1] > 0:
            return {
                "symbol": symbol,
                "action": "BUY",
                "price": recent_prices[-1],
                "confidence": 0.7,
                "strategy": "HFT_Volatility_Scalp",
                "type": "HFT"
            }
        elif vol_ratio > 2.0 and returns[-1] < 0:
            return {
                "symbol": symbol,
                "action": "SELL",
                "price": recent_prices[-1],
                "confidence": 0.7,
                "strategy": "HFT_Volatility_Scalp",
                "type": "HFT"
            }
        
        return None
    
    def get_hft_stats(self):
        """Get HFT performance statistics"""
        total_trades = sum([s["performance"]["trades"] for s in self.active_strategies])
        total_pnl = sum([s["performance"]["pnl"] for s in self.active_strategies])
        
        return {
            "active_strategies": len(self.active_strategies),
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "avg_trade_pnl": total_pnl / max(1, total_trades),
            "position_counts": self.position_counts
        }

class AlphaGenerator:
    """Advanced Alpha Generation Strategies for Professional Trading"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.strategies = {
            "statistical_arbitrage": self.statistical_arbitrage,
            "pairs_trading": self.pairs_trading,
            "market_neutral": self.market_neutral,
            "volatility_arbitrage": self.volatility_arbitrage,
            "event_driven": self.event_driven,
            "momentum_reversal": self.momentum_reversal,
            "mean_reversion": self.mean_reversion
        }
        self.pairs_cache = {}
        self.cointegration_cache = {}
        
    def statistical_arbitrage(self, symbols=None, lookback_days=30):
        """Multi-stock mean reversion strategy"""
        if symbols is None:
            symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
        
        # Get historical data
        price_data = {}
        for symbol in symbols:
            data = self.data_manager.get_stock_data(symbol, "1h")
            if data is not None and len(data) > 100:
                price_data[symbol] = data["Close"].values[-100:]
        
        if len(price_data) < 3:
            return []
        
        # Create price matrix
        price_df = pd.DataFrame(price_data)
        
        # Calculate z-scores for each stock
        z_scores = {}
        for symbol in price_df.columns:
            prices = price_df[symbol].values
            if len(prices) > 20:
                mean = np.mean(prices[-20:])
                std = np.std(prices[-20:])
                if std > 0:
                    z_scores[symbol] = (prices[-1] - mean) / std
        
        # Generate signals
        signals = []
        buy_threshold = -1.5
        sell_threshold = 1.5
        
        for symbol, z in z_scores.items():
            if z < buy_threshold:
                signals.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "price": price_df[symbol].iloc[-1],
                    "confidence": min(0.95, 0.7 + abs(z) * 0.1),
                    "strategy": "Statistical_Arbitrage",
                    "z_score": z,
                    "type": "ALPHA"
                })
            elif z > sell_threshold:
                signals.append({
                    "symbol": symbol,
                    "action": "SELL",
                    "price": price_df[symbol].iloc[-1],
                    "confidence": min(0.95, 0.7 + abs(z) * 0.1),
                    "strategy": "Statistical_Arbitrage",
                    "z_score": z,
                    "type": "ALPHA"
                })
        
        return signals
    
    def pairs_trading(self, symbol1="RELIANCE.NS", symbol2="TCS.NS", lookback_days=60):
        """Cointegration-based pairs trading"""
        cache_key = f"{symbol1}_{symbol2}_{lookback_days}"
        
        if cache_key in self.pairs_cache:
            cached_data = self.pairs_cache[cache_key]
            if time.time() - cached_data["timestamp"] < 3600:  # 1 hour cache
                return cached_data["signals"]
        
        # Get historical data
        data1 = self.data_manager.get_stock_data(symbol1, "1h")
        data2 = self.data_manager.get_stock_data(symbol2, "1h")
        
        if data1 is None or data2 is None or len(data1) < 100 or len(data2) < 100:
            return []
        
        # Align data
        prices1 = data1["Close"].values[-100:]
        prices2 = data2["Close"].values[-100:]
        
        # Test for cointegration
        coint_test = self._test_cointegration(prices1, prices2)
        
        if not coint_test["cointegrated"]:
            return []
        
        # Calculate spread
        hedge_ratio = coint_test["hedge_ratio"]
        spread = prices1 - hedge_ratio * prices2
        
        # Calculate z-score of spread
        spread_mean = np.mean(spread[-20:])
        spread_std = np.std(spread[-20:])
        
        if spread_std == 0:
            return []
        
        current_spread = spread[-1]
        z_score = (current_spread - spread_mean) / spread_std
        
        # Generate signals
        signals = []
        entry_threshold = 2.0
        exit_threshold = 0.5
        
        if z_score > entry_threshold:
            # Spread too wide - short spread (sell symbol1, buy symbol2)
            signals.append({
                "symbol": symbol1,
                "action": "SELL",
                "price": prices1[-1],
                "confidence": min(0.90, 0.6 + abs(z_score) * 0.1),
                "strategy": "Pairs_Trading",
                "pair": symbol2,
                "hedge_ratio": hedge_ratio,
                "z_score": z_score,
                "type": "ALPHA_PAIR"
            })
            signals.append({
                "symbol": symbol2,
                "action": "BUY",
                "price": prices2[-1],
                "confidence": min(0.90, 0.6 + abs(z_score) * 0.1),
                "strategy": "Pairs_Trading",
                "pair": symbol1,
                "hedge_ratio": 1/hedge_ratio,
                "z_score": z_score,
                "type": "ALPHA_PAIR"
            })
        elif z_score < -entry_threshold:
            # Spread too narrow - long spread (buy symbol1, sell symbol2)
            signals.append({
                "symbol": symbol1,
                "action": "BUY",
                "price": prices1[-1],
                "confidence": min(0.90, 0.6 + abs(z_score) * 0.1),
                "strategy": "Pairs_Trading",
                "pair": symbol2,
                "hedge_ratio": hedge_ratio,
                "z_score": z_score,
                "type": "ALPHA_PAIR"
            })
            signals.append({
                "symbol": symbol2,
                "action": "SELL",
                "price": prices2[-1],
                "confidence": min(0.90, 0.6 + abs(z_score) * 0.1),
                "strategy": "Pairs_Trading",
                "pair": symbol1,
                "hedge_ratio": 1/hedge_ratio,
                "z_score": z_score,
                "type": "ALPHA_PAIR"
            })
        
        # Cache results
        self.pairs_cache[cache_key] = {
            "signals": signals,
            "timestamp": time.time(),
            "z_score": z_score
        }
        
        return signals
    
    def _test_cointegration(self, series1, series2):
        """Test if two series are cointegrated"""
        try:
            # Run OLS regression
            X = sm.add_constant(series2)
            model = sm.OLS(series1, X).fit()
            hedge_ratio = model.params[1]
            spread = series1 - hedge_ratio * series2
            
            # ADF test on spread
            adf_result = adfuller(spread)
            
            return {
                "cointegrated": adf_result[0] < adf_result[4]['5%'],
                "hedge_ratio": hedge_ratio,
                "adf_statistic": adf_result[0],
                "p_value": adf_result[1]
            }
        except:
            return {"cointegrated": False, "hedge_ratio": 1.0}
    
    def market_neutral(self, universe="Nifty 50", lookback_days=30):
        """Market-neutral portfolio construction"""
        symbols = NIFTY_50[:20]  # Top 20 Nifty stocks
        
        # Get factor exposures (simplified)
        signals = []
        
        for symbol in symbols:
            data = self.data_manager.get_stock_data(symbol, "1h")
            if data is None or len(data) < 50:
                continue
            
            # Calculate beta to market (simplified)
            market_returns = np.random.randn(50) * 0.01  # Simulated market returns
            stock_returns = data["Close"].pct_change().dropna().values[-50:]
            
            if len(stock_returns) < 20:
                continue
            
            # Simple factor model
            momentum = data["Close"].iloc[-1] / data["Close"].iloc[-20] - 1
            volatility = data["Close"].pct_change().std() * np.sqrt(252)
            volume_trend = data["Volume"].iloc[-1] / data["Volume"].rolling(20).mean().iloc[-1]
            
            # Score based on factors
            score = 0
            if momentum > 0.05:
                score += 1
            if volatility < 0.25:
                score += 1
            if volume_trend > 1.2:
                score += 1
            
            if score >= 2:
                signals.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "price": data["Close"].iloc[-1],
                    "confidence": 0.7,
                    "strategy": "Market_Neutral",
                    "type": "ALPHA"
                })
        
        return signals
    
    def volatility_arbitrage(self, symbol, lookback_days=20):
        """Volatility arbitrage strategy"""
        data = self.data_manager.get_stock_data(symbol, "1h")
        if data is None or len(data) < 100:
            return []
        
        # Calculate historical vs implied volatility
        historical_vol = data["Close"].pct_change().std() * np.sqrt(252)
        
        # Simplified implied volatility (using ATR as proxy)
        atr = data["ATR"].iloc[-1] if "ATR" in data.columns else data["Close"].iloc[-1] * 0.02
        implied_vol = (atr / data["Close"].iloc[-1]) * np.sqrt(252)
        
        vol_ratio = historical_vol / implied_vol if implied_vol > 0 else 1
        
        signals = []
        
        # Volatility mispricing
        if vol_ratio < 0.7:
            # Historical vol lower than implied - sell volatility
            signals.append({
                "symbol": symbol,
                "action": "SELL",
                "price": data["Close"].iloc[-1],
                "confidence": 0.75,
                "strategy": "Volatility_Arbitrage",
                "vol_ratio": vol_ratio,
                "type": "ALPHA"
            })
        elif vol_ratio > 1.3:
            # Historical vol higher than implied - buy volatility
            signals.append({
                "symbol": symbol,
                "action": "BUY",
                "price": data["Close"].iloc[-1],
                "confidence": 0.75,
                "strategy": "Volatility_Arbitrage",
                "vol_ratio": vol_ratio,
                "type": "ALPHA"
            })
        
        return signals
    
    def generate_all_alpha_signals(self, universe="Nifty 50"):
        """Generate signals from all alpha strategies"""
        all_signals = []
        
        # Statistical Arbitrage
        stat_arb_signals = self.statistical_arbitrage()
        all_signals.extend(stat_arb_signals)
        
        # Pairs Trading (top pairs)
        pairs = [("RELIANCE.NS", "TCS.NS"), ("HDFCBANK.NS", "ICICIBANK.NS"), 
                ("INFY.NS", "TCS.NS"), ("HINDUNILVR.NS", "ITC.NS")]
        
        for pair in pairs:
            pair_signals = self.pairs_trading(pair[0], pair[1])
            all_signals.extend(pair_signals)
        
        # Market Neutral
        neutral_signals = self.market_neutral()
        all_signals.extend(neutral_signals)
        
        # Volatility Arbitrage for top stocks
        for symbol in ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]:
            vol_signals = self.volatility_arbitrage(symbol)
            all_signals.extend(vol_signals)
        
        # Filter and rank signals
        ranked_signals = self._rank_signals(all_signals)
        
        return ranked_signals
    
    def _rank_signals(self, signals):
        """Rank signals by confidence and strategy weight"""
        if not signals:
            return []
        
        # Strategy weights
        strategy_weights = {
            "Pairs_Trading": 1.5,
            "Statistical_Arbitrage": 1.3,
            "Market_Neutral": 1.2,
            "Volatility_Arbitrage": 1.4,
            "HFT": 1.1,
            "Standard": 1.0
        }
        
        for signal in signals:
            strategy = signal.get("strategy", "Standard")
            weight = strategy_weights.get(strategy, 1.0)
            
            # Adjust confidence by strategy weight
            signal["weighted_confidence"] = signal.get("confidence", 0.5) * weight
            signal["score"] = signal.get("confidence", 0.5) * 10 * weight
        
        # Sort by weighted confidence
        signals.sort(key=lambda x: x.get("weighted_confidence", 0), reverse=True)
        
        return signals

class SmartOrderRouter:
    """Professional Order Execution with Smart Routing"""
    
    def __init__(self, kite_manager=None):
        self.kite_manager = kite_manager
        self.execution_algorithms = {
            "TWAP": self.twap_execution,
            "VWAP": self.vwap_execution,
            "POV": self.pov_execution,
            "Iceberg": self.iceberg_execution,
            "Sniper": self.sniper_execution
        }
        self.execution_history = []
        self.slippage_model = "proportional"  # proportional, fixed, random
        self.default_slippage = config.DEFAULT_SLIPPAGE
        self.commission_rate = config.COMMISSION_RATE
        
    def execute_order(self, symbol, action, quantity, price, algorithm="TWAP", **kwargs):
        """Execute order using specified algorithm"""
        if algorithm not in self.execution_algorithms:
            algorithm = "TWAP"
        
        # Get execution function
        exec_func = self.execution_algorithms[algorithm]
        
        # Execute with algorithm
        execution_result = exec_func(symbol, action, quantity, price, **kwargs)
        
        # Apply slippage and commission
        execution_result = self._apply_costs(execution_result)
        
        # Record execution
        self.execution_history.append({
            "timestamp": now_indian(),
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "algorithm": algorithm,
            "result": execution_result
        })
        
        return execution_result
    
    def twap_execution(self, symbol, action, quantity, price, duration_minutes=5):
        """Time-Weighted Average Price execution"""
        slices = max(1, int(duration_minutes))
        slice_qty = quantity // slices
        remainder = quantity % slices
        
        executions = []
        total_executed = 0
        avg_price = 0
        
        for i in range(slices):
            # Add remainder to last slice
            current_qty = slice_qty + (remainder if i == slices - 1 else 0)
            
            if current_qty <= 0:
                continue
            
            # Simulate execution with market impact
            executed_price = self._simulate_execution(symbol, action, current_qty, price)
            
            executions.append({
                "slice": i + 1,
                "quantity": current_qty,
                "price": executed_price,
                "timestamp": now_indian()
            })
            
            total_executed += current_qty
            avg_price = ((avg_price * (total_executed - current_qty)) + 
                        (executed_price * current_qty)) / total_executed
            
            # Wait between slices (simulated)
            time.sleep(60 / slices)  # Equal spacing over duration
        
        return {
            "algorithm": "TWAP",
            "total_quantity": total_executed,
            "avg_price": avg_price,
            "slices": len(executions),
            "executions": executions
        }
    
    def vwap_execution(self, symbol, action, quantity, price, lookback_minutes=30):
        """Volume-Weighted Average Price execution"""
        # Get volume profile
        volume_profile = self._get_volume_profile(symbol, lookback_minutes)
        
        if not volume_profile:
            # Fallback to TWAP
            return self.twap_execution(symbol, action, quantity, price)
        
        # Allocate based on volume distribution
        total_volume = sum(volume_profile.values())
        executions = []
        total_executed = 0
        avg_price = 0
        
        for minute, volume_pct in volume_profile.items():
            minute_qty = int(quantity * volume_pct)
            
            if minute_qty <= 0:
                continue
            
            executed_price = self._simulate_execution(symbol, action, minute_qty, price)
            
            executions.append({
                "minute": minute,
                "quantity": minute_qty,
                "price": executed_price,
                "volume_pct": volume_pct
            })
            
            total_executed += minute_qty
            avg_price = ((avg_price * (total_executed - minute_qty)) + 
                        (executed_price * minute_qty)) / total_executed
        
        return {
            "algorithm": "VWAP",
            "total_quantity": total_executed,
            "avg_price": avg_price,
            "volume_matched": total_executed / quantity,
            "executions": executions
        }
    
    def iceberg_execution(self, symbol, action, quantity, price, visible_qty=None):
        """Iceberg order execution"""
        if visible_qty is None:
            visible_qty = max(100, int(quantity * 0.1))  # Show 10% of order
        
        executions = []
        remaining_qty = quantity
        
        while remaining_qty > 0:
            current_qty = min(visible_qty, remaining_qty)
            
            executed_price = self._simulate_execution(symbol, action, current_qty, price)
            
            executions.append({
                "iceberg_slice": len(executions) + 1,
                "quantity": current_qty,
                "price": executed_price,
                "visible": current_qty == visible_qty
            })
            
            remaining_qty -= current_qty
            
            # Random delay between slices
            if remaining_qty > 0:
                time.sleep(random.uniform(2, 5))
        
        avg_price = sum(e["price"] * e["quantity"] for e in executions) / quantity
        
        return {
            "algorithm": "Iceberg",
            "total_quantity": quantity,
            "avg_price": avg_price,
            "slices": len(executions),
            "visible_qty": visible_qty,
            "executions": executions
        }
    
    def sniper_execution(self, symbol, action, quantity, price, tolerance=0.001):
        """Sniper execution - quick market order"""
        # Single execution attempt
        executed_price = self._simulate_execution(symbol, action, quantity, price)
        
        # Check if within tolerance
        price_diff = abs(executed_price - price) / price
        
        if price_diff > tolerance:
            return {
                "algorithm": "Sniper",
                "total_quantity": 0,
                "avg_price": 0,
                "status": "REJECTED",
                "reason": f"Price deviation {price_diff:.2%} > tolerance {tolerance:.2%}"
            }
        
        return {
            "algorithm": "Sniper",
            "total_quantity": quantity,
            "avg_price": executed_price,
            "status": "FILLED",
            "price_deviation": price_diff
        }
    
    def pov_execution(self, symbol, action, quantity, price, participation_rate=0.1):
        """Participation of Volume execution"""
        # Simplified POV - execute at fixed % of volume
        executions = []
        remaining_qty = quantity
        
        while remaining_qty > 0:
            # Get current minute volume (simulated)
            current_volume = random.randint(10000, 50000)
            max_qty = int(current_volume * participation_rate)
            current_qty = min(max_qty, remaining_qty)
            
            if current_qty <= 0:
                break
            
            executed_price = self._simulate_execution(symbol, action, current_qty, price)
            
            executions.append({
                "pov_slice": len(executions) + 1,
                "quantity": current_qty,
                "price": executed_price,
                "participation_rate": participation_rate,
                "volume_used": current_volume
            })
            
            remaining_qty -= current_qty
            time.sleep(60)  # Wait for next minute
        
        if remaining_qty > 0:
            # Execute remaining with market order
            executed_price = self._simulate_execution(symbol, action, remaining_qty, price)
            executions.append({
                "pov_slice": len(executions) + 1,
                "quantity": remaining_qty,
                "price": executed_price,
                "participation_rate": 1.0,  # Full participation for remainder
                "volume_used": remaining_qty * 10  # Simulated
            })
        
        total_executed = sum(e["quantity"] for e in executions)
        avg_price = sum(e["price"] * e["quantity"] for e in executions) / total_executed
        
        return {
            "algorithm": "POV",
            "total_quantity": total_executed,
            "avg_price": avg_price,
            "participation_rate": participation_rate,
            "executions": executions
        }
    
    def _simulate_execution(self, symbol, action, quantity, target_price):
        """Simulate order execution with market impact"""
        # Base execution price
        if action == "BUY":
            # Buying pushes price up
            impact_factor = min(0.001, quantity / 10000 * 0.0001)
            executed_price = target_price * (1 + impact_factor)
        else:
            # Selling pushes price down
            impact_factor = min(0.001, quantity / 10000 * 0.0001)
            executed_price = target_price * (1 - impact_factor)
        
        # Add random noise
        noise = random.uniform(-0.0005, 0.0005)
        executed_price *= (1 + noise)
        
        return round(executed_price, 2)
    
    def _get_volume_profile(self, symbol, lookback_minutes):
        """Get volume profile for VWAP (simulated)"""
        # Simulated volume profile (typically higher at open and close)
        profile = {}
        total_minutes = lookback_minutes
        
        for i in range(total_minutes):
            minute = i % 60
            if minute < 30:  # First half hour
                volume_pct = 0.4 / 30
            elif minute >= 55:  # Last 5 minutes
                volume_pct = 0.3 / 5
            else:  # Middle period
                volume_pct = 0.3 / 25
            
            profile[minute] = volume_pct
        
        return profile
    
    def _apply_costs(self, execution_result):
        """Apply slippage and commission costs"""
        if execution_result["total_quantity"] == 0:
            return execution_result
        
        # Apply slippage
        slippage = self.default_slippage
        if self.slippage_model == "proportional":
            # Slippage proportional to order size
            order_size = execution_result["total_quantity"] * execution_result["avg_price"]
            slippage = min(0.001, self.default_slippage * (1 + order_size / 1000000))
        
        # Adjust average price
        if execution_result.get("action") == "BUY":
            execution_result["final_price"] = execution_result["avg_price"] * (1 + slippage)
        else:
            execution_result["final_price"] = execution_result["avg_price"] * (1 - slippage)
        
        # Calculate commission
        order_value = execution_result["total_quantity"] * execution_result["final_price"]
        commission = order_value * self.commission_rate
        
        execution_result["slippage"] = slippage
        execution_result["commission"] = commission
        execution_result["net_price"] = execution_result["final_price"] * (
            1 + (self.commission_rate if execution_result.get("action") == "BUY" else -self.commission_rate)
        )
        
        return execution_result
    
    def get_execution_stats(self):
        """Get execution performance statistics"""
        if not self.execution_history:
            return {"total_orders": 0, "avg_slippage": 0, "success_rate": 0}
        
        total_orders = len(self.execution_history)
        successful_orders = sum(1 for h in self.execution_history 
                              if h["result"]["total_quantity"] > 0)
        
        slippages = [h["result"].get("slippage", 0) 
                    for h in self.execution_history 
                    if "slippage" in h["result"]]
        
        avg_slippage = np.mean(slippages) if slippages else 0
        
        return {
            "total_orders": total_orders,
            "successful_orders": successful_orders,
            "success_rate": successful_orders / total_orders if total_orders > 0 else 0,
            "avg_slippage": avg_slippage,
            "avg_commission": np.mean([h["result"].get("commission", 0) 
                                      for h in self.execution_history])
        }

class BacktestingEngine:
    """Professional Backtesting Engine with Walk-Forward Analysis"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.backtest_results = {}
        self.optimization_history = []
        self.commission_rate = config.COMMISSION_RATE
        self.slippage_model = "proportional"
        self.min_sample_size = config.MIN_SAMPLE_SIZE
        self.walk_forward_windows = config.WALK_FORWARD_WINDOWS
        
    def run_strategy_backtest(self, strategy_func, symbol, start_date, end_date, 
                             initial_capital=1000000, parameters=None):
        """Run comprehensive backtest for a strategy"""
        
        if parameters is None:
            parameters = {}
        
        # Generate trade signals
        trades = self._generate_trades(strategy_func, symbol, start_date, end_date, parameters)
        
        if not trades:
            return {"error": "No trades generated"}
        
        # Simulate trading
        results = self._simulate_trading(trades, initial_capital)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(results)
        
        # Store results
        test_id = f"{symbol}_{int(time.time())}"
        self.backtest_results[test_id] = {
            "strategy": strategy_func.__name__,
            "symbol": symbol,
            "parameters": parameters,
            "trades": trades,
            "results": results,
            "metrics": metrics,
            "timestamp": now_indian()
        }
        
        return metrics
    
    def run_walk_forward_analysis(self, strategy_func, symbol, start_date, end_date,
                                 optimization_windows=5, initial_capital=1000000):
        """Run walk-forward optimization and testing"""
        
        # Split data into windows
        windows = self._create_walk_forward_windows(start_date, end_date, optimization_windows)
        
        all_results = []
        best_parameters = {}
        
        for i, window in enumerate(windows):
            st.write(f"Window {i+1}/{len(windows)}: {window['train_start']} to {window['test_end']}")
            
            # Optimize on training data
            optimized_params = self._optimize_parameters(
                strategy_func, symbol, window["train_start"], window["train_end"]
            )
            
            # Test on out-of-sample data
            test_results = self.run_strategy_backtest(
                strategy_func, symbol, window["test_start"], window["test_end"],
                initial_capital, optimized_params
            )
            
            # Store results
            window_result = {
                "window": i + 1,
                "optimized_params": optimized_params,
                "test_results": test_results,
                "train_period": f"{window['train_start']} to {window['train_end']}",
                "test_period": f"{window['test_start']} to {window['test_end']}"
            }
            
            all_results.append(window_result)
            
            # Update best parameters
            if test_results.get("sharpe_ratio", 0) > best_parameters.get("sharpe_ratio", 0):
                best_parameters = {
                    "params": optimized_params,
                    "sharpe_ratio": test_results.get("sharpe_ratio", 0),
                    "window": i + 1
                }
        
        # Calculate consolidated metrics
        consolidated = self._consolidate_walk_forward_results(all_results)
        
        self.optimization_history.append({
            "strategy": strategy_func.__name__,
            "symbol": symbol,
            "walk_forward_results": all_results,
            "best_parameters": best_parameters,
            "consolidated": consolidated,
            "timestamp": now_indian()
        })
        
        return {
            "best_parameters": best_parameters,
            "consolidated_metrics": consolidated,
            "all_results": all_results
        }
    
    def _generate_trades(self, strategy_func, symbol, start_date, end_date, parameters):
        """Generate trades from strategy"""
        trades = []
        
        # Get historical data
        # Note: This is simplified - in reality, you'd get OHLC data for the period
        data = self.data_manager.get_stock_data(symbol, "1d")
        
        if data is None or len(data) < 100:
            return []
        
        # Simulate trading days
        for i in range(20, len(data) - 1):
            current_data = data.iloc[:i+1]
            
            # Generate signal (simplified)
            signal = self._simulate_strategy_signal(strategy_func, current_data, parameters)
            
            if signal:
                trades.append({
                    "date": current_data.index[-1],
                    "symbol": symbol,
                    "action": signal["action"],
                    "price": current_data["Close"].iloc[-1],
                    "stop_loss": signal.get("stop_loss", current_data["Close"].iloc[-1] * 0.98),
                    "target": signal.get("target", current_data["Close"].iloc[-1] * 1.02),
                    "confidence": signal.get("confidence", 0.7)
                })
        
        return trades
    
    def _simulate_strategy_signal(self, strategy_func, data, parameters):
        """Simulate strategy signal generation"""
        # Simplified signal generation
        # In reality, you'd call the actual strategy function
        
        if len(data) < 50:
            return None
        
        # Example: Simple moving average crossover
        short_ma = data["Close"].rolling(10).mean().iloc[-1]
        long_ma = data["Close"].rolling(30).mean().iloc[-1]
        
        prev_short_ma = data["Close"].rolling(10).mean().iloc[-2]
        prev_long_ma = data["Close"].rolling(30).mean().iloc[-2]
        
        if short_ma > long_ma and prev_short_ma <= prev_long_ma:
            return {
                "action": "BUY",
                "confidence": 0.7,
                "stop_loss": data["Close"].iloc[-1] * 0.97,
                "target": data["Close"].iloc[-1] * 1.03
            }
        elif short_ma < long_ma and prev_short_ma >= prev_long_ma:
            return {
                "action": "SELL",
                "confidence": 0.7,
                "stop_loss": data["Close"].iloc[-1] * 1.03,
                "target": data["Close"].iloc[-1] * 0.97
            }
        
        return None
    
    def _simulate_trading(self, trades, initial_capital):
        """Simulate trading with realistic constraints"""
        cash = initial_capital
        positions = {}
        trade_history = []
        portfolio_values = []
        
        for trade in trades:
            # Check if we can execute
            trade_value = trade["price"] * 100  # Assuming 100 shares per trade
            
            if trade["action"] == "BUY" and cash >= trade_value:
                # Execute buy
                cash -= trade_value
                positions[trade["symbol"]] = {
                    "entry_price": trade["price"],
                    "quantity": 100,
                    "entry_date": trade["date"],
                    "stop_loss": trade["stop_loss"],
                    "target": trade["target"]
                }
                
                trade_history.append({
                    **trade,
                    "quantity": 100,
                    "trade_value": trade_value,
                    "type": "ENTRY"
                })
            elif trade["action"] == "SELL" and trade["symbol"] in positions:
                # Execute sell
                position = positions.pop(trade["symbol"])
                sell_value = trade["price"] * position["quantity"]
                cash += sell_value
                
                pnl = (trade["price"] - position["entry_price"]) * position["quantity"]
                
                trade_history.append({
                    **trade,
                    "quantity": position["quantity"],
                    "trade_value": sell_value,
                    "pnl": pnl,
                    "type": "EXIT",
                    "holding_period": (trade["date"] - position["entry_date"]).days
                })
            
            # Calculate portfolio value
            position_value = sum(pos["entry_price"] * pos["quantity"] for pos in positions.values())
            portfolio_value = cash + position_value
            portfolio_values.append(portfolio_value)
        
        # Close any remaining positions at last price
        final_pnl = 0
        for symbol, position in positions.items():
            last_trade = trades[-1] if trades else {"price": position["entry_price"]}
            sell_value = last_trade["price"] * position["quantity"]
            cash += sell_value
            pnl = (last_trade["price"] - position["entry_price"]) * position["quantity"]
            final_pnl += pnl
        
        final_portfolio_value = cash
        
        return {
            "initial_capital": initial_capital,
            "final_portfolio_value": final_portfolio_value,
            "total_return": (final_portfolio_value - initial_capital) / initial_capital,
            "trade_history": trade_history,
            "portfolio_values": portfolio_values,
            "total_trades": len(trade_history),
            "final_pnl": final_pnl
        }
    
    def _calculate_performance_metrics(self, results):
        """Calculate professional performance metrics"""
        if not results.get("trade_history"):
            return {}
        
        trades = results["trade_history"]
        portfolio_values = results["portfolio_values"]
        
        # Basic metrics
        total_trades = len([t for t in trades if t["type"] == "EXIT"])
        winning_trades = len([t for t in trades if t.get("pnl", 0) > 0])
        losing_trades = total_trades - winning_trades
        
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        winning_pnl = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
        losing_pnl = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0)
        
        # Calculate returns
        returns = []
        for i in range(1, len(portfolio_values)):
            if portfolio_values[i-1] > 0:
                ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                returns.append(ret)
        
        # Performance metrics
        metrics = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "avg_win": winning_pnl / winning_trades if winning_trades > 0 else 0,
            "avg_loss": losing_pnl / losing_trades if losing_trades > 0 else 0,
            "profit_factor": abs(winning_pnl / losing_pnl) if losing_pnl < 0 else float('inf'),
            "total_return": results.get("total_return", 0)
        }
        
        # Risk metrics
        if returns:
            returns_array = np.array(returns)
            metrics["sharpe_ratio"] = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
            metrics["sortino_ratio"] = self._calculate_sortino_ratio(returns_array)
            
            # Maximum drawdown
            metrics["max_drawdown"] = self._calculate_max_drawdown(portfolio_values)
            metrics["calmar_ratio"] = metrics["total_return"] / abs(metrics["max_drawdown"]) if metrics["max_drawdown"] < 0 else 0
            
            # Volatility
            metrics["annual_volatility"] = np.std(returns_array) * np.sqrt(252)
            
            # Value at Risk (95%)
            metrics["var_95"] = np.percentile(returns_array, 5)
            
            # Expected Shortfall (95%)
            metrics["expected_shortfall_95"] = returns_array[returns_array <= metrics["var_95"]].mean() if any(returns_array <= metrics["var_95"]) else 0
        
        return metrics
    
    def _calculate_sortino_ratio(self, returns):
        """Calculate Sortino ratio (downside risk only)"""
        if len(returns) == 0:
            return 0
        
        target_return = 0
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return float('inf')
        
        excess_return = np.mean(returns) - target_return
        return excess_return / downside_std * np.sqrt(252)
    
    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        if len(portfolio_values) == 0:
            return 0
        
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return -max_dd
    
    def _create_walk_forward_windows(self, start_date, end_date, num_windows):
        """Create walk-forward optimization windows"""
        # Simplified window creation
        windows = []
        total_days = (end_date - start_date).days
        
        if total_days < num_windows * 30:
            # Not enough data
            return windows
        
        window_size = total_days // (num_windows + 1)
        
        for i in range(num_windows):
            train_start = start_date + timedelta(days=i * window_size)
            train_end = train_start + timedelta(days=window_size * 0.7)
            test_start = train_end
            test_end = test_start + timedelta(days=window_size * 0.3)
            
            if test_end > end_date:
                break
            
            windows.append({
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end
            })
        
        return windows
    
    def _optimize_parameters(self, strategy_func, symbol, start_date, end_date):
        """Optimize strategy parameters"""
        # Simplified parameter optimization
        # In reality, you'd use grid search, random search, or Bayesian optimization
        
        param_options = {
            "fast_period": [5, 10, 20],
            "slow_period": [20, 30, 50],
            "rsi_period": [14, 21, 28],
            "stop_loss_pct": [0.02, 0.03, 0.05],
            "target_pct": [0.03, 0.05, 0.08]
        }
        
        best_params = {}
        best_sharpe = -float('inf')
        
        # Simple grid search (limited for demonstration)
        for fast in param_options["fast_period"][:2]:
            for slow in param_options["slow_period"][:2]:
                params = {
                    "fast_period": fast,
                    "slow_period": slow,
                    "rsi_period": 14,
                    "stop_loss_pct": 0.03,
                    "target_pct": 0.05
                }
                
                # Run backtest with these parameters
                results = self.run_strategy_backtest(
                    strategy_func, symbol, start_date, end_date,
                    1000000, params
                )
                
                sharpe = results.get("sharpe_ratio", 0)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
        
        return best_params
    
    def _consolidate_walk_forward_results(self, all_results):
        """Consolidate walk-forward analysis results"""
        if not all_results:
            return {}
        
        metrics = ["sharpe_ratio", "win_rate", "total_return", "max_drawdown"]
        consolidated = {}
        
        for metric in metrics:
            values = [r["test_results"].get(metric, 0) for r in all_results]
            if values:
                consolidated[f"avg_{metric}"] = np.mean(values)
                consolidated[f"std_{metric}"] = np.std(values)
                consolidated[f"min_{metric}"] = np.min(values)
                consolidated[f"max_{metric}"] = np.max(values)
        
        # Calculate consistency
        positive_windows = sum(1 for r in all_results 
                              if r["test_results"].get("total_return", 0) > 0)
        consolidated["consistency"] = positive_windows / len(all_results) if all_results else 0
        
        return consolidated

class MarketMicrostructureAnalyzer:
    """Professional Market Microstructure Analysis"""
    
    def __init__(self):
        self.order_book_depth = config.ORDER_BOOK_DEPTH
        self.volume_profile_bins = config.VOLUME_PROFILE_BINS
        self.large_order_threshold = config.LARGE_ORDER_THRESHOLD
        self.tick_data_history = {}
        self.order_flow_metrics = {}
        
    def analyze_order_book(self, symbol, bids, asks):
        """Comprehensive order book analysis"""
        if not bids or not asks:
            return None
        
        # Convert to numpy arrays for analysis
        bid_prices = np.array([b[0] for b in bids[:self.order_book_depth]])
        bid_volumes = np.array([b[1] for b in bids[:self.order_book_depth]])
        ask_prices = np.array([a[0] for a in asks[:self.order_book_depth]])
        ask_volumes = np.array([a[1] for a in asks[:self.order_book_depth]])
        
        # Basic metrics
        best_bid = bid_prices[0]
        best_ask = ask_prices[0]
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0
        
        # Volume analysis
        total_bid_volume = np.sum(bid_volumes)
        total_ask_volume = np.sum(ask_volumes)
        volume_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0
        
        # Depth analysis
        bid_depth_value = np.sum(bid_prices * bid_volumes)
        ask_depth_value = np.sum(ask_prices * ask_volumes)
        depth_imbalance = (bid_depth_value - ask_depth_value) / (bid_depth_value + ask_depth_value) if (bid_depth_value + ask_depth_value) > 0 else 0
        
        # Large order detection
        large_bid_threshold = np.percentile(bid_volumes, self.large_order_threshold * 100)
        large_ask_threshold = np.percentile(ask_volumes, self.large_order_threshold * 100)
        
        large_bid_count = np.sum(bid_volumes > large_bid_threshold)
        large_ask_count = np.sum(ask_volumes > large_ask_threshold)
        
        # Order flow metrics
        order_flow_pressure = self._calculate_order_flow_pressure(bids, asks)
        
        # Market impact cost
        market_impact = self._estimate_market_impact(bid_volumes, ask_volumes, spread)
        
        analysis = {
            "symbol": symbol,
            "timestamp": now_indian(),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_bps": spread_bps,
            "mid_price": mid_price,
            "total_bid_volume": total_bid_volume,
            "total_ask_volume": total_ask_volume,
            "volume_imbalance": volume_imbalance,
            "bid_depth_value": bid_depth_value,
            "ask_depth_value": ask_depth_value,
            "depth_imbalance": depth_imbalance,
            "large_bid_count": int(large_bid_count),
            "large_ask_count": int(large_ask_count),
            "order_flow_pressure": order_flow_pressure,
            "market_impact_bps": market_impact * 10000,
            "liquidity_score": self._calculate_liquidity_score(total_bid_volume, total_ask_volume, spread_bps)
        }
        
        # Store for trend analysis
        if symbol not in self.order_flow_metrics:
            self.order_flow_metrics[symbol] = []
        
        self.order_flow_metrics[symbol].append(analysis)
        
        # Keep only recent data
        if len(self.order_flow_metrics[symbol]) > 1000:
            self.order_flow_metrics[symbol] = self.order_flow_metrics[symbol][-1000:]
        
        return analysis
    
    def _calculate_order_flow_pressure(self, bids, asks):
        """Calculate order flow pressure"""
        if not bids or not asks:
            return 0
        
        # Simplified order flow calculation
        # In reality, you'd track order flow over time
        
        recent_bid_volume = sum(b[1] for b in bids[:5])
        recent_ask_volume = sum(a[1] for a in asks[:5])
        
        if recent_bid_volume + recent_ask_volume == 0:
            return 0
        
        return (recent_bid_volume - recent_ask_volume) / (recent_bid_volume + recent_ask_volume)
    
    def _estimate_market_impact(self, bid_volumes, ask_volumes, spread):
        """Estimate market impact cost"""
        if len(bid_volumes) == 0 or len(ask_volumes) == 0:
            return 0.001  # Default 10 bps
        
        avg_bid_volume = np.mean(bid_volumes)
        avg_ask_volume = np.mean(ask_volumes)
        
        # Simplified market impact model
        # Based on Kyle's lambda
        lambda_param = 0.0001  # Market impact coefficient
        
        volume_ratio = avg_bid_volume / avg_ask_volume if avg_ask_volume > 0 else 1
        impact = lambda_param * np.sqrt(volume_ratio) * spread
        
        return min(impact, 0.01)  # Cap at 1%
    
    def _calculate_liquidity_score(self, bid_volume, ask_volume, spread_bps):
        """Calculate liquidity score (0-100)"""
        if bid_volume == 0 or ask_volume == 0:
            return 0
        
        # Volume component (0-50 points)
        avg_volume = (bid_volume + ask_volume) / 2
        volume_score = min(50, np.log10(avg_volume + 1) * 10)
        
        # Spread component (0-50 points)
        if spread_bps <= 1:
            spread_score = 50
        elif spread_bps <= 5:
            spread_score = 40
        elif spread_bps <= 10:
            spread_score = 30
        elif spread_bps <= 20:
            spread_score = 20
        elif spread_bps <= 50:
            spread_score = 10
        else:
            spread_score = 0
        
        return volume_score + spread_score
    
    def detect_market_manipulation(self, symbol, window=100):
        """Detect potential market manipulation patterns"""
        if symbol not in self.order_flow_metrics:
            return None
        
        metrics = self.order_flow_metrics[symbol]
        if len(metrics) < window:
            return None
        
        recent_metrics = metrics[-window:]
        
        # Check for spoofing (large orders that disappear)
        spoofing_signals = []
        for i in range(len(recent_metrics) - 5):
            window_metrics = recent_metrics[i:i+5]
            
            # Look for large orders that appear and disappear quickly
            large_order_changes = []
            for j in range(1, len(window_metrics)):
                bid_change = window_metrics[j]["large_bid_count"] - window_metrics[j-1]["large_bid_count"]
                ask_change = window_metrics[j]["large_ask_count"] - window_metrics[j-1]["large_ask_count"]
                large_order_changes.append((bid_change, ask_change))
            
            # Check for rapid large order placement and cancellation
            if any(abs(bc) > 2 for bc, _ in large_order_changes) or any(abs(ac) > 2 for _, ac in large_order_changes):
                spoofing_signals.append({
                    "timestamp": window_metrics[-1]["timestamp"],
                    "bid_changes": [bc for bc, _ in large_order_changes],
                    "ask_changes": [ac for _, ac in large_order_changes]
                })
        
        # Check for quote stuffing (rapid order placement/cancellation)
        quote_stuffing = len(spoofing_signals) > window * 0.1  # More than 10% of windows
        
        # Check for layering (multiple orders at different price levels)
        layering_signals = []
        for metric in recent_metrics:
            if metric["large_bid_count"] > 3 and metric["large_ask_count"] > 3:
                # Both sides have multiple large orders
                layering_signals.append(metric["timestamp"])
        
        detection_result = {
            "symbol": symbol,
            "analysis_window": window,
            "spoofing_detected": len(spoofing_signals) > 0,
            "spoofing_signals_count": len(spoofing_signals),
            "quote_stuffing_detected": quote_stuffing,
            "layering_detected": len(layering_signals) > 0,
            "layering_signals_count": len(layering_signals),
            "confidence": min(1.0, (len(spoofing_signals) + len(layering_signals)) / window),
            "timestamp": now_indian()
        }
        
        return detection_result
    
    def generate_market_quality_report(self, symbol):
        """Generate comprehensive market quality report"""
        if symbol not in self.order_flow_metrics or len(self.order_flow_metrics[symbol]) < 100:
            return None
        
        metrics = self.order_flow_metrics[symbol][-100:]  # Last 100 readings
        
        spreads = [m["spread_bps"] for m in metrics]
        volume_imbalances = [m["volume_imbalance"] for m in metrics]
        liquidity_scores = [m["liquidity_score"] for m in metrics]
        market_impacts = [m["market_impact_bps"] for m in metrics]
        
        report = {
            "symbol": symbol,
            "period": f"{metrics[0]['timestamp']} to {metrics[-1]['timestamp']}",
            "samples": len(metrics),
            "spread_analysis": {
                "avg_spread_bps": np.mean(spreads),
                "std_spread_bps": np.std(spreads),
                "min_spread_bps": np.min(spreads),
                "max_spread_bps": np.max(spreads),
                "spread_stability": 1 - (np.std(spreads) / np.mean(spreads)) if np.mean(spreads) > 0 else 0
            },
            "liquidity_analysis": {
                "avg_liquidity_score": np.mean(liquidity_scores),
                "liquidity_trend": self._calculate_trend(liquidity_scores),
                "volume_stability": np.std(volume_imbalances) / (np.max(volume_imbalances) - np.min(volume_imbalances)) if (np.max(volume_imbalances) - np.min(volume_imbalances)) > 0 else 0
            },
            "market_impact_analysis": {
                "avg_market_impact_bps": np.mean(market_impacts),
                "max_market_impact_bps": np.max(market_impacts),
                "impact_for_1cr_order": np.mean(market_impacts) * 2  # Rough estimate for ₹1Cr order
            },
            "order_flow_analysis": {
                "avg_volume_imbalance": np.mean(volume_imbalances),
                "order_flow_trend": self._calculate_trend(volume_imbalances),
                "buying_pressure_days": sum(1 for vi in volume_imbalances if vi > 0.1),
                "selling_pressure_days": sum(1 for vi in volume_imbalances if vi < -0.1)
            },
            "market_quality_score": self._calculate_market_quality_score(
                np.mean(spreads), np.mean(liquidity_scores), 
                np.std(volume_imbalances), np.mean(market_impacts)
            ),
            "recommendations": self._generate_market_quality_recommendations(
                np.mean(spreads), np.mean(liquidity_scores),
                np.std(volume_imbalances)
            )
        }
        
        return report
    
    def _calculate_trend(self, values):
        """Calculate trend of a time series"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression for trend
        slope, intercept = np.polyfit(x, y, 1)
        
        return slope
    
    def _calculate_market_quality_score(self, avg_spread, avg_liquidity, flow_volatility, market_impact):
        """Calculate overall market quality score (0-100)"""
        # Spread component (lower is better)
        if avg_spread <= 2:
            spread_score = 30
        elif avg_spread <= 5:
            spread_score = 25
        elif avg_spread <= 10:
            spread_score = 20
        elif avg_spread <= 20:
            spread_score = 15
        elif avg_spread <= 50:
            spread_score = 10
        else:
            spread_score = 5
        
        # Liquidity component (higher is better)
        liquidity_score = min(30, avg_liquidity * 0.3)
        
        # Flow stability component (lower volatility is better)
        if flow_volatility <= 0.1:
            flow_score = 20
        elif flow_volatility <= 0.2:
            flow_score = 15
        elif flow_volatility <= 0.3:
            flow_score = 10
        elif flow_volatility <= 0.5:
            flow_score = 5
        else:
            flow_score = 0
        
        # Market impact component (lower is better)
        if market_impact <= 5:
            impact_score = 20
        elif market_impact <= 10:
            impact_score = 15
        elif market_impact <= 20:
            impact_score = 10
        elif market_impact <= 50:
            impact_score = 5
        else:
            impact_score = 0
        
        total_score = spread_score + liquidity_score + flow_score + impact_score
        
        return min(100, total_score)
    
    def _generate_market_quality_recommendations(self, avg_spread, avg_liquidity, flow_volatility):
        """Generate trading recommendations based on market quality"""
        recommendations = []
        
        if avg_spread > 10:
            recommendations.append("High spreads - Consider limit orders only")
        elif avg_spread < 5:
            recommendations.append("Low spreads - Market orders acceptable")
        
        if avg_liquidity < 50:
            recommendations.append("Low liquidity - Reduce position sizes")
        elif avg_liquidity > 70:
            recommendations.append("High liquidity - Can trade larger sizes")
        
        if flow_volatility > 0.3:
            recommendations.append("High order flow volatility - Increased risk of slippage")
        elif flow_volatility < 0.1:
            recommendations.append("Stable order flow - Good execution conditions")
        
        if not recommendations:
            recommendations.append("Market conditions normal - Standard trading protocols")
        
        return recommendations

class MultiTimeframeAnalyzer:
    """Multi-Timeframe Analysis for Professional Trading"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.timeframes = config.TIMEFRAMES
        self.weights = config.TIMEFRAME_WEIGHTS
        self.analysis_cache = {}
        
    def analyze_symbol(self, symbol, include_indicators=True):
        """Comprehensive multi-timeframe analysis"""
        cache_key = f"{symbol}_{include_indicators}"
        
        if cache_key in self.analysis_cache:
            cached = self.analysis_cache[cache_key]
            if time.time() - cached["timestamp"] < 300:  # 5 minute cache
                return cached["analysis"]
        
        analysis = {
            "symbol": symbol,
            "timestamp": now_indian(),
            "timeframes": {},
            "consensus": {},
            "recommendation": None
        }
        
        # Analyze each timeframe
        for tf in self.timeframes:
            tf_analysis = self._analyze_timeframe(symbol, tf, include_indicators)
            if tf_analysis:
                analysis["timeframes"][tf] = tf_analysis
        
        if not analysis["timeframes"]:
            return None
        
        # Calculate consensus
        analysis["consensus"] = self._calculate_consensus(analysis["timeframes"])
        
        # Generate recommendation
        analysis["recommendation"] = self._generate_recommendation(analysis["consensus"])
        
        # Cache analysis
        self.analysis_cache[cache_key] = {
            "analysis": analysis,
            "timestamp": time.time()
        }
        
        return analysis
    
    def _analyze_timeframe(self, symbol, timeframe, include_indicators=True):
        """Analyze single timeframe"""
        # Map timeframe to yfinance interval
        tf_map = {
            "1m": "1m", "5m": "5m", "15m": "15m",
            "30m": "30m", "1h": "60m", "4h": "60m",  # 4h not directly available
            "1d": "1d"
        }
        
        if timeframe not in tf_map:
            return None
        
        # Get data
        data = self.data_manager.get_stock_data(symbol, tf_map[timeframe])
        if data is None or len(data) < 50:
            return None
        
        analysis = {
            "timeframe": timeframe,
            "current_price": float(data["Close"].iloc[-1]),
            "trend": self._analyze_trend(data),
            "momentum": self._analyze_momentum(data) if include_indicators else {},
            "volatility": self._analyze_volatility(data) if include_indicators else {},
            "volume": self._analyze_volume(data) if include_indicators else {},
            "support_resistance": self._analyze_support_resistance(data),
            "score": 0
        }
        
        # Calculate overall score
        scores = []
        
        # Trend score (0-30)
        trend_score = analysis["trend"]["score"] * 30
        scores.append(trend_score)
        
        # Momentum score (0-25)
        if analysis["momentum"]:
            momentum_score = analysis["momentum"]["score"] * 25
            scores.append(momentum_score)
        
        # Volume score (0-20)
        if analysis["volume"]:
            volume_score = analysis["volume"]["score"] * 20
            scores.append(volume_score)
        
        # Support/Resistance score (0-25)
        sr_score = analysis["support_resistance"]["score"] * 25
        scores.append(sr_score)
        
        analysis["score"] = sum(scores)
        
        # Determine bias
        if analysis["score"] >= 60:
            analysis["bias"] = "BULLISH"
        elif analysis["score"] <= 40:
            analysis["bias"] = "BEARISH"
        else:
            analysis["bias"] = "NEUTRAL"
        
        return analysis
    
    def _analyze_trend(self, data):
        """Analyze trend direction and strength"""
        if len(data) < 50:
            return {"direction": "NEUTRAL", "strength": 0, "score": 0.5}
        
        prices = data["Close"].values
        
        # Multiple moving averages
        ma_short = talib.SMA(prices, timeperiod=10) if TALIB_AVAILABLE else pd.Series(prices).rolling(10).mean()
        ma_medium = talib.SMA(prices, timeperiod=20) if TALIB_AVAILABLE else pd.Series(prices).rolling(20).mean()
        ma_long = talib.SMA(prices, timeperiod=50) if TALIB_AVAILABLE else pd.Series(prices).rolling(50).mean()
        
        # Current values
        current_price = prices[-1]
        current_short = ma_short[-1] if not pd.isna(ma_short[-1]) else current_price
        current_medium = ma_medium[-1] if not pd.isna(ma_medium[-1]) else current_price
        current_long = ma_long[-1] if not pd.isna(ma_long[-1]) else current_price
        
        # Previous values for crossover detection
        prev_short = ma_short[-2] if len(ma_short) > 1 and not pd.isna(ma_short[-2]) else current_short
        prev_medium = ma_medium[-2] if len(ma_medium) > 1 and not pd.isna(ma_medium[-2]) else current_medium
        
        # Trend determination
        bullish_conditions = 0
        bearish_conditions = 0
        
        if current_price > current_short > current_medium > current_long:
            bullish_conditions += 3
        elif current_price < current_short < current_medium < current_long:
            bearish_conditions += 3
        
        if current_short > current_medium and prev_short <= prev_medium:
            bullish_conditions += 2
        elif current_short < current_medium and prev_short >= prev_medium:
            bearish_conditions += 2
        
        if current_price > current_medium:
            bullish_conditions += 1
        else:
            bearish_conditions += 1
        
        # Calculate score
        total_conditions = 6
        bullish_score = bullish_conditions / total_conditions
        bearish_score = bearish_conditions / total_conditions
        
        if bullish_conditions > bearish_conditions:
            direction = "BULLISH"
            strength = (bullish_conditions - bearish_conditions) / total_conditions
            score = 0.5 + (strength / 2)
        elif bearish_conditions > bullish_conditions:
            direction = "BEARISH"
            strength = (bearish_conditions - bullish_conditions) / total_conditions
            score = 0.5 - (strength / 2)
        else:
            direction = "NEUTRAL"
            strength = 0
            score = 0.5
        
        return {
            "direction": direction,
            "strength": strength,
            "score": score,
            "price_vs_ma": {
                "vs_short": current_price / current_short - 1,
                "vs_medium": current_price / current_medium - 1,
                "vs_long": current_price / current_long - 1
            }
        }
    
    def _analyze_momentum(self, data):
        """Analyze momentum indicators"""
        if len(data) < 30:
            return {"score": 0.5}
        
        prices = data["Close"].values
        
        # RSI
        rsi_period = 14
        rsi_values = talib.RSI(prices, timeperiod=rsi_period) if TALIB_AVAILABLE else None
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(prices) if TALIB_AVAILABLE else (None, None, None)
        
        # Stochastic
        slowk, slowd = talib.STOCH(data["High"].values, data["Low"].values, prices) if TALIB_AVAILABLE else (None, None)
        
        # Calculate momentum score
        momentum_score = 0.5
        conditions = 0
        
        if rsi_values is not None:
            current_rsi = rsi_values[-1]
            if current_rsi > 70:
                momentum_score -= 0.2
                conditions += 1
            elif current_rsi < 30:
                momentum_score += 0.2
                conditions += 1
        
        if macd is not None and macd_signal is not None:
            current_macd = macd[-1]
            current_signal = macd_signal[-1]
            if current_macd > current_signal:
                momentum_score += 0.1
                conditions += 1
            else:
                momentum_score -= 0.1
                conditions += 1
        
        if slowk is not None and slowd is not None:
            current_k = slowk[-1]
            current_d = slowd[-1]
            if current_k > 80 and current_d > 80:
                momentum_score -= 0.15
                conditions += 1
            elif current_k < 20 and current_d < 20:
                momentum_score += 0.15
                conditions += 1
        
        # Normalize score
        if conditions > 0:
            momentum_score = max(0.1, min(0.9, momentum_score))
        
        return {
            "score": momentum_score,
            "rsi": float(current_rsi) if rsi_values is not None else None,
            "macd_bullish": current_macd > current_signal if macd is not None else None,
            "stochastic_overbought": current_k > 80 if slowk is not None else None,
            "stochastic_oversold": current_k < 20 if slowk is not None else None
        }
    
    def _analyze_volatility(self, data):
        """Analyze volatility characteristics"""
        if len(data) < 20:
            return {"score": 0.5}
        
        prices = data["Close"].values
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        if len(returns) < 10:
            return {"score": 0.5}
        
        # Volatility metrics
        current_vol = np.std(returns[-10:]) * np.sqrt(252)  # Annualized
        historical_vol = np.std(returns) * np.sqrt(252)
        
        # ATR
        atr = talib.ATR(data["High"].values, data["Low"].values, prices, timeperiod=14) if TALIB_AVAILABLE else None
        current_atr = atr[-1] if atr is not None else None
        
        # Volatility score (lower volatility is generally better for trend following)
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1
        
        if vol_ratio < 0.7:
            # Low volatility regime
            score = 0.7
        elif vol_ratio > 1.3:
            # High volatility regime
            score = 0.3
        else:
            # Normal volatility
            score = 0.5
        
        return {
            "score": score,
            "current_volatility": current_vol,
            "volatility_ratio": vol_ratio,
            "atr": current_atr,
            "regime": "LOW" if vol_ratio < 0.7 else "HIGH" if vol_ratio > 1.3 else "NORMAL"
        }
    
    def _analyze_volume(self, data):
        """Analyze volume patterns"""
        if len(data) < 20:
            return {"score": 0.5}
        
        volumes = data["Volume"].values
        prices = data["Close"].values
        
        # Volume indicators
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-20:])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Volume-price correlation
        if len(prices) >= 20:
            price_changes = np.diff(prices[-20:])
            volume_changes = np.diff(volumes[-20:])
            
            if len(price_changes) > 1 and len(volume_changes) > 1:
                correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
            else:
                correlation = 0
        else:
            correlation = 0
        
        # Volume score
        score = 0.5
        
        if volume_ratio > 1.5:
            # High volume
            if correlation > 0.3:
                # Volume confirms price movement
                score = 0.8
            elif correlation < -0.3:
                # Volume contradicts price movement
                score = 0.3
            else:
                score = 0.6
        elif volume_ratio < 0.5:
            # Low volume
            score = 0.4
        
        return {
            "score": score,
            "volume_ratio": volume_ratio,
            "volume_price_correlation": correlation,
            "current_volume": current_volume,
            "avg_volume": avg_volume
        }
    
    def _analyze_support_resistance(self, data):
        """Analyze support and resistance levels"""
        if len(data) < 50:
            return {"score": 0.5, "levels": {}}
        
        prices = data["Close"].values
        highs = data["High"].values
        lows = data["Low"].values
        
        # Find recent swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(20, len(prices) - 20):
            if highs[i] == max(highs[i-20:i+21]):
                swing_highs.append(highs[i])
            if lows[i] == min(lows[i-20:i+21]):
                swing_lows.append(lows[i])
        
        # Current price position
        current_price = prices[-1]
        
        # Find nearest support and resistance
        support_levels = [s for s in swing_lows if s < current_price]
        resistance_levels = [r for r in swing_highs if r > current_price]
        
        nearest_support = max(support_levels) if support_levels else current_price * 0.95
        nearest_resistance = min(resistance_levels) if resistance_levels else current_price * 1.05
        
        # Calculate distances
        support_distance = (current_price - nearest_support) / current_price
        resistance_distance = (nearest_resistance - current_price) / current_price
        
        # Score based on position between S/R
        total_range = support_distance + resistance_distance
        if total_range > 0:
            position_score = resistance_distance / total_range
        else:
            position_score = 0.5
        
        # Adjust score based on proximity to levels
        if support_distance < 0.02 or resistance_distance < 0.02:
            # Near a key level
            score = 0.4
        elif 0.4 < position_score < 0.6:
            # In the middle of range
            score = 0.5
        elif position_score > 0.7:
            # Near resistance
            score = 0.3
        elif position_score < 0.3:
            # Near support
            score = 0.7
        else:
            score = position_score
        
        return {
            "score": score,
            "levels": {
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "support_distance_pct": support_distance * 100,
                "resistance_distance_pct": resistance_distance * 100,
                "position_score": position_score
            }
        }
    
    def _calculate_consensus(self, timeframe_analyses):
        """Calculate consensus across timeframes"""
        if not timeframe_analyses:
            return {}
        
        # Weighted scores
        total_weighted_score = 0
        total_weight = 0
        
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for tf, analysis in timeframe_analyses.items():
            weight = self.weights.get(tf, 0.1)
            score = analysis.get("score", 50)
            
            total_weighted_score += score * weight
            total_weight += weight
            
            bias = analysis.get("bias", "NEUTRAL")
            if bias == "BULLISH":
                bullish_count += 1
            elif bias == "BEARISH":
                bearish_count += 1
            else:
                neutral_count += 1
        
        if total_weight > 0:
            consensus_score = total_weighted_score / total_weight
        else:
            consensus_score = 50
        
        # Determine consensus bias
        total_analyses = len(timeframe_analyses)
        bullish_pct = bullish_count / total_analyses
        bearish_pct = bearish_count / total_analyses
        
        if bullish_pct > 0.6:
            consensus_bias = "STRONG_BULLISH"
        elif bullish_pct > 0.4:
            consensus_bias = "BULLISH"
        elif bearish_pct > 0.6:
            consensus_bias = "STRONG_BEARISH"
        elif bearish_pct > 0.4:
            consensus_bias = "BEARISH"
        else:
            consensus_bias = "NEUTRAL"
        
        # Calculate agreement level
        max_count = max(bullish_count, bearish_count, neutral_count)
        agreement = max_count / total_analyses
        
        return {
            "score": consensus_score,
            "bias": consensus_bias,
            "agreement": agreement,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "timeframes_analyzed": total_analyses
        }
    
    def _generate_recommendation(self, consensus):
        """Generate trading recommendation based on consensus"""
        if not consensus:
            return "NO_CONSENSUS"
        
        score = consensus.get("score", 50)
        bias = consensus.get("bias", "NEUTRAL")
        agreement = consensus.get("agreement", 0.5)
        
        if agreement < 0.4:
            return "WAIT_FOR_CONFIRMATION"
        
        if bias in ["STRONG_BULLISH", "BULLISH"] and score > 60:
            if agreement > 0.7:
                return "STRONG_BUY"
            else:
                return "BUY"
        elif bias in ["STRONG_BEARISH", "BEARISH"] and score < 40:
            if agreement > 0.7:
                return "STRONG_SELL"
            else:
                return "SELL"
        elif 40 <= score <= 60:
            return "HOLD_NEUTRAL"
        else:
            return "NO_CLEAR_SIGNAL"

# ===================== ENHANCED TRADING ENGINE =====================
class ProfessionalMultiStrategyIntradayTrader(MultiStrategyIntradayTrader):
    """Enhanced trader with professional features"""
    
    def __init__(self, capital=CAPITAL):
        super().__init__(capital)
        
        # Professional components
        self.capital_allocator = DynamicCapitalAllocator(capital)
        self.hft_engine = HighFrequencyEngine(self.data_manager) if config.HFT_ENABLED else None
        self.alpha_generator = AlphaGenerator(self.data_manager)
        self.order_router = SmartOrderRouter()
        self.market_microstructure = MarketMicrostructureAnalyzer()
        self.multi_timeframe_analyzer = MultiTimeframeAnalyzer(self.data_manager)
        self.backtesting_engine = BacktestingEngine(self.data_manager)
        
        # Professional metrics
        self.alpha_signals = []
        self.hft_signals = []
        self.professional_stats = {
            "alpha_pnl": 0.0,
            "hft_pnl": 0.0,
            "arbitrage_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "winning_streak": 0,
            "losing_streak": 0
        }
    
    def execute_professional_trade(self, signal, algorithm="TWAP", **kwargs):
        """Execute trade with professional order routing"""
        symbol = signal['symbol']
        action = signal['action']
        
        # Calculate optimal position size
        position_size = self.capital_allocator.calculate_position_size(
            symbol=symbol,
            entry_price=signal['price'],
            stop_loss=signal['stop_loss'],
            confidence=signal['confidence'],
            strategy_stats=self.strategy_performance.get(signal.get('strategy', 'Manual'), {})
        )
        
        quantity = position_size['shares']
        
        if quantity < 1:
            return False, f"Position size too small: {quantity} shares"
        
        # Execute with smart order routing
        execution_result = self.order_router.execute_order(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=signal['price'],
            algorithm=algorithm,
            **kwargs
        )
        
        if execution_result["total_quantity"] == 0:
            return False, "Order execution failed"
        
        # Execute trade in trading system
        success, msg = self.execute_trade(
            symbol=symbol,
            action=action,
            quantity=execution_result["total_quantity"],
            price=execution_result["net_price"],
            stop_loss=signal['stop_loss'],
            target=signal['target'],
            win_probability=signal.get('win_probability', 0.75),
            auto_trade=signal.get('type', 'STANDARD') != 'MANUAL',
            strategy=signal.get('strategy', 'Professional')
        )
        
        if success:
            # Update deployed capital
            self.capital_allocator.update_deployed_capital(
                position_size['position_value'], 
                action="add"
            )
            
            # Record professional execution
            signal['execution_details'] = {
                'algorithm': algorithm,
                'avg_price': execution_result['avg_price'],
                'net_price': execution_result['net_price'],
                'slippage': execution_result.get('slippage', 0),
                'commission': execution_result.get('commission', 0),
                'position_size': position_size
            }
            
            if signal.get('type') == 'ALPHA':
                self.alpha_signals.append(signal)
            elif signal.get('type') == 'HFT':
                self.hft_signals.append(signal)
        
        return success, f"{msg} | Algorithm: {algorithm} | Slippage: {execution_result.get('slippage', 0):.4f}"
    
    def scan_alpha_signals(self):
        """Scan for alpha generation signals"""
        signals = self.alpha_generator.generate_all_alpha_signals()
        self.alpha_signals = signals
        return signals
    
    def process_hft_ticks(self, symbol, tick_data):
        """Process HFT tick data"""
        if not self.hft_engine:
            return []
        
        signals = self.hft_engine.process_tick(symbol, tick_data)
        self.hft_signals.extend(signals)
        return signals
    
    def analyze_market_microstructure(self, symbol, bids, asks):
        """Analyze market microstructure"""
        return self.market_microstructure.analyze_order_book(symbol, bids, asks)
    
    def multi_timeframe_analysis(self, symbol):
        """Perform multi-timeframe analysis"""
        return self.multi_timeframe_analyzer.analyze_symbol(symbol)
    
    def run_strategy_backtest(self, strategy_name, symbol, start_date, end_date, parameters=None):
        """Run professional backtest"""
        # Map strategy name to function
        strategy_map = {
            "Moving Average Crossover": self._ma_crossover_strategy,
            "RSI Mean Reversion": self._rsi_mean_reversion_strategy,
            "Bollinger Band Reversion": self._bollinger_reversion_strategy
        }
        
        strategy_func = strategy_map.get(strategy_name)
        if not strategy_func:
            return {"error": f"Strategy {strategy_name} not found"}
        
        return self.backtesting_engine.run_strategy_backtest(
            strategy_func, symbol, start_date, end_date, 
            self.initial_capital, parameters
        )
    
    def run_walk_forward_analysis(self, strategy_name, symbol, start_date, end_date):
        """Run walk-forward analysis"""
        strategy_map = {
            "Moving Average Crossover": self._ma_crossover_strategy,
            "RSI Mean Reversion": self._rsi_mean_reversion_strategy,
            "Bollinger Band Reversion": self._bollinger_reversion_strategy
        }
        
        strategy_func = strategy_map.get(strategy_name)
        if not strategy_func:
            return {"error": f"Strategy {strategy_name} not found"}
        
        return self.backtesting_engine.run_walk_forward_analysis(
            strategy_func, symbol, start_date, end_date
        )
    
    def _ma_crossover_strategy(self, data, parameters):
        """MA Crossover strategy for backtesting"""
        # Simplified for demonstration
        return None
    
    def _rsi_mean_reversion_strategy(self, data, parameters):
        """RSI Mean Reversion strategy for backtesting"""
        # Simplified for demonstration
        return None
    
    def _bollinger_reversion_strategy(self, data, parameters):
        """Bollinger Band Reversion strategy for backtesting"""
        # Simplified for demonstration
        return None
    
    def get_professional_stats(self):
        """Get professional trading statistics"""
        basic_stats = self.get_performance_stats()
        
        # Calculate advanced metrics
        if self.trade_log:
            closed_trades = [t for t in self.trade_log if t.get("status") == "CLOSED"]
            if closed_trades:
                returns = [t.get("closed_pnl", 0) / (t.get("entry_price", 1) * t.get("quantity", 1)) 
                          for t in closed_trades]
                
                if returns and len(returns) > 1:
                    returns_array = np.array(returns)
                    self.professional_stats["sharpe_ratio"] = (
                        np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) 
                        if np.std(returns_array) > 0 else 0
                    )
                    
                    # Calculate drawdown
                    cumulative = np.cumsum(returns_array)
                    running_max = np.maximum.accumulate(cumulative)
                    drawdown = (cumulative - running_max) / (running_max + 1e-10)
                    self.professional_stats["max_drawdown"] = np.min(drawdown)
        
        # Combine stats
        return {
            **basic_stats,
            **self.professional_stats,
            "capital_utilization": self.capital_allocator.get_capital_utilization(),
            "alpha_signals_generated": len(self.alpha_signals),
            "hft_signals_generated": len(self.hft_signals),
            "execution_stats": self.order_router.get_execution_stats(),
            "hft_stats": self.hft_engine.get_hft_stats() if self.hft_engine else {}
        }

# ===================== UPDATED MAIN APPLICATION =====================
def main():
    # Display Professional Header
    st.markdown("""
    <div class="logo-container" style="background: linear-gradient(135deg, #ff8c00 0%, #ff6b00 100%);">
        <h1 style="color: white; margin: 10px 0 0 0; font-size: 36px;">📈 RANTV TERMINAL PRO</h1>
        <p style="color: white; margin: 5px 0; font-size: 18px;">PROFESSIONAL EDITION - Institutional Grade Trading</p>
        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 10px;">
            <span style="background: white; color: #ff8c00; padding: 5px 15px; border-radius: 20px; font-weight: bold;">High-Frequency Ready</span>
            <span style="background: white; color: #ff8c00; padding: 5px 15px; border-radius: 20px; font-weight: bold;">Alpha Generation</span>
            <span style="background: white; color: #ff8c00; padding: 5px 15px; border-radius: 20px; font-weight: bold;">Large Capital</span>
            <span style="background: white; color: #ff8c00; padding: 5px 15px; border-radius: 20px; font-weight: bold;">Risk Managed</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align:center; color: #ff8c00;'>Professional Trading Suite</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color: #6b7280;'>Complete with HFT, Alpha Strategies & Institutional Execution</h4>", unsafe_allow_html=True)
    
    # Initialize professional components
    data_manager = EnhancedDataManager()
    
    if st.session_state.trader is None:
        st.session_state.trader = ProfessionalMultiStrategyIntradayTrader()
    
    trader = st.session_state.trader
    
    if st.session_state.kite_manager is None:
        st.session_state.kite_manager = KiteConnectManager(KITE_API_KEY, KITE_API_SECRET)
    
    kite_manager = st.session_state.kite_manager
    
    if st.session_state.algo_engine is None:
        st.session_state.algo_engine = AlgoEngine(trader=trader)
    
    algo_engine = st.session_state.algo_engine
    
    # Auto-refresh for professional trading
    st_autorefresh(interval=PRICE_REFRESH_MS, key="professional_auto_refresh")
    st.session_state.refresh_count += 1
    
    # Professional Sidebar
    with st.sidebar:
        st.header("⚙️ Professional Configuration")
        
        # Trading Mode
        trading_mode = st.selectbox(
            "Trading Mode",
            ["Alpha Generation", "High-Frequency", "Swing Trading", "Arbitrage", "Market Making"],
            key="trading_mode_select"
        )
        
        # Capital Management
        st.subheader("💰 Capital Management")
        
        col1, col2 = st.columns(2)
        with col1:
            risk_per_trade = st.slider(
                "Risk per Trade (%)",
                min_value=0.1, max_value=5.0,
                value=config.RISK_PER_TRADE * 100,
                step=0.1,
                key="risk_per_trade_slider"
            )
        
        with col2:
            max_capital_utilization = st.slider(
                "Max Capital Utilization (%)",
                min_value=10, max_value=100,
                value=int(config.MAX_CAPITAL_UTILIZATION * 100),
                step=5,
                key="max_capital_utilization_slider"
            )
        
        # Strategy Selection
        st.subheader("🎯 Active Strategies")
        
        alpha_strategies = st.multiselect(
            "Alpha Strategies",
            config.ENABLED_STRATEGIES,
            default=["pairs_trading", "statistical_arbitrage", "volatility_arbitrage"],
            key="alpha_strategies_select"
        )
        
        # Execution Settings
        st.subheader("🚀 Execution Settings")
        
        default_algorithm = st.selectbox(
            "Default Execution Algorithm",
            config.EXECUTION_ALGORITHMS,
            index=0,
            key="execution_algorithm_select"
        )
        
        max_slippage = st.slider(
            "Max Slippage (bps)",
            min_value=1, max_value=50,
            value=int(config.DEFAULT_SLIPPAGE * 10000),
            step=1,
            key="max_slippage_slider"
        )
        
        # Professional Features Toggle
        st.subheader("🛠️ Professional Features")
        
        enable_hft = st.checkbox("Enable HFT Engine", value=config.HFT_ENABLED, key="enable_hft_checkbox")
        enable_alpha = st.checkbox("Enable Alpha Generation", value=True, key="enable_alpha_checkbox")
        enable_microstructure = st.checkbox("Enable Market Microstructure Analysis", value=True, key="enable_microstructure_checkbox")
        enable_backtesting = st.checkbox("Enable Backtesting Engine", value=True, key="enable_backtesting_checkbox")
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Run Alpha Scan", key="quick_alpha_scan_btn"):
                with st.spinner("Scanning for alpha opportunities..."):
                    signals = trader.scan_alpha_signals()
                    if signals:
                        st.success(f"Found {len(signals)} alpha signals")
                    else:
                        st.info("No alpha signals found")
        
        with col2:
            if st.button("📈 Update All Analysis", key="update_all_analysis_btn"):
                st.info("Updating all professional analyses...")
                # This would trigger updates in the main interface
        
        # System Status
        st.divider()
        st.subheader("🛠️ System Status")
        
        st.write(f"✅ Capital: ₹{trader.capital_allocator.total_capital:,.0f}")
        st.write(f"📊 Utilized: {trader.capital_allocator.get_capital_utilization():.1%}")
        st.write(f"🚀 HFT: {'Active' if trader.hft_engine else 'Inactive'}")
        st.write(f"🎯 Alpha: {len(trader.alpha_signals)} signals")
        st.write(f"📈 Positions: {len(trader.positions)} open")
        st.write(f"🔄 Refresh: {st.session_state.refresh_count}")
        
        # Professional Reports
        if st.button("📋 Generate Professional Report", key="professional_report_btn"):
            st.info("Generating professional trading report...")
            # This would generate a comprehensive report
    
    # Main Interface - Professional Tabs
    tabs = st.tabs(["📊 Professional Dashboard", "🚀 Alpha Generation", "⚡ High-Frequency", 
                   "📈 Multi-Timeframe", "🔬 Market Microstructure", "📊 Backtesting Lab",
                   "🤖 Algo Trading", "📋 Trade Desk"])
    
    # Tab 1: Professional Dashboard
    with tabs[0]:
        st.subheader("📊 Professional Trading Dashboard")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Capital", f"₹{trader.capital_allocator.total_capital:,.0f}")
        
        with col2:
            utilization = trader.capital_allocator.get_capital_utilization()
            st.metric("Capital Utilization", f"{utilization:.1%}")
        
        with col3:
            stats = trader.get_professional_stats()
            st.metric("Sharpe Ratio", f"{stats.get('sharpe_ratio', 0):.2f}")
        
        with col4:
            st.metric("Max Drawdown", f"{stats.get('max_drawdown', 0):.2%}")
        
        # Performance Charts
        st.subheader("Performance Analytics")
        
        # Create performance visualization
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            # Strategy Performance
            strategy_data = []
            for strategy, config in TRADING_STRATEGIES.items():
                if strategy in trader.strategy_performance:
                    perf = trader.strategy_performance[strategy]
                    if perf["trades"] > 0:
                        win_rate = perf["wins"] / perf["trades"]
                        strategy_data.append({
                            "Strategy": config["name"],
                            "Trades": perf["trades"],
                            "Win Rate": win_rate,
                            "P&L": perf["pnl"]
                        })
            
            if strategy_data:
                df_strategies = pd.DataFrame(strategy_data)
                st.dataframe(df_strategies, use_container_width=True)
        
        with perf_col2:
            # Capital Allocation
            deployed = trader.capital_allocator.deployed_capital
            available = trader.capital_allocator.total_capital - deployed
            
            fig = go.Figure(data=[go.Pie(
                labels=['Deployed', 'Available'],
                values=[deployed, available],
                hole=0.4,
                marker_colors=['#ff8c00', '#e5e7eb']
            )])
            fig.update_layout(title="Capital Allocation")
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent Activity
        st.subheader("Recent Professional Activity")
        
        activity_cols = st.columns(3)
        
        with activity_cols[0]:
            st.metric("Alpha Signals", len(trader.alpha_signals))
        
        with activity_cols[1]:
            st.metric("HFT Signals", len(trader.hft_signals))
        
        with activity_cols[2]:
            exec_stats = trader.order_router.get_execution_stats()
            st.metric("Execution Success", f"{exec_stats.get('success_rate', 0):.1%}")
    
    # Tab 2: Alpha Generation
    with tabs[1]:
        st.subheader("🚀 Alpha Generation Engine")
        
        # Alpha Strategy Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            scan_type = st.selectbox(
                "Scan Type",
                ["Statistical Arbitrage", "Pairs Trading", "Volatility Arb", "Market Neutral", "All Alpha"],
                key="alpha_scan_type"
            )
        
        with col2:
            min_confidence = st.slider(
                "Min Confidence",
                min_value=0.6, max_value=0.95,
                value=0.75, step=0.05,
                key="alpha_min_confidence"
            )
        
        with col3:
            max_signals = st.number_input(
                "Max Signals",
                min_value=1, max_value=50,
                value=10,
                key="alpha_max_signals"
            )
        
        # Run Alpha Scan
        if st.button("🎯 Run Alpha Scan", type="primary", key="run_alpha_scan_btn"):
            with st.spinner("Running alpha generation..."):
                signals = trader.scan_alpha_signals()
                
                if signals:
                    # Filter by confidence
                    filtered_signals = [s for s in signals if s.get('confidence', 0) >= min_confidence]
                    filtered_signals = filtered_signals[:max_signals]
                    
                    # Display signals
                    signal_data = []
                    for i, signal in enumerate(filtered_signals):
                        signal_type = signal.get('type', 'ALPHA')
                        signal_data.append({
                            "#": i+1,
                            "Symbol": signal['symbol'].replace('.NS', ''),
                            "Type": signal_type,
                            "Action": signal['action'],
                            "Price": f"₹{signal['price']:.2f}",
                            "Confidence": f"{signal['confidence']:.1%}",
                            "Strategy": signal['strategy'],
                            "Z-Score": signal.get('z_score', 'N/A')
                        })
                    
                    df_alpha = pd.DataFrame(signal_data)
                    st.dataframe(df_alpha, use_container_width=True)
                    
                    # Execution controls
                    st.subheader("Alpha Execution")
                    
                    exec_cols = st.columns(4)
                    
                    with exec_cols[0]:
                        if st.button("🎯 Execute Top Alpha", key="execute_top_alpha_btn"):
                            if filtered_signals:
                                success, msg = trader.execute_professional_trade(
                                    filtered_signals[0],
                                    algorithm=default_algorithm
                                )
                                if success:
                                    st.success(msg)
                                else:
                                    st.error(msg)
                    
                    with exec_cols[1]:
                        if st.button("📈 Execute All BUY", key="execute_all_buy_alpha_btn"):
                            buy_signals = [s for s in filtered_signals if s['action'] == 'BUY']
                            executed = 0
                            for signal in buy_signals[:3]:  # Limit to 3
                                success, msg = trader.execute_professional_trade(signal)
                                if success:
                                    executed += 1
                            st.success(f"Executed {executed} BUY signals")
                    
                    with exec_cols[2]:
                        if st.button("📉 Execute All SELL", key="execute_all_sell_alpha_btn"):
                            sell_signals = [s for s in filtered_signals if s['action'] == 'SELL']
                            executed = 0
                            for signal in sell_signals[:3]:
                                success, msg = trader.execute_professional_trade(signal)
                                if success:
                                    executed += 1
                            st.success(f"Executed {executed} SELL signals")
                    
                    with exec_cols[3]:
                        algorithm = st.selectbox(
                            "Execution Algorithm",
                            config.EXECUTION_ALGORITHMS,
                            key="alpha_exec_algorithm"
                        )
                
                else:
                    st.info("No alpha signals generated. Try adjusting parameters.")
        
        # Alpha Performance
        st.subheader("Alpha Strategy Performance")
        
        if trader.alpha_signals:
            alpha_stats = []
            for signal in trader.alpha_signals[-10:]:  # Last 10
                alpha_stats.append({
                    "Time": signal.get('timestamp', now_indian()).strftime("%H:%M"),
                    "Symbol": signal['symbol'].replace('.NS', ''),
                    "Strategy": signal['strategy'],
                    "Action": signal['action'],
                    "Confidence": f"{signal['confidence']:.1%}"
                })
            
            df_alpha_stats = pd.DataFrame(alpha_stats)
            st.dataframe(df_alpha_stats, use_container_width=True)
        else:
            st.info("No alpha signals executed yet.")
    
    # Tab 3: High-Frequency Trading
    with tabs[2]:
        st.subheader("⚡ High-Frequency Trading Engine")
        
        if not trader.hft_engine:
            st.warning("HFT Engine is disabled. Enable in sidebar.")
        else:
            # HFT Controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                hft_mode = st.selectbox(
                    "HFT Mode",
                    ["Tick Momentum", "Order Flow", "Statistical Arb", "Volatility Scalping"],
                    key="hft_mode_select"
                )
            
            with col2:
                hft_aggressiveness = st.slider(
                    "Aggressiveness",
                    min_value=1, max_value=10,
                    value=5,
                    key="hft_aggressiveness"
                )
            
            with col3:
                max_hft_positions = st.number_input(
                    "Max HFT Positions",
                    min_value=1, max_value=config.HFT_MAX_POSITIONS,
                    value=5,
                    key="max_hft_positions"
                )
            
            # Simulated HFT Feed
            st.subheader("📊 HFT Market Feed")
            
            # Simulate tick data
            if st.button("🔄 Simulate HFT Ticks", key="simulate_hft_ticks_btn"):
                symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
                
                for symbol in symbols:
                    # Generate simulated tick
                    tick_data = {
                        'symbol': symbol,
                        'price': random.uniform(1000, 5000),
                        'volume': random.randint(1000, 10000),
                        'bid_volume': random.randint(500, 5000),
                        'ask_volume': random.randint(500, 5000),
                        'timestamp': now_indian()
                    }
                    
                    # Process HFT tick
                    signals = trader.process_hft_ticks(symbol, tick_data)
                    
                    if signals:
                        st.success(f"{symbol}: Generated {len(signals)} HFT signals")
            
            # HFT Statistics
            st.subheader("HFT Performance")
            
            hft_stats = trader.hft_engine.get_hft_stats() if trader.hft_engine else {}
            
            stats_cols = st.columns(4)
            stats_cols[0].metric("Active Strategies", hft_stats.get("active_strategies", 0))
            stats_cols[1].metric("Total HFT Trades", hft_stats.get("total_trades", 0))
            stats_cols[2].metric("HFT P&L", f"₹{hft_stats.get('total_pnl', 0):.2f}")
            stats_cols[3].metric("Avg Trade P&L", f"₹{hft_stats.get('avg_trade_pnl', 0):.2f}")
            
            # Recent HFT Signals
            if trader.hft_signals:
                st.subheader("Recent HFT Signals")
                
                hft_data = []
                for signal in trader.hft_signals[-10:]:
                    hft_data.append({
                        "Symbol": signal['symbol'].replace('.NS', ''),
                        "Action": signal['action'],
                        "Price": f"₹{signal['price']:.2f}",
                        "Strategy": signal['strategy'],
                        "Confidence": f"{signal['confidence']:.1%}"
                    })
                
                df_hft = pd.DataFrame(hft_data)
                st.dataframe(df_hft, use_container_width=True)
    
    # Tab 4: Multi-Timeframe Analysis
    with tabs[3]:
        st.subheader("📈 Multi-Timeframe Analysis")
        
        symbol = st.selectbox(
            "Select Symbol",
            NIFTY_50[:10],
            key="mta_symbol_select"
        )
        
        if st.button("🔍 Analyze Timeframes", key="analyze_timeframes_btn"):
            with st.spinner(f"Analyzing {symbol} across timeframes..."):
                analysis = trader.multi_timeframe_analysis(symbol)
                
                if analysis:
                    # Display consensus
                    consensus = analysis["consensus"]
                    
                    st.subheader("📊 Consensus Analysis")
                    
                    cons_cols = st.columns(3)
                    cons_cols[0].metric("Consensus Score", f"{consensus['score']:.1f}")
                    cons_cols[1].metric("Bias", consensus['bias'])
                    cons_cols[2].metric("Agreement", f"{consensus['agreement']:.0%}")
                    
                    # Display timeframe analysis
                    st.subheader("📈 Timeframe Breakdown")
                    
                    for tf, tf_analysis in analysis["timeframes"].items():
                        with st.expander(f"{tf} - {tf_analysis['bias']} (Score: {tf_analysis['score']:.1f})"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Price**: ₹{tf_analysis['current_price']:.2f}")
                                st.write(f"**Trend**: {tf_analysis['trend']['direction']}")
                                st.write(f"**Trend Strength**: {tf_analysis['trend']['strength']:.2f}")
                            
                            with col2:
                                if tf_analysis.get('momentum'):
                                    st.write(f"**RSI**: {tf_analysis['momentum'].get('rsi', 'N/A')}")
                                    st.write(f"**MACD**: {'Bullish' if tf_analysis['momentum'].get('macd_bullish') else 'Bearish'}")
                    
                    # Recommendation
                    st.subheader("🎯 Trading Recommendation")
                    
                    recommendation = analysis["recommendation"]
                    if recommendation in ["STRONG_BUY", "BUY"]:
                        st.success(f"✅ {recommendation}")
                    elif recommendation in ["STRONG_SELL", "SELL"]:
                        st.error(f"❌ {recommendation}")
                    else:
                        st.info(f"ℹ️ {recommendation}")
    
    # Tab 5: Market Microstructure
    with tabs[5]:
        st.subheader("🔬 Market Microstructure Analysis")
        
        symbol = st.selectbox(
            "Select Symbol for Microstructure",
            NIFTY_50[:5],
            key="microstructure_symbol"
        )
        
        # Simulated order book
        st.subheader("📊 Simulated Order Book")
        
        # Generate simulated bids and asks
        current_price = trader.data_manager._validate_live_price(symbol)
        
        bids = []
        asks = []
        
        for i in range(1, 6):
            bid_price = current_price * (1 - i * 0.001)
            ask_price = current_price * (1 + i * 0.001)
            
            bids.append((round(bid_price, 2), random.randint(100, 1000)))
            asks.append((round(ask_price, 2), random.randint(100, 1000)))
        
        # Display order book
        ob_cols = st.columns(2)
        
        with ob_cols[0]:
            st.markdown("**Bid Side**")
            for price, qty in bids:
                st.write(f"₹{price:.2f} | {qty:>6}")
        
        with ob_cols[1]:
            st.markdown("**Ask Side**")
            for price, qty in asks:
                st.write(f"₹{price:.2f} | {qty:>6}")
        
        # Analyze microstructure
        if st.button("🔍 Analyze Microstructure", key="analyze_microstructure_btn"):
            analysis = trader.analyze_market_microstructure(symbol, bids, asks)
            
            if analysis:
                st.subheader("📈 Microstructure Analysis")
                
                # Key metrics
                metrics_cols = st.columns(4)
                metrics_cols[0].metric("Spread", f"{analysis['spread_bps']:.1f} bps")
                metrics_cols[1].metric("Volume Imbalance", f"{analysis['volume_imbalance']:.2f}")
                metrics_cols[2].metric("Liquidity Score", f"{analysis['liquidity_score']:.0f}")
                metrics_cols[3].metric("Market Impact", f"{analysis['market_impact_bps']:.1f} bps")
                
                # Detailed analysis
                with st.expander("Detailed Analysis", expanded=False):
                    st.write(f"**Best Bid**: ₹{analysis['best_bid']:.2f}")
                    st.write(f"**Best Ask**: ₹{analysis['best_ask']:.2f}")
                    st.write(f"**Mid Price**: ₹{analysis['mid_price']:.2f}")
                    st.write(f"**Total Bid Volume**: {analysis['total_bid_volume']:,}")
                    st.write(f"**Total Ask Volume**: {analysis['total_ask_volume']:,}")
                    st.write(f"**Large Bids**: {analysis['large_bid_count']}")
                    st.write(f"**Large Asks**: {analysis['large_ask_count']}")
                    st.write(f"**Order Flow Pressure**: {analysis['order_flow_pressure']:.2f}")
    
    # Tab 6: Backtesting Lab
    with tabs[6]:
        st.subheader("📊 Backtesting Laboratory")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strategy = st.selectbox(
                "Strategy",
                ["Moving Average Crossover", "RSI Mean Reversion", "Bollinger Band Reversion"],
                key="backtest_strategy"
            )
        
        with col2:
            symbol = st.selectbox(
                "Symbol",
                NIFTY_50[:10],
                key="backtest_symbol"
            )
        
        with col3:
            capital = st.number_input(
                "Test Capital (₹)",
                min_value=100000,
                max_value=10000000,
                value=1000000,
                step=100000,
                key="backtest_capital"
            )
        
        # Date range
        col4, col5 = st.columns(2)
        
        with col4:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        
        with col5:
            end_date = st.date_input("End Date", value=datetime.now())
        
        # Run backtest
        if st.button("🚀 Run Backtest", type="primary", key="run_backtest_btn"):
            with st.spinner(f"Backtesting {strategy} on {symbol}..."):
                results = trader.run_strategy_backtest(
                    strategy, symbol, start_date, end_date, capital
                )
                
                if "error" in results:
                    st.error(results["error"])
                else:
                    st.success("Backtest completed successfully!")
                    
                    # Display results
                    st.subheader("📈 Backtest Results")
                    
                    # Key metrics
                    res_cols = st.columns(4)
                    res_cols[0].metric("Total Return", f"{results.get('total_return', 0):.1%}")
                    res_cols[1].metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
                    res_cols[2].metric("Win Rate", f"{results.get('win_rate', 0):.1%}")
                    res_cols[3].metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}")
                    
                    # Detailed metrics
                    with st.expander("Detailed Performance Metrics", expanded=False):
                        detail_cols = st.columns(3)
                        
                        detail_cols[0].write(f"**Total Trades**: {results.get('total_trades', 0)}")
                        detail_cols[0].write(f"**Winning Trades**: {results.get('winning_trades', 0)}")
                        detail_cols[0].write(f"**Losing Trades**: {results.get('losing_trades', 0)}")
                        
                        detail_cols[1].write(f"**Total P&L**: ₹{results.get('total_pnl', 0):.2f}")
                        detail_cols[1].write(f"**Avg Win**: ₹{results.get('avg_win', 0):.2f}")
                        detail_cols[1].write(f"**Avg Loss**: ₹{results.get('avg_loss', 0):.2f}")
                        
                        detail_cols[2].write(f"**Profit Factor**: {results.get('profit_factor', 0):.2f}")
                        detail_cols[2].write(f"**Sortino Ratio**: {results.get('sortino_ratio', 0):.2f}")
                        detail_cols[2].write(f"**Calmar Ratio**: {results.get('calmar_ratio', 0):.2f}")
        
        # Walk-Forward Analysis
        st.subheader("🔄 Walk-Forward Analysis")
        
        if st.button("📊 Run Walk-Forward Analysis", key="run_wfa_btn"):
            with st.spinner("Running walk-forward analysis..."):
                wfa_results = trader.run_walk_forward_analysis(strategy, symbol, start_date, end_date)
                
                if "error" in wfa_results:
                    st.error(wfa_results["error"])
                else:
                    st.success("Walk-forward analysis completed!")
                    
                    best_params = wfa_results.get("best_parameters", {})
                    consolidated = wfa_results.get("consolidated_metrics", {})
                    
                    st.write(f"**Best Parameters**: {best_params.get('params', {})}")
                    st.write(f"**Best Sharpe Ratio**: {best_params.get('sharpe_ratio', 0):.2f}")
                    
                    st.write(f"**Average Sharpe**: {consolidated.get('avg_sharpe_ratio', 0):.2f}")
                    st.write(f"**Average Win Rate**: {consolidated.get('avg_win_rate', 0):.1%}")
                    st.write(f"**Consistency**: {consolidated.get('consistency', 0):.1%}")
    
    # Tab 7: Algo Trading (Updated)
    with tabs[7]:
        create_algo_trading_tab(algo_engine)
    
    # Tab 8: Trade Desk
    with tabs[8]:
        st.subheader("📋 Professional Trade Desk")
        
        # Current Positions
        st.subheader("📊 Current Positions")
        
        positions = trader.get_open_positions_data()
        if positions:
            df_positions = pd.DataFrame(positions)
            st.dataframe(df_positions, use_container_width=True)
        else:
            st.info("No open positions")
        
        # Trade Execution
        st.subheader("🎯 Manual Trade Execution")
        
        exec_cols = st.columns(4)
        
        with exec_cols[0]:
            exec_symbol = st.selectbox("Symbol", NIFTY_50[:20], key="exec_symbol")
        
        with exec_cols[1]:
            exec_action = st.selectbox("Action", ["BUY", "SELL"], key="exec_action")
        
        with exec_cols[2]:
            exec_quantity = st.number_input("Quantity", min_value=1, value=100, key="exec_quantity")
        
        with exec_cols[3]:
            exec_algorithm = st.selectbox("Algorithm", config.EXECUTION_ALGORITHMS, key="exec_algorithm")
        
        # Get current price
        current_price = trader.data_manager._validate_live_price(exec_symbol)
        
        if st.button("🚀 Execute Professional Trade", type="primary", key="execute_pro_trade_btn"):
            # Create signal
            signal = {
                'symbol': exec_symbol,
                'action': exec_action,
                'price': current_price,
                'stop_loss': current_price * 0.98 if exec_action == 'BUY' else current_price * 1.02,
                'target': current_price * 1.03 if exec_action == 'BUY' else current_price * 0.97,
                'confidence': 0.8,
                'strategy': 'Manual_Professional',
                'type': 'MANUAL'
            }
            
            success, msg = trader.execute_professional_trade(
                signal,
                algorithm=exec_algorithm,
                quantity=exec_quantity
            )
            
            if success:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")
        
        # Trade History
        st.subheader("📋 Trade History")
        
        if trader.trade_log:
            history_data = []
            for trade in trader.trade_log[-20:]:  # Last 20 trades
                history_data.append({
                    "Symbol": trade['symbol'].replace('.NS', ''),
                    "Action": trade['action'],
                    "Quantity": trade['quantity'],
                    "Entry": f"₹{trade['entry_price']:.2f}",
                    "Exit": f"₹{trade.get('exit_price', '')}",
                    "P&L": f"₹{trade.get('closed_pnl', trade.get('current_pnl', 0)):+.2f}",
                    "Strategy": trade.get('strategy', 'Manual'),
                    "Auto": "Yes" if trade.get('auto_trade') else "No"
                })
            
            df_history = pd.DataFrame(history_data)
            st.dataframe(df_history, use_container_width=True)
    
    # Professional Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; font-size: 12px;">
        <strong>Rantv Terminal Pro - Professional Edition</strong> | Institutional Grade Trading Suite | © 2024 | 
        Capital: ₹{trader.capital_allocator.total_capital:,.0f} | Utilized: {trader.capital_allocator.get_capital_utilization():.1%} | 
        Alpha Signals: {len(trader.alpha_signals)} | HFT Mode: {config.HFT_ENABLED} | 
        {now_indian().strftime("%H:%M:%S")} | Daily Exit: 3:35 PM | Professional Reports: rantv2002@gmail.com
    </div>
    """, unsafe_allow_html=True)

# Run the professional application
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Professional application error: {str(e)}")
        st.code(traceback.format_exc())
