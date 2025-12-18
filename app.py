import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache

@dataclass(frozen=True)
class MarketData:
    """Immutable market data structure for better caching"""
    timestamp: int
    price: float
    volume: float
    symbol: str

class OptimizedStrategy:
    def __init__(self, config: Dict):
        # Convert lists to numpy arrays for vectorized operations
        self.param_array = np.array(config['parameters'])
        self.weights = np.array(config.get('weights', []))
        self.threshold = config.get('threshold', 0.5)
        
        # Precompute frequently used values
        self.sma_period = config.get('sma_period', 20)
        self.ema_alpha = 2 / (self.sma_period + 1)
        
        # Use local references for frequently accessed attributes
        self._cache = {}
        self._last_calc = {}
        
        # Initialize numpy arrays for time series data
        self.price_history = np.zeros(1000, dtype=np.float64)
        self.volume_history = np.zeros(1000, dtype=np.float64)
        self.index = 0
        self.max_history = 1000

    def update_market_data(self, data: MarketData) -> None:
        """Efficient market data update using numpy arrays"""
        idx = self.index % self.max_history
        self.price_history[idx] = data.price
        self.volume_history[idx] = data.volume
        self.index += 1
        
        # Clear cache when new data arrives
        self._cache.clear()

    @lru_cache(maxsize=128)
    def calculate_signal(self, lookback: int) -> float:
        """Memoized signal calculation"""
        if self.index < lookback:
            return 0.0
            
        start = max(0, self.index - lookback)
        recent_prices = self.price_history[start:self.index]
        
        if len(recent_prices) < 2:
            return 0.0
            
        # Vectorized calculations
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns) if len(returns) > 1 else 0.0
        trend = np.mean(np.sign(np.diff(recent_prices)))
        
        # Precompute weights if not done
        if not hasattr(self, '_signal_weights'):
            self._signal_weights = np.linspace(0.1, 1.0, len(recent_prices))
        
        weighted_signal = np.sum(self._signal_weights[-len(recent_prices):] * recent_prices)
        normalized_signal = weighted_signal / np.sum(self._signal_weights[-len(recent_prices):])
        
        return normalized_signal * trend * (1.0 - volatility)

    def generate_signals(self, market_data: List[MarketData]) -> np.ndarray:
        """Vectorized signal generation"""
        if not market_data:
            return np.array([])
        
        # Convert to numpy arrays in one operation
        prices = np.array([d.price for d in market_data])
        volumes = np.array([d.volume for d in market_data])
        
        # Vectorized calculations
        signals = np.zeros_like(prices)
        
        # Precompute rolling windows
        if len(prices) >= self.sma_period:
            # Use convolution for SMA (much faster than manual loops)
            window = np.ones(self.sma_period) / self.sma_period
            sma = np.convolve(prices, window, mode='valid')
            
            # Vectorized comparison
            sma_full = np.full_like(prices, np.nan)
            sma_full[self.sma_period-1:] = sma
            
            # Generate signals based on price vs SMA
            signals = np.where(prices > sma_full, 1.0, 
                              np.where(prices < sma_full, -1.0, 0.0))
        
        return signals

    def calculate_portfolio_weights(self, signals: np.ndarray, 
                                  risk_budget: np.ndarray) -> np.ndarray:
        """Efficient portfolio weight calculation"""
        if len(signals) == 0:
            return np.array([])
        
        # Clip signals to avoid extreme values
        clipped_signals = np.clip(signals, -2.0, 2.0)
        
        # Normalize using efficient operations
        abs_signals = np.abs(clipped_signals)
        sum_abs = np.sum(abs_signals)
        
        if sum_abs > 0:
            # Use vectorized operations
            weights = clipped_signals / sum_abs
            weights *= risk_budget[:len(weights)]
            
            # Ensure weights sum to approximately 1
            weight_sum = np.sum(np.abs(weights))
            if weight_sum > 0:
                weights = weights / weight_sum
        else:
            weights = np.zeros_like(clipped_signals)
        
        return weights

    def process_batch(self, data_batch: List[MarketData]) -> Dict:
        """Optimized batch processing"""
        if not data_batch:
            return {"signals": [], "weights": []}
        
        # Single pass operations
        signals = self.generate_signals(data_batch)
        
        # Pre-allocate risk budget array
        n_assets = len(signals)
        risk_budget = np.full(n_assets, 1.0 / max(n_assets, 1))
        
        weights = self.calculate_portfolio_weights(signals, risk_budget)
        
        # Use dictionary comprehension for output
        return {
            "timestamp": data_batch[-1].timestamp,
            "signals": signals.tolist(),
            "weights": weights.tolist(),
            "num_assets": n_assets
        }

    def should_enter_trade(self, current_price: float, 
                         entry_price: float, 
                         stop_loss: float) -> bool:
        """Fast trade entry check with early exits"""
        if current_price <= stop_loss:
            return False
            
        # Precompute values for quick comparison
        price_ratio = current_price / entry_price
        
        # Early exit conditions
        if price_ratio > 1.1:  # Already up 10%
            return False
            
        # Single boolean expression
        return (current_price > entry_price * 1.01 and 
                current_price < entry_price * 1.05)

    # Helper methods for common calculations
    @staticmethod
    def normalize_array(arr: np.ndarray) -> np.ndarray:
        """Efficient array normalization"""
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        
        if arr_max - arr_min > 0:
            return (arr - arr_min) / (arr_max - arr_min)
        return np.zeros_like(arr)

    def cleanup(self):
        """Clean up resources"""
        self._cache.clear()
        self._last_calc.clear()

# Usage optimization: Pre-compile regular expressions if used
import re
PATTERN_CACHE = {}

def get_cached_pattern(pattern: str) -> re.Pattern:
    """Cache compiled regex patterns"""
    if pattern not in PATTERN_CACHE:
        PATTERN_CACHE[pattern] = re.compile(pattern)
    return PATTERN_CACHE[pattern]
