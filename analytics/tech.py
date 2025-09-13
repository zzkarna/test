"""Technical analysis indicators."""

import numpy as np
import pandas as pd
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Technical analysis indicators."""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average."""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def wma(data: pd.Series, window: int) -> pd.Series:
        """Weighted Moving Average."""
        weights = np.arange(1, window + 1)
        return data.rolling(window=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)."""
        try:
            ema_fast = TechnicalAnalyzer.ema(data, fast)
            ema_slow = TechnicalAnalyzer.ema(data, slow)
            
            macd_line = ema_fast - ema_slow
            signal_line = TechnicalAnalyzer.ema(macd_line, signal)
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return pd.Series(), pd.Series(), pd.Series()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series()
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range."""
        try:
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=window).mean()
            
            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series()
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        try:
            sma = TechnicalAnalyzer.sma(data, window)
            std = data.rolling(window=window).std()
            
            upper_band = sma + (std * num_std)
            lower_band = sma - (std * num_std)
            
            return upper_band, sma, lower_band
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return pd.Series(), pd.Series(), pd.Series()
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator."""
        try:
            lowest_low = low.rolling(window=k_window).min()
            highest_high = high.rolling(window=k_window).max()
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_window).mean()
            
            return k_percent, d_percent
        except Exception as e:
            logger.error(f"Error calculating Stochastic Oscillator: {e}")
            return pd.Series(), pd.Series()
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R."""
        try:
            highest_high = high.rolling(window=window).max()
            lowest_low = low.rolling(window=window).min()
            
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            
            return williams_r
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {e}")
            return pd.Series()
    
    @staticmethod
    def pivot_points(high: float, low: float, close: float) -> dict:
        """Calculate pivot points."""
        try:
            pivot = (high + low + close) / 3
            
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                'pivot': pivot,
                'r1': r1, 'r2': r2, 'r3': r3,
                's1': s1, 's2': s2, 's3': s3
            }
        except Exception as e:
            logger.error(f"Error calculating pivot points: {e}")
            return {}
    
    @staticmethod
    def detect_crossovers(fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
        """Detect moving average crossovers."""
        try:
            # 1 for bullish crossover, -1 for bearish crossover, 0 for no crossover
            crossovers = pd.Series(0, index=fast_ma.index)
            
            # Bullish crossover: fast MA crosses above slow MA
            bullish = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
            crossovers[bullish] = 1
            
            # Bearish crossover: fast MA crosses below slow MA
            bearish = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
            crossovers[bearish] = -1
            
            return crossovers
        except Exception as e:
            logger.error(f"Error detecting crossovers: {e}")
            return pd.Series()
    
    @staticmethod
    def detect_price_crossovers(price: pd.Series, ma: pd.Series) -> pd.Series:
        """Detect price crossovers with moving average."""
        try:
            crossovers = pd.Series(0, index=price.index)
            
            # Bullish: price crosses above MA
            bullish = (price > ma) & (price.shift(1) <= ma.shift(1))
            crossovers[bullish] = 1
            
            # Bearish: price crosses below MA
            bearish = (price < ma) & (price.shift(1) >= ma.shift(1))
            crossovers[bearish] = -1
            
            return crossovers
        except Exception as e:
            logger.error(f"Error detecting price crossovers: {e}")
            return pd.Series()
    
    @staticmethod
    def support_resistance_levels(data: pd.Series, window: int = 20, 
                                min_touches: int = 2) -> Tuple[List[float], List[float]]:
        """Identify support and resistance levels."""
        try:
            # Find local minima and maxima
            local_min = data[(data.shift(1) > data) & (data.shift(-1) > data)]
            local_max = data[(data.shift(1) < data) & (data.shift(-1) < data)]
            
            # Group similar levels
            support_levels = []
            resistance_levels = []
            
            tolerance = data.std() * 0.01  # 1% of standard deviation
            
            # Process local minima for support
            for level in local_min:
                similar_levels = local_min[abs(local_min - level) <= tolerance]
                if len(similar_levels) >= min_touches:
                    support_levels.append(level)
            
            # Process local maxima for resistance
            for level in local_max:
                similar_levels = local_max[abs(local_max - level) <= tolerance]
                if len(similar_levels) >= min_touches:
                    resistance_levels.append(level)
            
            # Remove duplicates and sort
            support_levels = sorted(list(set(support_levels)))
            resistance_levels = sorted(list(set(resistance_levels)))
            
            return support_levels, resistance_levels
        except Exception as e:
            logger.error(f"Error identifying support/resistance levels: {e}")
            return [], []
