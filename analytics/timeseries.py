"""Time series analysis and calculations."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import logging

logger = logging.getLogger(__name__)


class TimeSeriesAnalyzer:
    """Time series analysis for financial data."""
    
    @staticmethod
    def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
        """Calculate returns from price series."""
        if method == 'simple':
            return prices.pct_change().dropna()
        elif method == 'log':
            return np.log(prices / prices.shift(1)).dropna()
        else:
            raise ValueError("Method must be 'simple' or 'log'")
    
    @staticmethod
    def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns."""
        return (1 + returns).cumprod() - 1
    
    @staticmethod
    def calculate_rolling_stats(data: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate rolling mean and standard deviation."""
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        return rolling_mean, rolling_std
    
    @staticmethod
    def calculate_realized_volatility(returns: pd.Series, annualize: bool = True) -> float:
        """Calculate realized volatility."""
        vol = returns.std()
        if annualize:
            # Assume 252 trading days per year
            vol *= np.sqrt(252)
        return vol
    
    @staticmethod
    def calculate_ewma_volatility(returns: pd.Series, lambda_param: float = 0.94, 
                                annualize: bool = True) -> float:
        """Calculate EWMA (Exponentially Weighted Moving Average) volatility."""
        ewma_var = returns.ewm(alpha=1-lambda_param).var().iloc[-1]
        vol = np.sqrt(ewma_var)
        if annualize:
            vol *= np.sqrt(252)
        return vol
    
    @staticmethod
    def calculate_garman_klass_volatility(ohlc_data: pd.DataFrame, 
                                        annualize: bool = True) -> float:
        """Calculate Garman-Klass volatility estimator."""
        try:
            high = ohlc_data['high']
            low = ohlc_data['low']
            close = ohlc_data['close']
            open_price = ohlc_data['open']
            
            # Garman-Klass formula
            gk = 0.5 * (np.log(high / low)) ** 2 - (2 * np.log(2) - 1) * (np.log(close / open_price)) ** 2
            vol = np.sqrt(gk.mean())
            
            if annualize:
                vol *= np.sqrt(252)
            
            return vol
        except Exception as e:
            logger.error(f"Error calculating Garman-Klass volatility: {e}")
            return np.nan
    
    @staticmethod
    def calculate_parkinson_volatility(ohlc_data: pd.DataFrame, 
                                     annualize: bool = True) -> float:
        """Calculate Parkinson volatility estimator."""
        try:
            high = ohlc_data['high']
            low = ohlc_data['low']
            
            # Parkinson formula
            park = (1 / (4 * np.log(2))) * (np.log(high / low)) ** 2
            vol = np.sqrt(park.mean())
            
            if annualize:
                vol *= np.sqrt(252)
            
            return vol
        except Exception as e:
            logger.error(f"Error calculating Parkinson volatility: {e}")
            return np.nan
    
    @staticmethod
    def calculate_rogers_satchell_volatility(ohlc_data: pd.DataFrame, 
                                           annualize: bool = True) -> float:
        """Calculate Rogers-Satchell volatility estimator."""
        try:
            high = ohlc_data['high']
            low = ohlc_data['low']
            close = ohlc_data['close']
            open_price = ohlc_data['open']
            
            # Rogers-Satchell formula
            rs = (np.log(high / close) * np.log(high / open_price) + 
                  np.log(low / close) * np.log(low / open_price))
            vol = np.sqrt(rs.mean())
            
            if annualize:
                vol *= np.sqrt(252)
            
            return vol
        except Exception as e:
            logger.error(f"Error calculating Rogers-Satchell volatility: {e}")
            return np.nan
    
    @staticmethod
    def calculate_autocorrelation(data: pd.Series, lag: int = 1) -> float:
        """Calculate autocorrelation at specified lag."""
        return data.autocorr(lag=lag)
    
    @staticmethod
    def adf_test(data: pd.Series) -> Tuple[float, float, bool]:
        """Perform Augmented Dickey-Fuller test for stationarity."""
        try:
            result = adfuller(data.dropna())
            adf_statistic = result[0]
            p_value = result[1]
            is_stationary = p_value < 0.05
            return adf_statistic, p_value, is_stationary
        except Exception as e:
            logger.error(f"Error in ADF test: {e}")
            return np.nan, np.nan, False
    
    @staticmethod
    def calculate_beta_alpha(returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float]:
        """Calculate beta and alpha using CAPM model."""
        try:
            # Remove NaN values and align series
            aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
            if len(aligned_data) < 10:  # Need minimum data points
                return np.nan, np.nan
            
            asset_returns = aligned_data.iloc[:, 0]
            market_returns = aligned_data.iloc[:, 1]
            
            # Calculate beta using covariance
            covariance = np.cov(asset_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance
            
            # Calculate alpha
            asset_mean = asset_returns.mean()
            market_mean = market_returns.mean()
            alpha = asset_mean - beta * market_mean
            
            # Annualize alpha
            alpha *= 252
            
            return beta, alpha
        except Exception as e:
            logger.error(f"Error calculating beta/alpha: {e}")
            return np.nan, np.nan
    
    @staticmethod
    def calculate_correlation_matrix(returns_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for multiple assets."""
        return returns_data.corr()
    
    @staticmethod
    def rolling_correlation(series1: pd.Series, series2: pd.Series, 
                          window: int = 60) -> pd.Series:
        """Calculate rolling correlation between two series."""
        return series1.rolling(window=window).corr(series2)
