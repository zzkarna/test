import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from analytics.timeseries import TimeSeriesAnalytics
from analytics.risk import RiskAnalytics
from analytics.tech import TechnicalAnalytics
from analytics.options import OptionsAnalytics

class TestTimeSeriesAnalytics:
    """Test time series analytics functions."""
    
    @pytest.fixture
    def sample_prices(self):
        """Generate sample price data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 252)))
        return pd.Series(prices, index=dates)
    
    def test_simple_returns(self, sample_prices):
        """Test simple returns calculation."""
        analytics = TimeSeriesAnalytics()
        returns = analytics.simple_returns(sample_prices)
        
        assert len(returns) == len(sample_prices) - 1
        assert not returns.isna().any()
        assert abs(returns.mean()) < 0.1  # Reasonable mean return
    
    def test_log_returns(self, sample_prices):
        """Test log returns calculation."""
        analytics = TimeSeriesAnalytics()
        returns = analytics.log_returns(sample_prices)
        
        assert len(returns) == len(sample_prices) - 1
        assert not returns.isna().any()
    
    def test_volatility_estimators(self, sample_prices):
        """Test various volatility estimators."""
        analytics = TimeSeriesAnalytics()
        
        # Create OHLC data
        ohlc_data = pd.DataFrame({
            'open': sample_prices * np.random.uniform(0.99, 1.01, len(sample_prices)),
            'high': sample_prices * np.random.uniform(1.00, 1.05, len(sample_prices)),
            'low': sample_prices * np.random.uniform(0.95, 1.00, len(sample_prices)),
            'close': sample_prices
        })
        
        gk_vol = analytics.garman_klass_volatility(ohlc_data)
        pk_vol = analytics.parkinson_volatility(ohlc_data)
        rs_vol = analytics.rogers_satchell_volatility(ohlc_data)
        
        assert gk_vol > 0
        assert pk_vol > 0
        assert rs_vol > 0
        assert all(vol < 2.0 for vol in [gk_vol, pk_vol, rs_vol])  # Reasonable volatility

class TestRiskAnalytics:
    """Test risk analytics functions."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data."""
        np.random.seed(42)
        return pd.Series(np.random.normal(0.001, 0.02, 252))
    
    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        analytics = RiskAnalytics()
        sharpe = analytics.sharpe_ratio(sample_returns, risk_free_rate=0.02)
        
        assert isinstance(sharpe, float)
        assert -5 < sharpe < 5  # Reasonable range
    
    def test_var_calculation(self, sample_returns):
        """Test VaR calculation."""
        analytics = RiskAnalytics()
        var_95 = analytics.value_at_risk(sample_returns, confidence=0.05)
        var_99 = analytics.value_at_risk(sample_returns, confidence=0.01)
        
        assert var_95 < 0  # VaR should be negative (loss)
        assert var_99 < var_95  # 99% VaR should be more extreme
    
    def test_max_drawdown(self, sample_returns):
        """Test maximum drawdown calculation."""
        analytics = RiskAnalytics()
        
        # Create cumulative returns
        cum_returns = (1 + sample_returns).cumprod()
        max_dd = analytics.max_drawdown(cum_returns)
        
        assert max_dd <= 0  # Drawdown should be negative or zero
        assert max_dd >= -1  # Cannot lose more than 100%

class TestTechnicalAnalytics:
    """Test technical analysis functions."""
    
    @pytest.fixture
    def sample_ohlcv(self):
        """Generate sample OHLCV data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        close = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
        
        return pd.DataFrame({
            'open': close * np.random.uniform(0.99, 1.01, 100),
            'high': close * np.random.uniform(1.00, 1.05, 100),
            'low': close * np.random.uniform(0.95, 1.00, 100),
            'close': close,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def test_moving_averages(self, sample_ohlcv):
        """Test moving average calculations."""
        analytics = TechnicalAnalytics()
        
        sma_20 = analytics.sma(sample_ohlcv['close'], 20)
        ema_20 = analytics.ema(sample_ohlcv['close'], 20)
        
        assert len(sma_20) == len(sample_ohlcv)
        assert len(ema_20) == len(sample_ohlcv)
        assert sma_20.iloc[-1] > 0
        assert ema_20.iloc[-1] > 0
    
    def test_rsi(self, sample_ohlcv):
        """Test RSI calculation."""
        analytics = TechnicalAnalytics()
        rsi = analytics.rsi(sample_ohlcv['close'])
        
        assert len(rsi) == len(sample_ohlcv)
        assert (rsi >= 0).all() and (rsi <= 100).all()
    
    def test_macd(self, sample_ohlcv):
        """Test MACD calculation."""
        analytics = TechnicalAnalytics()
        macd_line, signal_line, histogram = analytics.macd(sample_ohlcv['close'])
        
        assert len(macd_line) == len(sample_ohlcv)
        assert len(signal_line) == len(sample_ohlcv)
        assert len(histogram) == len(sample_ohlcv)

class TestOptionsAnalytics:
    """Test options analytics functions."""
    
    def test_black_scholes_call(self):
        """Test Black-Scholes call option pricing."""
        analytics = OptionsAnalytics()
        
        price = analytics.black_scholes_call(
            S=100, K=100, T=0.25, r=0.05, sigma=0.2
        )
        
        assert price > 0
        assert price < 100  # Call price should be less than stock price for ATM
    
    def test_black_scholes_put(self):
        """Test Black-Scholes put option pricing."""
        analytics = OptionsAnalytics()
        
        price = analytics.black_scholes_put(
            S=100, K=100, T=0.25, r=0.05, sigma=0.2
        )
        
        assert price > 0
        assert price < 100  # Put price should be reasonable
    
    def test_greeks_calculation(self):
        """Test Greeks calculation."""
        analytics = OptionsAnalytics()
        
        greeks = analytics.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type='call'
        )
        
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        assert 'vega' in greeks
        assert 'rho' in greeks
        
        # Delta should be around 0.5 for ATM call
        assert 0.4 < greeks['delta'] < 0.6
        
        # Gamma should be positive
        assert greeks['gamma'] > 0
        
        # Theta should be negative (time decay)
        assert greeks['theta'] < 0
    
    def test_implied_volatility(self):
        """Test implied volatility calculation."""
        analytics = OptionsAnalytics()
        
        # First calculate a theoretical price
        theoretical_price = analytics.black_scholes_call(
            S=100, K=100, T=0.25, r=0.05, sigma=0.2
        )
        
        # Then solve for implied volatility
        iv = analytics.implied_volatility(
            market_price=theoretical_price,
            S=100, K=100, T=0.25, r=0.05,
            option_type='call'
        )
        
        # Should recover the original volatility (within tolerance)
        assert abs(iv - 0.2) < 0.01

if __name__ == "__main__":
    pytest.main([__file__])
