"""FastAPI application for analytics service."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
import asyncio
from contextlib import asynccontextmanager

from common.config import config
from common.db import db_manager
from models import (
    TimeSeriesMetrics, RiskMetrics, TechnicalIndicators, 
    OptionsGreeks, PortfolioMetrics, TradingSignal, HealthStatus
)
from timeseries import TimeSeriesAnalyzer
from risk import RiskAnalyzer
from tech import TechnicalAnalyzer
from options import OptionsAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for caching
signal_cache: List[TradingSignal] = []
last_signal_check = datetime.now()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting analytics service")
    
    # Start background tasks
    asyncio.create_task(generate_signals_periodically())
    
    yield
    
    logger.info("Shutting down analytics service")


app = FastAPI(
    title="Quant Dashboard Analytics API",
    description="Analytics engine for quantitative finance calculations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        db_connected = True
        try:
            db_manager.query("SELECT 1")
        except Exception:
            db_connected = False
        
        # Check data freshness
        data_freshness = {}
        try:
            # Check latest OHLCV data
            latest_ohlcv = db_manager.query("""
                SELECT source, MAX(ts) as latest_ts 
                FROM ohlcv 
                GROUP BY source
            """)
            
            for _, row in latest_ohlcv.iterrows():
                data_freshness[f"ohlcv_{row['source']}"] = row['latest_ts']
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
        
        # Get active symbols
        active_symbols = []
        try:
            symbols_df = db_manager.query("""
                SELECT DISTINCT symbol 
                FROM ohlcv 
                WHERE ts >= CURRENT_DATE - INTERVAL 7 DAY
            """)
            active_symbols = symbols_df['symbol'].tolist()
        except Exception as e:
            logger.error(f"Error getting active symbols: {e}")
        
        return HealthStatus(
            status="healthy" if db_connected else "unhealthy",
            timestamp=datetime.now(),
            database_connected=db_connected,
            data_freshness=data_freshness,
            active_symbols=active_symbols
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/{symbol}", response_model=TimeSeriesMetrics)
async def get_time_series_metrics(
    symbol: str,
    lookback_days: int = Query(default=252, ge=30, le=1000),
    source: str = Query(default="yfinance")
):
    """Get time series metrics for a symbol."""
    try:
        # Fetch OHLCV data
        sql = """
            SELECT ts, close 
            FROM ohlcv 
            WHERE symbol = ? AND source = ?
                AND ts >= CURRENT_DATE - INTERVAL ? DAY
            ORDER BY ts
        """
        
        df = db_manager.query(sql.replace('?', f"'{symbol}', '{source}', {lookback_days}"))
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        df['ts'] = pd.to_datetime(df['ts'])
        df.set_index('ts', inplace=True)
        
        prices = df['close']
        
        # Calculate metrics
        returns_simple = TimeSeriesAnalyzer.calculate_returns(prices, 'simple')
        returns_log = TimeSeriesAnalyzer.calculate_returns(prices, 'log')
        cumulative_returns = TimeSeriesAnalyzer.calculate_cumulative_returns(returns_simple)
        
        rolling_mean, rolling_std = TimeSeriesAnalyzer.calculate_rolling_stats(returns_simple, 30)
        
        realized_vol = TimeSeriesAnalyzer.calculate_realized_volatility(returns_simple)
        ewma_vol = TimeSeriesAnalyzer.calculate_ewma_volatility(returns_simple)
        autocorr = TimeSeriesAnalyzer.calculate_autocorrelation(returns_simple, 1)
        
        return TimeSeriesMetrics(
            symbol=symbol,
            returns_simple=returns_simple.fillna(0).tolist(),
            returns_log=returns_log.fillna(0).tolist(),
            cumulative_returns=cumulative_returns.fillna(0).tolist(),
            rolling_mean=rolling_mean.fillna(0).tolist(),
            rolling_std=rolling_std.fillna(0).tolist(),
            realized_volatility=realized_vol,
            ewma_volatility=ewma_vol,
            autocorrelation_lag1=autocorr,
            timestamps=df.index.tolist()
        )
    except Exception as e:
        logger.error(f"Error getting time series metrics for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk/{symbol}", response_model=RiskMetrics)
async def get_risk_metrics(
    symbol: str,
    lookback_days: int = Query(default=252, ge=30, le=1000),
    source: str = Query(default="yfinance"),
    benchmark: str = Query(default="SPY")
):
    """Get risk metrics for a symbol."""
    try:
        # Fetch symbol data
        sql = """
            SELECT ts, close 
            FROM ohlcv 
            WHERE symbol = ? AND source = ?
                AND ts >= CURRENT_DATE - INTERVAL ? DAY
            ORDER BY ts
        """
        
        df = db_manager.query(sql.replace('?', f"'{symbol}', '{source}', {lookback_days}"))
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        df['ts'] = pd.to_datetime(df['ts'])
        df.set_index('ts', inplace=True)
        
        returns = TimeSeriesAnalyzer.calculate_returns(df['close'], 'simple')
        
        # Fetch benchmark data if different from symbol
        benchmark_returns = None
        beta = None
        alpha = None
        
        if benchmark != symbol:
            bench_sql = sql.replace('?', f"'{benchmark}', '{source}', {lookback_days}")
            bench_df = db_manager.query(bench_sql)
            
            if not bench_df.empty:
                bench_df['ts'] = pd.to_datetime(bench_df['ts'])
                bench_df.set_index('ts', inplace=True)
                benchmark_returns = TimeSeriesAnalyzer.calculate_returns(bench_df['close'], 'simple')
                beta, alpha = TimeSeriesAnalyzer.calculate_beta_alpha(returns, benchmark_returns)
        
        # Calculate risk metrics
        sharpe = RiskAnalyzer.calculate_sharpe_ratio(returns)
        sortino = RiskAnalyzer.calculate_sortino_ratio(returns)
        calmar = RiskAnalyzer.calculate_calmar_ratio(returns)
        
        info_ratio = np.nan
        if benchmark_returns is not None:
            info_ratio = RiskAnalyzer.calculate_information_ratio(returns, benchmark_returns)
        
        max_dd, dd_series = RiskAnalyzer.calculate_max_drawdown(returns)
        current_dd = dd_series.iloc[-1] if not dd_series.empty else 0
        
        var_95 = RiskAnalyzer.calculate_var(returns, 0.05)
        var_99 = RiskAnalyzer.calculate_var(returns, 0.01)
        es_95 = RiskAnalyzer.calculate_expected_shortfall(returns, 0.05)
        es_99 = RiskAnalyzer.calculate_expected_shortfall(returns, 0.01)
        
        return RiskMetrics(
            symbol=symbol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            beta=beta,
            alpha=alpha
        )
    except Exception as e:
        logger.error(f"Error getting risk metrics for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/technical/{symbol}", response_model=TechnicalIndicators)
async def get_technical_indicators(
    symbol: str,
    lookback_days: int = Query(default=252, ge=30, le=1000),
    source: str = Query(default="yfinance")
):
    """Get technical indicators for a symbol."""
    try:
        # Fetch OHLCV data
        sql = """
            SELECT ts, open, high, low, close, volume
            FROM ohlcv 
            WHERE symbol = ? AND source = ?
                AND ts >= CURRENT_DATE - INTERVAL ? DAY
            ORDER BY ts
        """
        
        df = db_manager.query(sql.replace('?', f"'{symbol}', '{source}', {lookback_days}"))
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        df['ts'] = pd.to_datetime(df['ts'])
        df.set_index('ts', inplace=True)
        
        # Calculate technical indicators
        sma_20 = TechnicalAnalyzer.sma(df['close'], 20)
        sma_50 = TechnicalAnalyzer.sma(df['close'], 50)
        ema_12 = TechnicalAnalyzer.ema(df['close'], 12)
        ema_26 = TechnicalAnalyzer.ema(df['close'], 26)
        
        macd, macd_signal, _ = TechnicalAnalyzer.macd(df['close'])
        rsi = TechnicalAnalyzer.rsi(df['close'])
        atr = TechnicalAnalyzer.atr(df['high'], df['low'], df['close'])
        
        bb_upper, bb_middle, bb_lower = TechnicalAnalyzer.bollinger_bands(df['close'])
        
        return TechnicalIndicators(
            symbol=symbol,
            sma_20=sma_20.fillna(0).tolist(),
            sma_50=sma_50.fillna(0).tolist(),
            ema_12=ema_12.fillna(0).tolist(),
            ema_26=ema_26.fillna(0).tolist(),
            macd=macd.fillna(0).tolist(),
            macd_signal=macd_signal.fillna(0).tolist(),
            rsi=rsi.fillna(50).tolist(),
            atr=atr.fillna(0).tolist(),
            bb_upper=bb_upper.fillna(0).tolist(),
            bb_middle=bb_middle.fillna(0).tolist(),
            bb_lower=bb_lower.fillna(0).tolist(),
            timestamps=df.index.tolist()
        )
    except Exception as e:
        logger.error(f"Error getting technical indicators for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/options/{underlying}")
async def get_options_analytics(
    underlying: str,
    risk_free_rate: float = Query(default=0.05, ge=0, le=0.2)
):
    """Get options analytics for an underlying."""
    try:
        # Get latest spot price
        spot_sql = """
            SELECT close as spot_price
            FROM ohlcv 
            WHERE symbol = ? AND source = 'yfinance'
            ORDER BY ts DESC 
            LIMIT 1
        """
        
        spot_df = db_manager.query(spot_sql.replace('?', f"'{underlying}'"))
        
        if spot_df.empty:
            raise HTTPException(status_code=404, detail=f"No spot price found for {underlying}")
        
        spot_price = float(spot_df.iloc[0]['spot_price'])
        
        # Get options data
        options_sql = """
            SELECT *
            FROM options_chains 
            WHERE underlying = ?
                AND ts >= CURRENT_DATE - INTERVAL 1 DAY
            ORDER BY expiry, strike
        """
        
        options_df = db_manager.query(options_sql.replace('?', f"'{underlying}'"))
        
        if options_df.empty:
            raise HTTPException(status_code=404, detail=f"No options data found for {underlying}")
        
        # Calculate IV surface
        iv_surface = OptionsAnalyzer.calculate_iv_surface(options_df, spot_price, risk_free_rate)
        
        # Calculate Greeks for each option
        greeks_data = []
        
        for _, row in options_df.iterrows():
            expiry_date = pd.to_datetime(row['expiry'])
            current_date = pd.to_datetime(row['ts'])
            T = (expiry_date - current_date).days / 365.0
            
            if T > 0 and row['mid'] > 0:
                # Use implied volatility if available, otherwise estimate
                if pd.notna(row['implied_vol']) and row['implied_vol'] > 0:
                    sigma = row['implied_vol']
                else:
                    sigma = OptionsAnalyzer.implied_volatility(
                        row['mid'], spot_price, row['strike'], T, 
                        risk_free_rate, row['option_type']
                    )
                
                if not np.isnan(sigma) and sigma > 0:
                    greeks = OptionsAnalyzer.calculate_all_greeks(
                        spot_price, row['strike'], T, risk_free_rate, sigma, row['option_type']
                    )
                    
                    theoretical_price = OptionsAnalyzer.black_scholes_price(
                        spot_price, row['strike'], T, risk_free_rate, sigma, row['option_type']
                    )
                    
                    greeks_data.append({
                        'underlying': underlying,
                        'expiry': expiry_date,
                        'strike': row['strike'],
                        'option_type': row['option_type'],
                        'spot_price': spot_price,
                        'risk_free_rate': risk_free_rate,
                        'implied_vol': sigma,
                        'theoretical_price': theoretical_price,
                        'market_price': row['mid'],
                        **greeks
                    })
        
        # Calculate skew metrics
        skew_metrics = OptionsAnalyzer.calculate_skew_metrics(iv_surface)
        
        return {
            'underlying': underlying,
            'spot_price': spot_price,
            'greeks': greeks_data,
            'iv_surface': iv_surface.to_dict('records') if not iv_surface.empty else [],
            'skew_metrics': skew_metrics,
            'total_contracts': len(options_df)
        }
    except Exception as e:
        logger.error(f"Error getting options analytics for {underlying}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals", response_model=List[TradingSignal])
async def get_trading_signals(
    limit: int = Query(default=50, ge=1, le=200),
    signal_type: Optional[str] = Query(default=None)
):
    """Get recent trading signals."""
    try:
        signals = signal_cache.copy()
        
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
        
        # Sort by timestamp (newest first) and limit
        signals.sort(key=lambda x: x.timestamp, reverse=True)
        
        return signals[:limit]
    except Exception as e:
        logger.error(f"Error getting trading signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_signals_periodically():
    """Background task to generate trading signals."""
    global signal_cache, last_signal_check
    
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            
            current_time = datetime.now()
            
            # Generate signals for all configured symbols
            for symbol in config.symbols.equities + config.symbols.crypto:
                await generate_signals_for_symbol(symbol)
            
            last_signal_check = current_time
            
        except Exception as e:
            logger.error(f"Error in signal generation: {e}")


async def generate_signals_for_symbol(symbol: str):
    """Generate signals for a specific symbol."""
    try:
        # Determine source based on symbol type
        source = 'binance' if symbol in config.symbols.crypto else 'yfinance'
        
        # Get recent price data
        sql = """
            SELECT ts, close 
            FROM ohlcv 
            WHERE symbol = ? AND source = ?
                AND ts >= CURRENT_DATE - INTERVAL 100 DAY
            ORDER BY ts
        """
        
        df = db_manager.query(sql.replace('?', f"'{symbol}', '{source}'"))
        
        if df.empty or len(df) < 50:
            return
        
        df['ts'] = pd.to_datetime(df['ts'])
        df.set_index('ts', inplace=True)
        
        prices = df['close']
        
        # Check for SMA crossovers
        sma_20 = TechnicalAnalyzer.sma(prices, 20)
        sma_50 = TechnicalAnalyzer.sma(prices, 50)
        
        crossovers = TechnicalAnalyzer.detect_crossovers(sma_20, sma_50)
        
        # Check for recent crossovers (last 2 periods)
        recent_crossovers = crossovers.tail(2)
        
        for timestamp, signal_value in recent_crossovers.items():
            if signal_value != 0:
                signal_id = f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}_crossover"
                
                # Check if signal already exists
                if not any(s.signal_id == signal_id for s in signal_cache):
                    signal_type = "bullish" if signal_value > 0 else "bearish"
                    
                    signal = TradingSignal(
                        signal_id=signal_id,
                        symbol=symbol,
                        signal_type="crossover",
                        timestamp=timestamp,
                        message=f"SMA crossover: {signal_type} signal for {symbol}",
                        strength=0.7,
                        metadata={
                            'sma_20': float(sma_20.loc[timestamp]),
                            'sma_50': float(sma_50.loc[timestamp]),
                            'price': float(prices.loc[timestamp]),
                            'direction': signal_type
                        }
                    )
                    
                    signal_cache.append(signal)
        
        # Check for volatility regime changes
        returns = TimeSeriesAnalyzer.calculate_returns(prices, 'simple')
        current_vol = TimeSeriesAnalyzer.calculate_realized_volatility(returns.tail(30), annualize=True)
        historical_vol = TimeSeriesAnalyzer.calculate_realized_volatility(returns.tail(60), annualize=True)
        
        vol_percentile = np.percentile(returns.rolling(60).std().dropna() * np.sqrt(252), 75)
        
        if current_vol > vol_percentile and current_vol > historical_vol * 1.5:
            signal_id = f"{symbol}_{datetime.now().strftime('%Y%m%d')}_vol_regime"
            
            if not any(s.signal_id == signal_id for s in signal_cache):
                signal = TradingSignal(
                    signal_id=signal_id,
                    symbol=symbol,
                    signal_type="volatility_regime",
                    timestamp=datetime.now(),
                    message=f"High volatility regime detected for {symbol}",
                    strength=0.8,
                    metadata={
                        'current_vol': current_vol,
                        'historical_vol': historical_vol,
                        'vol_percentile': vol_percentile
                    }
                )
                
                signal_cache.append(signal)
        
        # Keep only recent signals (last 7 days)
        cutoff_date = datetime.now() - timedelta(days=7)
        signal_cache[:] = [s for s in signal_cache if s.timestamp >= cutoff_date]
        
    except Exception as e:
        logger.error(f"Error generating signals for {symbol}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.analytics.port)
