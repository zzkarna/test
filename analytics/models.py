"""Pydantic models for analytics API."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class TimeSeriesMetrics(BaseModel):
    """Time series analysis metrics."""
    symbol: str
    returns_simple: List[float]
    returns_log: List[float]
    cumulative_returns: List[float]
    rolling_mean: List[float]
    rolling_std: List[float]
    realized_volatility: float
    ewma_volatility: float
    autocorrelation_lag1: float
    timestamps: List[datetime]


class RiskMetrics(BaseModel):
    """Risk analysis metrics."""
    symbol: str
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    max_drawdown: float
    current_drawdown: float
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    beta: Optional[float] = None
    alpha: Optional[float] = None


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators."""
    symbol: str
    sma_20: List[float]
    sma_50: List[float]
    ema_12: List[float]
    ema_26: List[float]
    macd: List[float]
    macd_signal: List[float]
    rsi: List[float]
    atr: List[float]
    bb_upper: List[float]
    bb_middle: List[float]
    bb_lower: List[float]
    timestamps: List[datetime]


class OptionsGreeks(BaseModel):
    """Options Greeks calculations."""
    underlying: str
    expiry: datetime
    strike: float
    option_type: str
    spot_price: float
    risk_free_rate: float
    implied_vol: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    theoretical_price: float


class PortfolioMetrics(BaseModel):
    """Portfolio analysis metrics."""
    symbols: List[str]
    weights: List[float]
    portfolio_return: float
    portfolio_volatility: float
    portfolio_sharpe: float
    correlation_matrix: List[List[float]]
    contribution_to_risk: List[float]
    kelly_fractions: List[float]


class SignalType(str, Enum):
    """Signal types."""
    CROSSOVER = "crossover"
    VOLATILITY_REGIME = "volatility_regime"
    OPTIONS_SKEW = "options_skew"


class TradingSignal(BaseModel):
    """Trading signal model."""
    signal_id: str
    symbol: str
    signal_type: SignalType
    timestamp: datetime
    message: str
    strength: float = Field(ge=0, le=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    """API health status."""
    status: str
    timestamp: datetime
    database_connected: bool
    data_freshness: Dict[str, datetime]
    active_symbols: List[str]
