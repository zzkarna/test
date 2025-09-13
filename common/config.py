"""Configuration management for the quant dashboard."""

import yaml
from pathlib import Path
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class SymbolsConfig(BaseModel):
    crypto: List[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    equities: List[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "TSLA", "SPY"])


class OptionsConfig(BaseModel):
    equities: List[str] = Field(default_factory=lambda: ["AAPL", "SPY"])
    expiries_to_pull: int = 3


class BinanceConfig(BaseModel):
    streams: List[str] = Field(default_factory=lambda: ["trades", "bookTicker", "kline_1m"])


class StorageConfig(BaseModel):
    base_path: str = "./data"


class DashboardConfig(BaseModel):
    port: int = 8501


class AnalyticsConfig(BaseModel):
    port: int = 8000


class RiskConfig(BaseModel):
    lookback_days: int = 252
    var_confidence: float = 0.05
    ewma_lambda: float = 0.94


class Config(BaseModel):
    symbols: SymbolsConfig = Field(default_factory=SymbolsConfig)
    options: OptionsConfig = Field(default_factory=OptionsConfig)
    binance: BinanceConfig = Field(default_factory=BinanceConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        return Config(**config_data)
    else:
        # Return default config if file doesn't exist
        return Config()


# Global config instance
config = load_config()
