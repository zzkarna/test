"""API client for communicating with analytics service."""

import requests
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
import streamlit as st
from datetime import datetime

logger = logging.getLogger(__name__)


class AnalyticsAPIClient:
    """Client for analytics API."""
    
    def __init__(self, base_url: str = "http://analytics:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request to analytics API."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            st.error(f"Failed to fetch data from analytics service: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error in API request: {e}")
            st.error(f"Unexpected error: {e}")
            return {}
    
    def get_health(self) -> Dict[str, Any]:
        """Get service health status."""
        return self._make_request("/health")
    
    def get_time_series_metrics(self, symbol: str, lookback_days: int = 252, 
                              source: str = "yfinance") -> Dict[str, Any]:
        """Get time series metrics for a symbol."""
        params = {
            "lookback_days": lookback_days,
            "source": source
        }
        return self._make_request(f"/metrics/{symbol}", params)
    
    def get_risk_metrics(self, symbol: str, lookback_days: int = 252, 
                        source: str = "yfinance", benchmark: str = "SPY") -> Dict[str, Any]:
        """Get risk metrics for a symbol."""
        params = {
            "lookback_days": lookback_days,
            "source": source,
            "benchmark": benchmark
        }
        return self._make_request(f"/risk/{symbol}", params)
    
    def get_technical_indicators(self, symbol: str, lookback_days: int = 252, 
                               source: str = "yfinance") -> Dict[str, Any]:
        """Get technical indicators for a symbol."""
        params = {
            "lookback_days": lookback_days,
            "source": source
        }
        return self._make_request(f"/technical/{symbol}", params)
    
    def get_options_analytics(self, underlying: str, risk_free_rate: float = 0.05) -> Dict[str, Any]:
        """Get options analytics for an underlying."""
        params = {"risk_free_rate": risk_free_rate}
        return self._make_request(f"/options/{underlying}", params)
    
    def get_trading_signals(self, limit: int = 50, signal_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get trading signals."""
        params = {"limit": limit}
        if signal_type:
            params["signal_type"] = signal_type
        
        result = self._make_request("/signals", params)
        return result if isinstance(result, list) else []


# Global API client instance
@st.cache_resource
def get_api_client():
    """Get cached API client instance."""
    return AnalyticsAPIClient()
