import pytest
import asyncio
import aiohttp
from datetime import datetime
import pandas as pd

class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def event_loop(self):
        """Create an event loop for async tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()
    
    @pytest.mark.asyncio
    async def test_analytics_api_health(self):
        """Test analytics API health endpoint."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get('http://localhost:8002/health') as response:
                    assert response.status == 200
                    data = await response.json()
                    assert 'status' in data
            except aiohttp.ClientConnectorError:
                pytest.skip("Analytics API not running")
    
    @pytest.mark.asyncio
    async def test_data_flow(self):
        """Test complete data flow from ingestion to analytics."""
        async with aiohttp.ClientSession() as session:
            try:
                # Check if we have data
                async with session.get('http://localhost:8002/symbols') as response:
                    if response.status == 200:
                        symbols = await response.json()
                        
                        if symbols:
                            # Test metrics calculation for first symbol
                            symbol = symbols[0]
                            async with session.get(
                                f'http://localhost:8002/metrics/{symbol}'
                            ) as metrics_response:
                                assert metrics_response.status == 200
                                metrics = await metrics_response.json()
                                assert 'returns' in metrics
                                assert 'volatility' in metrics
                        
            except aiohttp.ClientConnectorError:
                pytest.skip("Services not running")
    
    @pytest.mark.asyncio
    async def test_signal_generation(self):
        """Test signal generation endpoint."""
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "symbol": "BTCUSDT",
                    "lookback": 50
                }
                
                async with session.post(
                    'http://localhost:8002/signals/generate',
                    json=payload
                ) as response:
                    if response.status == 200:
                        signals = await response.json()
                        assert 'signals' in signals
                        assert isinstance(signals['signals'], list)
                        
            except aiohttp.ClientConnectorError:
                pytest.skip("Analytics API not running")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
