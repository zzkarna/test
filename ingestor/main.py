"""Main ingestion orchestrator."""

import asyncio
import logging
import signal
import sys
from typing import List
from concurrent.futures import ThreadPoolExecutor

from crypto_binance import BinanceWebSocketClient
from equities_yf import YFinanceIngestor
from options_yf import OptionsIngestor

logger = logging.getLogger(__name__)


class IngestionOrchestrator:
    """Orchestrates all data ingestion services."""
    
    def __init__(self):
        self.binance_client = BinanceWebSocketClient()
        self.yfinance_ingestor = YFinanceIngestor()
        self.options_ingestor = OptionsIngestor()
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    async def start(self) -> None:
        """Start all ingestion services."""
        self.running = True
        logger.info("Starting ingestion orchestrator")
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        tasks = []
        
        # Start Binance WebSocket client (async)
        tasks.append(asyncio.create_task(self.binance_client.start()))
        
        # Start periodic yfinance data fetching
        tasks.append(asyncio.create_task(self._run_periodic_yfinance()))
        
        # Start periodic options data fetching
        tasks.append(asyncio.create_task(self._run_periodic_options()))
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in ingestion orchestrator: {e}")
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop all ingestion services."""
        self.running = False
        logger.info("Stopping ingestion orchestrator")
        
        # Stop Binance client
        await self.binance_client.stop()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def _run_periodic_yfinance(self) -> None:
        """Run yfinance data fetching periodically."""
        while self.running:
            try:
                logger.info("Starting yfinance data fetch")
                
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                
                # Fetch daily data (once per day)
                await loop.run_in_executor(
                    self.executor, 
                    self.yfinance_ingestor.fetch_daily_ohlcv, 
                    252
                )
                
                # Fetch intraday data (every 15 minutes)
                await loop.run_in_executor(
                    self.executor,
                    self.yfinance_ingestor.fetch_intraday_ohlcv,
                    "1d",
                    "1m"
                )
                
                # Fetch fundamentals (once per day)
                await loop.run_in_executor(
                    self.executor,
                    self.yfinance_ingestor.fetch_fundamentals
                )
                
                # Fetch corporate actions (once per day)
                await loop.run_in_executor(
                    self.executor,
                    self.yfinance_ingestor.fetch_corporate_actions,
                    365
                )
                
                logger.info("Completed yfinance data fetch")
                
                # Wait 15 minutes before next fetch
                await asyncio.sleep(900)  # 15 minutes
                
            except Exception as e:
                logger.error(f"Error in periodic yfinance fetch: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _run_periodic_options(self) -> None:
        """Run options data fetching periodically."""
        while self.running:
            try:
                logger.info("Starting options data fetch")
                
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    self.options_ingestor.fetch_options_chains
                )
                
                logger.info("Completed options data fetch")
                
                # Wait 30 minutes before next fetch
                await asyncio.sleep(1800)  # 30 minutes
                
            except Exception as e:
                logger.error(f"Error in periodic options fetch: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error


async def main():
    """Main function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    orchestrator = IngestionOrchestrator()
    
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
