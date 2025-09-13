import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import aiohttp
from common.config import Config
from common.db import DuckDBManager

logger = logging.getLogger(__name__)

class SchedulerJobs:
    """Handles all scheduled jobs for the quant dashboard."""
    
    def __init__(self, config: Config):
        self.config = config
        self.db = DuckDBManager(config.database_url)
        self.analytics_url = config.analytics_api_url
        
    async def cleanup_old_data(self) -> Dict[str, Any]:
        """Clean up old data beyond retention period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.cleanup_days)
            
            # Clean up old trades
            trades_deleted = await self.db.execute(
                "DELETE FROM trades WHERE timestamp < ?", 
                (cutoff_date,)
            )
            
            # Clean up old klines
            klines_deleted = await self.db.execute(
                "DELETE FROM klines WHERE timestamp < ?", 
                (cutoff_date,)
            )
            
            # Clean up old signals
            signals_deleted = await self.db.execute(
                "DELETE FROM signals WHERE timestamp < ?", 
                (cutoff_date,)
            )
            
            logger.info(f"Cleanup completed: {trades_deleted} trades, {klines_deleted} klines, {signals_deleted} signals")
            
            return {
                "status": "success",
                "trades_deleted": trades_deleted,
                "klines_deleted": klines_deleted,
                "signals_deleted": signals_deleted,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def generate_signals(self) -> Dict[str, Any]:
        """Generate trading signals for all active symbols."""
        try:
            async with aiohttp.ClientSession() as session:
                # Get active symbols
                symbols_response = await session.get(f"{self.analytics_url}/symbols")
                symbols = await symbols_response.json()
                
                signals_generated = 0
                for symbol in symbols:
                    try:
                        # Generate signals for each symbol
                        signal_response = await session.post(
                            f"{self.analytics_url}/signals/generate",
                            json={"symbol": symbol, "lookback": 100}
                        )
                        
                        if signal_response.status == 200:
                            signals_generated += 1
                            
                    except Exception as e:
                        logger.warning(f"Signal generation failed for {symbol}: {e}")
                
                logger.info(f"Generated signals for {signals_generated} symbols")
                return {
                    "status": "success",
                    "signals_generated": signals_generated,
                    "total_symbols": len(symbols)
                }
                
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def update_risk_metrics(self) -> Dict[str, Any]:
        """Update risk metrics for all portfolios."""
        try:
            async with aiohttp.ClientSession() as session:
                # Trigger risk metric updates
                response = await session.post(f"{self.analytics_url}/risk/update")
                
                if response.status == 200:
                    result = await response.json()
                    logger.info("Risk metrics updated successfully")
                    return {"status": "success", "metrics_updated": result.get("count", 0)}
                else:
                    error_msg = f"Risk update failed with status {response.status}"
                    logger.error(error_msg)
                    return {"status": "error", "message": error_msg}
                    
        except Exception as e:
            logger.error(f"Risk metrics update failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health checks."""
        try:
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "database": "unknown",
                "analytics_api": "unknown",
                "data_freshness": "unknown"
            }
            
            # Check database connectivity
            try:
                await self.db.execute("SELECT 1")
                health_status["database"] = "healthy"
            except Exception as e:
                health_status["database"] = f"error: {str(e)}"
            
            # Check analytics API
            try:
                async with aiohttp.ClientSession() as session:
                    response = await session.get(f"{self.analytics_url}/health")
                    if response.status == 200:
                        health_status["analytics_api"] = "healthy"
                    else:
                        health_status["analytics_api"] = f"error: status {response.status}"
            except Exception as e:
                health_status["analytics_api"] = f"error: {str(e)}"
            
            # Check data freshness
            try:
                latest_data = await self.db.fetch_one(
                    "SELECT MAX(timestamp) as latest FROM trades"
                )
                if latest_data and latest_data[0]:
                    latest_time = datetime.fromisoformat(latest_data[0])
                    age_minutes = (datetime.now() - latest_time).total_seconds() / 60
                    
                    if age_minutes < 10:
                        health_status["data_freshness"] = "fresh"
                    elif age_minutes < 60:
                        health_status["data_freshness"] = f"stale ({age_minutes:.1f}m old)"
                    else:
                        health_status["data_freshness"] = f"very_stale ({age_minutes/60:.1f}h old)"
                else:
                    health_status["data_freshness"] = "no_data"
            except Exception as e:
                health_status["data_freshness"] = f"error: {str(e)}"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}
