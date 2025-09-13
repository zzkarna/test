import asyncio
import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from common.config import Config
from scheduler.jobs import SchedulerJobs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuantScheduler:
    """Main scheduler service for the quant dashboard."""
    
    def __init__(self):
        self.config = Config()
        self.jobs = SchedulerJobs(self.config)
        self.scheduler = AsyncIOScheduler()
        
    def setup_jobs(self):
        """Setup all scheduled jobs."""
        
        # Data cleanup - daily at 2 AM
        self.scheduler.add_job(
            self.jobs.cleanup_old_data,
            CronTrigger(hour=2, minute=0),
            id='cleanup_old_data',
            name='Clean up old data',
            replace_existing=True
        )
        
        # Signal generation - every 5 minutes
        self.scheduler.add_job(
            self.jobs.generate_signals,
            IntervalTrigger(minutes=5),
            id='generate_signals',
            name='Generate trading signals',
            replace_existing=True
        )
        
        # Risk metrics update - every 15 minutes
        self.scheduler.add_job(
            self.jobs.update_risk_metrics,
            IntervalTrigger(minutes=15),
            id='update_risk_metrics',
            name='Update risk metrics',
            replace_existing=True
        )
        
        # Health check - every minute
        self.scheduler.add_job(
            self.jobs.health_check,
            IntervalTrigger(minutes=1),
            id='health_check',
            name='System health check',
            replace_existing=True
        )
        
        logger.info("Scheduled jobs configured")
    
    async def start(self):
        """Start the scheduler service."""
        logger.info("Starting Quant Dashboard Scheduler...")
        
        self.setup_jobs()
        self.scheduler.start()
        
        logger.info("Scheduler started successfully")
        logger.info(f"Active jobs: {len(self.scheduler.get_jobs())}")
        
        # Keep the service running
        try:
            while True:
                await asyncio.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        finally:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")

async def main():
    """Main entry point."""
    scheduler = QuantScheduler()
    await scheduler.start()

if __name__ == "__main__":
    asyncio.run(main())
