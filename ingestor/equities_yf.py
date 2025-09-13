"""yfinance data ingestion for equity markets."""

import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
import time
import uuid

from common.config import config
from common.db import db_manager
from common.io import parquet_manager

logger = logging.getLogger(__name__)


class YFinanceIngestor:
    """yfinance data ingestor for equity markets."""
    
    def __init__(self):
        self.symbols = config.symbols.equities
        self.ingest_run_id = str(uuid.uuid4())
        
        # Rate limiting
        self.request_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def fetch_daily_ohlcv(self, lookback_days: int = 252) -> None:
        """Fetch daily OHLCV data for all equity symbols."""
        logger.info(f"Fetching daily OHLCV for {len(self.symbols)} symbols")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        for symbol in self.symbols:
            try:
                self._rate_limit()
                
                ticker = yf.Ticker(symbol)
                hist = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d'
                )
                
                if hist.empty:
                    logger.warning(f"No daily data found for {symbol}")
                    continue
                
                # Prepare DataFrame
                df = hist.reset_index()
                df['symbol'] = symbol
                df['ts'] = pd.to_datetime(df['Date'])
                df['open'] = df['Open']
                df['high'] = df['High']
                df['low'] = df['Low']
                df['close'] = df['Close']
                df['volume'] = df['Volume']
                df['source'] = 'yfinance'
                df['fetched_at'] = datetime.now(timezone.utc)
                df['ingest_run_id'] = self.ingest_run_id
                
                # Select required columns
                ohlcv_df = df[['symbol', 'ts', 'open', 'high', 'low', 'close', 
                              'volume', 'source', 'fetched_at', 'ingest_run_id']]
                
                # Write to database and Parquet
                db_manager.insert_dataframe(ohlcv_df, 'ohlcv')
                parquet_manager.write_ohlcv(ohlcv_df, symbol, 'yfinance')
                
                logger.info(f"Fetched {len(ohlcv_df)} daily records for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching daily data for {symbol}: {e}")
    
    def fetch_intraday_ohlcv(self, period: str = "1d", interval: str = "1m") -> None:
        """Fetch intraday OHLCV data for all equity symbols."""
        logger.info(f"Fetching {interval} OHLCV for {len(self.symbols)} symbols")
        
        for symbol in self.symbols:
            try:
                self._rate_limit()
                
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=interval)
                
                if hist.empty:
                    logger.warning(f"No intraday data found for {symbol}")
                    continue
                
                # Prepare DataFrame
                df = hist.reset_index()
                df['symbol'] = symbol
                df['ts'] = pd.to_datetime(df['Datetime'])
                df['open'] = df['Open']
                df['high'] = df['High']
                df['low'] = df['Low']
                df['close'] = df['Close']
                df['volume'] = df['Volume']
                df['source'] = f'yfinance_{interval}'
                df['fetched_at'] = datetime.now(timezone.utc)
                df['ingest_run_id'] = self.ingest_run_id
                
                # Select required columns
                ohlcv_df = df[['symbol', 'ts', 'open', 'high', 'low', 'close', 
                              'volume', 'source', 'fetched_at', 'ingest_run_id']]
                
                # Write to database and Parquet
                db_manager.insert_dataframe(ohlcv_df, 'ohlcv')
                parquet_manager.write_ohlcv(ohlcv_df, symbol, f'yfinance_{interval}')
                
                logger.info(f"Fetched {len(ohlcv_df)} {interval} records for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching {interval} data for {symbol}: {e}")
    
    def fetch_fundamentals(self) -> None:
        """Fetch fundamental data for all equity symbols."""
        logger.info(f"Fetching fundamentals for {len(self.symbols)} symbols")
        
        for symbol in self.symbols:
            try:
                self._rate_limit()
                
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if not info:
                    logger.warning(f"No fundamental data found for {symbol}")
                    continue
                
                # Extract key fundamental metrics
                fundamental_data = {
                    'symbol': symbol,
                    'ts': datetime.now(timezone.utc),
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'beta': info.get('beta'),
                    'sector': info.get('sector'),
                    'source': 'yfinance',
                    'fetched_at': datetime.now(timezone.utc),
                    'ingest_run_id': self.ingest_run_id
                }
                
                df = pd.DataFrame([fundamental_data])
                
                # Write to database
                db_manager.insert_dataframe(df, 'fundamentals')
                
                logger.info(f"Fetched fundamentals for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching fundamentals for {symbol}: {e}")
    
    def fetch_corporate_actions(self, lookback_days: int = 365) -> None:
        """Fetch corporate actions (dividends, splits) for all equity symbols."""
        logger.info(f"Fetching corporate actions for {len(self.symbols)} symbols")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        for symbol in self.symbols:
            try:
                self._rate_limit()
                
                ticker = yf.Ticker(symbol)
                
                # Fetch dividends
                dividends = ticker.dividends
                if not dividends.empty:
                    div_df = dividends.reset_index()
                    div_df['symbol'] = symbol
                    div_df['ts'] = pd.to_datetime(div_df['Date'])
                    div_df['dividend'] = div_df['Dividends']
                    div_df['split_ratio'] = None
                    div_df['source'] = 'yfinance'
                    div_df['fetched_at'] = datetime.now(timezone.utc)
                    div_df['ingest_run_id'] = self.ingest_run_id
                    
                    # Filter by date range
                    div_df = div_df[div_df['ts'] >= start_date]
                    
                    if not div_df.empty:
                        corp_df = div_df[['symbol', 'ts', 'dividend', 'split_ratio', 
                                        'source', 'fetched_at', 'ingest_run_id']]
                        db_manager.insert_dataframe(corp_df, 'corporate_actions')
                        logger.info(f"Fetched {len(corp_df)} dividends for {symbol}")
                
                # Fetch stock splits
                splits = ticker.splits
                if not splits.empty:
                    split_df = splits.reset_index()
                    split_df['symbol'] = symbol
                    split_df['ts'] = pd.to_datetime(split_df['Date'])
                    split_df['dividend'] = None
                    split_df['split_ratio'] = split_df['Stock Splits']
                    split_df['source'] = 'yfinance'
                    split_df['fetched_at'] = datetime.now(timezone.utc)
                    split_df['ingest_run_id'] = self.ingest_run_id
                    
                    # Filter by date range
                    split_df = split_df[split_df['ts'] >= start_date]
                    
                    if not split_df.empty:
                        corp_df = split_df[['symbol', 'ts', 'dividend', 'split_ratio', 
                                          'source', 'fetched_at', 'ingest_run_id']]
                        db_manager.insert_dataframe(corp_df, 'corporate_actions')
                        logger.info(f"Fetched {len(corp_df)} splits for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching corporate actions for {symbol}: {e}")


def main():
    """Main function to run yfinance data ingestion."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    ingestor = YFinanceIngestor()
    
    try:
        # Fetch different types of data
        ingestor.fetch_daily_ohlcv(lookback_days=252)
        ingestor.fetch_intraday_ohlcv(period="1d", interval="1m")
        ingestor.fetch_fundamentals()
        ingestor.fetch_corporate_actions(lookback_days=365)
        
        logger.info("yfinance data ingestion completed")
        
    except Exception as e:
        logger.error(f"Error in yfinance ingestion: {e}")


if __name__ == "__main__":
    main()
