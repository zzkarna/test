"""yfinance options data ingestion."""

import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
import time
import uuid
import asyncio

from common.config import config
from common.db import db_manager
from common.io import parquet_manager

logger = logging.getLogger(__name__)


class OptionsIngestor:
    """Options data ingestor using yfinance."""
    
    def __init__(self):
        self.symbols = config.options.equities
        self.expiries_to_pull = config.options.expiries_to_pull
        self.ingest_run_id = str(uuid.uuid4())
        
        # Rate limiting
        self.request_delay = 0.2  # 200ms between requests for options
        self.last_request_time = 0
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def fetch_options_chains(self) -> None:
        """Fetch options chains for all configured symbols."""
        logger.info(f"Fetching options chains for {len(self.symbols)} symbols")
        
        for symbol in self.symbols:
            try:
                self._fetch_symbol_options(symbol)
            except Exception as e:
                logger.error(f"Error fetching options for {symbol}: {e}")
    
    def _fetch_symbol_options(self, symbol: str) -> None:
        """Fetch options chain for a single symbol."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expiry_dates = ticker.options
            
            if not expiry_dates:
                logger.warning(f"No options expiry dates found for {symbol}")
                return
            
            # Limit to nearest N expiries
            expiry_dates = expiry_dates[:self.expiries_to_pull]
            
            logger.info(f"Fetching options for {symbol} with {len(expiry_dates)} expiries")
            
            all_options_data = []
            
            for expiry in expiry_dates:
                try:
                    self._rate_limit()
                    
                    # Get options chain for this expiry
                    options_chain = ticker.option_chain(expiry)
                    
                    # Process calls
                    if not options_chain.calls.empty:
                        calls_df = self._process_options_data(
                            options_chain.calls, symbol, expiry, 'call'
                        )
                        all_options_data.append(calls_df)
                    
                    # Process puts
                    if not options_chain.puts.empty:
                        puts_df = self._process_options_data(
                            options_chain.puts, symbol, expiry, 'put'
                        )
                        all_options_data.append(puts_df)
                    
                    logger.info(f"Fetched options for {symbol} expiry {expiry}")
                    
                except Exception as e:
                    logger.error(f"Error fetching options for {symbol} expiry {expiry}: {e}")
                    continue
            
            if all_options_data:
                # Combine all options data
                combined_df = pd.concat(all_options_data, ignore_index=True)
                
                # Write to database and Parquet
                db_manager.insert_dataframe(combined_df, 'options_chains')
                parquet_manager.write_options(combined_df, symbol, 'yfinance')
                
                logger.info(f"Stored {len(combined_df)} options records for {symbol}")
            
        except Exception as e:
            logger.error(f"Error in _fetch_symbol_options for {symbol}: {e}")
    
    def _process_options_data(self, options_df: pd.DataFrame, symbol: str, 
                            expiry: str, option_type: str) -> pd.DataFrame:
        """Process raw options data into standardized format."""
        
        # Create standardized DataFrame
        processed_df = pd.DataFrame()
        
        processed_df['underlying'] = symbol
        processed_df['expiry'] = pd.to_datetime(expiry)
        processed_df['strike'] = options_df['strike']
        processed_df['option_type'] = option_type
        processed_df['last'] = options_df['lastPrice']
        processed_df['bid'] = options_df['bid']
        processed_df['ask'] = options_df['ask']
        
        # Calculate mid price
        processed_df['mid'] = (processed_df['bid'] + processed_df['ask']) / 2
        
        # Implied volatility (if available)
        processed_df['implied_vol'] = options_df.get('impliedVolatility', None)
        
        # Open interest and volume
        processed_df['open_interest'] = options_df.get('openInterest', 0)
        processed_df['volume'] = options_df.get('volume', 0)
        
        # Metadata
        processed_df['ts'] = datetime.now(timezone.utc)
        processed_df['source'] = 'yfinance'
        processed_df['fetched_at'] = datetime.now(timezone.utc)
        processed_df['ingest_run_id'] = self.ingest_run_id
        
        # Clean up data
        processed_df = processed_df.dropna(subset=['strike', 'last'])
        processed_df = processed_df[processed_df['strike'] > 0]
        processed_df = processed_df[processed_df['last'] > 0]
        
        return processed_df
    
    def get_options_summary(self, symbol: str) -> Dict:
        """Get summary statistics for options data."""
        try:
            # Query recent options data
            sql = """
                SELECT 
                    expiry,
                    option_type,
                    COUNT(*) as contract_count,
                    AVG(implied_vol) as avg_iv,
                    SUM(volume) as total_volume,
                    SUM(open_interest) as total_oi
                FROM options_chains 
                WHERE underlying = ? 
                    AND ts >= CURRENT_DATE - INTERVAL 1 DAY
                GROUP BY expiry, option_type
                ORDER BY expiry, option_type
            """
            
            df = db_manager.query(sql.replace('?', f"'{symbol}'"))
            
            summary = {
                'symbol': symbol,
                'expiries': df['expiry'].unique().tolist(),
                'total_contracts': df['contract_count'].sum(),
                'avg_implied_vol': df['avg_iv'].mean(),
                'total_volume': df['total_volume'].sum(),
                'total_open_interest': df['total_oi'].sum()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting options summary for {symbol}: {e}")
            return {}


async def main():
    """Main function to run options data ingestion."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    ingestor = OptionsIngestor()
    
    try:
        ingestor.fetch_options_chains()
        
        # Print summary for each symbol
        for symbol in config.options.equities:
            summary = ingestor.get_options_summary(symbol)
            if summary:
                logger.info(f"Options summary for {symbol}: {summary}")
        
        logger.info("Options data ingestion completed")
        
    except Exception as e:
        logger.error(f"Error in options ingestion: {e}")


if __name__ == "__main__":
    asyncio.run(main())
