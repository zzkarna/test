"""I/O utilities for Parquet file operations."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ParquetManager:
    """Manager for Parquet file operations."""
    
    def __init__(self, base_path: str = "./data/parquet"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def write_ohlcv(self, df: pd.DataFrame, symbol: str, source: str) -> None:
        """Write OHLCV data to partitioned Parquet files."""
        if df.empty:
            return
        
        # Create partition path: symbol/source/year/month
        df['year'] = pd.to_datetime(df['ts']).dt.year
        df['month'] = pd.to_datetime(df['ts']).dt.month
        
        partition_path = self.base_path / "ohlcv" / symbol / source
        partition_path.mkdir(parents=True, exist_ok=True)
        
        # Group by year/month and write separate files
        for (year, month), group_df in df.groupby(['year', 'month']):
            file_path = partition_path / f"{year}_{month:02d}.parquet"
            
            # Remove partition columns before writing
            write_df = group_df.drop(['year', 'month'], axis=1)
            
            if file_path.exists():
                # Append to existing file
                existing_df = pd.read_parquet(file_path)
                combined_df = pd.concat([existing_df, write_df]).drop_duplicates(
                    subset=['symbol', 'ts'], keep='last'
                )
                combined_df.to_parquet(file_path, index=False)
            else:
                write_df.to_parquet(file_path, index=False)
        
        logger.info(f"Wrote {len(df)} OHLCV records for {symbol} ({source})")
    
    def write_trades(self, df: pd.DataFrame, symbol: str, source: str) -> None:
        """Write trades data to partitioned Parquet files."""
        if df.empty:
            return
        
        df['date'] = pd.to_datetime(df['ts']).dt.date
        
        partition_path = self.base_path / "trades" / symbol / source
        partition_path.mkdir(parents=True, exist_ok=True)
        
        for date, group_df in df.groupby('date'):
            file_path = partition_path / f"{date}.parquet"
            write_df = group_df.drop(['date'], axis=1)
            
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                combined_df = pd.concat([existing_df, write_df]).drop_duplicates(
                    subset=['trade_id'], keep='last'
                )
                combined_df.to_parquet(file_path, index=False)
            else:
                write_df.to_parquet(file_path, index=False)
        
        logger.info(f"Wrote {len(df)} trade records for {symbol} ({source})")
    
    def write_options(self, df: pd.DataFrame, underlying: str, source: str) -> None:
        """Write options data to partitioned Parquet files."""
        if df.empty:
            return
        
        df['expiry_str'] = pd.to_datetime(df['expiry']).dt.strftime('%Y-%m-%d')
        
        partition_path = self.base_path / "options" / underlying / source
        partition_path.mkdir(parents=True, exist_ok=True)
        
        for expiry_str, group_df in df.groupby('expiry_str'):
            file_path = partition_path / f"{expiry_str}.parquet"
            write_df = group_df.drop(['expiry_str'], axis=1)
            
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                combined_df = pd.concat([existing_df, write_df]).drop_duplicates(
                    subset=['underlying', 'expiry', 'strike', 'option_type', 'ts'], 
                    keep='last'
                )
                combined_df.to_parquet(file_path, index=False)
            else:
                write_df.to_parquet(file_path, index=False)
        
        logger.info(f"Wrote {len(df)} options records for {underlying} ({source})")
    
    def read_ohlcv(self, symbol: str, source: str, 
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Read OHLCV data from Parquet files."""
        partition_path = self.base_path / "ohlcv" / symbol / source
        
        if not partition_path.exists():
            return pd.DataFrame()
        
        # Read all parquet files in the partition
        parquet_files = list(partition_path.glob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()
        
        dfs = []
        for file_path in parquet_files:
            df = pd.read_parquet(file_path)
            dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['ts'] = pd.to_datetime(combined_df['ts'])
        
        # Filter by date range if provided
        if start_date:
            combined_df = combined_df[combined_df['ts'] >= start_date]
        if end_date:
            combined_df = combined_df[combined_df['ts'] <= end_date]
        
        return combined_df.sort_values('ts')
    
    def get_available_symbols(self) -> Dict[str, list]:
        """Get available symbols by data type."""
        symbols = {}
        
        for data_type in ['ohlcv', 'trades', 'options']:
            type_path = self.base_path / data_type
            if type_path.exists():
                symbols[data_type] = [d.name for d in type_path.iterdir() if d.is_dir()]
            else:
                symbols[data_type] = []
        
        return symbols


# Global Parquet manager instance
parquet_manager = ParquetManager()
