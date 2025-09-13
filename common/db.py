"""Database utilities for DuckDB operations."""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DuckDBManager:
    """Manager for DuckDB operations."""
    
    def __init__(self, db_path: str = "./data/duckdb/market.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
    
    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
            self._initialize_tables()
        return self._conn
    
    def _initialize_tables(self) -> None:
        """Initialize database tables if they don't exist."""
        
        # OHLCV table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol VARCHAR,
                ts TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                source VARCHAR,
                fetched_at TIMESTAMP,
                ingest_run_id VARCHAR,
                PRIMARY KEY (symbol, ts, source)
            )
        """)
        
        # Trades table (crypto)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                symbol VARCHAR,
                ts TIMESTAMP,
                price DOUBLE,
                qty DOUBLE,
                side VARCHAR,
                trade_id VARCHAR,
                source VARCHAR,
                fetched_at TIMESTAMP,
                ingest_run_id VARCHAR,
                PRIMARY KEY (symbol, trade_id, source)
            )
        """)
        
        # Book ticker table (crypto)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS book_ticker (
                symbol VARCHAR,
                ts TIMESTAMP,
                best_bid DOUBLE,
                best_bid_qty DOUBLE,
                best_ask DOUBLE,
                best_ask_qty DOUBLE,
                source VARCHAR,
                fetched_at TIMESTAMP,
                ingest_run_id VARCHAR,
                PRIMARY KEY (symbol, ts, source)
            )
        """)
        
        # Corporate actions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS corporate_actions (
                symbol VARCHAR,
                ts TIMESTAMP,
                dividend DOUBLE,
                split_ratio DOUBLE,
                source VARCHAR,
                fetched_at TIMESTAMP,
                ingest_run_id VARCHAR,
                PRIMARY KEY (symbol, ts, source)
            )
        """)
        
        # Fundamentals table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                symbol VARCHAR,
                ts TIMESTAMP,
                market_cap DOUBLE,
                pe_ratio DOUBLE,
                beta DOUBLE,
                sector VARCHAR,
                source VARCHAR,
                fetched_at TIMESTAMP,
                ingest_run_id VARCHAR,
                PRIMARY KEY (symbol, ts, source)
            )
        """)
        
        # Options chains table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS options_chains (
                underlying VARCHAR,
                expiry DATE,
                strike DOUBLE,
                option_type VARCHAR,
                last DOUBLE,
                bid DOUBLE,
                ask DOUBLE,
                mid DOUBLE,
                implied_vol DOUBLE,
                open_interest INTEGER,
                volume INTEGER,
                ts TIMESTAMP,
                source VARCHAR,
                fetched_at TIMESTAMP,
                ingest_run_id VARCHAR,
                PRIMARY KEY (underlying, expiry, strike, option_type, ts, source)
            )
        """)
        
        logger.info("Database tables initialized")
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        """Insert DataFrame into specified table."""
        try:
            self.conn.register('temp_df', df)
            self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
            self.conn.unregister('temp_df')
            logger.info(f"Inserted {len(df)} rows into {table_name}")
        except Exception as e:
            logger.error(f"Error inserting data into {table_name}: {e}")
            raise
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        try:
            return self.conn.execute(sql).df()
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def get_latest_ohlcv(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Get latest OHLCV data for a symbol."""
        sql = """
            SELECT * FROM ohlcv 
            WHERE symbol = ? 
            ORDER BY ts DESC 
            LIMIT ?
        """
        return self.conn.execute(sql, [symbol, limit]).df()
    
    def get_symbol_list(self, source: Optional[str] = None) -> List[str]:
        """Get list of available symbols."""
        sql = "SELECT DISTINCT symbol FROM ohlcv"
        if source:
            sql += " WHERE source = ?"
            return self.conn.execute(sql, [source]).df()['symbol'].tolist()
        return self.conn.execute(sql).df()['symbol'].tolist()
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Global database manager instance
db_manager = DuckDBManager()
