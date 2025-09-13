"""Binance WebSocket data ingestion for crypto markets."""

import asyncio
import json
import logging
import websockets
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
import pandas as pd
from dataclasses import dataclass
import uuid

from common.config import config
from common.db import db_manager
from common.io import parquet_manager

logger = logging.getLogger(__name__)


@dataclass
class TradeData:
    """Trade data structure."""
    symbol: str
    timestamp: datetime
    price: float
    quantity: float
    side: str
    trade_id: str


@dataclass
class BookTickerData:
    """Book ticker data structure."""
    symbol: str
    timestamp: datetime
    best_bid: float
    best_bid_qty: float
    best_ask: float
    best_ask_qty: float


@dataclass
class KlineData:
    """Kline (candlestick) data structure."""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float


class BinanceWebSocketClient:
    """Binance WebSocket client for real-time data streaming."""
    
    BASE_WS_URL = "wss://stream.binance.com:9443/ws/"
    
    def __init__(self):
        self.symbols = config.symbols.crypto
        self.streams = config.binance.streams
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.running = False
        self.ingest_run_id = str(uuid.uuid4())
        
        # Data buffers for batch processing
        self.trade_buffer: List[TradeData] = []
        self.book_buffer: List[BookTickerData] = []
        self.kline_buffer: List[KlineData] = []
        
        self.buffer_size = 100  # Batch size for database writes
        
    async def start(self) -> None:
        """Start WebSocket connections for all configured streams."""
        self.running = True
        logger.info(f"Starting Binance WebSocket client for symbols: {self.symbols}")
        
        tasks = []
        
        if "trades" in self.streams:
            tasks.append(self._connect_trades())
        if "bookTicker" in self.streams:
            tasks.append(self._connect_book_ticker())
        if "kline_1m" in self.streams:
            tasks.append(self._connect_klines())
            
        # Start buffer flush task
        tasks.append(self._flush_buffers_periodically())
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self) -> None:
        """Stop all WebSocket connections."""
        self.running = False
        logger.info("Stopping Binance WebSocket client")
        
        for connection in self.connections.values():
            await connection.close()
        
        # Flush remaining data
        await self._flush_all_buffers()
    
    async def _connect_trades(self) -> None:
        """Connect to trade streams."""
        stream_names = [f"{symbol.lower()}@trade" for symbol in self.symbols]
        stream_url = self.BASE_WS_URL + "/".join(stream_names)
        
        await self._connect_stream(stream_url, self._handle_trade_message, "trades")
    
    async def _connect_book_ticker(self) -> None:
        """Connect to book ticker streams."""
        stream_names = [f"{symbol.lower()}@bookTicker" for symbol in self.symbols]
        stream_url = self.BASE_WS_URL + "/".join(stream_names)
        
        await self._connect_stream(stream_url, self._handle_book_ticker_message, "bookTicker")
    
    async def _connect_klines(self) -> None:
        """Connect to 1-minute kline streams."""
        stream_names = [f"{symbol.lower()}@kline_1m" for symbol in self.symbols]
        stream_url = self.BASE_WS_URL + "/".join(stream_names)
        
        await self._connect_stream(stream_url, self._handle_kline_message, "klines")
    
    async def _connect_stream(self, url: str, handler: Callable, stream_type: str) -> None:
        """Generic stream connection with reconnection logic."""
        while self.running:
            try:
                logger.info(f"Connecting to {stream_type} stream: {url}")
                
                async with websockets.connect(url) as websocket:
                    self.connections[stream_type] = websocket
                    logger.info(f"Connected to {stream_type} stream")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        try:
                            data = json.loads(message)
                            await handler(data)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error in {stream_type}: {e}")
                        except Exception as e:
                            logger.error(f"Error handling {stream_type} message: {e}")
                            
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"{stream_type} connection closed, reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in {stream_type} stream: {e}")
                await asyncio.sleep(5)
    
    async def _handle_trade_message(self, data: Dict) -> None:
        """Handle trade stream messages."""
        try:
            trade = TradeData(
                symbol=data['s'],
                timestamp=datetime.fromtimestamp(data['T'] / 1000, tz=timezone.utc),
                price=float(data['p']),
                quantity=float(data['q']),
                side='buy' if data['m'] else 'sell',  # m=true means buyer is market maker
                trade_id=str(data['t'])
            )
            
            self.trade_buffer.append(trade)
            
            if len(self.trade_buffer) >= self.buffer_size:
                await self._flush_trade_buffer()
                
        except KeyError as e:
            logger.error(f"Missing key in trade data: {e}")
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
    
    async def _handle_book_ticker_message(self, data: Dict) -> None:
        """Handle book ticker stream messages."""
        try:
            book_ticker = BookTickerData(
                symbol=data['s'],
                timestamp=datetime.now(timezone.utc),  # Binance doesn't provide timestamp
                best_bid=float(data['b']),
                best_bid_qty=float(data['B']),
                best_ask=float(data['a']),
                best_ask_qty=float(data['A'])
            )
            
            self.book_buffer.append(book_ticker)
            
            if len(self.book_buffer) >= self.buffer_size:
                await self._flush_book_buffer()
                
        except KeyError as e:
            logger.error(f"Missing key in book ticker data: {e}")
        except Exception as e:
            logger.error(f"Error processing book ticker data: {e}")
    
    async def _handle_kline_message(self, data: Dict) -> None:
        """Handle kline stream messages."""
        try:
            kline_data = data['k']
            
            # Only process closed klines
            if not kline_data['x']:
                return
            
            kline = KlineData(
                symbol=kline_data['s'],
                timestamp=datetime.fromtimestamp(kline_data['t'] / 1000, tz=timezone.utc),
                open_price=float(kline_data['o']),
                high_price=float(kline_data['h']),
                low_price=float(kline_data['l']),
                close_price=float(kline_data['c']),
                volume=float(kline_data['v'])
            )
            
            self.kline_buffer.append(kline)
            
            if len(self.kline_buffer) >= self.buffer_size:
                await self._flush_kline_buffer()
                
        except KeyError as e:
            logger.error(f"Missing key in kline data: {e}")
        except Exception as e:
            logger.error(f"Error processing kline data: {e}")
    
    async def _flush_trade_buffer(self) -> None:
        """Flush trade buffer to database and Parquet."""
        if not self.trade_buffer:
            return
        
        try:
            # Convert to DataFrame
            trades_data = []
            for trade in self.trade_buffer:
                trades_data.append({
                    'symbol': trade.symbol,
                    'ts': trade.timestamp,
                    'price': trade.price,
                    'qty': trade.quantity,
                    'side': trade.side,
                    'trade_id': trade.trade_id,
                    'source': 'binance',
                    'fetched_at': datetime.now(timezone.utc),
                    'ingest_run_id': self.ingest_run_id
                })
            
            df = pd.DataFrame(trades_data)
            
            # Write to database
            db_manager.insert_dataframe(df, 'trades')
            
            # Write to Parquet (group by symbol)
            for symbol, symbol_df in df.groupby('symbol'):
                parquet_manager.write_trades(symbol_df, symbol, 'binance')
            
            logger.info(f"Flushed {len(self.trade_buffer)} trades to storage")
            self.trade_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing trade buffer: {e}")
    
    async def _flush_book_buffer(self) -> None:
        """Flush book ticker buffer to database and Parquet."""
        if not self.book_buffer:
            return
        
        try:
            # Convert to DataFrame
            book_data = []
            for book in self.book_buffer:
                book_data.append({
                    'symbol': book.symbol,
                    'ts': book.timestamp,
                    'best_bid': book.best_bid,
                    'best_bid_qty': book.best_bid_qty,
                    'best_ask': book.best_ask,
                    'best_ask_qty': book.best_ask_qty,
                    'source': 'binance',
                    'fetched_at': datetime.now(timezone.utc),
                    'ingest_run_id': self.ingest_run_id
                })
            
            df = pd.DataFrame(book_data)
            
            # Write to database
            db_manager.insert_dataframe(df, 'book_ticker')
            
            logger.info(f"Flushed {len(self.book_buffer)} book tickers to storage")
            self.book_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing book buffer: {e}")
    
    async def _flush_kline_buffer(self) -> None:
        """Flush kline buffer to database and Parquet."""
        if not self.kline_buffer:
            return
        
        try:
            # Convert to DataFrame
            ohlcv_data = []
            for kline in self.kline_buffer:
                ohlcv_data.append({
                    'symbol': kline.symbol,
                    'ts': kline.timestamp,
                    'open': kline.open_price,
                    'high': kline.high_price,
                    'low': kline.low_price,
                    'close': kline.close_price,
                    'volume': kline.volume,
                    'source': 'binance',
                    'fetched_at': datetime.now(timezone.utc),
                    'ingest_run_id': self.ingest_run_id
                })
            
            df = pd.DataFrame(ohlcv_data)
            
            # Write to database
            db_manager.insert_dataframe(df, 'ohlcv')
            
            # Write to Parquet (group by symbol)
            for symbol, symbol_df in df.groupby('symbol'):
                parquet_manager.write_ohlcv(symbol_df, symbol, 'binance')
            
            logger.info(f"Flushed {len(self.kline_buffer)} klines to storage")
            self.kline_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing kline buffer: {e}")
    
    async def _flush_all_buffers(self) -> None:
        """Flush all buffers."""
        await asyncio.gather(
            self._flush_trade_buffer(),
            self._flush_book_buffer(),
            self._flush_kline_buffer(),
            return_exceptions=True
        )
    
    async def _flush_buffers_periodically(self) -> None:
        """Periodically flush buffers every 30 seconds."""
        while self.running:
            await asyncio.sleep(30)
            await self._flush_all_buffers()


async def main():
    """Main function to run Binance WebSocket client."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    client = BinanceWebSocketClient()
    
    try:
        await client.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
