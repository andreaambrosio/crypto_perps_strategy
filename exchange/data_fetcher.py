"""
Data Fetcher for Historical and Live Market Data

Provides unified data fetching capabilities with caching, rate limiting,
and support for multiple timeframes. Integrates with exchange interfaces
for seamless data retrieval.
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from .base_exchange import (
    BaseExchange,
    OHLCV,
    Ticker,
    FundingRate,
    Trade,
    ExchangeError,
)

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for data caching."""
    enabled: bool = True
    cache_dir: str = ".cache/market_data"
    max_age_hours: int = 24
    compress: bool = True


@dataclass
class FetchConfig:
    """Configuration for data fetching."""
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 1000
    rate_limit_delay: float = 0.1
    timeout: int = 30


@dataclass
class LiveDataConfig:
    """Configuration for live data streaming."""
    buffer_size: int = 1000
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10


class DataCache:
    """
    File-based caching for market data.

    Supports automatic cache invalidation and compression.
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize data cache.

        Args:
            config: Cache configuration
        """
        self.config = config
        self.cache_dir = Path(config.cache_dir)

        if config.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        """
        Get data from cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found/expired
        """
        if not self.config.enabled:
            return None

        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        # Check age
        file_age = time.time() - cache_path.stat().st_mtime
        max_age_seconds = self.config.max_age_hours * 3600

        if file_age > max_age_seconds:
            cache_path.unlink()
            return None

        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache {key}: {e}")
            return None

    def set(self, key: str, data: Any) -> None:
        """
        Store data in cache.

        Args:
            key: Cache key
            data: Data to cache
        """
        if not self.config.enabled:
            return

        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {key}: {e}")

    def invalidate(self, key: str) -> None:
        """
        Invalidate a cache entry.

        Args:
            key: Cache key to invalidate
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()

    def clear(self) -> None:
        """Clear all cached data."""
        if self.cache_dir.exists():
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()


class LiveDataBuffer:
    """
    Circular buffer for storing live market data.

    Thread-safe implementation for real-time data streaming.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize live data buffer.

        Args:
            max_size: Maximum number of items to store
        """
        self.max_size = max_size
        self._data: List[Any] = []
        self._lock = asyncio.Lock()

    async def append(self, item: Any) -> None:
        """
        Append item to buffer.

        Args:
            item: Item to append
        """
        async with self._lock:
            self._data.append(item)
            if len(self._data) > self.max_size:
                self._data.pop(0)

    async def get_all(self) -> List[Any]:
        """
        Get all items in buffer.

        Returns:
            List of buffered items
        """
        async with self._lock:
            return list(self._data)

    async def get_latest(self, n: int = 1) -> List[Any]:
        """
        Get latest n items.

        Args:
            n: Number of items to retrieve

        Returns:
            List of latest items
        """
        async with self._lock:
            return list(self._data[-n:])

    async def clear(self) -> None:
        """Clear the buffer."""
        async with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


class DataFetcher:
    """
    Unified data fetching interface for historical and live market data.

    Features:
    - Historical OHLCV data with pagination
    - Live data streaming with WebSocket
    - Automatic caching and rate limiting
    - Multiple timeframe support
    - DataFrame conversion utilities
    """

    # Timeframe to milliseconds mapping
    TIMEFRAME_MS = {
        "1m": 60000,
        "3m": 180000,
        "5m": 300000,
        "15m": 900000,
        "30m": 1800000,
        "1h": 3600000,
        "2h": 7200000,
        "4h": 14400000,
        "6h": 21600000,
        "8h": 28800000,
        "12h": 43200000,
        "1d": 86400000,
        "3d": 259200000,
        "1w": 604800000,
    }

    def __init__(
        self,
        exchange: BaseExchange,
        cache_config: Optional[CacheConfig] = None,
        fetch_config: Optional[FetchConfig] = None,
        live_config: Optional[LiveDataConfig] = None,
    ):
        """
        Initialize data fetcher.

        Args:
            exchange: Exchange interface to use
            cache_config: Cache configuration
            fetch_config: Fetch configuration
            live_config: Live data configuration
        """
        self.exchange = exchange
        self.cache_config = cache_config or CacheConfig()
        self.fetch_config = fetch_config or FetchConfig()
        self.live_config = live_config or LiveDataConfig()

        self.cache = DataCache(self.cache_config)

        # Live data state
        self._live_buffers: Dict[str, LiveDataBuffer] = {}
        self._live_subscriptions: Dict[str, bool] = {}
        self._live_callbacks: Dict[str, List[Callable]] = {}

    # ========== Historical Data Fetching ==========

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        limit: Optional[int] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start: Start datetime or string
            end: End datetime or string
            limit: Maximum number of candles
            use_cache: Whether to use cache

        Returns:
            DataFrame with OHLCV data
        """
        # Parse dates
        start_dt = self._parse_datetime(start) if start else None
        end_dt = self._parse_datetime(end) if end else datetime.now(timezone.utc)

        # Check cache
        cache_key = self.cache._get_cache_key(
            "ohlcv", symbol, timeframe, str(start_dt), str(end_dt)
        )

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {symbol} {timeframe}")
                return cached

        # Fetch data
        candles = await self._fetch_ohlcv_paginated(
            symbol=symbol,
            timeframe=timeframe,
            start=start_dt,
            end=end_dt,
            limit=limit,
        )

        # Convert to DataFrame
        df = self._ohlcv_to_dataframe(candles)

        # Cache result
        if use_cache and not df.empty:
            self.cache.set(cache_key, df)

        return df

    async def _fetch_ohlcv_paginated(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[OHLCV]:
        """
        Fetch OHLCV data with pagination for large date ranges.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start: Start datetime
            end: End datetime
            limit: Maximum number of candles

        Returns:
            List of OHLCV candles
        """
        all_candles: List[OHLCV] = []
        batch_size = self.fetch_config.batch_size
        timeframe_ms = self.TIMEFRAME_MS.get(timeframe, 3600000)

        current_start = start
        end_ts = end.timestamp() * 1000 if end else time.time() * 1000

        while True:
            try:
                # Calculate how many candles to fetch
                fetch_limit = min(batch_size, limit - len(all_candles)) if limit else batch_size

                candles = await self.exchange.get_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_start,
                    limit=fetch_limit,
                )

                if not candles:
                    break

                all_candles.extend(candles)
                logger.debug(f"Fetched {len(candles)} candles for {symbol}, total: {len(all_candles)}")

                # Check if we've reached the end
                last_ts = candles[-1].timestamp.timestamp() * 1000
                if last_ts >= end_ts:
                    break

                # Check limit
                if limit and len(all_candles) >= limit:
                    break

                # Move to next batch
                current_start = datetime.fromtimestamp(
                    (last_ts + timeframe_ms) / 1000, tz=timezone.utc
                )

                # Rate limiting
                await asyncio.sleep(self.fetch_config.rate_limit_delay)

            except ExchangeError as e:
                logger.error(f"Error fetching OHLCV data: {e}")
                break

        # Filter to exact date range if specified
        if start or end:
            filtered = []
            for candle in all_candles:
                ts = candle.timestamp.timestamp() * 1000
                if start and ts < start.timestamp() * 1000:
                    continue
                if end and ts > end_ts:
                    continue
                filtered.append(candle)
            all_candles = filtered

        # Apply limit
        if limit:
            all_candles = all_candles[:limit]

        return all_candles

    async def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols concurrently.

        Args:
            symbols: List of trading pair symbols
            timeframe: Candle timeframe
            start: Start datetime
            end: End datetime
            limit: Maximum candles per symbol

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        tasks = [
            self.fetch_ohlcv(symbol, timeframe, start, end, limit)
            for symbol in symbols
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {symbol}: {result}")
                data[symbol] = pd.DataFrame()
            else:
                data[symbol] = result

        return data

    async def fetch_multiple_timeframes(
        self,
        symbol: str,
        timeframes: List[str],
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple timeframes.

        Args:
            symbol: Trading pair symbol
            timeframes: List of timeframes
            start: Start datetime
            end: End datetime
            limit: Maximum candles per timeframe

        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        tasks = [
            self.fetch_ohlcv(symbol, tf, start, end, limit)
            for tf in timeframes
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        for tf, result in zip(timeframes, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {symbol} {tf}: {result}")
                data[tf] = pd.DataFrame()
            else:
                data[tf] = result

        return data

    async def fetch_funding_rates(
        self,
        symbol: str,
        start: Optional[Union[datetime, str]] = None,
        limit: int = 500,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates.

        Args:
            symbol: Trading pair symbol
            start: Start datetime
            limit: Maximum number of records
            use_cache: Whether to use cache

        Returns:
            DataFrame with funding rate data
        """
        start_dt = self._parse_datetime(start) if start else None

        # Check cache
        cache_key = self.cache._get_cache_key("funding", symbol, str(start_dt), limit)

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Fetch data
        funding_rates = await self.exchange.get_funding_rate_history(
            symbol=symbol,
            since=start_dt,
            limit=limit,
        )

        # Convert to DataFrame
        df = self._funding_to_dataframe(funding_rates)

        # Cache result
        if use_cache and not df.empty:
            self.cache.set(cache_key, df)

        return df

    async def fetch_ticker(self, symbol: str) -> Ticker:
        """
        Fetch current ticker for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current ticker data
        """
        return await self.exchange.get_ticker(symbol)

    async def fetch_tickers(
        self,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Ticker]:
        """
        Fetch tickers for multiple symbols.

        Args:
            symbols: List of symbols or None for all

        Returns:
            Dictionary mapping symbol to Ticker
        """
        return await self.exchange.get_tickers(symbols)

    async def fetch_orderbook(
        self,
        symbol: str,
        depth: int = 100
    ) -> Dict[str, Any]:
        """
        Fetch current order book.

        Args:
            symbol: Trading pair symbol
            depth: Number of price levels

        Returns:
            Order book data
        """
        return await self.exchange.get_orderbook(symbol, depth)

    async def fetch_recent_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Trade]:
        """
        Fetch recent public trades.

        Args:
            symbol: Trading pair symbol
            limit: Number of trades

        Returns:
            List of recent trades
        """
        return await self.exchange.get_recent_trades(symbol, limit)

    # ========== Live Data Streaming ==========

    async def start_live_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        callback: Optional[Callable[[OHLCV], None]] = None,
    ) -> None:
        """
        Start live OHLCV data streaming.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            callback: Optional callback for new candles
        """
        key = f"ohlcv:{symbol}:{timeframe}"

        # Create buffer if not exists
        if key not in self._live_buffers:
            self._live_buffers[key] = LiveDataBuffer(self.live_config.buffer_size)

        # Register callback
        if callback:
            if key not in self._live_callbacks:
                self._live_callbacks[key] = []
            self._live_callbacks[key].append(callback)

        # Start subscription
        if key not in self._live_subscriptions or not self._live_subscriptions[key]:
            self._live_subscriptions[key] = True

            async def on_candle(candle: OHLCV):
                await self._live_buffers[key].append(candle)
                for cb in self._live_callbacks.get(key, []):
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            await cb(candle)
                        else:
                            cb(candle)
                    except Exception as e:
                        logger.error(f"Error in live OHLCV callback: {e}")

            await self.exchange.subscribe_ohlcv(symbol, timeframe, on_candle)

    async def start_live_ticker(
        self,
        symbol: str,
        callback: Optional[Callable[[Ticker], None]] = None,
    ) -> None:
        """
        Start live ticker streaming.

        Args:
            symbol: Trading pair symbol
            callback: Optional callback for ticker updates
        """
        key = f"ticker:{symbol}"

        if key not in self._live_buffers:
            self._live_buffers[key] = LiveDataBuffer(self.live_config.buffer_size)

        if callback:
            if key not in self._live_callbacks:
                self._live_callbacks[key] = []
            self._live_callbacks[key].append(callback)

        if key not in self._live_subscriptions or not self._live_subscriptions[key]:
            self._live_subscriptions[key] = True

            async def on_ticker(ticker: Ticker):
                await self._live_buffers[key].append(ticker)
                for cb in self._live_callbacks.get(key, []):
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            await cb(ticker)
                        else:
                            cb(ticker)
                    except Exception as e:
                        logger.error(f"Error in live ticker callback: {e}")

            await self.exchange.subscribe_ticker(symbol, on_ticker)

    async def start_live_trades(
        self,
        symbol: str,
        callback: Optional[Callable[[Trade], None]] = None,
    ) -> None:
        """
        Start live trade streaming.

        Args:
            symbol: Trading pair symbol
            callback: Optional callback for trade updates
        """
        key = f"trades:{symbol}"

        if key not in self._live_buffers:
            self._live_buffers[key] = LiveDataBuffer(self.live_config.buffer_size)

        if callback:
            if key not in self._live_callbacks:
                self._live_callbacks[key] = []
            self._live_callbacks[key].append(callback)

        if key not in self._live_subscriptions or not self._live_subscriptions[key]:
            self._live_subscriptions[key] = True

            async def on_trade(trade: Trade):
                await self._live_buffers[key].append(trade)
                for cb in self._live_callbacks.get(key, []):
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            await cb(trade)
                        else:
                            cb(trade)
                    except Exception as e:
                        logger.error(f"Error in live trade callback: {e}")

            await self.exchange.subscribe_trades(symbol, on_trade)

    async def start_live_orderbook(
        self,
        symbol: str,
        depth: int = 20,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """
        Start live order book streaming.

        Args:
            symbol: Trading pair symbol
            depth: Number of price levels
            callback: Optional callback for orderbook updates
        """
        key = f"orderbook:{symbol}"

        if key not in self._live_buffers:
            self._live_buffers[key] = LiveDataBuffer(100)  # Keep fewer orderbook snapshots

        if callback:
            if key not in self._live_callbacks:
                self._live_callbacks[key] = []
            self._live_callbacks[key].append(callback)

        if key not in self._live_subscriptions or not self._live_subscriptions[key]:
            self._live_subscriptions[key] = True

            async def on_orderbook(orderbook: Dict[str, Any]):
                await self._live_buffers[key].append(orderbook)
                for cb in self._live_callbacks.get(key, []):
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            await cb(orderbook)
                        else:
                            cb(orderbook)
                    except Exception as e:
                        logger.error(f"Error in live orderbook callback: {e}")

            await self.exchange.subscribe_orderbook(symbol, on_orderbook, depth)

    async def stop_live_stream(self, stream_type: str, symbol: str) -> None:
        """
        Stop a live data stream.

        Args:
            stream_type: Type of stream (ohlcv, ticker, trades, orderbook)
            symbol: Trading pair symbol
        """
        key = f"{stream_type}:{symbol}"
        self._live_subscriptions[key] = False

        await self.exchange.unsubscribe(stream_type, symbol)

        # Clear callbacks
        if key in self._live_callbacks:
            del self._live_callbacks[key]

    async def get_live_buffer(
        self,
        stream_type: str,
        symbol: str,
        timeframe: Optional[str] = None,
    ) -> List[Any]:
        """
        Get buffered live data.

        Args:
            stream_type: Type of stream
            symbol: Trading pair symbol
            timeframe: Timeframe for OHLCV

        Returns:
            List of buffered data
        """
        if stream_type == "ohlcv" and timeframe:
            key = f"ohlcv:{symbol}:{timeframe}"
        else:
            key = f"{stream_type}:{symbol}"

        if key in self._live_buffers:
            return await self._live_buffers[key].get_all()
        return []

    # ========== DataFrame Conversion ==========

    def _ohlcv_to_dataframe(self, candles: List[OHLCV]) -> pd.DataFrame:
        """
        Convert OHLCV list to DataFrame.

        Args:
            candles: List of OHLCV candles

        Returns:
            DataFrame with OHLCV data
        """
        if not candles:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        data = [
            {
                "timestamp": c.timestamp,
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume),
            }
            for c in candles
        ]

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        return df

    def _funding_to_dataframe(self, funding_rates: List[FundingRate]) -> pd.DataFrame:
        """
        Convert funding rates to DataFrame.

        Args:
            funding_rates: List of funding rate records

        Returns:
            DataFrame with funding rate data
        """
        if not funding_rates:
            return pd.DataFrame(columns=["timestamp", "funding_rate", "mark_price", "index_price"])

        data = [
            {
                "timestamp": f.funding_time,
                "funding_rate": float(f.funding_rate),
                "mark_price": float(f.mark_price),
                "index_price": float(f.index_price),
            }
            for f in funding_rates
        ]

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        return df

    def _parse_datetime(self, dt: Union[datetime, str]) -> datetime:
        """
        Parse datetime from various formats.

        Args:
            dt: Datetime object or string

        Returns:
            Parsed datetime
        """
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt

        # Try various string formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d",
        ]

        for fmt in formats:
            try:
                parsed = datetime.strptime(dt, fmt)
                return parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        raise ValueError(f"Unable to parse datetime: {dt}")

    # ========== Data Processing Utilities ==========

    def resample_ohlcv(
        self,
        df: pd.DataFrame,
        target_timeframe: str,
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to a different timeframe.

        Args:
            df: Source DataFrame with OHLCV data
            target_timeframe: Target timeframe (1h, 4h, 1d, etc.)

        Returns:
            Resampled DataFrame
        """
        # Map timeframe to pandas frequency
        freq_map = {
            "1m": "1min",
            "3m": "3min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
            "8h": "8h",
            "12h": "12h",
            "1d": "1D",
            "3d": "3D",
            "1w": "1W",
        }

        freq = freq_map.get(target_timeframe)
        if not freq:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")

        resampled = df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        return resampled

    def calculate_returns(
        self,
        df: pd.DataFrame,
        period: int = 1,
        log_returns: bool = False,
    ) -> pd.Series:
        """
        Calculate returns from OHLCV data.

        Args:
            df: DataFrame with close prices
            period: Number of periods for return calculation
            log_returns: Use log returns instead of simple returns

        Returns:
            Series of returns
        """
        close = df["close"]

        if log_returns:
            return np.log(close / close.shift(period))
        return close.pct_change(period)

    def add_technical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical analysis columns.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with additional columns
        """
        df = df.copy()

        # Returns
        df["returns"] = self.calculate_returns(df)
        df["log_returns"] = self.calculate_returns(df, log_returns=True)

        # True Range
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            )
        )

        # Typical price
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

        # VWAP-like (approximation)
        df["vwap"] = (df["typical_price"] * df["volume"]).cumsum() / df["volume"].cumsum()

        return df

    def merge_timeframes(
        self,
        dfs: Dict[str, pd.DataFrame],
        base_timeframe: str,
    ) -> pd.DataFrame:
        """
        Merge multiple timeframe DataFrames.

        Args:
            dfs: Dictionary mapping timeframe to DataFrame
            base_timeframe: Base timeframe to use for index

        Returns:
            Merged DataFrame with suffixed columns
        """
        if base_timeframe not in dfs:
            raise ValueError(f"Base timeframe {base_timeframe} not in provided DataFrames")

        result = dfs[base_timeframe].copy()

        for tf, df in dfs.items():
            if tf == base_timeframe:
                continue

            # Forward fill higher timeframe data
            df_reindexed = df.reindex(result.index, method="ffill")

            # Add suffix and merge
            for col in df_reindexed.columns:
                result[f"{col}_{tf}"] = df_reindexed[col]

        return result

    # ========== Cache Management ==========

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache information
        """
        cache_files = list(self.cache.cache_dir.glob("*.pkl")) if self.cache.cache_dir.exists() else []

        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "enabled": self.cache_config.enabled,
            "cache_dir": str(self.cache.cache_dir),
            "num_entries": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "max_age_hours": self.cache_config.max_age_hours,
        }


# ========== Convenience Functions ==========

async def create_data_fetcher(
    exchange: BaseExchange,
    cache_dir: str = ".cache/market_data",
    cache_enabled: bool = True,
) -> DataFetcher:
    """
    Create and initialize a data fetcher.

    Args:
        exchange: Exchange interface
        cache_dir: Cache directory path
        cache_enabled: Enable caching

    Returns:
        Initialized DataFetcher
    """
    cache_config = CacheConfig(
        enabled=cache_enabled,
        cache_dir=cache_dir,
    )

    fetcher = DataFetcher(exchange, cache_config=cache_config)
    return fetcher


async def fetch_historical_data(
    exchange: BaseExchange,
    symbol: str,
    timeframe: str = "1h",
    days: int = 30,
) -> pd.DataFrame:
    """
    Convenience function to fetch historical data.

    Args:
        exchange: Exchange interface
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        days: Number of days to fetch

    Returns:
        DataFrame with OHLCV data
    """
    fetcher = DataFetcher(exchange)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    return await fetcher.fetch_ohlcv(symbol, timeframe, start, end)
