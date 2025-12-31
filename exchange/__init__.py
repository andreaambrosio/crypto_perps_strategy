"""
Exchange Integration Layer for Crypto Perpetual Futures Trading

This module provides a unified interface for connecting to cryptocurrency
exchanges for perpetual futures trading. It includes:

- BaseExchange: Abstract base class defining the exchange interface
- BinanceFutures: Binance USD-M Futures implementation
- DataFetcher: Historical and live data fetching with caching

Usage:
    from exchange import BinanceFutures, DataFetcher

    # Create exchange instance
    exchange = BinanceFutures(
        api_key="your_api_key",
        api_secret="your_api_secret",
        testnet=True,  # Use testnet for testing
    )

    # Connect and fetch data
    async with exchange:
        # Create data fetcher
        fetcher = DataFetcher(exchange)

        # Fetch historical OHLCV data
        df = await fetcher.fetch_ohlcv("BTC/USDT", "1h", days=30)

        # Get current ticker
        ticker = await exchange.get_ticker("BTC/USDT")

        # Create a market order
        order = await exchange.create_market_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.001"),
        )
"""

from .base_exchange import (
    # Enums
    OrderType,
    OrderSide,
    PositionSide,
    OrderStatus,
    MarginType,
    # Data Classes
    OrderRequest,
    Order,
    Position,
    Balance,
    AccountInfo,
    Ticker,
    OHLCV,
    Trade,
    FundingRate,
    # Exceptions
    ExchangeError,
    AuthenticationError,
    InsufficientFundsError,
    OrderNotFoundError,
    RateLimitError,
    NetworkError,
    InvalidOrderError,
    # Base Class
    BaseExchange,
)

from .binance_futures import BinanceFutures

from .data_fetcher import (
    DataFetcher,
    DataCache,
    LiveDataBuffer,
    CacheConfig,
    FetchConfig,
    LiveDataConfig,
    create_data_fetcher,
    fetch_historical_data,
)

__all__ = [
    # Enums
    "OrderType",
    "OrderSide",
    "PositionSide",
    "OrderStatus",
    "MarginType",
    # Data Classes
    "OrderRequest",
    "Order",
    "Position",
    "Balance",
    "AccountInfo",
    "Ticker",
    "OHLCV",
    "Trade",
    "FundingRate",
    # Exceptions
    "ExchangeError",
    "AuthenticationError",
    "InsufficientFundsError",
    "OrderNotFoundError",
    "RateLimitError",
    "NetworkError",
    "InvalidOrderError",
    # Exchange Classes
    "BaseExchange",
    "BinanceFutures",
    # Data Fetching
    "DataFetcher",
    "DataCache",
    "LiveDataBuffer",
    "CacheConfig",
    "FetchConfig",
    "LiveDataConfig",
    "create_data_fetcher",
    "fetch_historical_data",
]

__version__ = "1.0.0"
