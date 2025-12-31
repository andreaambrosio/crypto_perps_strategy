"""
Binance Futures Exchange Implementation

Provides connectivity to Binance USD-M Futures using ccxt library.
Supports both REST API and WebSocket streams for real-time data.
"""

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional
import json

try:
    import ccxt.async_support as ccxt
    from ccxt.async_support.base.exchange import Exchange as CCXTExchange
except ImportError:
    raise ImportError("ccxt library is required. Install with: pip install ccxt")

from .base_exchange import (
    BaseExchange,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    Balance,
    AccountInfo,
    Ticker,
    OHLCV,
    Trade,
    FundingRate,
    MarginType,
    ExchangeError,
    AuthenticationError,
    InsufficientFundsError,
    OrderNotFoundError,
    RateLimitError,
    NetworkError,
    InvalidOrderError,
)

logger = logging.getLogger(__name__)


class BinanceFutures(BaseExchange):
    """
    Binance USD-M Perpetual Futures exchange implementation.

    Features:
    - Full REST API support via ccxt
    - WebSocket streams for real-time data
    - Rate limiting and retry logic
    - Position and order management
    - Hedge mode support
    """

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    RETRY_MULTIPLIER = 2.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        sandbox: bool = False,
        rate_limit: bool = True,
        timeout: int = 30000,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Binance Futures interface.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use Binance Futures testnet
            sandbox: Alias for testnet
            rate_limit: Enable built-in rate limiting
            timeout: Request timeout in milliseconds
            options: Additional ccxt options
        """
        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            sandbox=sandbox,
            rate_limit=rate_limit,
            timeout=timeout,
            options=options,
        )

        # Initialize ccxt exchange
        exchange_options = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": rate_limit,
            "timeout": timeout,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
                "recvWindow": 60000,
                **(options or {}),
            },
        }

        if self.testnet:
            exchange_options["options"]["sandboxMode"] = True

        self._exchange: CCXTExchange = ccxt.binanceusdm(exchange_options)

        # WebSocket state
        self._ws_client = None
        self._ws_task: Optional[asyncio.Task] = None
        self._user_stream_key: Optional[str] = None
        self._user_stream_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None

        # Market info cache
        self._markets_loaded = False
        self._symbol_info: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Exchange name identifier."""
        return "binance_futures"

    @property
    def supported_timeframes(self) -> List[str]:
        """List of supported OHLCV timeframes."""
        return ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]

    # ========== Connection Management ==========

    async def connect(self) -> None:
        """Establish connection and load markets."""
        try:
            await self._load_markets()
            self._is_connected = True
            logger.info(f"Connected to Binance Futures {'(testnet)' if self.testnet else ''}")
        except Exception as e:
            raise NetworkError(f"Failed to connect to Binance Futures: {e}")

    async def disconnect(self) -> None:
        """Close all connections."""
        # Cancel WebSocket tasks
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._user_stream_task and not self._user_stream_task.done():
            self._user_stream_task.cancel()
            try:
                await self._user_stream_task
            except asyncio.CancelledError:
                pass

        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass

        # Close ccxt connection
        if self._exchange:
            await self._exchange.close()

        self._is_connected = False
        logger.info("Disconnected from Binance Futures")

    async def check_connection(self) -> bool:
        """Check if connection is alive."""
        try:
            await self._exchange.fetch_time()
            return True
        except Exception:
            return False

    async def _load_markets(self) -> None:
        """Load market information from exchange."""
        if not self._markets_loaded:
            await self._exchange.load_markets()
            self._markets_loaded = True

            # Cache symbol info for quick lookups
            for symbol, market in self._exchange.markets.items():
                if market.get("type") == "swap":
                    self._symbol_info[symbol] = market

    async def _retry_request(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a request with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        last_exception = None
        delay = self.RETRY_DELAY

        for attempt in range(self.MAX_RETRIES):
            try:
                return await func(*args, **kwargs)
            except ccxt.RateLimitExceeded as e:
                last_exception = RateLimitError(str(e))
                wait_time = delay * (self.RETRY_MULTIPLIER ** attempt)
                logger.warning(f"Rate limit exceeded, waiting {wait_time}s (attempt {attempt + 1})")
                await asyncio.sleep(wait_time)
            except ccxt.NetworkError as e:
                last_exception = NetworkError(str(e))
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = delay * (self.RETRY_MULTIPLIER ** attempt)
                    logger.warning(f"Network error, retrying in {wait_time}s (attempt {attempt + 1})")
                    await asyncio.sleep(wait_time)
            except ccxt.AuthenticationError as e:
                raise AuthenticationError(str(e))
            except ccxt.InsufficientFunds as e:
                raise InsufficientFundsError(str(e))
            except ccxt.OrderNotFound as e:
                raise OrderNotFoundError(str(e))
            except ccxt.InvalidOrder as e:
                raise InvalidOrderError(str(e))
            except ccxt.ExchangeError as e:
                raise ExchangeError(str(e))

        raise last_exception or ExchangeError("Request failed after retries")

    # ========== Market Data ==========

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current ticker for a symbol."""
        ticker = await self._retry_request(self._exchange.fetch_ticker, symbol)
        return self._parse_ticker(ticker)

    async def get_tickers(self, symbols: Optional[List[str]] = None) -> Dict[str, Ticker]:
        """Get tickers for multiple symbols."""
        tickers = await self._retry_request(self._exchange.fetch_tickers, symbols)
        return {symbol: self._parse_ticker(t) for symbol, t in tickers.items()}

    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book for a symbol."""
        orderbook = await self._retry_request(
            self._exchange.fetch_order_book, symbol, limit
        )
        return {
            "symbol": symbol,
            "bids": [(Decimal(str(p)), Decimal(str(q))) for p, q in orderbook["bids"]],
            "asks": [(Decimal(str(p)), Decimal(str(q))) for p, q in orderbook["asks"]],
            "timestamp": datetime.fromtimestamp(orderbook["timestamp"] / 1000, tz=timezone.utc)
            if orderbook.get("timestamp")
            else datetime.now(timezone.utc),
            "nonce": orderbook.get("nonce"),
        }

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[OHLCV]:
        """Get OHLCV candlestick data."""
        since_ts = int(since.timestamp() * 1000) if since else None

        candles = await self._retry_request(
            self._exchange.fetch_ohlcv,
            symbol,
            timeframe,
            since_ts,
            limit,
        )

        return [self._parse_ohlcv(c) for c in candles]

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Get recent public trades."""
        trades = await self._retry_request(
            self._exchange.fetch_trades, symbol, None, limit
        )
        return [self._parse_public_trade(t, symbol) for t in trades]

    async def get_funding_rate(self, symbol: str) -> FundingRate:
        """Get current funding rate for perpetual contract."""
        funding = await self._retry_request(
            self._exchange.fetch_funding_rate, symbol
        )
        return self._parse_funding_rate(funding)

    async def get_funding_rate_history(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[FundingRate]:
        """Get historical funding rates."""
        since_ts = int(since.timestamp() * 1000) if since else None

        history = await self._retry_request(
            self._exchange.fetch_funding_rate_history,
            symbol,
            since_ts,
            limit,
        )

        return [self._parse_funding_rate(f) for f in history]

    # ========== Account & Balance ==========

    async def get_balance(self) -> Dict[str, Balance]:
        """Get account balances."""
        balance = await self._retry_request(self._exchange.fetch_balance)

        result = {}
        for currency, data in balance.items():
            if isinstance(data, dict) and "total" in data:
                result[currency] = Balance(
                    currency=currency,
                    total=Decimal(str(data.get("total", 0) or 0)),
                    available=Decimal(str(data.get("free", 0) or 0)),
                    used=Decimal(str(data.get("used", 0) or 0)),
                    raw=data,
                )

        return result

    async def get_account_info(self) -> AccountInfo:
        """Get comprehensive account information."""
        # Fetch balance and positions
        balance = await self._retry_request(self._exchange.fetch_balance)
        positions = await self.get_positions()

        # Extract USDT balance (primary margin currency)
        usdt_balance = balance.get("USDT", {})

        # Parse balances
        balances = {}
        for currency, data in balance.items():
            if isinstance(data, dict) and "total" in data:
                balances[currency] = Balance(
                    currency=currency,
                    total=Decimal(str(data.get("total", 0) or 0)),
                    available=Decimal(str(data.get("free", 0) or 0)),
                    used=Decimal(str(data.get("used", 0) or 0)),
                    raw=data,
                )

        # Calculate totals
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_margin = Decimal(str(usdt_balance.get("total", 0) or 0))
        available_margin = Decimal(str(usdt_balance.get("free", 0) or 0))
        used_margin = Decimal(str(usdt_balance.get("used", 0) or 0))

        return AccountInfo(
            balances=balances,
            total_margin_balance=total_margin,
            available_margin=available_margin,
            used_margin=used_margin,
            total_unrealized_pnl=total_unrealized_pnl,
            total_realized_pnl=Decimal(0),  # Not available in standard API
            margin_ratio=None,
            maintenance_margin=None,
            positions=positions,
            updated_at=datetime.now(timezone.utc),
            raw=balance,
        )

    # ========== Position Management ==========

    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions."""
        positions = await self._retry_request(
            self._exchange.fetch_positions, [symbol] if symbol else None
        )

        result = []
        for pos in positions:
            parsed = self._parse_position(pos)
            if parsed and parsed.quantity != 0:
                result.append(parsed)

        return result

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        positions = await self.get_positions(symbol)
        return positions[0] if positions else None

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        try:
            await self._retry_request(
                self._exchange.set_leverage, leverage, symbol
            )
            logger.info(f"Set leverage for {symbol} to {leverage}x")
            return True
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
            return False

    async def set_margin_type(self, symbol: str, margin_type: MarginType) -> bool:
        """Set margin type (isolated/cross) for a symbol."""
        try:
            await self._retry_request(
                self._exchange.set_margin_mode,
                margin_type.value.upper(),
                symbol,
            )
            logger.info(f"Set margin type for {symbol} to {margin_type.value}")
            return True
        except ccxt.ExchangeError as e:
            # Binance returns error if margin type is already set
            if "No need to change margin type" in str(e):
                return True
            logger.error(f"Failed to set margin type: {e}")
            return False

    # ========== Order Management ==========

    async def create_order(self, request: OrderRequest) -> Order:
        """Create a new order."""
        # Map order type to ccxt
        order_type_map = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_MARKET: "stop_market",
            OrderType.STOP_LIMIT: "stop",
            OrderType.TAKE_PROFIT_MARKET: "take_profit_market",
            OrderType.TAKE_PROFIT_LIMIT: "take_profit",
            OrderType.TRAILING_STOP: "trailing_stop_market",
        }

        params = dict(request.params)

        # Handle position side
        if request.position_side != PositionSide.BOTH:
            params["positionSide"] = request.position_side.value.upper()

        # Handle reduce only
        if request.reduce_only:
            params["reduceOnly"] = True

        # Handle stop price
        if request.stop_price:
            params["stopPrice"] = float(request.stop_price)

        # Handle time in force
        if request.time_in_force:
            params["timeInForce"] = request.time_in_force

        # Handle client order ID
        if request.client_order_id:
            params["newClientOrderId"] = request.client_order_id

        try:
            result = await self._retry_request(
                self._exchange.create_order,
                request.symbol,
                order_type_map.get(request.order_type, request.order_type.value),
                request.side.value,
                float(request.quantity),
                float(request.price) if request.price else None,
                params,
            )

            order = self._parse_order(result)
            logger.info(f"Created {request.order_type.value} order: {order.id} for {request.symbol}")
            return order

        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        """Cancel an existing order."""
        result = await self._retry_request(
            self._exchange.cancel_order, order_id, symbol
        )

        order = self._parse_order(result)
        logger.info(f"Canceled order: {order_id} for {symbol}")
        return order

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Cancel all open orders."""
        if symbol:
            result = await self._retry_request(
                self._exchange.cancel_all_orders, symbol
            )
        else:
            # Cancel orders for all symbols
            positions = await self.get_positions()
            symbols = list(set([p.symbol for p in positions]))
            open_orders = await self.get_open_orders()
            symbols.extend([o.symbol for o in open_orders])
            symbols = list(set(symbols))

            result = []
            for sym in symbols:
                try:
                    orders = await self._retry_request(
                        self._exchange.cancel_all_orders, sym
                    )
                    result.extend(orders)
                except Exception as e:
                    logger.warning(f"Failed to cancel orders for {sym}: {e}")

        canceled = [self._parse_order(o) for o in result] if result else []
        logger.info(f"Canceled {len(canceled)} orders")
        return canceled

    async def get_order(self, order_id: str, symbol: str) -> Order:
        """Get order by ID."""
        result = await self._retry_request(
            self._exchange.fetch_order, order_id, symbol
        )
        return self._parse_order(result)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        result = await self._retry_request(
            self._exchange.fetch_open_orders, symbol
        )
        return [self._parse_order(o) for o in result]

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Order]:
        """Get order history."""
        since_ts = int(since.timestamp() * 1000) if since else None

        result = await self._retry_request(
            self._exchange.fetch_orders,
            symbol,
            since_ts,
            limit,
        )
        return [self._parse_order(o) for o in result]

    async def get_my_trades(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Trade]:
        """Get user's trade history."""
        since_ts = int(since.timestamp() * 1000) if since else None

        result = await self._retry_request(
            self._exchange.fetch_my_trades,
            symbol,
            since_ts,
            limit,
        )
        return [self._parse_my_trade(t) for t in result]

    # ========== WebSocket Subscriptions ==========

    async def subscribe_ticker(
        self,
        symbol: str,
        callback: Callable[[Ticker], None]
    ) -> None:
        """Subscribe to ticker updates via WebSocket."""
        channel = f"ticker:{symbol}"
        self.register_callback(channel, callback)

        if channel not in self._ws_subscriptions:
            self._ws_subscriptions[channel] = True
            asyncio.create_task(self._watch_ticker(symbol, channel))

    async def _watch_ticker(self, symbol: str, channel: str) -> None:
        """Watch ticker stream."""
        while self._ws_subscriptions.get(channel, False):
            try:
                ticker = await self._exchange.watch_ticker(symbol)
                parsed = self._parse_ticker(ticker)
                await self._emit(channel, parsed)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ticker stream for {symbol}: {e}")
                await asyncio.sleep(1)

    async def subscribe_orderbook(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None],
        depth: int = 20
    ) -> None:
        """Subscribe to order book updates."""
        channel = f"orderbook:{symbol}"
        self.register_callback(channel, callback)

        if channel not in self._ws_subscriptions:
            self._ws_subscriptions[channel] = True
            asyncio.create_task(self._watch_orderbook(symbol, channel, depth))

    async def _watch_orderbook(self, symbol: str, channel: str, depth: int) -> None:
        """Watch orderbook stream."""
        while self._ws_subscriptions.get(channel, False):
            try:
                orderbook = await self._exchange.watch_order_book(symbol, depth)
                parsed = {
                    "symbol": symbol,
                    "bids": [(Decimal(str(p)), Decimal(str(q))) for p, q in orderbook["bids"][:depth]],
                    "asks": [(Decimal(str(p)), Decimal(str(q))) for p, q in orderbook["asks"][:depth]],
                    "timestamp": datetime.now(timezone.utc),
                }
                await self._emit(channel, parsed)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in orderbook stream for {symbol}: {e}")
                await asyncio.sleep(1)

    async def subscribe_trades(
        self,
        symbol: str,
        callback: Callable[[Trade], None]
    ) -> None:
        """Subscribe to trade updates."""
        channel = f"trades:{symbol}"
        self.register_callback(channel, callback)

        if channel not in self._ws_subscriptions:
            self._ws_subscriptions[channel] = True
            asyncio.create_task(self._watch_trades(symbol, channel))

    async def _watch_trades(self, symbol: str, channel: str) -> None:
        """Watch trades stream."""
        while self._ws_subscriptions.get(channel, False):
            try:
                trades = await self._exchange.watch_trades(symbol)
                for trade in trades:
                    parsed = self._parse_public_trade(trade, symbol)
                    await self._emit(channel, parsed)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trades stream for {symbol}: {e}")
                await asyncio.sleep(1)

    async def subscribe_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable[[OHLCV], None]
    ) -> None:
        """Subscribe to OHLCV/kline updates."""
        channel = f"ohlcv:{symbol}:{timeframe}"
        self.register_callback(channel, callback)

        if channel not in self._ws_subscriptions:
            self._ws_subscriptions[channel] = True
            asyncio.create_task(self._watch_ohlcv(symbol, timeframe, channel))

    async def _watch_ohlcv(self, symbol: str, timeframe: str, channel: str) -> None:
        """Watch OHLCV stream."""
        while self._ws_subscriptions.get(channel, False):
            try:
                ohlcv = await self._exchange.watch_ohlcv(symbol, timeframe)
                for candle in ohlcv:
                    parsed = self._parse_ohlcv(candle)
                    await self._emit(channel, parsed)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in OHLCV stream for {symbol}: {e}")
                await asyncio.sleep(1)

    async def subscribe_user_data(
        self,
        on_order_update: Optional[Callable[[Order], None]] = None,
        on_position_update: Optional[Callable[[Position], None]] = None,
        on_balance_update: Optional[Callable[[Balance], None]] = None,
    ) -> None:
        """Subscribe to user data updates."""
        if on_order_update:
            self.register_callback("user:order", on_order_update)
        if on_position_update:
            self.register_callback("user:position", on_position_update)
        if on_balance_update:
            self.register_callback("user:balance", on_balance_update)

        if "user_data" not in self._ws_subscriptions:
            self._ws_subscriptions["user_data"] = True
            asyncio.create_task(self._watch_user_data())

    async def _watch_user_data(self) -> None:
        """Watch user data streams."""
        while self._ws_subscriptions.get("user_data", False):
            try:
                # Watch orders
                order_task = asyncio.create_task(self._watch_orders())
                # Watch balance
                balance_task = asyncio.create_task(self._watch_balance())

                await asyncio.gather(order_task, balance_task)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in user data stream: {e}")
                await asyncio.sleep(1)

    async def _watch_orders(self) -> None:
        """Watch order updates."""
        while self._ws_subscriptions.get("user_data", False):
            try:
                orders = await self._exchange.watch_orders()
                for order in orders:
                    parsed = self._parse_order(order)
                    await self._emit("user:order", parsed)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching orders: {e}")
                await asyncio.sleep(1)

    async def _watch_balance(self) -> None:
        """Watch balance updates."""
        while self._ws_subscriptions.get("user_data", False):
            try:
                balance = await self._exchange.watch_balance()
                for currency, data in balance.items():
                    if isinstance(data, dict) and "total" in data:
                        parsed = Balance(
                            currency=currency,
                            total=Decimal(str(data.get("total", 0) or 0)),
                            available=Decimal(str(data.get("free", 0) or 0)),
                            used=Decimal(str(data.get("used", 0) or 0)),
                            raw=data,
                        )
                        await self._emit("user:balance", parsed)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching balance: {e}")
                await asyncio.sleep(1)

    async def unsubscribe(self, channel: str, symbol: Optional[str] = None) -> None:
        """Unsubscribe from a channel."""
        if symbol:
            key = f"{channel}:{symbol}"
        else:
            key = channel

        self._ws_subscriptions[key] = False

        # Clear callbacks
        if key in self._ws_callbacks:
            del self._ws_callbacks[key]

    # ========== Utility Methods ==========

    async def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get exchange/market information."""
        await self._load_markets()

        if symbol:
            return self._exchange.market(symbol)

        return {
            "markets": self._exchange.markets,
            "currencies": self._exchange.currencies,
            "timeframes": self.supported_timeframes,
        }

    async def get_server_time(self) -> datetime:
        """Get exchange server time."""
        timestamp = await self._retry_request(self._exchange.fetch_time)
        return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)

    # ========== Parser Methods ==========

    def _parse_ticker(self, data: Dict[str, Any]) -> Ticker:
        """Parse raw ticker data to Ticker object."""
        return Ticker(
            symbol=data["symbol"],
            bid=Decimal(str(data.get("bid", 0) or 0)),
            ask=Decimal(str(data.get("ask", 0) or 0)),
            last=Decimal(str(data.get("last", 0) or 0)),
            high_24h=Decimal(str(data.get("high", 0) or 0)),
            low_24h=Decimal(str(data.get("low", 0) or 0)),
            volume_24h=Decimal(str(data.get("baseVolume", 0) or 0)),
            quote_volume_24h=Decimal(str(data.get("quoteVolume", 0) or 0)),
            change_24h=Decimal(str(data.get("change", 0) or 0)),
            change_pct_24h=Decimal(str(data.get("percentage", 0) or 0)),
            timestamp=datetime.fromtimestamp(data["timestamp"] / 1000, tz=timezone.utc)
            if data.get("timestamp")
            else datetime.now(timezone.utc),
            raw=data,
        )

    def _parse_ohlcv(self, data: List) -> OHLCV:
        """Parse raw OHLCV data to OHLCV object."""
        return OHLCV(
            timestamp=datetime.fromtimestamp(data[0] / 1000, tz=timezone.utc),
            open=Decimal(str(data[1])),
            high=Decimal(str(data[2])),
            low=Decimal(str(data[3])),
            close=Decimal(str(data[4])),
            volume=Decimal(str(data[5])),
        )

    def _parse_order(self, data: Dict[str, Any]) -> Order:
        """Parse raw order data to Order object."""
        status_map = {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
        }

        type_map = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP_LIMIT,
            "stop_market": OrderType.STOP_MARKET,
            "take_profit": OrderType.TAKE_PROFIT_LIMIT,
            "take_profit_market": OrderType.TAKE_PROFIT_MARKET,
            "trailing_stop_market": OrderType.TRAILING_STOP,
        }

        # Determine position side
        position_side = PositionSide.BOTH
        if data.get("info", {}).get("positionSide"):
            ps = data["info"]["positionSide"].lower()
            if ps == "long":
                position_side = PositionSide.LONG
            elif ps == "short":
                position_side = PositionSide.SHORT

        return Order(
            id=data["id"],
            client_order_id=data.get("clientOrderId"),
            symbol=data["symbol"],
            side=OrderSide.BUY if data["side"] == "buy" else OrderSide.SELL,
            order_type=type_map.get(data.get("type", "").lower(), OrderType.MARKET),
            quantity=Decimal(str(data.get("amount", 0) or 0)),
            filled_quantity=Decimal(str(data.get("filled", 0) or 0)),
            price=Decimal(str(data["price"])) if data.get("price") else None,
            average_price=Decimal(str(data["average"])) if data.get("average") else None,
            stop_price=Decimal(str(data["stopPrice"])) if data.get("stopPrice") else None,
            status=status_map.get(data.get("status", "open"), OrderStatus.OPEN),
            position_side=position_side,
            reduce_only=data.get("reduceOnly", False),
            time_in_force=data.get("timeInForce", "GTC"),
            created_at=datetime.fromtimestamp(data["timestamp"] / 1000, tz=timezone.utc)
            if data.get("timestamp")
            else datetime.now(timezone.utc),
            updated_at=datetime.fromtimestamp(data["lastTradeTimestamp"] / 1000, tz=timezone.utc)
            if data.get("lastTradeTimestamp")
            else datetime.now(timezone.utc),
            fee=Decimal(str(data["fee"]["cost"])) if data.get("fee", {}).get("cost") else None,
            fee_currency=data.get("fee", {}).get("currency"),
            raw=data,
        )

    def _parse_position(self, data: Dict[str, Any]) -> Optional[Position]:
        """Parse raw position data to Position object."""
        contracts = Decimal(str(data.get("contracts", 0) or 0))
        if contracts == 0:
            return None

        # Determine position side
        side = PositionSide.BOTH
        if data.get("side") == "long":
            side = PositionSide.LONG
        elif data.get("side") == "short":
            side = PositionSide.SHORT

        margin_type = MarginType.CROSS
        if data.get("marginMode") == "isolated":
            margin_type = MarginType.ISOLATED

        return Position(
            symbol=data["symbol"],
            side=side,
            quantity=contracts,
            entry_price=Decimal(str(data.get("entryPrice", 0) or 0)),
            mark_price=Decimal(str(data.get("markPrice", 0) or 0)),
            liquidation_price=Decimal(str(data["liquidationPrice"]))
            if data.get("liquidationPrice")
            else None,
            unrealized_pnl=Decimal(str(data.get("unrealizedPnl", 0) or 0)),
            realized_pnl=Decimal(str(data.get("realizedPnl", 0) or 0)),
            leverage=int(data.get("leverage", 1) or 1),
            margin_type=margin_type,
            margin=Decimal(str(data.get("initialMargin", 0) or 0)),
            notional_value=Decimal(str(data.get("notional", 0) or 0)),
            raw=data,
        )

    def _parse_public_trade(self, data: Dict[str, Any], symbol: str) -> Trade:
        """Parse public trade data."""
        return Trade(
            id=str(data.get("id", "")),
            order_id="",
            symbol=symbol,
            side=OrderSide.BUY if data.get("side") == "buy" else OrderSide.SELL,
            price=Decimal(str(data.get("price", 0))),
            quantity=Decimal(str(data.get("amount", 0))),
            fee=Decimal(0),
            fee_currency="",
            timestamp=datetime.fromtimestamp(data["timestamp"] / 1000, tz=timezone.utc)
            if data.get("timestamp")
            else datetime.now(timezone.utc),
            is_maker=False,
            raw=data,
        )

    def _parse_my_trade(self, data: Dict[str, Any]) -> Trade:
        """Parse user's trade data."""
        return Trade(
            id=str(data.get("id", "")),
            order_id=str(data.get("order", "")),
            symbol=data.get("symbol", ""),
            side=OrderSide.BUY if data.get("side") == "buy" else OrderSide.SELL,
            price=Decimal(str(data.get("price", 0))),
            quantity=Decimal(str(data.get("amount", 0))),
            fee=Decimal(str(data.get("fee", {}).get("cost", 0) or 0)),
            fee_currency=data.get("fee", {}).get("currency", ""),
            timestamp=datetime.fromtimestamp(data["timestamp"] / 1000, tz=timezone.utc)
            if data.get("timestamp")
            else datetime.now(timezone.utc),
            is_maker=data.get("takerOrMaker") == "maker",
            raw=data,
        )

    def _parse_funding_rate(self, data: Dict[str, Any]) -> FundingRate:
        """Parse funding rate data."""
        return FundingRate(
            symbol=data.get("symbol", ""),
            funding_rate=Decimal(str(data.get("fundingRate", 0) or 0)),
            funding_time=datetime.fromtimestamp(data["fundingTimestamp"] / 1000, tz=timezone.utc)
            if data.get("fundingTimestamp")
            else datetime.now(timezone.utc),
            mark_price=Decimal(str(data.get("markPrice", 0) or 0)),
            index_price=Decimal(str(data.get("indexPrice", 0) or 0)),
            raw=data,
        )

    async def __aenter__(self) -> "BinanceFutures":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()
