"""
Abstract Base Class for Exchange Interfaces

Provides a unified interface for cryptocurrency perpetual futures exchanges.
All exchange implementations should inherit from this base class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio
import logging

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by the exchange."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side (direction)."""
    BUY = "buy"
    SELL = "sell"


class PositionSide(Enum):
    """Position side for hedge mode."""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"  # One-way mode


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class MarginType(Enum):
    """Margin mode type."""
    ISOLATED = "isolated"
    CROSS = "cross"


@dataclass
class OrderRequest:
    """Request to create a new order."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    position_side: PositionSide = PositionSide.BOTH
    reduce_only: bool = False
    time_in_force: str = "GTC"
    client_order_id: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    """Represents an order on the exchange."""
    id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    filled_quantity: Decimal
    price: Optional[Decimal]
    average_price: Optional[Decimal]
    stop_price: Optional[Decimal]
    status: OrderStatus
    position_side: PositionSide
    reduce_only: bool
    time_in_force: str
    created_at: datetime
    updated_at: datetime
    fee: Optional[Decimal] = None
    fee_currency: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> Decimal:
        """Calculate remaining quantity to fill."""
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING)


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    liquidation_price: Optional[Decimal]
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    leverage: int
    margin_type: MarginType
    margin: Decimal
    notional_value: Decimal
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == PositionSide.LONG or (
            self.side == PositionSide.BOTH and self.quantity > 0
        )

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == PositionSide.SHORT or (
            self.side == PositionSide.BOTH and self.quantity < 0
        )

    @property
    def pnl_percentage(self) -> Decimal:
        """Calculate PnL as percentage of margin."""
        if self.margin == 0:
            return Decimal(0)
        return (self.unrealized_pnl / self.margin) * 100


@dataclass
class Balance:
    """Account balance information."""
    currency: str
    total: Decimal
    available: Decimal
    used: Decimal
    unrealized_pnl: Decimal = Decimal(0)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccountInfo:
    """Comprehensive account information."""
    balances: Dict[str, Balance]
    total_margin_balance: Decimal
    available_margin: Decimal
    used_margin: Decimal
    total_unrealized_pnl: Decimal
    total_realized_pnl: Decimal
    margin_ratio: Optional[Decimal]
    maintenance_margin: Optional[Decimal]
    positions: List[Position]
    updated_at: datetime
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Ticker:
    """Market ticker data."""
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    high_24h: Decimal
    low_24h: Decimal
    volume_24h: Decimal
    quote_volume_24h: Decimal
    change_24h: Decimal
    change_pct_24h: Decimal
    timestamp: datetime
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> Decimal:
        """Calculate spread as percentage of mid price."""
        mid = self.mid_price
        if mid == 0:
            return Decimal(0)
        return (self.spread / mid) * 100


@dataclass
class OHLCV:
    """OHLCV candlestick data."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
        }


@dataclass
class Trade:
    """Individual trade/fill information."""
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    price: Decimal
    quantity: Decimal
    fee: Decimal
    fee_currency: str
    timestamp: datetime
    is_maker: bool
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FundingRate:
    """Funding rate information."""
    symbol: str
    funding_rate: Decimal
    funding_time: datetime
    mark_price: Decimal
    index_price: Decimal
    raw: Dict[str, Any] = field(default_factory=dict)


class ExchangeError(Exception):
    """Base exception for exchange errors."""
    pass


class AuthenticationError(ExchangeError):
    """Authentication failed."""
    pass


class InsufficientFundsError(ExchangeError):
    """Insufficient funds for operation."""
    pass


class OrderNotFoundError(ExchangeError):
    """Order not found."""
    pass


class RateLimitError(ExchangeError):
    """Rate limit exceeded."""
    pass


class NetworkError(ExchangeError):
    """Network connectivity error."""
    pass


class InvalidOrderError(ExchangeError):
    """Invalid order parameters."""
    pass


class BaseExchange(ABC):
    """
    Abstract base class for exchange interfaces.

    All exchange implementations must inherit from this class and implement
    the abstract methods for unified exchange connectivity.
    """

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
        Initialize the exchange interface.

        Args:
            api_key: API key for authentication
            api_secret: API secret for authentication
            testnet: Use testnet/sandbox environment
            sandbox: Alias for testnet
            rate_limit: Enable rate limiting
            timeout: Request timeout in milliseconds
            options: Additional exchange-specific options
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet or sandbox
        self.rate_limit_enabled = rate_limit
        self.timeout = timeout
        self.options = options or {}

        self._is_connected = False
        self._ws_callbacks: Dict[str, List[Callable]] = {}
        self._ws_subscriptions: Dict[str, bool] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Exchange name identifier."""
        pass

    @property
    @abstractmethod
    def supported_timeframes(self) -> List[str]:
        """List of supported OHLCV timeframes."""
        pass

    @property
    def is_connected(self) -> bool:
        """Check if exchange is connected."""
        return self._is_connected

    # ========== Connection Management ==========

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the exchange."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the exchange."""
        pass

    @abstractmethod
    async def check_connection(self) -> bool:
        """Check if connection is alive."""
        pass

    # ========== Market Data ==========

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """
        Get current ticker for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")

        Returns:
            Ticker with current market data
        """
        pass

    @abstractmethod
    async def get_tickers(self, symbols: Optional[List[str]] = None) -> Dict[str, Ticker]:
        """
        Get tickers for multiple symbols.

        Args:
            symbols: List of symbols, or None for all

        Returns:
            Dictionary mapping symbol to Ticker
        """
        pass

    @abstractmethod
    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get order book for a symbol.

        Args:
            symbol: Trading pair symbol
            limit: Number of price levels

        Returns:
            Order book with bids and asks
        """
        pass

    @abstractmethod
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[OHLCV]:
        """
        Get OHLCV candlestick data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            since: Start time for data
            limit: Maximum number of candles

        Returns:
            List of OHLCV candles
        """
        pass

    @abstractmethod
    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Trade]:
        """
        Get recent public trades.

        Args:
            symbol: Trading pair symbol
            limit: Number of trades to fetch

        Returns:
            List of recent trades
        """
        pass

    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> FundingRate:
        """
        Get current funding rate for perpetual contract.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current funding rate information
        """
        pass

    @abstractmethod
    async def get_funding_rate_history(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[FundingRate]:
        """
        Get historical funding rates.

        Args:
            symbol: Trading pair symbol
            since: Start time
            limit: Number of records

        Returns:
            List of historical funding rates
        """
        pass

    # ========== Account & Balance ==========

    @abstractmethod
    async def get_balance(self) -> Dict[str, Balance]:
        """
        Get account balances.

        Returns:
            Dictionary mapping currency to Balance
        """
        pass

    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """
        Get comprehensive account information.

        Returns:
            Account information including balances, margin, and positions
        """
        pass

    # ========== Position Management ==========

    @abstractmethod
    async def get_positions(
        self,
        symbol: Optional[str] = None
    ) -> List[Position]:
        """
        Get open positions.

        Args:
            symbol: Specific symbol, or None for all positions

        Returns:
            List of open positions
        """
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Position if exists, None otherwise
        """
        pass

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol.

        Args:
            symbol: Trading pair symbol
            leverage: Leverage multiplier

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def set_margin_type(self, symbol: str, margin_type: MarginType) -> bool:
        """
        Set margin type (isolated/cross) for a symbol.

        Args:
            symbol: Trading pair symbol
            margin_type: Margin type to set

        Returns:
            True if successful
        """
        pass

    # ========== Order Management ==========

    @abstractmethod
    async def create_order(self, request: OrderRequest) -> Order:
        """
        Create a new order.

        Args:
            request: Order request parameters

        Returns:
            Created order
        """
        pass

    async def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        reduce_only: bool = False,
        position_side: PositionSide = PositionSide.BOTH,
    ) -> Order:
        """
        Convenience method to create a market order.

        Args:
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            reduce_only: Only reduce existing position
            position_side: Position side for hedge mode

        Returns:
            Created order
        """
        request = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            reduce_only=reduce_only,
            position_side=position_side,
        )
        return await self.create_order(request)

    async def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        reduce_only: bool = False,
        position_side: PositionSide = PositionSide.BOTH,
        time_in_force: str = "GTC",
    ) -> Order:
        """
        Convenience method to create a limit order.

        Args:
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            price: Limit price
            reduce_only: Only reduce existing position
            position_side: Position side for hedge mode
            time_in_force: Time in force (GTC, IOC, FOK)

        Returns:
            Created order
        """
        request = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            reduce_only=reduce_only,
            position_side=position_side,
            time_in_force=time_in_force,
        )
        return await self.create_order(request)

    async def create_stop_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        stop_price: Decimal,
        reduce_only: bool = True,
        position_side: PositionSide = PositionSide.BOTH,
    ) -> Order:
        """
        Convenience method to create a stop market order.

        Args:
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            stop_price: Trigger price
            reduce_only: Only reduce existing position
            position_side: Position side for hedge mode

        Returns:
            Created order
        """
        request = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_MARKET,
            quantity=quantity,
            stop_price=stop_price,
            reduce_only=reduce_only,
            position_side=position_side,
        )
        return await self.create_order(request)

    async def create_take_profit_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        stop_price: Decimal,
        reduce_only: bool = True,
        position_side: PositionSide = PositionSide.BOTH,
    ) -> Order:
        """
        Convenience method to create a take profit market order.

        Args:
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            stop_price: Trigger price
            reduce_only: Only reduce existing position
            position_side: Position side for hedge mode

        Returns:
            Created order
        """
        request = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            quantity=quantity,
            stop_price=stop_price,
            reduce_only=reduce_only,
            position_side=position_side,
        )
        return await self.create_order(request)

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        """
        Cancel an existing order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading pair symbol

        Returns:
            Canceled order
        """
        pass

    @abstractmethod
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Cancel all open orders.

        Args:
            symbol: Specific symbol, or None for all

        Returns:
            List of canceled orders
        """
        pass

    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> Order:
        """
        Get order by ID.

        Args:
            order_id: Order ID
            symbol: Trading pair symbol

        Returns:
            Order information
        """
        pass

    @abstractmethod
    async def get_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[Order]:
        """
        Get all open orders.

        Args:
            symbol: Specific symbol, or None for all

        Returns:
            List of open orders
        """
        pass

    @abstractmethod
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Order]:
        """
        Get order history.

        Args:
            symbol: Specific symbol, or None for all
            since: Start time
            limit: Maximum number of orders

        Returns:
            List of historical orders
        """
        pass

    @abstractmethod
    async def get_my_trades(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Trade]:
        """
        Get user's trade history.

        Args:
            symbol: Specific symbol, or None for all
            since: Start time
            limit: Maximum number of trades

        Returns:
            List of trades
        """
        pass

    # ========== WebSocket Subscriptions ==========

    @abstractmethod
    async def subscribe_ticker(
        self,
        symbol: str,
        callback: Callable[[Ticker], None]
    ) -> None:
        """
        Subscribe to ticker updates.

        Args:
            symbol: Trading pair symbol
            callback: Function to call with ticker updates
        """
        pass

    @abstractmethod
    async def subscribe_orderbook(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None],
        depth: int = 20
    ) -> None:
        """
        Subscribe to order book updates.

        Args:
            symbol: Trading pair symbol
            callback: Function to call with orderbook updates
            depth: Number of price levels
        """
        pass

    @abstractmethod
    async def subscribe_trades(
        self,
        symbol: str,
        callback: Callable[[Trade], None]
    ) -> None:
        """
        Subscribe to trade updates.

        Args:
            symbol: Trading pair symbol
            callback: Function to call with trade updates
        """
        pass

    @abstractmethod
    async def subscribe_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable[[OHLCV], None]
    ) -> None:
        """
        Subscribe to OHLCV/kline updates.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            callback: Function to call with candle updates
        """
        pass

    @abstractmethod
    async def subscribe_user_data(
        self,
        on_order_update: Optional[Callable[[Order], None]] = None,
        on_position_update: Optional[Callable[[Position], None]] = None,
        on_balance_update: Optional[Callable[[Balance], None]] = None,
    ) -> None:
        """
        Subscribe to user data updates (orders, positions, balance).

        Args:
            on_order_update: Callback for order updates
            on_position_update: Callback for position updates
            on_balance_update: Callback for balance updates
        """
        pass

    @abstractmethod
    async def unsubscribe(self, channel: str, symbol: Optional[str] = None) -> None:
        """
        Unsubscribe from a channel.

        Args:
            channel: Channel name (ticker, orderbook, trades, ohlcv, user)
            symbol: Symbol for market data channels
        """
        pass

    # ========== Utility Methods ==========

    @abstractmethod
    async def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get exchange/market information.

        Args:
            symbol: Specific symbol, or None for all

        Returns:
            Exchange and market information
        """
        pass

    @abstractmethod
    async def get_server_time(self) -> datetime:
        """
        Get exchange server time.

        Returns:
            Server timestamp
        """
        pass

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for an event.

        Args:
            event: Event name
            callback: Callback function
        """
        if event not in self._ws_callbacks:
            self._ws_callbacks[event] = []
        self._ws_callbacks[event].append(callback)

    def unregister_callback(self, event: str, callback: Callable) -> None:
        """
        Unregister a callback.

        Args:
            event: Event name
            callback: Callback function to remove
        """
        if event in self._ws_callbacks and callback in self._ws_callbacks[event]:
            self._ws_callbacks[event].remove(callback)

    async def _emit(self, event: str, data: Any) -> None:
        """
        Emit an event to registered callbacks.

        Args:
            event: Event name
            data: Data to pass to callbacks
        """
        if event in self._ws_callbacks:
            for callback in self._ws_callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for event {event}: {e}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(testnet={self.testnet})"
