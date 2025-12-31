"""
Event-Driven Backtesting Engine for Crypto Perpetual Futures Trading

Features:
- Event-driven architecture with support for multiple timeframes
- Proper leverage and margin handling for perpetual futures
- Funding rate simulation
- Slippage and commission modeling
- Comprehensive performance metrics
- Trade logging and analysis

Author: Generated for crypto_perps_strategy
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class EventType(Enum):
    """Types of events in the backtesting engine."""
    MARKET_DATA = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()
    FUNDING = auto()
    LIQUIDATION = auto()
    MARGIN_CALL = auto()


class OrderType(Enum):
    """Order types supported."""
    MARKET = auto()
    LIMIT = auto()
    STOP_MARKET = auto()
    TAKE_PROFIT_MARKET = auto()


class OrderSide(Enum):
    """Order side."""
    BUY = auto()
    SELL = auto()


class PositionSide(Enum):
    """Position side for perpetual futures."""
    LONG = auto()
    SHORT = auto()
    FLAT = auto()


class Timeframe(Enum):
    """Supported timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

    @classmethod
    def to_minutes(cls, tf: "Timeframe") -> int:
        """Convert timeframe to minutes."""
        mapping = {
            cls.M1: 1, cls.M5: 5, cls.M15: 15, cls.M30: 30,
            cls.H1: 60, cls.H4: 240, cls.D1: 1440, cls.W1: 10080
        }
        return mapping[tf]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Event:
    """Base event class."""
    event_type: EventType
    timestamp: pd.Timestamp
    symbol: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketDataEvent(Event):
    """Market data event containing OHLCV data."""
    timeframe: Timeframe = Timeframe.H1
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0

    def __post_init__(self):
        self.event_type = EventType.MARKET_DATA


@dataclass
class SignalEvent(Event):
    """Trading signal event."""
    signal_type: PositionSide = PositionSide.FLAT
    strength: float = 0.0
    target_leverage: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def __post_init__(self):
        self.event_type = EventType.SIGNAL


@dataclass
class OrderEvent(Event):
    """Order event."""
    order_type: OrderType = OrderType.MARKET
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    price: Optional[float] = None
    leverage: float = 1.0
    reduce_only: bool = False
    order_id: str = ""

    def __post_init__(self):
        self.event_type = EventType.ORDER


@dataclass
class FillEvent(Event):
    """Order fill event."""
    order_id: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0

    def __post_init__(self):
        self.event_type = EventType.FILL


@dataclass
class FundingEvent(Event):
    """Funding rate event for perpetual futures."""
    funding_rate: float = 0.0
    payment: float = 0.0

    def __post_init__(self):
        self.event_type = EventType.FUNDING


@dataclass
class Trade:
    """Represents a completed trade (round trip)."""
    trade_id: int
    symbol: str
    side: PositionSide
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    leverage: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage_cost: float
    funding_paid: float
    duration: pd.Timedelta
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    leverage: float
    entry_time: pd.Timestamp
    margin_used: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    liquidation_price: float = 0.0
    commission_paid: float = 0.0
    funding_paid: float = 0.0
    max_price: float = 0.0
    min_price: float = 0.0

    def __post_init__(self):
        self.max_price = self.entry_price
        self.min_price = self.entry_price


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    cagr: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # in bars
    volatility: float = 0.0
    downside_volatility: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_loss_ratio: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0

    # PnL breakdown
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_funding: float = 0.0

    # Additional metrics
    expectancy: float = 0.0
    sqn: float = 0.0  # System Quality Number
    recovery_factor: float = 0.0
    risk_reward_ratio: float = 0.0


# =============================================================================
# PROTOCOLS AND ABSTRACT CLASSES
# =============================================================================

class Strategy(Protocol):
    """Protocol for trading strategies."""

    def on_data(self, event: MarketDataEvent) -> Optional[SignalEvent]:
        """Process market data and generate signals."""
        ...

    def on_fill(self, event: FillEvent) -> None:
        """Handle fill events."""
        ...


class DataHandler(ABC):
    """Abstract base class for data handling."""

    @abstractmethod
    def get_next_bar(self) -> Optional[MarketDataEvent]:
        """Get the next bar of data."""
        pass

    @abstractmethod
    def get_latest_bars(self, n: int = 1) -> pd.DataFrame:
        """Get the latest n bars."""
        pass

    @abstractmethod
    def has_more_data(self) -> bool:
        """Check if there's more data to process."""
        pass


# =============================================================================
# DATA HANDLER IMPLEMENTATION
# =============================================================================

class MultiTimeframeDataHandler(DataHandler):
    """
    Handles multiple timeframe data for backtesting.
    Efficiently manages large datasets using pandas.
    """

    def __init__(
        self,
        data: Dict[str, Dict[str, pd.DataFrame]],
        primary_timeframe: Timeframe = Timeframe.H4,
    ):
        """
        Initialize the data handler.

        Args:
            data: Nested dict {symbol: {timeframe: DataFrame}}
                  DataFrames should have DatetimeIndex and OHLCV columns
            primary_timeframe: The main timeframe for iteration
        """
        self.data = data
        self.primary_timeframe = primary_timeframe
        self.symbols = list(data.keys())

        # Build unified timeline from primary timeframe
        self._build_timeline()

        self.current_idx = 0
        self._precompute_indices()

    def _build_timeline(self) -> None:
        """Build a unified timeline from all symbols' primary timeframe data."""
        all_timestamps = set()
        for symbol in self.symbols:
            tf_key = self.primary_timeframe.value
            if tf_key in self.data[symbol]:
                all_timestamps.update(self.data[symbol][tf_key].index)

        self.timeline = sorted(all_timestamps)
        self.total_bars = len(self.timeline)

    def _precompute_indices(self) -> None:
        """Precompute index mappings for efficient lookups."""
        self._index_maps: Dict[str, Dict[str, Dict[pd.Timestamp, int]]] = {}

        for symbol in self.symbols:
            self._index_maps[symbol] = {}
            for tf_key, df in self.data[symbol].items():
                self._index_maps[symbol][tf_key] = {
                    ts: idx for idx, ts in enumerate(df.index)
                }

    def get_next_bar(self) -> Optional[List[MarketDataEvent]]:
        """Get the next bar(s) of data for all symbols."""
        if not self.has_more_data():
            return None

        timestamp = self.timeline[self.current_idx]
        events = []

        for symbol in self.symbols:
            tf_key = self.primary_timeframe.value
            if tf_key not in self.data[symbol]:
                continue

            df = self.data[symbol][tf_key]
            if timestamp in self._index_maps[symbol][tf_key]:
                idx = self._index_maps[symbol][tf_key][timestamp]
                row = df.iloc[idx]

                event = MarketDataEvent(
                    event_type=EventType.MARKET_DATA,
                    timestamp=timestamp,
                    symbol=symbol,
                    timeframe=self.primary_timeframe,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row.get("volume", 0.0),
                )
                events.append(event)

        self.current_idx += 1
        return events if events else None

    def get_latest_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        n: int = 1,
        current_time: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Get the latest n bars for a symbol/timeframe up to current time.
        Efficient O(1) lookup using precomputed indices.
        """
        tf_key = timeframe.value
        if symbol not in self.data or tf_key not in self.data[symbol]:
            return pd.DataFrame()

        df = self.data[symbol][tf_key]

        if current_time is None:
            if self.current_idx > 0:
                current_time = self.timeline[self.current_idx - 1]
            else:
                return pd.DataFrame()

        # Binary search for the current time index
        mask = df.index <= current_time
        valid_df = df.loc[mask]

        if len(valid_df) == 0:
            return pd.DataFrame()

        return valid_df.tail(n).copy()

    def get_bar_at_time(
        self,
        symbol: str,
        timeframe: Timeframe,
        timestamp: pd.Timestamp,
    ) -> Optional[pd.Series]:
        """Get bar data at a specific timestamp."""
        tf_key = timeframe.value
        if symbol not in self.data or tf_key not in self.data[symbol]:
            return None

        if timestamp in self._index_maps[symbol][tf_key]:
            idx = self._index_maps[symbol][tf_key][timestamp]
            return self.data[symbol][tf_key].iloc[idx]
        return None

    def has_more_data(self) -> bool:
        """Check if there's more data to process."""
        return self.current_idx < self.total_bars

    def reset(self) -> None:
        """Reset the data handler to the beginning."""
        self.current_idx = 0


# =============================================================================
# EXECUTION ENGINE
# =============================================================================

class ExecutionEngine:
    """
    Handles order execution with realistic slippage and commission modeling.
    """

    def __init__(
        self,
        slippage_pct: float = 0.001,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0004,
        slippage_model: str = "percentage",  # 'percentage', 'volume_based', 'fixed'
    ):
        self.slippage_pct = slippage_pct
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_model = slippage_model

    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        side: OrderSide,
        volume: float = 0.0,
    ) -> float:
        """
        Calculate slippage based on the selected model.

        Returns the slippage amount (always positive).
        """
        if self.slippage_model == "percentage":
            return price * self.slippage_pct

        elif self.slippage_model == "volume_based":
            # Slippage increases with order size relative to volume
            if volume > 0:
                volume_ratio = (quantity * price) / volume
                # Quadratic slippage model
                slippage_mult = 1 + (volume_ratio ** 2) * 10
                return price * self.slippage_pct * slippage_mult
            return price * self.slippage_pct

        else:  # fixed
            return price * self.slippage_pct

    def calculate_commission(
        self,
        notional_value: float,
        order_type: OrderType,
    ) -> float:
        """Calculate commission based on order type."""
        if order_type == OrderType.LIMIT:
            return notional_value * self.maker_fee
        return notional_value * self.taker_fee

    def execute_order(
        self,
        order: OrderEvent,
        current_price: float,
        current_volume: float = 0.0,
    ) -> FillEvent:
        """
        Execute an order and return a fill event.
        """
        # Calculate slippage
        slippage = self.calculate_slippage(
            current_price,
            order.quantity,
            order.side,
            current_volume,
        )

        # Apply slippage based on order side
        if order.side == OrderSide.BUY:
            fill_price = current_price + slippage
        else:
            fill_price = current_price - slippage

        # For limit orders, check if price is favorable
        if order.order_type == OrderType.LIMIT and order.price is not None:
            if order.side == OrderSide.BUY and current_price > order.price:
                return None  # Order not filled
            if order.side == OrderSide.SELL and current_price < order.price:
                return None  # Order not filled
            fill_price = order.price  # Fill at limit price

        # Calculate commission
        notional_value = fill_price * order.quantity
        commission = self.calculate_commission(notional_value, order.order_type)

        return FillEvent(
            event_type=EventType.FILL,
            timestamp=order.timestamp,
            symbol=order.symbol,
            order_id=order.order_id,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage * order.quantity,
        )


# =============================================================================
# PORTFOLIO / POSITION MANAGER
# =============================================================================

class PortfolioManager:
    """
    Manages positions, margin, and P&L for perpetual futures trading.
    """

    def __init__(
        self,
        initial_capital: float,
        max_leverage: float = 10.0,
        maintenance_margin_rate: float = 0.005,  # 0.5%
        liquidation_fee_rate: float = 0.006,  # 0.6%
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_leverage = max_leverage
        self.maintenance_margin_rate = maintenance_margin_rate
        self.liquidation_fee_rate = liquidation_fee_rate

        # Positions: {symbol: Position}
        self.positions: Dict[str, Position] = {}

        # Trade history
        self.trades: List[Trade] = []
        self.trade_counter = 0

        # Equity tracking
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.cash_flow: List[Tuple[pd.Timestamp, str, float]] = []

        # Funding tracking
        self.total_funding_paid = 0.0

    @property
    def total_equity(self) -> float:
        """Calculate total equity (cash + unrealized P&L)."""
        unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.positions.values()
        )
        return self.cash + unrealized_pnl

    @property
    def total_margin_used(self) -> float:
        """Calculate total margin used by open positions."""
        return sum(pos.margin_used for pos in self.positions.values())

    @property
    def available_margin(self) -> float:
        """Calculate available margin for new positions."""
        return self.cash - self.total_margin_used

    @property
    def margin_ratio(self) -> float:
        """Calculate margin ratio (used margin / equity)."""
        if self.total_equity <= 0:
            return float("inf")
        return self.total_margin_used / self.total_equity

    def calculate_liquidation_price(
        self,
        entry_price: float,
        leverage: float,
        side: PositionSide,
    ) -> float:
        """
        Calculate liquidation price for a position.
        Liquidation occurs when margin ratio exceeds maintenance margin.
        """
        # Simplified liquidation price calculation
        # Actual exchanges have more complex formulas
        maintenance_rate = self.maintenance_margin_rate
        initial_margin_rate = 1 / leverage

        if side == PositionSide.LONG:
            liq_price = entry_price * (1 - initial_margin_rate + maintenance_rate)
        else:
            liq_price = entry_price * (1 + initial_margin_rate - maintenance_rate)

        return liq_price

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        entry_price: float,
        quantity: float,
        leverage: float,
        timestamp: pd.Timestamp,
        commission: float = 0.0,
    ) -> Optional[Position]:
        """Open a new position or add to existing."""
        notional_value = entry_price * quantity
        required_margin = notional_value / leverage

        # Check if we have enough margin
        if required_margin > self.available_margin:
            logger.warning(
                f"Insufficient margin for position. Required: {required_margin:.2f}, "
                f"Available: {self.available_margin:.2f}"
            )
            return None

        # Check if position already exists
        if symbol in self.positions:
            existing = self.positions[symbol]
            if existing.side == side:
                # Add to existing position (average in)
                total_quantity = existing.quantity + quantity
                avg_price = (
                    existing.entry_price * existing.quantity +
                    entry_price * quantity
                ) / total_quantity

                existing.entry_price = avg_price
                existing.quantity = total_quantity
                existing.margin_used += required_margin
                existing.commission_paid += commission
                existing.liquidation_price = self.calculate_liquidation_price(
                    avg_price, leverage, side
                )
                return existing
            else:
                # Close existing and potentially open opposite
                # This is handled as a close + open
                logger.warning(
                    f"Position flip not handled in open_position. "
                    f"Close existing position first."
                )
                return None

        # Create new position
        liquidation_price = self.calculate_liquidation_price(
            entry_price, leverage, side
        )

        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            leverage=leverage,
            entry_time=timestamp,
            margin_used=required_margin,
            liquidation_price=liquidation_price,
            commission_paid=commission,
        )

        self.positions[symbol] = position
        self.cash_flow.append((timestamp, "margin_lock", -required_margin))

        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        quantity: Optional[float] = None,
        timestamp: Optional[pd.Timestamp] = None,
        commission: float = 0.0,
    ) -> Optional[Trade]:
        """Close a position (fully or partially) and record the trade."""
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return None

        position = self.positions[symbol]

        # Determine quantity to close
        close_quantity = quantity if quantity else position.quantity
        close_quantity = min(close_quantity, position.quantity)

        # Calculate P&L
        if position.side == PositionSide.LONG:
            pnl = (exit_price - position.entry_price) * close_quantity
        else:
            pnl = (position.entry_price - exit_price) * close_quantity

        # Apply leverage to P&L (this is already the actual P&L for futures)
        # The leverage affects the position size, not the P&L calculation

        # Calculate P&L percentage (on margin used)
        margin_portion = position.margin_used * (close_quantity / position.quantity)
        pnl_pct = pnl / margin_portion if margin_portion > 0 else 0

        # Create trade record
        self.trade_counter += 1
        trade = Trade(
            trade_id=self.trade_counter,
            symbol=symbol,
            side=position.side,
            entry_time=position.entry_time,
            exit_time=timestamp or pd.Timestamp.now(),
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=close_quantity,
            leverage=position.leverage,
            pnl=pnl - commission - (position.commission_paid * close_quantity / position.quantity),
            pnl_pct=pnl_pct,
            commission=commission + (position.commission_paid * close_quantity / position.quantity),
            slippage_cost=0.0,  # Tracked separately
            funding_paid=position.funding_paid * (close_quantity / position.quantity),
            duration=timestamp - position.entry_time if timestamp else pd.Timedelta(0),
            max_favorable_excursion=(
                (position.max_price - position.entry_price) / position.entry_price
                if position.side == PositionSide.LONG
                else (position.entry_price - position.min_price) / position.entry_price
            ),
            max_adverse_excursion=(
                (position.entry_price - position.min_price) / position.entry_price
                if position.side == PositionSide.LONG
                else (position.max_price - position.entry_price) / position.entry_price
            ),
        )

        self.trades.append(trade)

        # Update cash and position
        self.cash += margin_portion + pnl - commission
        self.cash_flow.append((timestamp, "pnl", pnl))
        self.cash_flow.append((timestamp, "commission", -commission))

        if close_quantity >= position.quantity:
            # Fully closed
            del self.positions[symbol]
        else:
            # Partially closed
            position.quantity -= close_quantity
            position.margin_used -= margin_portion
            position.funding_paid *= (position.quantity / (position.quantity + close_quantity))

        return trade

    def update_positions(
        self,
        current_prices: Dict[str, float],
        timestamp: pd.Timestamp,
    ) -> List[str]:
        """
        Update unrealized P&L and check for liquidations.
        Returns list of liquidated symbols.
        """
        liquidated = []

        for symbol, position in list(self.positions.items()):
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]

            # Update price extremes
            position.max_price = max(position.max_price, price)
            position.min_price = min(position.min_price, price)

            # Calculate unrealized P&L
            if position.side == PositionSide.LONG:
                position.unrealized_pnl = (price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - price) * position.quantity

            # Check for liquidation
            if position.side == PositionSide.LONG and price <= position.liquidation_price:
                logger.warning(f"LIQUIDATION: {symbol} LONG at {price:.2f}")
                self._liquidate_position(symbol, price, timestamp)
                liquidated.append(symbol)
            elif position.side == PositionSide.SHORT and price >= position.liquidation_price:
                logger.warning(f"LIQUIDATION: {symbol} SHORT at {price:.2f}")
                self._liquidate_position(symbol, price, timestamp)
                liquidated.append(symbol)

        # Record equity
        self.equity_curve.append((timestamp, self.total_equity))

        return liquidated

    def _liquidate_position(
        self,
        symbol: str,
        price: float,
        timestamp: pd.Timestamp,
    ) -> None:
        """Handle position liquidation."""
        position = self.positions[symbol]

        # Liquidation fee
        notional = price * position.quantity
        liquidation_fee = notional * self.liquidation_fee_rate

        # Close at liquidation price with fee
        self.close_position(
            symbol=symbol,
            exit_price=price,
            timestamp=timestamp,
            commission=liquidation_fee,
        )

    def apply_funding(
        self,
        symbol: str,
        funding_rate: float,
        mark_price: float,
        timestamp: pd.Timestamp,
    ) -> Optional[FundingEvent]:
        """
        Apply funding payment to a position.
        Positive funding rate: longs pay shorts
        Negative funding rate: shorts pay longs
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        notional_value = mark_price * position.quantity

        # Calculate funding payment
        if position.side == PositionSide.LONG:
            payment = -notional_value * funding_rate  # Longs pay when positive
        else:
            payment = notional_value * funding_rate  # Shorts receive when positive

        # Apply to cash and position tracking
        self.cash += payment
        position.funding_paid -= payment  # Track as paid, so negative if we receive
        self.total_funding_paid -= payment

        self.cash_flow.append((timestamp, "funding", payment))

        return FundingEvent(
            event_type=EventType.FUNDING,
            timestamp=timestamp,
            symbol=symbol,
            funding_rate=funding_rate,
            payment=payment,
        )

    def get_equity_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame(columns=["timestamp", "equity"])

        df = pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])
        df.set_index("timestamp", inplace=True)
        return df

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "side": t.side.name,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "leverage": t.leverage,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "commission": t.commission,
                "slippage_cost": t.slippage_cost,
                "funding_paid": t.funding_paid,
                "duration": t.duration,
                "mfe": t.max_favorable_excursion,
                "mae": t.max_adverse_excursion,
            })

        return pd.DataFrame(records)


# =============================================================================
# FUNDING RATE SIMULATOR
# =============================================================================

class FundingRateSimulator:
    """
    Simulates funding rate payments for perpetual futures.
    Funding is typically paid every 8 hours on most exchanges.
    """

    def __init__(
        self,
        funding_interval_hours: int = 8,
        base_rate: float = 0.0001,  # 0.01% base rate
        historical_rates: Optional[pd.DataFrame] = None,
    ):
        self.funding_interval_hours = funding_interval_hours
        self.base_rate = base_rate
        self.historical_rates = historical_rates
        self._last_funding_times: Dict[str, pd.Timestamp] = {}

    def get_funding_rate(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
    ) -> Optional[float]:
        """
        Get funding rate at a specific timestamp.
        Returns rate if funding should be applied, None otherwise.
        """
        # Check if it's time for funding
        if symbol not in self._last_funding_times:
            self._last_funding_times[symbol] = timestamp
            return None

        hours_since_last = (
            timestamp - self._last_funding_times[symbol]
        ).total_seconds() / 3600

        if hours_since_last < self.funding_interval_hours:
            return None

        self._last_funding_times[symbol] = timestamp

        # Use historical rates if available
        if self.historical_rates is not None:
            if symbol in self.historical_rates.columns:
                # Find the closest rate
                mask = self.historical_rates.index <= timestamp
                if mask.any():
                    return self.historical_rates.loc[mask, symbol].iloc[-1]

        # Otherwise, return base rate (can be enhanced with simulation)
        return self.base_rate

    def reset(self) -> None:
        """Reset funding state."""
        self._last_funding_times.clear()


# =============================================================================
# PERFORMANCE CALCULATOR
# =============================================================================

class PerformanceCalculator:
    """
    Calculates comprehensive performance metrics.
    """

    TRADING_DAYS_PER_YEAR = 365  # Crypto trades 24/7

    @classmethod
    def calculate_metrics(
        cls,
        equity_curve: pd.DataFrame,
        trades: List[Trade],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 365 * 24,  # Hourly data
    ) -> PerformanceMetrics:
        """Calculate all performance metrics."""
        metrics = PerformanceMetrics()

        if equity_curve.empty or len(trades) == 0:
            return metrics

        # Ensure equity is properly formatted
        if isinstance(equity_curve, pd.DataFrame):
            equity = equity_curve["equity"].values
        else:
            equity = equity_curve.values

        # Calculate returns
        returns = pd.Series(equity).pct_change().dropna()

        # Total return
        metrics.total_return = (equity[-1] - equity[0]) / equity[0]

        # Annualized return
        n_periods = len(equity)
        years = n_periods / periods_per_year
        if years > 0:
            metrics.annualized_return = (1 + metrics.total_return) ** (1 / years) - 1
            metrics.cagr = metrics.annualized_return

        # Volatility
        if len(returns) > 1:
            metrics.volatility = returns.std() * np.sqrt(periods_per_year)

            # Downside volatility (for Sortino)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                metrics.downside_volatility = downside_returns.std() * np.sqrt(periods_per_year)

        # Sharpe Ratio
        if metrics.volatility > 0:
            excess_return = metrics.annualized_return - risk_free_rate
            metrics.sharpe_ratio = excess_return / metrics.volatility

        # Sortino Ratio
        if metrics.downside_volatility > 0:
            excess_return = metrics.annualized_return - risk_free_rate
            metrics.sortino_ratio = excess_return / metrics.downside_volatility

        # Maximum Drawdown
        metrics.max_drawdown, metrics.max_drawdown_duration = cls._calculate_max_drawdown(equity)

        # Calmar Ratio
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)

        # Trade statistics
        cls._calculate_trade_stats(metrics, trades)

        # Recovery Factor
        if metrics.max_drawdown > 0:
            metrics.recovery_factor = metrics.net_profit / abs(metrics.max_drawdown * equity[0])

        return metrics

    @classmethod
    def _calculate_max_drawdown(
        cls,
        equity: np.ndarray,
    ) -> Tuple[float, int]:
        """Calculate maximum drawdown and its duration."""
        peak = equity[0]
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_start = 0

        for i, value in enumerate(equity):
            if value > peak:
                peak = value
                current_dd_start = i
            else:
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
                    max_dd_duration = i - current_dd_start

        return max_dd, max_dd_duration

    @classmethod
    def _calculate_trade_stats(
        cls,
        metrics: PerformanceMetrics,
        trades: List[Trade],
    ) -> None:
        """Calculate trade-level statistics."""
        if not trades:
            return

        pnls = [t.pnl for t in trades]
        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p < 0]

        metrics.total_trades = len(trades)
        metrics.winning_trades = len(winning_pnls)
        metrics.losing_trades = len(losing_pnls)

        # Win rate
        metrics.win_rate = metrics.winning_trades / metrics.total_trades

        # Profit factor
        metrics.gross_profit = sum(winning_pnls) if winning_pnls else 0
        metrics.gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0
        if metrics.gross_loss > 0:
            metrics.profit_factor = metrics.gross_profit / metrics.gross_loss
        else:
            metrics.profit_factor = float("inf") if metrics.gross_profit > 0 else 0

        # Average win/loss
        metrics.avg_win = np.mean(winning_pnls) if winning_pnls else 0
        metrics.avg_loss = abs(np.mean(losing_pnls)) if losing_pnls else 0
        if metrics.avg_loss > 0:
            metrics.avg_win_loss_ratio = metrics.avg_win / metrics.avg_loss

        # Largest win/loss
        metrics.largest_win = max(winning_pnls) if winning_pnls else 0
        metrics.largest_loss = abs(min(losing_pnls)) if losing_pnls else 0

        # Net profit
        metrics.net_profit = sum(pnls)

        # Total costs
        metrics.total_commission = sum(t.commission for t in trades)
        metrics.total_slippage = sum(t.slippage_cost for t in trades)
        metrics.total_funding = sum(t.funding_paid for t in trades)

        # Average trade duration
        durations = [t.duration.total_seconds() / 3600 for t in trades]  # In hours
        metrics.avg_trade_duration = np.mean(durations) if durations else 0

        # Expectancy
        metrics.expectancy = (
            metrics.win_rate * metrics.avg_win -
            (1 - metrics.win_rate) * metrics.avg_loss
        )

        # System Quality Number (SQN)
        if len(pnls) > 0:
            pnl_std = np.std(pnls)
            if pnl_std > 0:
                metrics.sqn = np.sqrt(len(pnls)) * np.mean(pnls) / pnl_std

        # Risk-Reward Ratio
        if metrics.avg_loss > 0:
            metrics.risk_reward_ratio = metrics.avg_win / metrics.avg_loss


# =============================================================================
# MAIN BACKTESTING ENGINE
# =============================================================================

class BacktestEngine:
    """
    Main event-driven backtesting engine for perpetual futures.
    """

    def __init__(
        self,
        data_handler: MultiTimeframeDataHandler,
        strategy: Strategy,
        initial_capital: float = 10000.0,
        max_leverage: float = 10.0,
        slippage_pct: float = 0.001,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0004,
        funding_interval_hours: int = 8,
        base_funding_rate: float = 0.0001,
        historical_funding_rates: Optional[pd.DataFrame] = None,
    ):
        self.data_handler = data_handler
        self.strategy = strategy
        self.initial_capital = initial_capital  # Store for position sizing

        # Initialize components
        self.portfolio = PortfolioManager(
            initial_capital=initial_capital,
            max_leverage=max_leverage,
        )

        self.execution = ExecutionEngine(
            slippage_pct=slippage_pct,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
        )

        self.funding_simulator = FundingRateSimulator(
            funding_interval_hours=funding_interval_hours,
            base_rate=base_funding_rate,
            historical_rates=historical_funding_rates,
        )

        # Event queue
        self.events: List[Event] = []

        # State tracking
        self.current_prices: Dict[str, float] = {}
        self.current_volumes: Dict[str, float] = {}
        self.order_counter = 0

        # Performance
        self.metrics: Optional[PerformanceMetrics] = None

        # Logging
        self.event_log: List[Dict[str, Any]] = []
        self.verbose = False

    def run(self, verbose: bool = False) -> PerformanceMetrics:
        """
        Run the backtest.

        Args:
            verbose: If True, print progress and events

        Returns:
            PerformanceMetrics object with results
        """
        self.verbose = verbose

        if verbose:
            logger.info("Starting backtest...")
            logger.info(f"Initial capital: ${self.portfolio.initial_capital:,.2f}")

        # Reset components
        self.data_handler.reset()
        self.funding_simulator.reset()

        bar_count = 0
        total_bars = self.data_handler.total_bars

        # Main event loop
        while self.data_handler.has_more_data():
            # Get next market data events
            market_events = self.data_handler.get_next_bar()

            if market_events is None:
                continue

            bar_count += 1

            if verbose and bar_count % 1000 == 0:
                progress = bar_count / total_bars * 100
                logger.info(
                    f"Progress: {progress:.1f}% | "
                    f"Equity: ${self.portfolio.total_equity:,.2f}"
                )

            # Process each market event
            for event in market_events:
                self._process_market_event(event)

        # Calculate final metrics
        # Determine periods per year based on timeframe
        tf_minutes = Timeframe.to_minutes(self.data_handler.primary_timeframe)
        periods_per_year = (365 * 24 * 60) // tf_minutes  # Crypto trades 24/7

        self.metrics = PerformanceCalculator.calculate_metrics(
            equity_curve=self.portfolio.get_equity_df(),
            trades=self.portfolio.trades,
            periods_per_year=periods_per_year,
        )

        if verbose:
            self._print_summary()

        return self.metrics

    def _process_market_event(self, event: MarketDataEvent) -> None:
        """Process a market data event."""
        # Update current prices
        self.current_prices[event.symbol] = event.close
        self.current_volumes[event.symbol] = event.volume

        # Log event
        self._log_event(event)

        # Update positions and check for liquidations
        liquidated = self.portfolio.update_positions(
            self.current_prices,
            event.timestamp,
        )

        # Apply funding if due
        self._apply_funding(event)

        # Generate trading signal
        signal = self.strategy.on_data(event)

        if signal is not None:
            self._process_signal(signal, event)

    def _apply_funding(self, event: MarketDataEvent) -> None:
        """Apply funding rate payments."""
        funding_rate = self.funding_simulator.get_funding_rate(
            event.symbol,
            event.timestamp,
        )

        if funding_rate is not None:
            funding_event = self.portfolio.apply_funding(
                symbol=event.symbol,
                funding_rate=funding_rate,
                mark_price=event.close,
                timestamp=event.timestamp,
            )

            if funding_event:
                self._log_event(funding_event)

    def _process_signal(
        self,
        signal: SignalEvent,
        market_event: MarketDataEvent,
    ) -> None:
        """Process a trading signal and generate orders."""
        self._log_event(signal)

        symbol = signal.symbol
        current_price = self.current_prices[symbol]

        # Check if we have an existing position
        has_position = symbol in self.portfolio.positions
        current_position = self.portfolio.positions.get(symbol)

        # Determine action based on signal and current position
        if signal.signal_type == PositionSide.FLAT:
            # Close any existing position
            if has_position:
                self._create_close_order(
                    symbol, current_position, market_event.timestamp
                )

        elif signal.signal_type == PositionSide.LONG:
            if has_position:
                if current_position.side == PositionSide.SHORT:
                    # Close short and open long
                    self._create_close_order(
                        symbol, current_position, market_event.timestamp
                    )
                    self._create_open_order(
                        symbol, signal, market_event.timestamp, current_price
                    )
                # If already long, we could add to position (not implemented here)
            else:
                self._create_open_order(
                    symbol, signal, market_event.timestamp, current_price
                )

        elif signal.signal_type == PositionSide.SHORT:
            if has_position:
                if current_position.side == PositionSide.LONG:
                    # Close long and open short
                    self._create_close_order(
                        symbol, current_position, market_event.timestamp
                    )
                    self._create_open_order(
                        symbol, signal, market_event.timestamp, current_price
                    )
                # If already short, we could add to position (not implemented here)
            else:
                self._create_open_order(
                    symbol, signal, market_event.timestamp, current_price
                )

    def _create_open_order(
        self,
        symbol: str,
        signal: SignalEvent,
        timestamp: pd.Timestamp,
        current_price: float,
    ) -> None:
        """Create an order to open a position."""
        # Calculate position size based on risk management
        leverage = min(signal.target_leverage, self.portfolio.max_leverage)

        # CRITICAL: Use INITIAL capital for position sizing to prevent exponential growth
        # This ensures position sizes stay reasonable regardless of equity curve
        base_capital = self.initial_capital

        # Position sizing parameters (conservative for perpetual futures)
        risk_per_trade = 0.02  # Risk 2% of initial capital per trade
        max_position_margin = base_capital * 0.15  # Max 15% of initial capital as margin per position

        # Calculate target margin based on risk
        target_margin = base_capital * risk_per_trade

        # Ensure we don't exceed available margin
        available = self.portfolio.available_margin
        if available <= 0:
            return

        # Use the smaller of target margin and what's available (with buffer)
        margin_to_use = min(target_margin, max_position_margin, available * 0.80)

        if margin_to_use <= 0:
            return

        # Notional value we can take
        max_notional = margin_to_use * leverage
        quantity = max_notional / current_price

        # Ensure minimum notional value
        if max_notional < 100:  # Min $100 position notional
            return

        # Create order
        self.order_counter += 1
        order = OrderEvent(
            event_type=EventType.ORDER,
            timestamp=timestamp,
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY if signal.signal_type == PositionSide.LONG else OrderSide.SELL,
            quantity=quantity,
            leverage=leverage,
            order_id=f"O{self.order_counter:06d}",
        )

        self._execute_order(order, signal.signal_type)

    def _create_close_order(
        self,
        symbol: str,
        position: Position,
        timestamp: pd.Timestamp,
    ) -> None:
        """Create an order to close a position."""
        self.order_counter += 1
        order = OrderEvent(
            event_type=EventType.ORDER,
            timestamp=timestamp,
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
            quantity=position.quantity,
            leverage=position.leverage,
            reduce_only=True,
            order_id=f"O{self.order_counter:06d}",
        )

        self._execute_order(order)

    def _execute_order(
        self,
        order: OrderEvent,
        new_position_side: Optional[PositionSide] = None,
    ) -> None:
        """Execute an order through the execution engine."""
        self._log_event(order)

        current_price = self.current_prices[order.symbol]
        current_volume = self.current_volumes.get(order.symbol, 0)

        # Execute order
        fill = self.execution.execute_order(order, current_price, current_volume)

        if fill is None:
            return  # Order not filled (e.g., limit order conditions not met)

        self._log_event(fill)

        # Update portfolio
        if order.reduce_only:
            # Close position
            self.portfolio.close_position(
                symbol=order.symbol,
                exit_price=fill.fill_price,
                quantity=fill.quantity,
                timestamp=fill.timestamp,
                commission=fill.commission,
            )
        else:
            # Open new position
            if new_position_side:
                self.portfolio.open_position(
                    symbol=order.symbol,
                    side=new_position_side,
                    entry_price=fill.fill_price,
                    quantity=fill.quantity,
                    leverage=order.leverage,
                    timestamp=fill.timestamp,
                    commission=fill.commission,
                )

        # Notify strategy
        self.strategy.on_fill(fill)

    def _log_event(self, event: Event) -> None:
        """Log an event."""
        log_entry = {
            "timestamp": event.timestamp,
            "event_type": event.event_type.name,
            "symbol": event.symbol,
        }

        if isinstance(event, MarketDataEvent):
            log_entry.update({
                "open": event.open,
                "high": event.high,
                "low": event.low,
                "close": event.close,
                "volume": event.volume,
            })
        elif isinstance(event, SignalEvent):
            log_entry.update({
                "signal_type": event.signal_type.name,
                "strength": event.strength,
                "leverage": event.target_leverage,
            })
        elif isinstance(event, OrderEvent):
            log_entry.update({
                "order_type": event.order_type.name,
                "side": event.side.name,
                "quantity": event.quantity,
                "order_id": event.order_id,
            })
        elif isinstance(event, FillEvent):
            log_entry.update({
                "fill_price": event.fill_price,
                "quantity": event.quantity,
                "commission": event.commission,
                "slippage": event.slippage,
            })
        elif isinstance(event, FundingEvent):
            log_entry.update({
                "funding_rate": event.funding_rate,
                "payment": event.payment,
            })

        self.event_log.append(log_entry)

    def _print_summary(self) -> None:
        """Print backtest summary."""
        m = self.metrics

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        print(f"\n--- Returns ---")
        print(f"Total Return:      {m.total_return * 100:>10.2f}%")
        print(f"Annualized Return: {m.annualized_return * 100:>10.2f}%")
        print(f"CAGR:              {m.cagr * 100:>10.2f}%")

        print(f"\n--- Risk Metrics ---")
        print(f"Sharpe Ratio:      {m.sharpe_ratio:>10.2f}")
        print(f"Sortino Ratio:     {m.sortino_ratio:>10.2f}")
        print(f"Calmar Ratio:      {m.calmar_ratio:>10.2f}")
        print(f"Max Drawdown:      {m.max_drawdown * 100:>10.2f}%")
        print(f"Volatility:        {m.volatility * 100:>10.2f}%")

        print(f"\n--- Trade Statistics ---")
        print(f"Total Trades:      {m.total_trades:>10d}")
        print(f"Win Rate:          {m.win_rate * 100:>10.2f}%")
        print(f"Profit Factor:     {m.profit_factor:>10.2f}")
        print(f"Avg Win:           ${m.avg_win:>9.2f}")
        print(f"Avg Loss:          ${m.avg_loss:>9.2f}")
        print(f"Win/Loss Ratio:    {m.avg_win_loss_ratio:>10.2f}")
        print(f"Expectancy:        ${m.expectancy:>9.2f}")
        print(f"SQN:               {m.sqn:>10.2f}")

        print(f"\n--- Costs ---")
        print(f"Total Commission:  ${m.total_commission:>9.2f}")
        print(f"Total Slippage:    ${m.total_slippage:>9.2f}")
        print(f"Total Funding:     ${m.total_funding:>9.2f}")
        print(f"Net Profit:        ${m.net_profit:>9.2f}")

        print("\n" + "=" * 60)

    def get_event_log_df(self) -> pd.DataFrame:
        """Get event log as DataFrame."""
        return pd.DataFrame(self.event_log)

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve."""
        return self.portfolio.get_equity_df()

    def get_trades(self) -> pd.DataFrame:
        """Get trade history."""
        return self.portfolio.get_trades_df()


# =============================================================================
# EXAMPLE STRATEGY (For demonstration)
# =============================================================================

class SimpleMovingAverageCrossStrategy:
    """
    Simple MA crossover strategy for demonstration.
    """

    def __init__(
        self,
        data_handler: MultiTimeframeDataHandler,
        fast_period: int = 12,
        slow_period: int = 26,
        leverage: float = 5.0,
    ):
        self.data_handler = data_handler
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.leverage = leverage

        # Track previous signal to avoid duplicate signals
        self._last_signal: Dict[str, PositionSide] = {}

    def on_data(self, event: MarketDataEvent) -> Optional[SignalEvent]:
        """Generate trading signal based on MA crossover."""
        # Get enough data for calculation
        bars = self.data_handler.get_latest_bars(
            symbol=event.symbol,
            timeframe=event.timeframe,
            n=self.slow_period + 1,
            current_time=event.timestamp,
        )

        if len(bars) < self.slow_period:
            return None

        # Calculate MAs
        closes = bars["close"].values
        fast_ma = np.mean(closes[-self.fast_period:])
        slow_ma = np.mean(closes[-self.slow_period:])

        # Previous values
        prev_fast_ma = np.mean(closes[-self.fast_period - 1:-1])
        prev_slow_ma = np.mean(closes[-self.slow_period - 1:-1])

        # Detect crossover
        signal_type = PositionSide.FLAT
        strength = 0.0

        # Bullish crossover
        if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
            signal_type = PositionSide.LONG
            strength = (fast_ma - slow_ma) / slow_ma

        # Bearish crossover
        elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
            signal_type = PositionSide.SHORT
            strength = (slow_ma - fast_ma) / slow_ma

        # Only emit signal on change
        last_signal = self._last_signal.get(event.symbol, PositionSide.FLAT)
        if signal_type != PositionSide.FLAT and signal_type != last_signal:
            self._last_signal[event.symbol] = signal_type
            return SignalEvent(
                event_type=EventType.SIGNAL,
                timestamp=event.timestamp,
                symbol=event.symbol,
                signal_type=signal_type,
                strength=abs(strength),
                target_leverage=self.leverage,
            )

        # Close signal when flat
        if signal_type == PositionSide.FLAT and last_signal != PositionSide.FLAT:
            self._last_signal[event.symbol] = PositionSide.FLAT
            return SignalEvent(
                event_type=EventType.SIGNAL,
                timestamp=event.timestamp,
                symbol=event.symbol,
                signal_type=PositionSide.FLAT,
                strength=0.0,
                target_leverage=1.0,
            )

        return None

    def on_fill(self, event: FillEvent) -> None:
        """Handle fill events."""
        pass  # Can be used for tracking or adjustments


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_sample_data(
    symbols: List[str] = None,
    timeframes: List[str] = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Create sample OHLCV data for testing.
    In production, you would load actual historical data.
    """
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT"]

    if timeframes is None:
        timeframes = ["4h", "1h"]

    np.random.seed(seed)

    data = {}

    for symbol in symbols:
        data[symbol] = {}

        # Starting prices
        base_price = 50000 if "BTC" in symbol else 3000

        for tf in timeframes:
            # Determine number of bars
            # Use lowercase time aliases (pandas 2.0+ compatible)
            freq_map = {
                "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                "1h": "1h", "4h": "4h", "1d": "1D", "1w": "1W"
            }
            freq = freq_map.get(tf, "1h")

            dates = pd.date_range(start=start_date, end=end_date, freq=freq)
            n_bars = len(dates)

            # Generate random walk for close prices
            returns = np.random.normal(0.0002, 0.02, n_bars)
            prices = base_price * np.cumprod(1 + returns)

            # Generate OHLC from close
            high_mult = 1 + np.abs(np.random.normal(0.005, 0.003, n_bars))
            low_mult = 1 - np.abs(np.random.normal(0.005, 0.003, n_bars))

            opens = np.roll(prices, 1)
            opens[0] = base_price

            df = pd.DataFrame({
                "open": opens,
                "high": prices * high_mult,
                "low": prices * low_mult,
                "close": prices,
                "volume": np.random.lognormal(20, 1, n_bars),
            }, index=dates)

            # Ensure OHLC consistency
            df["high"] = df[["open", "high", "close"]].max(axis=1)
            df["low"] = df[["open", "low", "close"]].min(axis=1)

            data[symbol][tf] = df

    return data


def run_backtest_example():
    """
    Example of running a backtest with the engine.
    """
    # Create sample data
    print("Generating sample data...")
    data = create_sample_data(
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframes=["4h"],
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    # Initialize data handler
    data_handler = MultiTimeframeDataHandler(
        data=data,
        primary_timeframe=Timeframe.H4,
    )

    # Initialize strategy
    strategy = SimpleMovingAverageCrossStrategy(
        data_handler=data_handler,
        fast_period=12,
        slow_period=26,
        leverage=5.0,
    )

    # Initialize backtest engine
    engine = BacktestEngine(
        data_handler=data_handler,
        strategy=strategy,
        initial_capital=10000.0,
        max_leverage=10.0,
        slippage_pct=0.001,
        maker_fee=0.0002,
        taker_fee=0.0004,
        funding_interval_hours=8,
        base_funding_rate=0.0001,
    )

    # Run backtest
    metrics = engine.run(verbose=True)

    # Get additional data
    trades_df = engine.get_trades()
    equity_df = engine.get_equity_curve()

    print(f"\nTrades DataFrame shape: {trades_df.shape}")
    print(f"Equity curve length: {len(equity_df)}")

    return engine, metrics


if __name__ == "__main__":
    engine, metrics = run_backtest_example()
