"""
Abstract Base Strategy for Crypto Perpetual Futures Trading

This module provides the foundational abstract base class for all trading strategies.
It defines the interface and common functionality that all strategy implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np


class SignalType(Enum):
    """Trading signal types for perpetual futures."""
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    HOLD = "HOLD"
    NO_SIGNAL = "NO_SIGNAL"


class TimeFrame(Enum):
    """Supported timeframes for analysis."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


@dataclass
class Signal:
    """
    Trading signal with metadata and confidence scoring.

    Attributes:
        signal_type: The type of trading signal (LONG, SHORT, CLOSE, etc.)
        symbol: Trading pair symbol
        timestamp: When the signal was generated
        confidence: Confidence score from 0.0 to 1.0
        entry_price: Suggested entry price
        stop_loss: Suggested stop loss price
        take_profit: Suggested take profit price
        timeframe: The timeframe this signal was generated on
        metadata: Additional signal information
    """
    signal_type: SignalType
    symbol: str
    timestamp: datetime
    confidence: float = 0.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe: TimeFrame = TimeFrame.H4
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal parameters."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

    @property
    def is_entry_signal(self) -> bool:
        """Check if this is an entry signal."""
        return self.signal_type in (SignalType.LONG, SignalType.SHORT)

    @property
    def is_exit_signal(self) -> bool:
        """Check if this is an exit signal."""
        return self.signal_type in (SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT)

    @property
    def is_actionable(self) -> bool:
        """Check if this signal requires action."""
        return self.signal_type not in (SignalType.HOLD, SignalType.NO_SIGNAL)

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk/reward ratio if stop loss and take profit are set."""
        if all([self.entry_price, self.stop_loss, self.take_profit]):
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.take_profit - self.entry_price)
            if risk > 0:
                return reward / risk
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            "signal_type": self.signal_type.value,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timeframe": self.timeframe.value,
            "risk_reward_ratio": self.risk_reward_ratio,
            "metadata": self.metadata
        }


@dataclass
class IndicatorValues:
    """
    Container for indicator values at a specific point in time.

    This class stores computed indicator values for easy access
    and passing between strategy components.
    """
    timestamp: datetime
    # Trend indicators
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    sma: Optional[float] = None

    # Momentum indicators
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None

    # Volatility indicators
    atr: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_middle: Optional[float] = None

    # Trend strength
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None

    # Volume
    volume: Optional[float] = None
    volume_sma: Optional[float] = None
    volume_ratio: Optional[float] = None

    # Supertrend
    supertrend: Optional[float] = None
    supertrend_direction: Optional[int] = None  # 1 for bullish, -1 for bearish

    # Price
    close: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    This class defines the interface that all strategy implementations must follow.
    It provides common functionality for indicator calculation, signal generation,
    and multi-timeframe analysis.

    Attributes:
        name: Strategy name for identification
        config: Strategy configuration parameters
        indicators: Dictionary of computed indicators by timeframe
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base strategy.

        Args:
            name: Unique name for the strategy
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._indicators_cache: Dict[str, Dict[TimeFrame, pd.DataFrame]] = {}
        self._last_signals: Dict[str, Signal] = {}

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame, timeframe: TimeFrame) -> pd.DataFrame:
        """
        Calculate all required indicators for the strategy.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            timeframe: The timeframe of the data

        Returns:
            DataFrame with additional indicator columns
        """
        pass

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: TimeFrame,
        current_position: Optional[str] = None
    ) -> Signal:
        """
        Generate a trading signal based on current market data.

        Args:
            df: DataFrame with OHLCV data and calculated indicators
            symbol: Trading pair symbol
            timeframe: The timeframe of the analysis
            current_position: Current position state ("LONG", "SHORT", or None)

        Returns:
            Signal object with trading recommendation
        """
        pass

    @abstractmethod
    def get_entry_conditions(self, indicators: IndicatorValues) -> Tuple[bool, bool, float]:
        """
        Evaluate entry conditions based on indicator values.

        Args:
            indicators: Current indicator values

        Returns:
            Tuple of (long_condition, short_condition, confidence_score)
        """
        pass

    @abstractmethod
    def get_exit_conditions(
        self,
        indicators: IndicatorValues,
        position_type: str,
        entry_price: float
    ) -> Tuple[bool, str]:
        """
        Evaluate exit conditions for an existing position.

        Args:
            indicators: Current indicator values
            position_type: "LONG" or "SHORT"
            entry_price: The entry price of the position

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        pass

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        signal_type: SignalType,
        multiplier: float = 2.0
    ) -> float:
        """
        Calculate dynamic stop loss based on ATR.

        Args:
            entry_price: The entry price
            atr: Current ATR value
            signal_type: LONG or SHORT
            multiplier: ATR multiplier for stop distance

        Returns:
            Stop loss price
        """
        stop_distance = atr * multiplier

        if signal_type == SignalType.LONG:
            return entry_price - stop_distance
        elif signal_type == SignalType.SHORT:
            return entry_price + stop_distance
        else:
            raise ValueError(f"Invalid signal type for stop loss: {signal_type}")

    def calculate_take_profit(
        self,
        entry_price: float,
        atr: float,
        signal_type: SignalType,
        multiplier: float = 3.0
    ) -> float:
        """
        Calculate dynamic take profit based on ATR.

        Args:
            entry_price: The entry price
            atr: Current ATR value
            signal_type: LONG or SHORT
            multiplier: ATR multiplier for target distance

        Returns:
            Take profit price
        """
        target_distance = atr * multiplier

        if signal_type == SignalType.LONG:
            return entry_price + target_distance
        elif signal_type == SignalType.SHORT:
            return entry_price - target_distance
        else:
            raise ValueError(f"Invalid signal type for take profit: {signal_type}")

    def calculate_confidence_score(self, factors: Dict[str, Tuple[bool, float]]) -> float:
        """
        Calculate overall confidence score from multiple factors.

        Args:
            factors: Dictionary of factor_name -> (condition_met, weight)

        Returns:
            Weighted confidence score between 0 and 1
        """
        total_weight = sum(weight for _, weight in factors.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            weight if condition else 0
            for condition, weight in factors.values()
        )

        return weighted_sum / total_weight

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame has required columns.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, raises ValueError otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if df.empty:
            raise ValueError("DataFrame is empty")

        return True

    def get_latest_indicators(self, df: pd.DataFrame) -> IndicatorValues:
        """
        Extract the latest indicator values from a DataFrame.

        Args:
            df: DataFrame with indicator columns

        Returns:
            IndicatorValues object with latest values
        """
        if df.empty:
            raise ValueError("DataFrame is empty")

        latest = df.iloc[-1]
        timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()

        return IndicatorValues(
            timestamp=timestamp,
            ema_fast=latest.get('ema_fast'),
            ema_slow=latest.get('ema_slow'),
            sma=latest.get('sma'),
            rsi=latest.get('rsi'),
            macd=latest.get('macd'),
            macd_signal=latest.get('macd_signal'),
            macd_histogram=latest.get('macd_histogram'),
            atr=latest.get('atr'),
            bollinger_upper=latest.get('bb_upper'),
            bollinger_lower=latest.get('bb_lower'),
            bollinger_middle=latest.get('bb_middle'),
            adx=latest.get('adx'),
            plus_di=latest.get('plus_di'),
            minus_di=latest.get('minus_di'),
            volume=latest.get('volume'),
            volume_sma=latest.get('volume_sma'),
            volume_ratio=latest.get('volume_ratio'),
            supertrend=latest.get('supertrend'),
            supertrend_direction=latest.get('supertrend_direction'),
            close=latest.get('close'),
            high=latest.get('high'),
            low=latest.get('low'),
            open=latest.get('open')
        )

    def multi_timeframe_analysis(
        self,
        data: Dict[TimeFrame, pd.DataFrame],
        symbol: str,
        primary_tf: TimeFrame,
        secondary_tf: TimeFrame
    ) -> Signal:
        """
        Perform multi-timeframe analysis combining signals from different timeframes.

        Args:
            data: Dictionary mapping timeframes to DataFrames
            symbol: Trading pair symbol
            primary_tf: Primary timeframe for trend direction
            secondary_tf: Secondary timeframe for entry timing

        Returns:
            Combined Signal object
        """
        if primary_tf not in data or secondary_tf not in data:
            raise ValueError("Required timeframe data not provided")

        # Calculate indicators for both timeframes
        primary_df = self.calculate_indicators(data[primary_tf], primary_tf)
        secondary_df = self.calculate_indicators(data[secondary_tf], secondary_tf)

        # Generate signals from both timeframes
        primary_signal = self.generate_signal(primary_df, symbol, primary_tf)
        secondary_signal = self.generate_signal(secondary_df, symbol, secondary_tf)

        # Combine signals - primary determines direction, secondary confirms entry
        return self._combine_signals(primary_signal, secondary_signal)

    def _combine_signals(self, primary: Signal, secondary: Signal) -> Signal:
        """
        Combine signals from multiple timeframes.

        Args:
            primary: Signal from primary (higher) timeframe
            secondary: Signal from secondary (lower) timeframe

        Returns:
            Combined Signal with adjusted confidence
        """
        # If primary has no signal, return no signal
        if primary.signal_type in (SignalType.NO_SIGNAL, SignalType.HOLD):
            return primary

        # If signals align, boost confidence
        if primary.signal_type == secondary.signal_type:
            combined_confidence = min(1.0, (primary.confidence + secondary.confidence) / 1.5)
        # If secondary confirms direction but not exact signal
        elif (primary.signal_type == SignalType.LONG and
              secondary.signal_type in (SignalType.LONG, SignalType.HOLD)):
            combined_confidence = primary.confidence
        elif (primary.signal_type == SignalType.SHORT and
              secondary.signal_type in (SignalType.SHORT, SignalType.HOLD)):
            combined_confidence = primary.confidence
        # If signals conflict, reduce confidence
        else:
            combined_confidence = primary.confidence * 0.5

        return Signal(
            signal_type=primary.signal_type,
            symbol=primary.symbol,
            timestamp=secondary.timestamp,  # Use more recent timestamp
            confidence=combined_confidence,
            entry_price=secondary.entry_price or primary.entry_price,
            stop_loss=primary.stop_loss,
            take_profit=primary.take_profit,
            timeframe=primary.timeframe,
            metadata={
                "primary_confidence": primary.confidence,
                "secondary_confidence": secondary.confidence,
                "signals_aligned": primary.signal_type == secondary.signal_type,
                **primary.metadata
            }
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
