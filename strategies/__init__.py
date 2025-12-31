"""
Crypto Perpetual Futures Trading Strategies

This module contains the core trading strategies for perpetual futures trading.
"""

from .base_strategy import (
    BaseStrategy,
    Signal,
    SignalType,
    TimeFrame,
    IndicatorValues
)

from .trend_momentum_strategy import (
    TrendMomentumStrategy,
    StrategyConfig
)

__all__ = [
    # Base classes and types
    'BaseStrategy',
    'Signal',
    'SignalType',
    'TimeFrame',
    'IndicatorValues',

    # Strategies
    'TrendMomentumStrategy',
    'StrategyConfig',
]
