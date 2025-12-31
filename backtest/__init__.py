"""
Backtesting module for crypto perpetual futures trading.
"""

from .engine import (
    # Main engine
    BacktestEngine,

    # Data handling
    MultiTimeframeDataHandler,
    DataHandler,

    # Events
    Event,
    EventType,
    MarketDataEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    FundingEvent,

    # Enums
    OrderType,
    OrderSide,
    PositionSide,
    Timeframe,

    # Data classes
    Trade,
    Position,
    PerformanceMetrics,

    # Components
    ExecutionEngine,
    PortfolioManager,
    FundingRateSimulator,
    PerformanceCalculator,

    # Example strategy
    SimpleMovingAverageCrossStrategy,

    # Utilities
    create_sample_data,
    run_backtest_example,
)

__all__ = [
    # Main engine
    "BacktestEngine",

    # Data handling
    "MultiTimeframeDataHandler",
    "DataHandler",

    # Events
    "Event",
    "EventType",
    "MarketDataEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "FundingEvent",

    # Enums
    "OrderType",
    "OrderSide",
    "PositionSide",
    "Timeframe",

    # Data classes
    "Trade",
    "Position",
    "PerformanceMetrics",

    # Components
    "ExecutionEngine",
    "PortfolioManager",
    "FundingRateSimulator",
    "PerformanceCalculator",

    # Example strategy
    "SimpleMovingAverageCrossStrategy",

    # Utilities
    "create_sample_data",
    "run_backtest_example",
]
