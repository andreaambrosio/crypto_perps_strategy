"""
Risk Management Module for Crypto Perpetual Futures Trading.

This module provides comprehensive risk management functionality including:
- Position sizing (Kelly Criterion, volatility-adjusted)
- Dynamic stop-loss and take-profit calculation using ATR
- Portfolio-level risk limits
- Position correlation analysis
- Trailing stop logic
- Risk-adjusted return metrics (Sharpe, Sortino, Calmar)
"""

from .risk_manager import (
    # Core classes
    RiskManager,
    RiskMetrics,
    RiskLimits,
    Position,
    PositionSizeResult,
    # Enums
    RiskLevel,
    PositionSide,
    # Factory functions
    create_risk_manager,
    quick_position_size,
)

__all__ = [
    "RiskManager",
    "RiskMetrics",
    "RiskLimits",
    "Position",
    "PositionSizeResult",
    "RiskLevel",
    "PositionSide",
    "create_risk_manager",
    "quick_position_size",
]

__version__ = "1.0.0"
