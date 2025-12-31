"""
Configuration for the Crypto Perpetual Futures Trading Strategy

OPTIMIZED PARAMETERS for maximum risk-adjusted returns.
Based on backtesting across 2024 crypto market conditions.
"""
from dataclasses import dataclass
from typing import List

@dataclass
class TradingConfig:
    # Trading pairs - Focus on high-liquidity perpetuals
    symbols: List[str] = None

    # Timeframes - 4h provides best signal quality for crypto
    primary_timeframe: str = "4h"
    secondary_timeframe: str = "1h"

    # OPTIMIZED EMA PARAMETERS
    # 8/21 is faster than default 12/26, better for crypto volatility
    ema_fast: int = 8
    ema_slow: int = 21

    # RSI PARAMETERS - Widened bands for crypto
    rsi_period: int = 14
    rsi_overbought: float = 75  # Higher for crypto (runs longer)
    rsi_oversold: float = 25    # Lower for crypto (runs longer)

    # ATR PARAMETERS
    atr_period: int = 14
    atr_multiplier: float = 2.5

    # ADX PARAMETERS
    adx_period: int = 14
    adx_threshold: float = 20  # Lower threshold catches more trends

    # Momentum parameters
    momentum_period: int = 10
    momentum_threshold: float = 0.02

    # Volume filter
    volume_ma_period: int = 20
    volume_multiplier: float = 1.2  # Lower = more trades

    # OPTIMIZED RISK MANAGEMENT
    max_position_size_pct: float = 0.02  # 2% risk per trade
    max_total_exposure: float = 0.30     # 30% max exposure
    stop_loss_atr_mult: float = 1.5      # Tighter stops
    take_profit_atr_mult: float = 3.0    # 2:1 risk/reward minimum

    # Max drawdown trigger
    max_drawdown_pct: float = 0.10  # 10% max drawdown

    # LEVERAGE - Conservative for sustainability
    default_leverage: int = 3  # Lower leverage = lower risk
    max_leverage: int = 5

    # Execution
    slippage_pct: float = 0.001
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004

    # Signal confidence threshold
    min_confidence: float = 0.55  # Lower = more trades

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]


@dataclass
class BacktestConfig:
    initial_capital: float = 10000.0
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    commission_pct: float = 0.0004


# Default configurations
DEFAULT_CONFIG = TradingConfig()
DEFAULT_BACKTEST_CONFIG = BacktestConfig()
