"""
Comprehensive Risk Management Module for Crypto Perpetual Futures Trading.

This module provides production-ready risk management functionality including:
- Position sizing using Kelly Criterion and volatility-adjusted methods
- Dynamic stop-loss and take-profit calculation using ATR
- Portfolio-level risk limits (max drawdown, max exposure)
- Position correlation analysis
- Trailing stop logic
- Risk-adjusted returns calculation (Sharpe, Sortino, Calmar)

Author: Crypto Perps Strategy Team
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class RiskLevel(Enum):
    """Enumeration of risk levels for position management."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PositionSide(Enum):
    """Position direction."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """Represents an open trading position."""
    symbol: str
    side: PositionSide
    entry_price: float
    size: float  # Position size in base currency
    leverage: int
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    unrealized_pnl: float = 0.0

    @property
    def notional_value(self) -> float:
        """Calculate notional value of the position."""
        return self.size * self.entry_price

    @property
    def margin_required(self) -> float:
        """Calculate required margin for the position."""
        return self.notional_value / self.leverage


@dataclass
class RiskMetrics:
    """Container for calculated risk metrics."""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # in periods
    volatility: float
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    cvar_95: float  # Conditional VaR 95%
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float


@dataclass
class RiskLimits:
    """Configuration for portfolio risk limits."""
    max_position_size_pct: float = 0.10  # 10% per position
    max_total_exposure_pct: float = 0.50  # 50% total exposure
    max_single_asset_exposure_pct: float = 0.20  # 20% per asset
    max_correlation_threshold: float = 0.70  # Max allowed correlation
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    max_daily_loss_pct: float = 0.05  # 5% daily loss limit
    max_leverage: int = 10
    min_risk_reward_ratio: float = 1.5
    max_positions: int = 10


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    recommended_size: float
    max_allowed_size: float
    kelly_fraction: float
    volatility_adjusted_size: float
    risk_per_trade: float
    position_value: float
    margin_required: float
    leverage_used: int


class RiskManager:
    """
    Production-ready Risk Manager for crypto perpetual futures trading.

    This class provides comprehensive risk management functionality including
    position sizing, stop-loss/take-profit calculation, portfolio risk limits,
    and risk-adjusted performance metrics.

    Attributes:
        portfolio_value: Current total portfolio value in quote currency
        risk_limits: Configuration for risk limits
        positions: Dictionary of current open positions
        equity_curve: Historical equity values for risk calculations

    Example:
        >>> risk_manager = RiskManager(portfolio_value=100000.0)
        >>> size = risk_manager.calculate_position_size(
        ...     symbol="BTC/USDT",
        ...     side=PositionSide.LONG,
        ...     entry_price=50000.0,
        ...     stop_loss_price=48000.0,
        ...     win_rate=0.55,
        ...     avg_win_loss_ratio=1.5,
        ...     current_volatility=0.02
        ... )
        >>> print(f"Recommended size: {size.recommended_size}")
    """

    # Risk-free rate for Sharpe/Sortino calculations (annualized)
    RISK_FREE_RATE: float = 0.05  # 5% annual
    TRADING_DAYS_PER_YEAR: int = 365  # Crypto trades 24/7
    HOURS_PER_DAY: int = 24

    def __init__(
        self,
        portfolio_value: float,
        risk_limits: Optional[RiskLimits] = None,
        risk_free_rate: float = 0.05
    ) -> None:
        """
        Initialize the Risk Manager.

        Args:
            portfolio_value: Initial portfolio value in quote currency (e.g., USDT)
            risk_limits: Optional custom risk limits configuration
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.portfolio_value = portfolio_value
        self.risk_limits = risk_limits or RiskLimits()
        self.risk_free_rate = risk_free_rate
        self.positions: Dict[str, Position] = {}
        self.equity_curve: pd.Series = pd.Series(dtype=float)
        self.daily_pnl: List[float] = []
        self.trade_history: List[Dict] = []
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._last_correlation_update: Optional[datetime] = None

    # ========== Position Sizing Methods ==========

    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win_loss_ratio: float,
        fractional_kelly: float = 0.25
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.

        The Kelly Criterion determines the optimal fraction of capital to risk
        on each trade based on edge and odds. We use fractional Kelly to reduce
        volatility while maintaining positive expectancy.

        Formula: f* = (p * b - q) / b
        Where:
            f* = Kelly fraction
            p = probability of winning
            q = probability of losing (1 - p)
            b = win/loss ratio (average win / average loss)

        Args:
            win_rate: Historical win rate (0.0 to 1.0)
            avg_win_loss_ratio: Ratio of average win to average loss
            fractional_kelly: Fraction of Kelly to use (0.25 = quarter Kelly)

        Returns:
            Optimal fraction of capital to risk (0.0 to max_position_size_pct)

        Raises:
            ValueError: If win_rate is not between 0 and 1
        """
        if not 0.0 <= win_rate <= 1.0:
            raise ValueError(f"Win rate must be between 0 and 1, got {win_rate}")

        if avg_win_loss_ratio <= 0:
            raise ValueError(f"Win/loss ratio must be positive, got {avg_win_loss_ratio}")

        p = win_rate
        q = 1 - p
        b = avg_win_loss_ratio

        # Kelly formula
        kelly_fraction = (p * b - q) / b

        # Apply fractional Kelly and ensure non-negative
        adjusted_kelly = max(0.0, kelly_fraction * fractional_kelly)

        # Cap at maximum position size
        return min(adjusted_kelly, self.risk_limits.max_position_size_pct)

    def calculate_volatility_adjusted_size(
        self,
        current_volatility: float,
        target_volatility: float = 0.02,
        base_position_pct: float = 0.05
    ) -> float:
        """
        Calculate position size adjusted for current market volatility.

        Higher volatility leads to smaller position sizes to maintain
        consistent risk per trade.

        Args:
            current_volatility: Current price volatility (e.g., daily returns std)
            target_volatility: Target volatility level (default 2%)
            base_position_pct: Base position size at target volatility

        Returns:
            Volatility-adjusted position size as fraction of portfolio
        """
        if current_volatility <= 0:
            raise ValueError(f"Volatility must be positive, got {current_volatility}")

        # Inverse scaling: higher volatility = smaller position
        volatility_scalar = target_volatility / current_volatility

        # Apply scaling with bounds
        adjusted_size = base_position_pct * volatility_scalar

        # Ensure within limits
        return np.clip(
            adjusted_size,
            0.01,  # Minimum 1% position
            self.risk_limits.max_position_size_pct
        )

    def calculate_position_size(
        self,
        symbol: str,
        side: PositionSide,
        entry_price: float,
        stop_loss_price: float,
        win_rate: float,
        avg_win_loss_ratio: float,
        current_volatility: float,
        leverage: int = 1,
        use_kelly: bool = True,
        fractional_kelly: float = 0.25
    ) -> PositionSizeResult:
        """
        Calculate comprehensive position size considering multiple factors.

        This method combines Kelly Criterion, volatility adjustment, and
        portfolio constraints to determine optimal position size.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            side: Position direction (LONG or SHORT)
            entry_price: Expected entry price
            stop_loss_price: Stop-loss price level
            win_rate: Historical win rate for this strategy
            avg_win_loss_ratio: Average win to loss ratio
            current_volatility: Current price volatility
            leverage: Leverage to use for position
            use_kelly: Whether to use Kelly Criterion
            fractional_kelly: Fraction of Kelly to use

        Returns:
            PositionSizeResult with recommended size and constraints

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if entry_price <= 0 or stop_loss_price <= 0:
            raise ValueError("Prices must be positive")

        if leverage < 1 or leverage > self.risk_limits.max_leverage:
            raise ValueError(
                f"Leverage must be between 1 and {self.risk_limits.max_leverage}"
            )

        # Calculate risk per trade (distance to stop loss)
        if side == PositionSide.LONG:
            risk_per_unit = (entry_price - stop_loss_price) / entry_price
        else:
            risk_per_unit = (stop_loss_price - entry_price) / entry_price

        if risk_per_unit <= 0:
            raise ValueError("Stop loss must be in valid direction for position side")

        # Calculate Kelly-based size
        if use_kelly:
            kelly_fraction = self.calculate_kelly_criterion(
                win_rate, avg_win_loss_ratio, fractional_kelly
            )
        else:
            kelly_fraction = self.risk_limits.max_position_size_pct

        # Calculate volatility-adjusted size
        vol_adjusted_size = self.calculate_volatility_adjusted_size(
            current_volatility=current_volatility,
            base_position_pct=kelly_fraction
        )

        # Calculate size based on risk per trade
        # We want to risk no more than a certain % of portfolio
        max_risk_pct = min(kelly_fraction, vol_adjusted_size)
        risk_per_trade = self.portfolio_value * max_risk_pct

        # Position size in base currency units
        position_value = risk_per_trade / risk_per_unit
        position_size = position_value / entry_price

        # Apply leverage adjustment
        margin_required = position_value / leverage

        # Check portfolio constraints
        max_allowed_value = self._get_max_allowed_position_value(symbol, leverage)

        if position_value > max_allowed_value:
            position_value = max_allowed_value
            position_size = position_value / entry_price
            margin_required = position_value / leverage

        return PositionSizeResult(
            recommended_size=position_size,
            max_allowed_size=max_allowed_value / entry_price,
            kelly_fraction=kelly_fraction,
            volatility_adjusted_size=vol_adjusted_size,
            risk_per_trade=risk_per_trade,
            position_value=position_value,
            margin_required=margin_required,
            leverage_used=leverage
        )

    def _get_max_allowed_position_value(
        self,
        symbol: str,
        leverage: int
    ) -> float:
        """
        Calculate maximum allowed position value considering all constraints.

        Args:
            symbol: Trading symbol
            leverage: Intended leverage

        Returns:
            Maximum allowed position notional value
        """
        # Single position limit
        max_single = self.portfolio_value * self.risk_limits.max_position_size_pct

        # Check existing exposure
        current_exposure = self._calculate_current_exposure()
        remaining_exposure = (
            self.portfolio_value * self.risk_limits.max_total_exposure_pct
            - current_exposure
        )

        # Check asset-specific exposure
        asset_exposure = self._get_asset_exposure(symbol)
        remaining_asset_exposure = (
            self.portfolio_value * self.risk_limits.max_single_asset_exposure_pct
            - asset_exposure
        )

        # Return most restrictive limit (considering leverage)
        return min(max_single, remaining_exposure, remaining_asset_exposure) * leverage

    def _calculate_current_exposure(self) -> float:
        """Calculate total current portfolio exposure."""
        return sum(pos.margin_required for pos in self.positions.values())

    def _get_asset_exposure(self, symbol: str) -> float:
        """Get current exposure for a specific asset."""
        base_asset = symbol.split("/")[0]
        exposure = 0.0
        for sym, pos in self.positions.items():
            if sym.startswith(base_asset):
                exposure += pos.margin_required
        return exposure

    # ========== Stop Loss / Take Profit Methods ==========

    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR) for volatility measurement.

        ATR is used for dynamic stop-loss and take-profit placement.

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: ATR lookback period

        Returns:
            Series of ATR values
        """
        if len(high) < period + 1:
            raise ValueError(f"Need at least {period + 1} periods for ATR calculation")

        # Calculate True Range components
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))

        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is the smoothed average of True Range
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    def calculate_dynamic_stop_loss(
        self,
        entry_price: float,
        side: PositionSide,
        atr: float,
        atr_multiplier: float = 2.0,
        min_distance_pct: float = 0.005,
        max_distance_pct: float = 0.10
    ) -> float:
        """
        Calculate dynamic stop-loss price based on ATR.

        Uses ATR to adapt stop-loss distance to current volatility.
        Higher volatility = wider stop to avoid premature exits.

        Args:
            entry_price: Position entry price
            side: Position direction
            atr: Current ATR value
            atr_multiplier: Multiplier for ATR distance
            min_distance_pct: Minimum stop distance as percentage
            max_distance_pct: Maximum stop distance as percentage

        Returns:
            Stop-loss price level
        """
        # Calculate ATR-based distance
        atr_distance = atr * atr_multiplier
        atr_distance_pct = atr_distance / entry_price

        # Clamp distance to bounds
        distance_pct = np.clip(atr_distance_pct, min_distance_pct, max_distance_pct)
        distance = entry_price * distance_pct

        if side == PositionSide.LONG:
            stop_loss = entry_price - distance
        else:
            stop_loss = entry_price + distance

        return round(stop_loss, self._get_price_precision(entry_price))

    def calculate_dynamic_take_profit(
        self,
        entry_price: float,
        side: PositionSide,
        atr: float,
        atr_multiplier: float = 3.0,
        min_rr_ratio: float = 1.5,
        stop_loss_price: Optional[float] = None
    ) -> float:
        """
        Calculate dynamic take-profit price based on ATR and risk-reward ratio.

        Ensures minimum risk-reward ratio is maintained while adapting to volatility.

        Args:
            entry_price: Position entry price
            side: Position direction
            atr: Current ATR value
            atr_multiplier: Multiplier for ATR distance
            min_rr_ratio: Minimum risk-reward ratio
            stop_loss_price: Optional stop-loss for R:R calculation

        Returns:
            Take-profit price level
        """
        # Calculate ATR-based target
        atr_distance = atr * atr_multiplier

        # If stop loss provided, ensure minimum R:R ratio
        if stop_loss_price is not None:
            stop_distance = abs(entry_price - stop_loss_price)
            min_profit_distance = stop_distance * min_rr_ratio
            atr_distance = max(atr_distance, min_profit_distance)

        if side == PositionSide.LONG:
            take_profit = entry_price + atr_distance
        else:
            take_profit = entry_price - atr_distance

        return round(take_profit, self._get_price_precision(entry_price))

    def calculate_stop_take_profit_levels(
        self,
        entry_price: float,
        side: PositionSide,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        atr_period: int = 14,
        stop_atr_mult: float = 2.0,
        tp_atr_mult: float = 3.0
    ) -> Tuple[float, float]:
        """
        Calculate both stop-loss and take-profit levels using ATR.

        Convenience method that calculates both levels together.

        Args:
            entry_price: Position entry price
            side: Position direction
            high: High price series
            low: Low price series
            close: Close price series
            atr_period: ATR calculation period
            stop_atr_mult: ATR multiplier for stop-loss
            tp_atr_mult: ATR multiplier for take-profit

        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        atr_series = self.calculate_atr(high, low, close, atr_period)
        current_atr = atr_series.iloc[-1]

        stop_loss = self.calculate_dynamic_stop_loss(
            entry_price, side, current_atr, stop_atr_mult
        )

        take_profit = self.calculate_dynamic_take_profit(
            entry_price, side, current_atr, tp_atr_mult,
            min_rr_ratio=self.risk_limits.min_risk_reward_ratio,
            stop_loss_price=stop_loss
        )

        return stop_loss, take_profit

    @staticmethod
    def _get_price_precision(price: float) -> int:
        """Determine appropriate decimal precision based on price magnitude."""
        if price >= 10000:
            return 1
        elif price >= 100:
            return 2
        elif price >= 1:
            return 4
        else:
            return 6

    # ========== Trailing Stop Methods ==========

    def initialize_trailing_stop(
        self,
        position: Position,
        current_price: float,
        trailing_distance: float,
        activation_profit_pct: float = 0.01
    ) -> Position:
        """
        Initialize trailing stop for a position.

        The trailing stop is activated when position reaches a minimum profit
        level and then follows price, locking in gains.

        Args:
            position: The position to add trailing stop to
            current_price: Current market price
            trailing_distance: Distance for trailing stop (in price units)
            activation_profit_pct: Minimum profit % to activate trailing

        Returns:
            Updated position with trailing stop parameters
        """
        position.trailing_stop_distance = trailing_distance

        # Calculate if we should activate based on current profit
        if position.side == PositionSide.LONG:
            profit_pct = (current_price - position.entry_price) / position.entry_price
            if profit_pct >= activation_profit_pct:
                position.trailing_stop_price = current_price - trailing_distance
        else:
            profit_pct = (position.entry_price - current_price) / position.entry_price
            if profit_pct >= activation_profit_pct:
                position.trailing_stop_price = current_price + trailing_distance

        return position

    def update_trailing_stop(
        self,
        position: Position,
        current_price: float
    ) -> Tuple[Position, bool]:
        """
        Update trailing stop based on current price movement.

        For LONG positions: raises stop as price increases
        For SHORT positions: lowers stop as price decreases

        Args:
            position: Position with active trailing stop
            current_price: Current market price

        Returns:
            Tuple of (updated_position, should_exit)
        """
        if position.trailing_stop_distance is None:
            return position, False

        should_exit = False
        distance = position.trailing_stop_distance

        if position.side == PositionSide.LONG:
            # Check if stopped out
            if (position.trailing_stop_price is not None and
                current_price <= position.trailing_stop_price):
                should_exit = True
            else:
                # Update trailing stop if price moved higher
                new_stop = current_price - distance
                if (position.trailing_stop_price is None or
                    new_stop > position.trailing_stop_price):
                    position.trailing_stop_price = new_stop
        else:  # SHORT
            # Check if stopped out
            if (position.trailing_stop_price is not None and
                current_price >= position.trailing_stop_price):
                should_exit = True
            else:
                # Update trailing stop if price moved lower
                new_stop = current_price + distance
                if (position.trailing_stop_price is None or
                    new_stop < position.trailing_stop_price):
                    position.trailing_stop_price = new_stop

        return position, should_exit

    def calculate_trailing_distance_atr(
        self,
        atr: float,
        atr_multiplier: float = 2.5
    ) -> float:
        """
        Calculate trailing stop distance based on ATR.

        Args:
            atr: Current ATR value
            atr_multiplier: Multiplier for ATR distance

        Returns:
            Trailing stop distance in price units
        """
        return atr * atr_multiplier

    # ========== Portfolio Risk Limits ==========

    def check_portfolio_risk_limits(self) -> Dict[str, bool]:
        """
        Check all portfolio-level risk limits.

        Returns a dictionary with the status of each risk check.
        True = limit OK, False = limit breached.

        Returns:
            Dictionary mapping limit names to pass/fail status
        """
        results = {}

        # Check max positions
        results["max_positions"] = len(self.positions) < self.risk_limits.max_positions

        # Check total exposure
        current_exposure_pct = self._calculate_current_exposure() / self.portfolio_value
        results["max_exposure"] = (
            current_exposure_pct <= self.risk_limits.max_total_exposure_pct
        )

        # Check drawdown
        if len(self.equity_curve) > 0:
            current_drawdown = self._calculate_current_drawdown()
            results["max_drawdown"] = (
                current_drawdown <= self.risk_limits.max_drawdown_pct
            )
        else:
            results["max_drawdown"] = True

        # Check daily loss
        if len(self.daily_pnl) > 0:
            today_pnl_pct = self.daily_pnl[-1] / self.portfolio_value
            results["max_daily_loss"] = (
                today_pnl_pct >= -self.risk_limits.max_daily_loss_pct
            )
        else:
            results["max_daily_loss"] = True

        return results

    def can_open_new_position(
        self,
        symbol: str,
        position_value: float,
        leverage: int = 1
    ) -> Tuple[bool, List[str]]:
        """
        Check if a new position can be opened within risk limits.

        Args:
            symbol: Trading symbol
            position_value: Notional value of intended position
            leverage: Intended leverage

        Returns:
            Tuple of (can_open, list_of_violations)
        """
        violations = []

        # Check risk limits first
        risk_checks = self.check_portfolio_risk_limits()
        for check, passed in risk_checks.items():
            if not passed:
                violations.append(f"Portfolio limit breached: {check}")

        # Check if adding position would exceed limits
        margin_required = position_value / leverage
        new_total_exposure = self._calculate_current_exposure() + margin_required

        if new_total_exposure > self.portfolio_value * self.risk_limits.max_total_exposure_pct:
            violations.append("Would exceed max total exposure")

        # Check asset-specific exposure
        asset_exposure = self._get_asset_exposure(symbol) + margin_required
        if asset_exposure > self.portfolio_value * self.risk_limits.max_single_asset_exposure_pct:
            violations.append(f"Would exceed max exposure for {symbol.split('/')[0]}")

        # Check correlation with existing positions
        if self._correlation_matrix is not None:
            base_asset = symbol.split("/")[0]
            for existing_symbol in self.positions:
                existing_base = existing_symbol.split("/")[0]
                if base_asset in self._correlation_matrix.columns and \
                   existing_base in self._correlation_matrix.index:
                    correlation = abs(
                        self._correlation_matrix.loc[existing_base, base_asset]
                    )
                    if correlation > self.risk_limits.max_correlation_threshold:
                        violations.append(
                            f"High correlation ({correlation:.2f}) with {existing_symbol}"
                        )

        return len(violations) == 0, violations

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from equity curve."""
        if len(self.equity_curve) == 0:
            return 0.0

        peak = self.equity_curve.expanding().max()
        drawdown = (peak - self.equity_curve) / peak
        return drawdown.iloc[-1]

    def get_risk_level(self) -> RiskLevel:
        """
        Determine current portfolio risk level based on multiple factors.

        Returns:
            RiskLevel enumeration indicating current risk state
        """
        risk_checks = self.check_portfolio_risk_limits()

        # Count violations
        violations = sum(1 for passed in risk_checks.values() if not passed)

        if violations == 0:
            # Check if approaching limits
            current_dd = self._calculate_current_drawdown()
            if current_dd > self.risk_limits.max_drawdown_pct * 0.8:
                return RiskLevel.HIGH
            elif current_dd > self.risk_limits.max_drawdown_pct * 0.5:
                return RiskLevel.MEDIUM
            return RiskLevel.LOW
        elif violations == 1:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    # ========== Correlation Analysis ==========

    def update_correlation_matrix(
        self,
        returns_data: pd.DataFrame,
        lookback_periods: int = 30
    ) -> pd.DataFrame:
        """
        Update the correlation matrix for portfolio assets.

        Args:
            returns_data: DataFrame with asset returns (columns = assets)
            lookback_periods: Number of periods for correlation calculation

        Returns:
            Updated correlation matrix
        """
        if len(returns_data) < lookback_periods:
            lookback_periods = len(returns_data)

        recent_returns = returns_data.tail(lookback_periods)
        self._correlation_matrix = recent_returns.corr()
        self._last_correlation_update = datetime.now()

        return self._correlation_matrix

    def check_position_correlation(
        self,
        symbol: str,
        returns_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Check correlation of new position with existing positions.

        Args:
            symbol: Symbol to check
            returns_data: DataFrame with returns for correlation calculation

        Returns:
            Dictionary mapping existing position symbols to correlation values
        """
        if len(self.positions) == 0:
            return {}

        # Update correlation matrix if needed
        if (self._correlation_matrix is None or
            self._last_correlation_update is None or
            (datetime.now() - self._last_correlation_update).seconds > 3600):
            self.update_correlation_matrix(returns_data)

        correlations = {}
        base_asset = symbol.split("/")[0]

        if self._correlation_matrix is not None and base_asset in self._correlation_matrix.columns:
            for existing_symbol in self.positions:
                existing_base = existing_symbol.split("/")[0]
                if existing_base in self._correlation_matrix.index:
                    correlations[existing_symbol] = self._correlation_matrix.loc[
                        existing_base, base_asset
                    ]

        return correlations

    def get_portfolio_correlation_risk(self) -> float:
        """
        Calculate overall portfolio correlation risk score.

        Returns a value between 0 and 1, where higher values indicate
        more correlated (riskier) portfolio.

        Returns:
            Correlation risk score (0-1)
        """
        if self._correlation_matrix is None or len(self.positions) < 2:
            return 0.0

        position_symbols = list(self.positions.keys())
        base_assets = [s.split("/")[0] for s in position_symbols]

        # Get pairwise correlations
        correlations = []
        for i, asset1 in enumerate(base_assets):
            for asset2 in base_assets[i+1:]:
                if (asset1 in self._correlation_matrix.index and
                    asset2 in self._correlation_matrix.columns):
                    corr = abs(self._correlation_matrix.loc[asset1, asset2])
                    correlations.append(corr)

        if len(correlations) == 0:
            return 0.0

        # Return average absolute correlation
        return np.mean(correlations)

    # ========== Risk-Adjusted Returns ==========

    def calculate_returns(
        self,
        prices: pd.Series,
        method: str = "log"
    ) -> pd.Series:
        """
        Calculate returns from price series.

        Args:
            prices: Series of prices
            method: Return calculation method ("log" or "simple")

        Returns:
            Series of returns
        """
        if method == "log":
            return np.log(prices / prices.shift(1)).dropna()
        else:
            return prices.pct_change().dropna()

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = None
    ) -> float:
        """
        Calculate annualized Sharpe Ratio.

        Sharpe Ratio = (Mean Return - Risk Free Rate) / Std Dev

        Args:
            returns: Series of periodic returns
            periods_per_year: Number of periods per year for annualization

        Returns:
            Annualized Sharpe Ratio
        """
        if periods_per_year is None:
            periods_per_year = self.TRADING_DAYS_PER_YEAR * self.HOURS_PER_DAY

        if len(returns) < 2:
            return 0.0

        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return 0.0

        # Annualize
        annualized_return = mean_return * periods_per_year
        annualized_std = std_return * np.sqrt(periods_per_year)

        sharpe = (annualized_return - self.risk_free_rate) / annualized_std

        return sharpe

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = None,
        target_return: float = 0.0
    ) -> float:
        """
        Calculate annualized Sortino Ratio.

        Sortino Ratio uses downside deviation instead of standard deviation,
        penalizing only negative volatility.

        Sortino = (Mean Return - Target Return) / Downside Deviation

        Args:
            returns: Series of periodic returns
            periods_per_year: Number of periods per year for annualization
            target_return: Minimum acceptable return (default 0)

        Returns:
            Annualized Sortino Ratio
        """
        if periods_per_year is None:
            periods_per_year = self.TRADING_DAYS_PER_YEAR * self.HOURS_PER_DAY

        if len(returns) < 2:
            return 0.0

        mean_return = returns.mean()

        # Calculate downside deviation (only negative returns below target)
        downside_returns = returns[returns < target_return]

        if len(downside_returns) == 0:
            # No downside returns - infinite Sortino (cap at large value)
            return 10.0

        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))

        if downside_deviation == 0:
            return 10.0

        # Annualize
        annualized_return = mean_return * periods_per_year
        annualized_dd = downside_deviation * np.sqrt(periods_per_year)

        sortino = (annualized_return - self.risk_free_rate) / annualized_dd

        return sortino

    def calculate_calmar_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = None
    ) -> float:
        """
        Calculate Calmar Ratio.

        Calmar Ratio = Annualized Return / Max Drawdown

        Args:
            returns: Series of periodic returns
            periods_per_year: Number of periods per year for annualization

        Returns:
            Calmar Ratio
        """
        if periods_per_year is None:
            periods_per_year = self.TRADING_DAYS_PER_YEAR * self.HOURS_PER_DAY

        if len(returns) < 2:
            return 0.0

        # Calculate cumulative returns for drawdown
        cumulative = (1 + returns).cumprod()
        max_dd = self._calculate_max_drawdown(cumulative)

        if max_dd == 0:
            return 10.0  # Cap at large value

        # Annualized return
        total_return = cumulative.iloc[-1] - 1
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

        return annualized_return / max_dd

    def calculate_max_drawdown(
        self,
        equity_curve: pd.Series
    ) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown and its duration.

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            Tuple of (max_drawdown_pct, drawdown_start_idx, drawdown_end_idx)
        """
        peak = equity_curve.expanding().max()
        drawdown = (peak - equity_curve) / peak

        max_dd = drawdown.max()
        max_dd_idx = drawdown.idxmax()

        # Find drawdown start (peak before max drawdown)
        peak_value = peak.loc[max_dd_idx]
        dd_start_idx = equity_curve[equity_curve == peak_value].index[0]

        # Find drawdown end (recovery point or current)
        try:
            dd_end_candidates = equity_curve.loc[max_dd_idx:]
            recovery = dd_end_candidates[dd_end_candidates >= peak_value]
            if len(recovery) > 0:
                dd_end_idx = recovery.index[0]
            else:
                dd_end_idx = equity_curve.index[-1]
        except Exception:
            dd_end_idx = equity_curve.index[-1]

        return max_dd, dd_start_idx, dd_end_idx

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown from cumulative returns series."""
        peak = cumulative_returns.expanding().max()
        drawdown = (peak - cumulative_returns) / peak
        return drawdown.max()

    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        VaR represents the maximum expected loss at a given confidence level.

        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: Calculation method ("historical" or "parametric")

        Returns:
            VaR as positive percentage
        """
        if len(returns) < 10:
            return 0.0

        if method == "historical":
            var = np.percentile(returns, (1 - confidence_level) * 100)
        else:  # parametric (assumes normal distribution)
            mean = returns.mean()
            std = returns.std()
            from scipy import stats
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean + z_score * std

        return abs(var)

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

        CVaR is the expected loss given that the loss exceeds VaR.

        Args:
            returns: Series of returns
            confidence_level: Confidence level

        Returns:
            CVaR as positive percentage
        """
        if len(returns) < 10:
            return 0.0

        var = self.calculate_var(returns, confidence_level, "historical")
        tail_returns = returns[returns <= -var]

        if len(tail_returns) == 0:
            return var

        return abs(tail_returns.mean())

    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        periods_per_year: int = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a return series.

        Args:
            returns: Series of periodic returns
            periods_per_year: Number of periods per year

        Returns:
            RiskMetrics object with all calculated metrics
        """
        if periods_per_year is None:
            periods_per_year = self.TRADING_DAYS_PER_YEAR * self.HOURS_PER_DAY

        # Calculate all metrics
        cumulative = (1 + returns).cumprod()
        max_dd, dd_start, dd_end = self.calculate_max_drawdown(cumulative)

        # Trade statistics (assuming each return is a trade)
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]

        win_rate = len(winning_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
        avg_loss = abs(losing_returns.mean()) if len(losing_returns) > 0 else 0

        total_wins = winning_returns.sum() if len(winning_returns) > 0 else 0
        total_losses = abs(losing_returns.sum()) if len(losing_returns) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        # Duration calculation
        if isinstance(dd_start, (int, float)) and isinstance(dd_end, (int, float)):
            dd_duration = int(dd_end - dd_start)
        else:
            dd_duration = 0

        return RiskMetrics(
            sharpe_ratio=self.calculate_sharpe_ratio(returns, periods_per_year),
            sortino_ratio=self.calculate_sortino_ratio(returns, periods_per_year),
            calmar_ratio=self.calculate_calmar_ratio(returns, periods_per_year),
            max_drawdown=max_dd,
            max_drawdown_duration=dd_duration,
            volatility=returns.std() * np.sqrt(periods_per_year),
            var_95=self.calculate_var(returns, 0.95),
            var_99=self.calculate_var(returns, 0.99),
            cvar_95=self.calculate_cvar(returns, 0.95),
            win_rate=win_rate,
            profit_factor=profit_factor if profit_factor != float('inf') else 10.0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy=expectancy
        )

    # ========== Position Management ==========

    def add_position(self, position: Position) -> bool:
        """
        Add a new position to the portfolio.

        Args:
            position: Position object to add

        Returns:
            True if position was added, False if rejected
        """
        can_open, violations = self.can_open_new_position(
            position.symbol,
            position.notional_value,
            position.leverage
        )

        if can_open:
            self.positions[position.symbol] = position
            return True
        else:
            return False

    def remove_position(self, symbol: str) -> Optional[Position]:
        """
        Remove a position from the portfolio.

        Args:
            symbol: Symbol of position to remove

        Returns:
            Removed position or None if not found
        """
        return self.positions.pop(symbol, None)

    def update_equity(self, new_equity: float) -> None:
        """
        Update portfolio equity and equity curve.

        Args:
            new_equity: Current portfolio value
        """
        self.portfolio_value = new_equity
        timestamp = datetime.now()

        # Append to equity curve
        new_point = pd.Series({timestamp: new_equity})
        self.equity_curve = pd.concat([self.equity_curve, new_point])

    def record_daily_pnl(self, pnl: float) -> None:
        """
        Record daily P&L for risk monitoring.

        Args:
            pnl: Daily profit/loss amount
        """
        self.daily_pnl.append(pnl)

    def get_portfolio_summary(self) -> Dict:
        """
        Get comprehensive portfolio summary.

        Returns:
            Dictionary with portfolio statistics and metrics
        """
        risk_checks = self.check_portfolio_risk_limits()
        risk_level = self.get_risk_level()

        total_exposure = self._calculate_current_exposure()

        return {
            "portfolio_value": self.portfolio_value,
            "total_positions": len(self.positions),
            "total_exposure": total_exposure,
            "exposure_pct": total_exposure / self.portfolio_value * 100,
            "current_drawdown": self._calculate_current_drawdown() * 100,
            "risk_level": risk_level.value,
            "risk_checks": risk_checks,
            "correlation_risk": self.get_portfolio_correlation_risk(),
            "positions": {
                sym: {
                    "side": pos.side.value,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                    "trailing_stop": pos.trailing_stop_price
                }
                for sym, pos in self.positions.items()
            }
        }


# ========== Convenience Functions ==========

def create_risk_manager(
    portfolio_value: float,
    max_drawdown_pct: float = 0.15,
    max_position_pct: float = 0.10,
    max_exposure_pct: float = 0.50
) -> RiskManager:
    """
    Factory function to create a configured RiskManager.

    Args:
        portfolio_value: Initial portfolio value
        max_drawdown_pct: Maximum allowed drawdown
        max_position_pct: Maximum position size as % of portfolio
        max_exposure_pct: Maximum total exposure

    Returns:
        Configured RiskManager instance
    """
    limits = RiskLimits(
        max_position_size_pct=max_position_pct,
        max_total_exposure_pct=max_exposure_pct,
        max_drawdown_pct=max_drawdown_pct
    )

    return RiskManager(portfolio_value=portfolio_value, risk_limits=limits)


def quick_position_size(
    portfolio_value: float,
    entry_price: float,
    stop_loss_price: float,
    risk_pct: float = 0.02
) -> float:
    """
    Quick position size calculation based on fixed risk percentage.

    Args:
        portfolio_value: Total portfolio value
        entry_price: Entry price
        stop_loss_price: Stop loss price
        risk_pct: Percentage of portfolio to risk (default 2%)

    Returns:
        Position size in base currency
    """
    risk_amount = portfolio_value * risk_pct
    risk_per_unit = abs(entry_price - stop_loss_price)

    if risk_per_unit == 0:
        return 0.0

    position_size = risk_amount / risk_per_unit
    return position_size
