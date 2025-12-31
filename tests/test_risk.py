"""Unit tests for risk management module."""
import pytest
import numpy as np
from risk.risk_manager import (
    RiskManager, RiskLimits, RiskMetrics,
    PositionSide, PositionSizeResult
)


@pytest.fixture
def risk_limits():
    """Create default risk limits."""
    return RiskLimits(
        max_position_size_pct=0.1,
        max_total_exposure_pct=0.5,
        max_drawdown_pct=0.15,
        max_daily_loss_pct=0.05,
        max_leverage=10,
        min_risk_reward_ratio=1.5
    )


@pytest.fixture
def risk_manager(risk_limits):
    """Create a risk manager instance."""
    return RiskManager(
        portfolio_value=10000.0,
        risk_limits=risk_limits
    )


class TestRiskLimits:
    """Tests for RiskLimits configuration."""

    def test_default_limits(self):
        """Default limits should have sensible values."""
        limits = RiskLimits()
        assert 0 < limits.max_position_size_pct <= 1
        assert 0 < limits.max_total_exposure_pct <= 1
        assert 0 < limits.max_drawdown_pct <= 1
        assert limits.max_leverage > 0

    def test_custom_limits(self):
        """Custom limits should be stored correctly."""
        limits = RiskLimits(
            max_position_size_pct=0.05,
            max_leverage=5
        )
        assert limits.max_position_size_pct == 0.05
        assert limits.max_leverage == 5


class TestRiskManagerInitialization:
    """Tests for RiskManager initialization."""

    def test_creates_with_portfolio_value(self):
        """Should initialize with portfolio value."""
        rm = RiskManager(portfolio_value=50000.0)
        assert rm.portfolio_value == 50000.0

    def test_creates_with_default_limits(self):
        """Should create default limits if not provided."""
        rm = RiskManager(portfolio_value=10000.0)
        assert rm.risk_limits is not None

    def test_creates_with_custom_limits(self, risk_limits):
        """Should use provided limits."""
        rm = RiskManager(portfolio_value=10000.0, risk_limits=risk_limits)
        assert rm.risk_limits.max_leverage == 10


class TestPositionSizing:
    """Tests for position sizing calculations."""

    def test_calculate_position_size_returns_result(self, risk_manager):
        """Should return PositionSizeResult."""
        result = risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            side=PositionSide.LONG,
            entry_price=50000.0,
            stop_loss_price=48000.0,
            win_rate=0.5,
            avg_win_loss_ratio=2.0,
            current_volatility=0.02,
            leverage=5
        )
        assert isinstance(result, PositionSizeResult)

    def test_position_size_is_positive(self, risk_manager):
        """Position size should be positive."""
        result = risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            side=PositionSide.LONG,
            entry_price=50000.0,
            stop_loss_price=48000.0,
            win_rate=0.5,
            avg_win_loss_ratio=2.0,
            current_volatility=0.02,
            leverage=5
        )
        assert result.recommended_size >= 0

    def test_position_value_reasonable(self, risk_manager):
        """Position value should be reasonable relative to portfolio."""
        result = risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            side=PositionSide.LONG,
            entry_price=50000.0,
            stop_loss_price=48000.0,
            win_rate=0.5,
            avg_win_loss_ratio=2.0,
            current_volatility=0.02,
            leverage=5
        )
        # Position value should not exceed portfolio * max_exposure * leverage
        max_possible = 10000.0 * 0.5 * 5  # portfolio * max_exposure * leverage
        assert result.position_value <= max_possible

    def test_higher_volatility_reduces_size(self, risk_manager):
        """Higher volatility should result in smaller position."""
        low_vol_result = risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            side=PositionSide.LONG,
            entry_price=50000.0,
            stop_loss_price=48000.0,
            win_rate=0.5,
            avg_win_loss_ratio=2.0,
            current_volatility=0.01,  # Low volatility
            leverage=5
        )

        high_vol_result = risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            side=PositionSide.LONG,
            entry_price=50000.0,
            stop_loss_price=48000.0,
            win_rate=0.5,
            avg_win_loss_ratio=2.0,
            current_volatility=0.05,  # High volatility
            leverage=5
        )

        assert high_vol_result.volatility_adjusted_size <= low_vol_result.volatility_adjusted_size

    def test_margin_required_calculation(self, risk_manager):
        """Margin required should equal position_value / leverage."""
        result = risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            side=PositionSide.LONG,
            entry_price=50000.0,
            stop_loss_price=48000.0,
            win_rate=0.5,
            avg_win_loss_ratio=2.0,
            current_volatility=0.02,
            leverage=5
        )
        expected_margin = result.position_value / result.leverage_used
        assert abs(result.margin_required - expected_margin) < 0.01


class TestKellyCriterion:
    """Tests for Kelly Criterion position sizing."""

    def test_kelly_fraction_bounded(self, risk_manager):
        """Kelly fraction should be between 0 and 1."""
        result = risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            side=PositionSide.LONG,
            entry_price=50000.0,
            stop_loss_price=48000.0,
            win_rate=0.5,
            avg_win_loss_ratio=2.0,
            current_volatility=0.02,
            leverage=5,
            use_kelly=True
        )
        assert 0 <= result.kelly_fraction <= 1

    def test_higher_win_rate_increases_kelly(self, risk_manager):
        """Higher win rate should increase Kelly fraction."""
        low_wr = risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            side=PositionSide.LONG,
            entry_price=50000.0,
            stop_loss_price=48000.0,
            win_rate=0.4,
            avg_win_loss_ratio=2.0,
            current_volatility=0.02,
            leverage=5,
            use_kelly=True
        )

        high_wr = risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            side=PositionSide.LONG,
            entry_price=50000.0,
            stop_loss_price=48000.0,
            win_rate=0.6,
            avg_win_loss_ratio=2.0,
            current_volatility=0.02,
            leverage=5,
            use_kelly=True
        )

        assert high_wr.kelly_fraction >= low_wr.kelly_fraction

    def test_fractional_kelly_reduces_size(self, risk_manager):
        """Fractional Kelly should reduce position size."""
        full_kelly = risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            side=PositionSide.LONG,
            entry_price=50000.0,
            stop_loss_price=48000.0,
            win_rate=0.55,
            avg_win_loss_ratio=2.0,
            current_volatility=0.02,
            leverage=5,
            use_kelly=True,
            fractional_kelly=1.0
        )

        half_kelly = risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            side=PositionSide.LONG,
            entry_price=50000.0,
            stop_loss_price=48000.0,
            win_rate=0.55,
            avg_win_loss_ratio=2.0,
            current_volatility=0.02,
            leverage=5,
            use_kelly=True,
            fractional_kelly=0.5
        )

        assert half_kelly.recommended_size <= full_kelly.recommended_size


class TestDynamicStopLoss:
    """Tests for dynamic stop loss calculation."""

    def test_stop_loss_long_below_entry(self, risk_manager):
        """Stop loss for LONG should be below entry price."""
        entry = 50000.0
        atr = 1000.0
        multiplier = 2.0

        stop_loss = risk_manager.calculate_dynamic_stop_loss(
            entry_price=entry,
            side=PositionSide.LONG,
            atr=atr,
            atr_multiplier=multiplier
        )

        assert stop_loss < entry

    def test_stop_loss_short_above_entry(self, risk_manager):
        """Stop loss for SHORT should be above entry price."""
        entry = 50000.0
        atr = 1000.0
        multiplier = 2.0

        stop_loss = risk_manager.calculate_dynamic_stop_loss(
            entry_price=entry,
            side=PositionSide.SHORT,
            atr=atr,
            atr_multiplier=multiplier
        )

        assert stop_loss > entry

    def test_stop_loss_distance_equals_atr_mult(self, risk_manager):
        """Stop loss distance should equal ATR * multiplier."""
        entry = 50000.0
        atr = 1000.0
        multiplier = 2.0

        stop_loss = risk_manager.calculate_dynamic_stop_loss(
            entry_price=entry,
            side=PositionSide.LONG,
            atr=atr,
            atr_multiplier=multiplier
        )

        expected_distance = atr * multiplier
        actual_distance = entry - stop_loss
        assert abs(actual_distance - expected_distance) < 0.01


class TestRiskMetrics:
    """Tests for risk metrics calculations."""

    def test_sharpe_ratio_calculation(self, risk_manager):
        """Sharpe ratio should be calculated correctly."""
        returns = np.array([0.02, -0.01, 0.03, 0.01, -0.02, 0.02, 0.01])
        sharpe = risk_manager.calculate_sharpe_ratio(returns, periods_per_year=252)

        # Sharpe should be a real number
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)

    def test_sortino_ratio_calculation(self, risk_manager):
        """Sortino ratio should be calculated correctly."""
        returns = np.array([0.02, -0.01, 0.03, 0.01, -0.02, 0.02, 0.01])
        sortino = risk_manager.calculate_sortino_ratio(returns, periods_per_year=252)

        # Sortino should be >= Sharpe (penalizes only downside)
        sharpe = risk_manager.calculate_sharpe_ratio(returns, periods_per_year=252)
        assert sortino >= sharpe or abs(sortino - sharpe) < 0.1

    def test_max_drawdown_calculation(self, risk_manager):
        """Max drawdown should be calculated correctly."""
        # Equity curve: up, down, up
        equity = np.array([100, 110, 105, 115, 100, 120])
        max_dd = risk_manager.calculate_max_drawdown(equity)

        # Max drawdown from 115 to 100 = 13%
        assert 0 < max_dd < 1
        assert abs(max_dd - (15/115)) < 0.01


class TestLeverageLimits:
    """Tests for leverage limit enforcement."""

    def test_leverage_capped_at_max(self, risk_manager):
        """Leverage should not exceed max_leverage."""
        result = risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            side=PositionSide.LONG,
            entry_price=50000.0,
            stop_loss_price=48000.0,
            win_rate=0.5,
            avg_win_loss_ratio=2.0,
            current_volatility=0.02,
            leverage=20  # Request 20x, max is 10x
        )

        assert result.leverage_used <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
