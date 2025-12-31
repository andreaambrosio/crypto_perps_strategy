"""Unit tests for backtesting engine."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtest.engine import (
    BacktestEngine, MultiTimeframeDataHandler, PortfolioManager,
    ExecutionEngine, PerformanceCalculator, FundingRateSimulator,
    Timeframe, OrderType, OrderSide, PositionSide,
    MarketDataEvent, SignalEvent, OrderEvent, FillEvent,
    EventType, Position, Trade, PerformanceMetrics
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 200

    dates = pd.date_range(start='2024-01-01', periods=n, freq='4h', tz='UTC')
    close = 50000 + np.cumsum(np.random.randn(n) * 500)
    high = close + np.abs(np.random.randn(n)) * 200
    low = close - np.abs(np.random.randn(n)) * 200
    open_price = close + np.random.randn(n) * 100
    volume = np.random.randint(100, 1000, n).astype(float)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    return {'BTC/USDT:USDT': {'4h': df}}


@pytest.fixture
def data_handler(sample_data):
    """Create a data handler with sample data."""
    return MultiTimeframeDataHandler(sample_data, primary_timeframe=Timeframe.H4)


@pytest.fixture
def portfolio():
    """Create a portfolio manager."""
    return PortfolioManager(initial_capital=10000.0, max_leverage=10.0)


@pytest.fixture
def execution_engine():
    """Create an execution engine."""
    return ExecutionEngine(
        slippage_pct=0.001,
        maker_fee=0.0002,
        taker_fee=0.0004
    )


class TestMultiTimeframeDataHandler:
    """Tests for MultiTimeframeDataHandler."""

    def test_initialization(self, data_handler):
        """Should initialize with data."""
        assert data_handler is not None
        assert data_handler.has_more_data()

    def test_get_next_bar(self, data_handler):
        """Should return MarketDataEvent."""
        event = data_handler.get_next_bar()
        assert isinstance(event, MarketDataEvent)
        assert event.symbol == 'BTC/USDT:USDT'

    def test_get_latest_bars(self, data_handler):
        """Should return DataFrame with n bars."""
        # Advance a few bars first
        for _ in range(10):
            data_handler.get_next_bar()

        event = data_handler.get_next_bar()
        bars = data_handler.get_latest_bars(
            symbol='BTC/USDT:USDT',
            timeframe=Timeframe.H4,
            n=5,
            current_time=event.timestamp
        )

        assert isinstance(bars, pd.DataFrame)
        assert len(bars) <= 5

    def test_has_more_data(self, data_handler):
        """Should return False when data exhausted."""
        while data_handler.has_more_data():
            data_handler.get_next_bar()

        assert not data_handler.has_more_data()


class TestPortfolioManager:
    """Tests for PortfolioManager."""

    def test_initialization(self, portfolio):
        """Should initialize with correct capital."""
        assert portfolio.cash == 10000.0
        assert portfolio.total_equity == 10000.0

    def test_open_position(self, portfolio):
        """Should open a position correctly."""
        position = portfolio.open_position(
            symbol='BTC/USDT',
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
            leverage=5.0,
            timestamp=pd.Timestamp.now(tz='UTC'),
            commission=5.0
        )

        assert position is not None
        assert position.symbol == 'BTC/USDT'
        assert position.quantity == 0.1
        assert position.side == PositionSide.LONG

    def test_margin_used_after_open(self, portfolio):
        """Margin should be used after opening position."""
        initial_margin = portfolio.total_margin_used

        portfolio.open_position(
            symbol='BTC/USDT',
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=0.1,  # $5000 notional
            leverage=5.0,  # $1000 margin
            timestamp=pd.Timestamp.now(tz='UTC')
        )

        assert portfolio.total_margin_used > initial_margin

    def test_close_position(self, portfolio):
        """Should close position and calculate PnL."""
        timestamp = pd.Timestamp.now(tz='UTC')

        portfolio.open_position(
            symbol='BTC/USDT',
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
            leverage=5.0,
            timestamp=timestamp
        )

        trade = portfolio.close_position(
            symbol='BTC/USDT',
            exit_price=51000.0,  # $1000 profit on 0.1 BTC = $100
            timestamp=timestamp + timedelta(hours=1)
        )

        assert trade is not None
        assert trade.pnl > 0  # Should be profitable

    def test_close_position_pnl_calculation(self, portfolio):
        """PnL should be calculated correctly."""
        timestamp = pd.Timestamp.now(tz='UTC')

        portfolio.open_position(
            symbol='BTC/USDT',
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
            leverage=5.0,
            timestamp=timestamp
        )

        trade = portfolio.close_position(
            symbol='BTC/USDT',
            exit_price=51000.0,
            timestamp=timestamp + timedelta(hours=1)
        )

        # 0.1 BTC * (51000 - 50000) = $100
        expected_pnl = 0.1 * (51000 - 50000)
        assert abs(trade.pnl - expected_pnl) < 1  # Allow for commission

    def test_short_position_pnl(self, portfolio):
        """Short position PnL should be inverse."""
        timestamp = pd.Timestamp.now(tz='UTC')

        portfolio.open_position(
            symbol='BTC/USDT',
            side=PositionSide.SHORT,
            entry_price=50000.0,
            quantity=0.1,
            leverage=5.0,
            timestamp=timestamp
        )

        trade = portfolio.close_position(
            symbol='BTC/USDT',
            exit_price=49000.0,  # Price dropped = profit for short
            timestamp=timestamp + timedelta(hours=1)
        )

        assert trade.pnl > 0  # Short is profitable when price drops


class TestExecutionEngine:
    """Tests for ExecutionEngine."""

    def test_slippage_applied_to_buy(self, execution_engine):
        """Buy orders should have positive slippage (worse price)."""
        order = OrderEvent(
            event_type=EventType.ORDER,
            timestamp=pd.Timestamp.now(tz='UTC'),
            symbol='BTC/USDT',
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0.1,
            leverage=5.0
        )

        fill = execution_engine.execute_order(order, current_price=50000.0)

        assert fill.fill_price >= 50000.0  # Slippage makes price higher

    def test_slippage_applied_to_sell(self, execution_engine):
        """Sell orders should have negative slippage (worse price)."""
        order = OrderEvent(
            event_type=EventType.ORDER,
            timestamp=pd.Timestamp.now(tz='UTC'),
            symbol='BTC/USDT',
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=0.1,
            leverage=5.0
        )

        fill = execution_engine.execute_order(order, current_price=50000.0)

        assert fill.fill_price <= 50000.0  # Slippage makes price lower

    def test_commission_calculated(self, execution_engine):
        """Commission should be calculated on notional value."""
        order = OrderEvent(
            event_type=EventType.ORDER,
            timestamp=pd.Timestamp.now(tz='UTC'),
            symbol='BTC/USDT',
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0.1,  # $5000 notional at $50k
            leverage=5.0
        )

        fill = execution_engine.execute_order(order, current_price=50000.0)

        # Commission should be ~0.04% of $5000 = $2
        assert fill.commission > 0
        assert fill.commission < 50  # Reasonable commission


class TestPerformanceCalculator:
    """Tests for PerformanceCalculator."""

    def test_calculate_metrics_returns_object(self):
        """Should return PerformanceMetrics."""
        equity = pd.DataFrame({
            'equity': [10000, 10100, 10050, 10200, 10150]
        }, index=pd.date_range('2024-01-01', periods=5, freq='D'))

        trades = [
            Trade(
                trade_id=1, symbol='BTC/USDT', side=PositionSide.LONG,
                entry_time=pd.Timestamp('2024-01-01'),
                exit_time=pd.Timestamp('2024-01-02'),
                entry_price=50000, exit_price=51000,
                quantity=0.1, leverage=5,
                pnl=100, pnl_pct=0.1, commission=2,
                slippage_cost=0, funding_paid=0,
                duration=timedelta(days=1), mfe=0.02, mae=0.01
            )
        ]

        metrics = PerformanceCalculator.calculate_metrics(equity, trades)
        assert isinstance(metrics, PerformanceMetrics)

    def test_total_return_calculation(self):
        """Total return should be calculated correctly."""
        equity = pd.DataFrame({
            'equity': [10000, 11000]  # 10% gain
        }, index=pd.date_range('2024-01-01', periods=2, freq='D'))

        trades = [
            Trade(
                trade_id=1, symbol='BTC/USDT', side=PositionSide.LONG,
                entry_time=pd.Timestamp('2024-01-01'),
                exit_time=pd.Timestamp('2024-01-02'),
                entry_price=50000, exit_price=51000,
                quantity=0.2, leverage=5,
                pnl=1000, pnl_pct=0.1, commission=2,
                slippage_cost=0, funding_paid=0,
                duration=timedelta(days=1), mfe=0.02, mae=0.01
            )
        ]

        metrics = PerformanceCalculator.calculate_metrics(equity, trades)
        assert abs(metrics.total_return - 0.1) < 0.01

    def test_win_rate_calculation(self):
        """Win rate should be wins / total trades."""
        equity = pd.DataFrame({
            'equity': [10000, 10100, 10050]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D'))

        trades = [
            Trade(
                trade_id=1, symbol='BTC/USDT', side=PositionSide.LONG,
                entry_time=pd.Timestamp('2024-01-01'),
                exit_time=pd.Timestamp('2024-01-02'),
                entry_price=50000, exit_price=51000,
                quantity=0.1, leverage=5,
                pnl=100, pnl_pct=0.1, commission=0,
                slippage_cost=0, funding_paid=0,
                duration=timedelta(days=1), mfe=0.02, mae=0.01
            ),
            Trade(
                trade_id=2, symbol='BTC/USDT', side=PositionSide.LONG,
                entry_time=pd.Timestamp('2024-01-02'),
                exit_time=pd.Timestamp('2024-01-03'),
                entry_price=51000, exit_price=50500,
                quantity=0.1, leverage=5,
                pnl=-50, pnl_pct=-0.05, commission=0,
                slippage_cost=0, funding_paid=0,
                duration=timedelta(days=1), mfe=0.01, mae=0.02
            )
        ]

        metrics = PerformanceCalculator.calculate_metrics(equity, trades)
        assert abs(metrics.win_rate - 0.5) < 0.01  # 1 win, 1 loss = 50%


class TestFundingRateSimulator:
    """Tests for FundingRateSimulator."""

    def test_generate_funding_rate(self):
        """Should generate valid funding rates."""
        simulator = FundingRateSimulator(
            funding_interval_hours=8,
            base_rate=0.0001
        )

        rate = simulator.get_funding_rate(
            'BTC/USDT',
            pd.Timestamp('2024-01-01 08:00:00', tz='UTC')
        )

        assert isinstance(rate, float)
        assert -0.01 < rate < 0.01  # Reasonable rate range

    def test_funding_rate_at_intervals(self):
        """Funding should apply at correct intervals."""
        simulator = FundingRateSimulator(funding_interval_hours=8)

        # 08:00 should be funding time
        is_funding_time_8 = simulator.is_funding_time(
            pd.Timestamp('2024-01-01 08:00:00', tz='UTC')
        )

        # 10:00 should not be funding time
        is_funding_time_10 = simulator.is_funding_time(
            pd.Timestamp('2024-01-01 10:00:00', tz='UTC')
        )

        assert is_funding_time_8 == True
        assert is_funding_time_10 == False


class TestBacktestEngine:
    """Integration tests for BacktestEngine."""

    def test_engine_initialization(self, sample_data):
        """Engine should initialize correctly."""
        data_handler = MultiTimeframeDataHandler(sample_data)

        class DummyStrategy:
            def on_data(self, event):
                return None
            def on_fill(self, event):
                pass

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=DummyStrategy(),
            initial_capital=10000.0
        )

        assert engine is not None
        assert engine.portfolio.cash == 10000.0

    def test_engine_run_completes(self, sample_data):
        """Engine should complete a full backtest run."""
        data_handler = MultiTimeframeDataHandler(sample_data)

        class DummyStrategy:
            def on_data(self, event):
                return None
            def on_fill(self, event):
                pass

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=DummyStrategy(),
            initial_capital=10000.0
        )

        metrics = engine.run(verbose=False)

        assert isinstance(metrics, PerformanceMetrics)

    def test_engine_tracks_equity(self, sample_data):
        """Engine should track equity curve."""
        data_handler = MultiTimeframeDataHandler(sample_data)

        class DummyStrategy:
            def on_data(self, event):
                return None
            def on_fill(self, event):
                pass

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=DummyStrategy(),
            initial_capital=10000.0
        )

        engine.run(verbose=False)
        equity_df = engine.get_equity_curve()

        assert isinstance(equity_df, pd.DataFrame)
        assert len(equity_df) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
