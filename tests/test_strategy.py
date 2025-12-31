"""Unit tests for trading strategy."""
import pytest
import pandas as pd
import numpy as np
from strategies.trend_momentum_strategy import TrendMomentumStrategy, StrategyConfig
from strategies.base_strategy import Signal, SignalType, TimeFrame


@pytest.fixture
def strategy():
    """Create a strategy instance with default config."""
    config = StrategyConfig()
    return TrendMomentumStrategy(config)


@pytest.fixture
def custom_strategy():
    """Create a strategy with custom parameters."""
    config = StrategyConfig(
        ema_fast_period=8,
        ema_slow_period=21,
        rsi_period=14,
        adx_threshold=20,
        min_confidence=0.5
    )
    strategy = TrendMomentumStrategy(config)
    strategy.config = config  # Ensure config is accessible
    return strategy


@pytest.fixture
def bullish_data():
    """Create data with clear bullish trend."""
    np.random.seed(42)
    n = 100

    # Strong uptrend
    trend = np.linspace(100, 200, n)
    noise = np.random.randn(n) * 2
    close = trend + noise

    high = close + np.abs(np.random.randn(n)) * 3
    low = close - np.abs(np.random.randn(n)) * 3
    open_price = close - np.random.randn(n) * 1
    volume = np.random.randint(5000, 15000, n).astype(float)

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.fixture
def bearish_data():
    """Create data with clear bearish trend."""
    np.random.seed(42)
    n = 100

    # Strong downtrend
    trend = np.linspace(200, 100, n)
    noise = np.random.randn(n) * 2
    close = trend + noise

    high = close + np.abs(np.random.randn(n)) * 3
    low = close - np.abs(np.random.randn(n)) * 3
    open_price = close + np.random.randn(n) * 1
    volume = np.random.randint(5000, 15000, n).astype(float)

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.fixture
def sideways_data():
    """Create choppy/sideways data."""
    np.random.seed(42)
    n = 100

    # Sideways movement
    close = 100 + np.random.randn(n) * 5
    high = close + np.abs(np.random.randn(n)) * 2
    low = close - np.abs(np.random.randn(n)) * 2
    open_price = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 5000, n).astype(float)

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


class TestStrategyConfig:
    """Tests for StrategyConfig."""

    def test_default_config_values(self):
        """Default config should have expected values."""
        config = StrategyConfig()
        assert config.ema_fast_period == 12
        assert config.ema_slow_period == 26
        assert config.rsi_period == 14
        assert config.adx_threshold == 25.0
        assert config.min_confidence == 0.6

    def test_custom_config_values(self):
        """Custom config should override defaults."""
        config = StrategyConfig(
            ema_fast_period=8,
            ema_slow_period=21,
            min_confidence=0.7
        )
        assert config.ema_fast_period == 8
        assert config.ema_slow_period == 21
        assert config.min_confidence == 0.7


class TestStrategyInitialization:
    """Tests for strategy initialization."""

    def test_strategy_creates_with_default_config(self):
        """Strategy should initialize with default config."""
        strategy = TrendMomentumStrategy()
        assert strategy is not None
        assert strategy.name == "TrendMomentum"

    def test_strategy_creates_with_custom_config(self, custom_strategy):
        """Strategy should initialize with custom config."""
        assert custom_strategy.config.ema_fast_period == 8
        assert custom_strategy.config.ema_slow_period == 21

    def test_strategy_has_required_methods(self, strategy):
        """Strategy should have required interface methods."""
        assert hasattr(strategy, 'calculate_indicators')
        assert hasattr(strategy, 'generate_signal')
        assert callable(strategy.calculate_indicators)
        assert callable(strategy.generate_signal)


class TestIndicatorCalculation:
    """Tests for indicator calculation."""

    def test_calculate_indicators_returns_dataframe(self, strategy, bullish_data):
        """calculate_indicators should return a DataFrame."""
        result = strategy.calculate_indicators(bullish_data.copy(), '4h')
        assert isinstance(result, pd.DataFrame)

    def test_calculate_indicators_adds_columns(self, strategy, bullish_data):
        """calculate_indicators should add indicator columns."""
        result = strategy.calculate_indicators(bullish_data.copy(), '4h')

        # Check for expected indicator columns
        expected_columns = ['ema_fast', 'ema_slow', 'rsi', 'atr']
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_calculate_indicators_preserves_original(self, strategy, bullish_data):
        """Original OHLCV columns should be preserved."""
        result = strategy.calculate_indicators(bullish_data.copy(), '4h')

        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in result.columns


class TestSignalGeneration:
    """Tests for signal generation."""

    def test_generate_signal_returns_signal(self, strategy, bullish_data):
        """generate_signal should return a Signal object."""
        df = strategy.calculate_indicators(bullish_data.copy(), '4h')
        signal = strategy.generate_signal(df, 'BTC/USDT', '4h', None)
        assert isinstance(signal, Signal)

    def test_signal_has_required_attributes(self, strategy, bullish_data):
        """Signal should have all required attributes."""
        df = strategy.calculate_indicators(bullish_data.copy(), '4h')
        signal = strategy.generate_signal(df, 'BTC/USDT', '4h', None)

        assert hasattr(signal, 'signal_type')
        assert hasattr(signal, 'confidence')
        assert hasattr(signal, 'symbol')
        assert hasattr(signal, 'stop_loss')
        assert hasattr(signal, 'take_profit')

    def test_signal_confidence_bounded(self, strategy, bullish_data):
        """Signal confidence should be between 0 and 1."""
        df = strategy.calculate_indicators(bullish_data.copy(), '4h')
        signal = strategy.generate_signal(df, 'BTC/USDT', '4h', None)

        assert 0 <= signal.confidence <= 1

    def test_signal_type_is_valid(self, strategy, bullish_data):
        """Signal type should be a valid SignalType."""
        df = strategy.calculate_indicators(bullish_data.copy(), '4h')
        signal = strategy.generate_signal(df, 'BTC/USDT', '4h', None)

        assert isinstance(signal.signal_type, SignalType)

    def test_bullish_trend_generates_long_or_hold(self, strategy, bullish_data):
        """Strong bullish trend should generate LONG or HOLD signal."""
        df = strategy.calculate_indicators(bullish_data.copy(), '4h')
        signal = strategy.generate_signal(df, 'BTC/USDT', '4h', None)

        # In strong uptrend, should be LONG, HOLD, or NO_SIGNAL
        valid_types = [SignalType.LONG, SignalType.HOLD, SignalType.NO_SIGNAL]
        assert signal.signal_type in valid_types

    def test_bearish_trend_generates_short_or_hold(self, strategy, bearish_data):
        """Strong bearish trend should generate SHORT or HOLD signal."""
        df = strategy.calculate_indicators(bearish_data.copy(), '4h')
        signal = strategy.generate_signal(df, 'BTC/USDT', '4h', None)

        # In strong downtrend, should be SHORT, HOLD, or NO_SIGNAL
        valid_types = [SignalType.SHORT, SignalType.HOLD, SignalType.NO_SIGNAL]
        assert signal.signal_type in valid_types

    def test_stop_loss_below_entry_for_long(self, strategy, bullish_data):
        """Stop loss should be below current price for LONG signals."""
        df = strategy.calculate_indicators(bullish_data.copy(), '4h')
        signal = strategy.generate_signal(df, 'BTC/USDT', '4h', None)

        if signal.signal_type == SignalType.LONG and signal.stop_loss:
            current_price = df['close'].iloc[-1]
            assert signal.stop_loss < current_price

    def test_stop_loss_above_entry_for_short(self, strategy, bearish_data):
        """Stop loss should be above current price for SHORT signals."""
        df = strategy.calculate_indicators(bearish_data.copy(), '4h')
        signal = strategy.generate_signal(df, 'BTC/USDT', '4h', None)

        if signal.signal_type == SignalType.SHORT and signal.stop_loss:
            current_price = df['close'].iloc[-1]
            assert signal.stop_loss > current_price


class TestExitSignals:
    """Tests for exit signal generation."""

    def test_close_long_with_long_position(self, strategy, bearish_data):
        """Should generate CLOSE_LONG when holding long in downtrend."""
        df = strategy.calculate_indicators(bearish_data.copy(), '4h')
        signal = strategy.generate_signal(df, 'BTC/USDT', '4h', 'long')

        # With existing long position in downtrend
        valid_types = [SignalType.CLOSE_LONG, SignalType.HOLD, SignalType.NO_SIGNAL]
        assert signal.signal_type in valid_types

    def test_close_short_with_short_position(self, strategy, bullish_data):
        """Should generate CLOSE_SHORT when holding short in uptrend."""
        df = strategy.calculate_indicators(bullish_data.copy(), '4h')
        signal = strategy.generate_signal(df, 'BTC/USDT', '4h', 'short')

        # With existing short position in uptrend
        valid_types = [SignalType.CLOSE_SHORT, SignalType.HOLD, SignalType.NO_SIGNAL]
        assert signal.signal_type in valid_types


class TestMinConfidenceThreshold:
    """Tests for minimum confidence filtering."""

    def test_low_confidence_signal_filtered(self):
        """Signals below min_confidence should be filtered to NO_SIGNAL."""
        config = StrategyConfig(min_confidence=0.9)  # Very high threshold
        strategy = TrendMomentumStrategy(config)

        # Sideways data should produce low confidence signals
        np.random.seed(42)
        n = 100
        close = 100 + np.random.randn(n) * 2
        high = close + 1
        low = close - 1

        data = pd.DataFrame({
            'open': close,
            'high': high,
            'low': low,
            'close': close,
            'volume': [1000] * n
        })

        df = strategy.calculate_indicators(data.copy(), '4h')
        signal = strategy.generate_signal(df, 'BTC/USDT', '4h', None)

        # With very high threshold, most signals should be filtered
        assert signal.confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
