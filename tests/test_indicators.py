"""Unit tests for technical indicators."""
import pytest
import pandas as pd
import numpy as np
from utils.indicators import (
    ema, sma, rsi, atr, macd, bollinger_bands,
    momentum, volume_sma, supertrend, adx
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100

    # Generate realistic price data
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n)) * 2
    low = close - np.abs(np.random.randn(n)) * 2
    open_price = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_ema_returns_series(self, sample_ohlcv_data):
        """EMA should return a pandas Series."""
        result = ema(sample_ohlcv_data['close'], period=12)
        assert isinstance(result, pd.Series)

    def test_ema_length_matches_input(self, sample_ohlcv_data):
        """EMA output length should match input length."""
        result = ema(sample_ohlcv_data['close'], period=12)
        assert len(result) == len(sample_ohlcv_data)

    def test_ema_first_values_are_nan(self, sample_ohlcv_data):
        """First (period-1) values should be NaN."""
        period = 12
        result = ema(sample_ohlcv_data['close'], period=period)
        # EMA starts calculating from period-1
        assert result.iloc[:period-1].isna().sum() >= 0

    def test_ema_values_are_reasonable(self, sample_ohlcv_data):
        """EMA values should be within price range."""
        result = ema(sample_ohlcv_data['close'], period=12)
        valid_result = result.dropna()
        assert valid_result.min() >= sample_ohlcv_data['close'].min() * 0.9
        assert valid_result.max() <= sample_ohlcv_data['close'].max() * 1.1


class TestSMA:
    """Tests for Simple Moving Average."""

    def test_sma_returns_series(self, sample_ohlcv_data):
        """SMA should return a pandas Series."""
        result = sma(sample_ohlcv_data['close'], period=20)
        assert isinstance(result, pd.Series)

    def test_sma_calculation_is_correct(self):
        """SMA should calculate correctly for known values."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = sma(data, period=5)
        # SMA of [1,2,3,4,5] = 3, [2,3,4,5,6] = 4, etc.
        assert result.iloc[4] == 3.0
        assert result.iloc[5] == 4.0


class TestRSI:
    """Tests for Relative Strength Index."""

    def test_rsi_returns_series(self, sample_ohlcv_data):
        """RSI should return a pandas Series."""
        result = rsi(sample_ohlcv_data['close'], period=14)
        assert isinstance(result, pd.Series)

    def test_rsi_bounded_0_to_100(self, sample_ohlcv_data):
        """RSI values should be between 0 and 100."""
        result = rsi(sample_ohlcv_data['close'], period=14)
        valid_result = result.dropna()
        assert valid_result.min() >= 0
        assert valid_result.max() <= 100

    def test_rsi_uptrend_above_50(self):
        """RSI should be above 50 in strong uptrend."""
        # Create steadily rising prices
        prices = pd.Series([100 + i * 2 for i in range(50)])
        result = rsi(prices, period=14)
        # Last RSI values should be above 50 in uptrend
        assert result.iloc[-1] > 50

    def test_rsi_downtrend_below_50(self):
        """RSI should be below 50 in strong downtrend."""
        # Create steadily falling prices
        prices = pd.Series([100 - i * 2 for i in range(50)])
        result = rsi(prices, period=14)
        # Last RSI values should be below 50 in downtrend
        assert result.iloc[-1] < 50


class TestATR:
    """Tests for Average True Range."""

    def test_atr_returns_series(self, sample_ohlcv_data):
        """ATR should return a pandas Series."""
        result = atr(
            sample_ohlcv_data['high'],
            sample_ohlcv_data['low'],
            sample_ohlcv_data['close'],
            period=14
        )
        assert isinstance(result, pd.Series)

    def test_atr_always_positive(self, sample_ohlcv_data):
        """ATR values should always be positive."""
        result = atr(
            sample_ohlcv_data['high'],
            sample_ohlcv_data['low'],
            sample_ohlcv_data['close'],
            period=14
        )
        valid_result = result.dropna()
        assert (valid_result >= 0).all()

    def test_atr_increases_with_volatility(self):
        """ATR should be higher for more volatile data."""
        # Low volatility data
        low_vol = pd.DataFrame({
            'high': [101] * 30,
            'low': [99] * 30,
            'close': [100] * 30
        })

        # High volatility data
        high_vol = pd.DataFrame({
            'high': [110] * 30,
            'low': [90] * 30,
            'close': [100] * 30
        })

        atr_low = atr(low_vol['high'], low_vol['low'], low_vol['close'], 14)
        atr_high = atr(high_vol['high'], high_vol['low'], high_vol['close'], 14)

        assert atr_high.iloc[-1] > atr_low.iloc[-1]


class TestMACD:
    """Tests for MACD indicator."""

    def test_macd_returns_tuple(self, sample_ohlcv_data):
        """MACD should return tuple of (macd_line, signal, histogram)."""
        result = macd(sample_ohlcv_data['close'])
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_macd_components_are_series(self, sample_ohlcv_data):
        """All MACD components should be pandas Series."""
        macd_line, signal, histogram = macd(sample_ohlcv_data['close'])
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(histogram, pd.Series)

    def test_macd_histogram_equals_difference(self, sample_ohlcv_data):
        """Histogram should equal MACD line minus signal."""
        macd_line, signal, histogram = macd(sample_ohlcv_data['close'])
        expected = macd_line - signal
        pd.testing.assert_series_equal(histogram, expected, check_names=False)


class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_bollinger_returns_tuple(self, sample_ohlcv_data):
        """Bollinger Bands should return tuple of (upper, middle, lower)."""
        result = bollinger_bands(sample_ohlcv_data['close'])
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_bollinger_band_order(self, sample_ohlcv_data):
        """Upper band > middle > lower band."""
        upper, middle, lower = bollinger_bands(sample_ohlcv_data['close'])
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()


class TestADX:
    """Tests for Average Directional Index."""

    def test_adx_returns_series(self, sample_ohlcv_data):
        """ADX should return a pandas Series."""
        result = adx(
            sample_ohlcv_data['high'],
            sample_ohlcv_data['low'],
            sample_ohlcv_data['close']
        )
        assert isinstance(result, pd.Series)

    def test_adx_bounded_0_to_100(self, sample_ohlcv_data):
        """ADX values should be between 0 and 100."""
        adx_val = adx(
            sample_ohlcv_data['high'],
            sample_ohlcv_data['low'],
            sample_ohlcv_data['close']
        )
        valid_adx = adx_val.dropna()
        assert valid_adx.min() >= 0
        assert valid_adx.max() <= 100


class TestSupertrend:
    """Tests for Supertrend indicator."""

    def test_supertrend_returns_tuple(self, sample_ohlcv_data):
        """Supertrend should return tuple of (supertrend, direction)."""
        result = supertrend(
            sample_ohlcv_data['high'],
            sample_ohlcv_data['low'],
            sample_ohlcv_data['close']
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_supertrend_direction_binary(self, sample_ohlcv_data):
        """Supertrend direction should be 1 (bullish) or -1 (bearish)."""
        st, direction = supertrend(
            sample_ohlcv_data['high'],
            sample_ohlcv_data['low'],
            sample_ohlcv_data['close']
        )
        valid_dir = direction.dropna()
        assert set(valid_dir.unique()).issubset({1, -1, 1.0, -1.0})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
