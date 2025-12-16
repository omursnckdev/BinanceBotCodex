"""
Tests for technical indicators module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from signals.indicators import TechnicalIndicators, TechnicalSignal
from config import IndicatorConfig, Trend


@pytest.fixture
def indicator_config():
    """Create test indicator configuration."""
    return IndicatorConfig()


@pytest.fixture
def indicators(indicator_config):
    """Create TechnicalIndicators instance."""
    return TechnicalIndicators(indicator_config)


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 300

    # Generate realistic price data
    base_price = 50000
    returns = np.random.randn(n) * 0.01
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n) * 0.001),
        'high': prices * (1 + abs(np.random.randn(n)) * 0.005),
        'low': prices * (1 - abs(np.random.randn(n)) * 0.005),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n),
        'open_time': range(n),
        'close_time': range(1, n + 1),
    })

    # Ensure high >= open, close, low
    df['high'] = df[['open', 'high', 'close']].max(axis=1) * 1.001
    df['low'] = df[['open', 'low', 'close']].min(axis=1) * 0.999

    return df


class TestRSI:
    """Tests for RSI calculation."""

    def test_rsi_returns_series(self, indicators, sample_ohlcv):
        """RSI should return a pandas Series."""
        rsi = indicators.calculate_rsi(sample_ohlcv['close'])
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_ohlcv)

    def test_rsi_bounded(self, indicators, sample_ohlcv):
        """RSI values should be between 0 and 100."""
        rsi = indicators.calculate_rsi(sample_ohlcv['close'])
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_rsi_no_nan_after_warmup(self, indicators, sample_ohlcv):
        """RSI should not have NaN values after warmup period."""
        rsi = indicators.calculate_rsi(sample_ohlcv['close'])
        # After 2x period, should have no NaN
        warmup = 2 * indicators.config.rsi_period
        assert not rsi.iloc[warmup:].isna().any()


class TestMACD:
    """Tests for MACD calculation."""

    def test_macd_returns_tuple(self, indicators, sample_ohlcv):
        """MACD should return tuple of three Series."""
        result = indicators.calculate_macd(sample_ohlcv['close'])
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(s, pd.Series) for s in result)

    def test_macd_histogram_equals_diff(self, indicators, sample_ohlcv):
        """MACD histogram should equal MACD line minus signal line."""
        macd, signal, histogram = indicators.calculate_macd(sample_ohlcv['close'])
        calculated_hist = macd - signal
        np.testing.assert_array_almost_equal(histogram.values, calculated_hist.values)

    def test_macd_correct_length(self, indicators, sample_ohlcv):
        """MACD components should have correct length."""
        macd, signal, histogram = indicators.calculate_macd(sample_ohlcv['close'])
        assert len(macd) == len(sample_ohlcv)
        assert len(signal) == len(sample_ohlcv)
        assert len(histogram) == len(sample_ohlcv)


class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""

    def test_bollinger_returns_tuple(self, indicators, sample_ohlcv):
        """Bollinger Bands should return tuple of three Series."""
        result = indicators.calculate_bollinger_bands(sample_ohlcv['close'])
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_bollinger_ordering(self, indicators, sample_ohlcv):
        """Upper band should always be >= middle >= lower."""
        upper, middle, lower = indicators.calculate_bollinger_bands(sample_ohlcv['close'])
        # Skip NaN values at start
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()


class TestATR:
    """Tests for ATR calculation."""

    def test_atr_positive(self, indicators, sample_ohlcv):
        """ATR should always be positive."""
        atr = indicators.calculate_atr(sample_ohlcv)
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()

    def test_atr_correct_length(self, indicators, sample_ohlcv):
        """ATR should have correct length."""
        atr = indicators.calculate_atr(sample_ohlcv)
        assert len(atr) == len(sample_ohlcv)


class TestEMA:
    """Tests for EMA calculation."""

    def test_ema_follows_price(self, indicators, sample_ohlcv):
        """EMA should follow price trend."""
        ema = indicators.calculate_ema(sample_ohlcv['close'], 20)
        # Check correlation with price
        corr = ema.corr(sample_ohlcv['close'])
        assert corr > 0.9  # Strong positive correlation

    def test_fast_ema_more_reactive(self, indicators, sample_ohlcv):
        """Faster EMA should be more reactive to price changes."""
        fast_ema = indicators.calculate_ema(sample_ohlcv['close'], 10)
        slow_ema = indicators.calculate_ema(sample_ohlcv['close'], 50)

        # Fast EMA should be closer to current price
        price = sample_ohlcv['close']
        fast_diff = abs(fast_ema - price).mean()
        slow_diff = abs(slow_ema - price).mean()
        assert fast_diff < slow_diff


class TestAnalyze:
    """Tests for full technical analysis."""

    def test_analyze_returns_signal(self, indicators, sample_ohlcv):
        """Analyze should return TechnicalSignal."""
        signal = indicators.analyze(sample_ohlcv, "BTCUSDT")
        assert isinstance(signal, TechnicalSignal)

    def test_analyze_valid_output(self, indicators, sample_ohlcv):
        """Analyze should return valid signal with proper bounds."""
        signal = indicators.analyze(sample_ohlcv, "BTCUSDT")

        assert signal.is_valid
        assert -1 <= signal.tech_score <= 1
        assert 0 <= signal.tech_confidence <= 1
        assert signal.trend in [Trend.BULLISH, Trend.BEARISH, Trend.NEUTRAL]
        assert signal.atr_pct > 0
        assert 0 <= signal.rsi <= 100

    def test_analyze_insufficient_data(self, indicators):
        """Analyze should handle insufficient data gracefully."""
        small_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
        })

        signal = indicators.analyze(small_df, "BTCUSDT")
        assert not signal.is_valid
        assert "Insufficient data" in signal.error_message

    def test_analyze_missing_columns(self, indicators):
        """Analyze should handle missing columns gracefully."""
        df = pd.DataFrame({
            'close': [100, 101, 102],
        })

        signal = indicators.analyze(df, "BTCUSDT")
        assert not signal.is_valid


class TestTrendDetection:
    """Tests for trend detection."""

    def test_bullish_trend_detection(self, indicators):
        """Should detect bullish trend when fast EMA > slow EMA."""
        # Create uptrending data
        n = 300
        prices = np.linspace(100, 150, n)  # Clear uptrend

        df = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.ones(n) * 100,
        })

        signal = indicators.analyze(df, "TEST")
        assert signal.trend == Trend.BULLISH

    def test_bearish_trend_detection(self, indicators):
        """Should detect bearish trend when fast EMA < slow EMA."""
        # Create downtrending data
        n = 300
        prices = np.linspace(150, 100, n)  # Clear downtrend

        df = pd.DataFrame({
            'open': prices * 1.001,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.ones(n) * 100,
        })

        signal = indicators.analyze(df, "TEST")
        assert signal.trend == Trend.BEARISH
