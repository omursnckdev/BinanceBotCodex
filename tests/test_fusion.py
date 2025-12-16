"""
Tests for fusion strategy module.
"""

import pytest
from datetime import datetime

from strategy.fusion import FusionStrategy, FusionDecision
from signals.indicators import TechnicalSignal
from signals.whales import WhaleSignal
from signals.sentiment import SentimentSignal
from config import (
    FusionConfig, TradingAction, Trend, WhaleDefensiveAction
)


@pytest.fixture
def fusion_config():
    """Create test fusion configuration."""
    return FusionConfig(
        entry_threshold=0.3,
        min_confidence_for_entry=0.4,
        high_confidence_threshold=0.7,
        require_trend_confirmation=True,
    )


@pytest.fixture
def fusion(fusion_config):
    """Create FusionStrategy instance."""
    return FusionStrategy(fusion_config)


@pytest.fixture
def bullish_tech_signal():
    """Create bullish technical signal."""
    return TechnicalSignal(
        symbol="BTCUSDT",
        tech_score=0.6,
        tech_confidence=0.7,
        trend=Trend.BULLISH,
        atr_value=500,
        atr_pct=1.0,
        rsi=45,
        macd=100,
        macd_signal=80,
        macd_histogram=20,
        bb_upper=52000,
        bb_middle=50000,
        bb_lower=48000,
        bb_position=0.2,
        ema_fast=50500,
        ema_slow=49500,
        current_price=50000,
        is_valid=True,
    )


@pytest.fixture
def bearish_tech_signal():
    """Create bearish technical signal."""
    return TechnicalSignal(
        symbol="BTCUSDT",
        tech_score=-0.6,
        tech_confidence=0.7,
        trend=Trend.BEARISH,
        atr_value=500,
        atr_pct=1.0,
        rsi=55,
        macd=-100,
        macd_signal=-80,
        macd_histogram=-20,
        bb_upper=52000,
        bb_middle=50000,
        bb_lower=48000,
        bb_position=-0.2,
        ema_fast=49500,
        ema_slow=50500,
        current_price=50000,
        is_valid=True,
    )


@pytest.fixture
def neutral_tech_signal():
    """Create neutral technical signal."""
    return TechnicalSignal(
        symbol="BTCUSDT",
        tech_score=0.1,
        tech_confidence=0.4,
        trend=Trend.NEUTRAL,
        atr_value=500,
        atr_pct=1.0,
        rsi=50,
        macd=0,
        macd_signal=0,
        macd_histogram=0,
        bb_upper=52000,
        bb_middle=50000,
        bb_lower=48000,
        bb_position=0,
        ema_fast=50000,
        ema_slow=50000,
        current_price=50000,
        is_valid=True,
    )


@pytest.fixture
def neutral_whale_signal():
    """Create neutral whale signal."""
    return WhaleSignal(
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        whale_score=0.0,
        whale_confidence=0.3,
        whale_alert_opposite=False,
        recommended_action=WhaleDefensiveAction.NONE,
    )


@pytest.fixture
def alert_whale_signal():
    """Create whale signal with alert."""
    return WhaleSignal(
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        whale_score=-0.7,
        whale_confidence=0.8,
        whale_alert_opposite=True,
        recommended_action=WhaleDefensiveAction.CLOSE_MARKET,
        detection_reason="Large bearish whale activity",
    )


@pytest.fixture
def neutral_sentiment_signal():
    """Create neutral sentiment signal."""
    return SentimentSignal(
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        sentiment_score=0.0,
        confidence=0.5,
        global_sentiment_score=0.0,
        global_confidence=0.5,
    )


class TestFusionDecision:
    """Tests for fusion decision making."""

    def test_bullish_entry_with_trend(
        self, fusion, bullish_tech_signal, neutral_whale_signal, neutral_sentiment_signal
    ):
        """Should generate LONG signal with bullish trend confirmation."""
        decision = fusion.fuse_signals(
            symbol="BTCUSDT",
            tech_signal=bullish_tech_signal,
            whale_signal=neutral_whale_signal,
            sentiment_signal=neutral_sentiment_signal,
        )

        assert decision.action == TradingAction.LONG
        assert decision.final_score > 0
        assert decision.confidence > 0

    def test_bearish_entry_with_trend(
        self, fusion, bearish_tech_signal, neutral_whale_signal, neutral_sentiment_signal
    ):
        """Should generate SHORT signal with bearish trend confirmation."""
        decision = fusion.fuse_signals(
            symbol="BTCUSDT",
            tech_signal=bearish_tech_signal,
            whale_signal=neutral_whale_signal,
            sentiment_signal=neutral_sentiment_signal,
        )

        assert decision.action == TradingAction.SHORT
        assert decision.final_score < 0

    def test_flat_below_threshold(
        self, fusion, neutral_tech_signal, neutral_whale_signal, neutral_sentiment_signal
    ):
        """Should generate FLAT when score below threshold."""
        decision = fusion.fuse_signals(
            symbol="BTCUSDT",
            tech_signal=neutral_tech_signal,
            whale_signal=neutral_whale_signal,
            sentiment_signal=neutral_sentiment_signal,
        )

        assert decision.action == TradingAction.FLAT
        assert "threshold" in decision.decision_reason.lower()

    def test_flat_does_not_close(
        self, fusion, neutral_tech_signal, neutral_whale_signal, neutral_sentiment_signal
    ):
        """FLAT should not trigger position close - it means no new entry only."""
        decision = fusion.fuse_signals(
            symbol="BTCUSDT",
            tech_signal=neutral_tech_signal,
            whale_signal=neutral_whale_signal,
            sentiment_signal=neutral_sentiment_signal,
            current_position_side="LONG",  # Has existing position
        )

        # FLAT should not set exit_reason
        assert decision.action == TradingAction.FLAT
        assert decision.exit_reason is None
        assert decision.recommended_defensive_action is None


class TestWhaleDefensiveAction:
    """Tests for whale defensive actions."""

    def test_whale_alert_sets_defensive_action(
        self, fusion, bullish_tech_signal, alert_whale_signal, neutral_sentiment_signal
    ):
        """Whale alert should set explicit defensive action, not FLAT."""
        decision = fusion.fuse_signals(
            symbol="BTCUSDT",
            tech_signal=bullish_tech_signal,
            whale_signal=alert_whale_signal,
            sentiment_signal=neutral_sentiment_signal,
            current_position_side="LONG",  # In position opposite to whale activity
        )

        # Should have defensive action set
        assert decision.recommended_defensive_action == WhaleDefensiveAction.CLOSE_MARKET
        assert decision.exit_reason is not None

    def test_whale_alert_without_position(
        self, fusion, bullish_tech_signal, alert_whale_signal, neutral_sentiment_signal
    ):
        """Whale alert without position should not trigger defensive action."""
        decision = fusion.fuse_signals(
            symbol="BTCUSDT",
            tech_signal=bullish_tech_signal,
            whale_signal=alert_whale_signal,
            sentiment_signal=neutral_sentiment_signal,
            current_position_side=None,  # No position
        )

        # No defensive action when not in position
        assert decision.recommended_defensive_action is None


class TestTrendConfirmation:
    """Tests for trend confirmation logic."""

    def test_no_long_without_bullish_trend(self, fusion_config, neutral_whale_signal, neutral_sentiment_signal):
        """Should not go LONG without bullish trend (unless high confidence)."""
        fusion_config.require_trend_confirmation = True
        fusion = FusionStrategy(fusion_config)

        # Bullish signal but neutral trend
        tech_signal = TechnicalSignal(
            symbol="BTCUSDT",
            tech_score=0.5,
            tech_confidence=0.5,  # Not high enough to override
            trend=Trend.NEUTRAL,
            atr_value=500, atr_pct=1.0, rsi=45,
            macd=100, macd_signal=80, macd_histogram=20,
            bb_upper=52000, bb_middle=50000, bb_lower=48000, bb_position=0.2,
            ema_fast=50000, ema_slow=50000, current_price=50000,
            is_valid=True,
        )

        decision = fusion.fuse_signals(
            symbol="BTCUSDT",
            tech_signal=tech_signal,
            whale_signal=neutral_whale_signal,
            sentiment_signal=neutral_sentiment_signal,
        )

        # Should be FLAT due to no trend confirmation
        assert decision.action == TradingAction.FLAT


class TestPositionSizing:
    """Tests for position sizing calculations."""

    def test_leverage_scales_with_confidence(
        self, fusion, bullish_tech_signal, neutral_whale_signal, neutral_sentiment_signal
    ):
        """Leverage should scale with confidence."""
        # High confidence
        bullish_tech_signal.tech_confidence = 0.9
        high_conf_decision = fusion.fuse_signals(
            symbol="BTCUSDT",
            tech_signal=bullish_tech_signal,
            whale_signal=neutral_whale_signal,
            sentiment_signal=neutral_sentiment_signal,
        )

        # Lower confidence
        bullish_tech_signal.tech_confidence = 0.5
        low_conf_decision = fusion.fuse_signals(
            symbol="BTCUSDT",
            tech_signal=bullish_tech_signal,
            whale_signal=neutral_whale_signal,
            sentiment_signal=neutral_sentiment_signal,
        )

        assert high_conf_decision.leverage >= low_conf_decision.leverage

    def test_stop_loss_always_set(
        self, fusion, bullish_tech_signal, neutral_whale_signal, neutral_sentiment_signal
    ):
        """Stop loss should always be set for entry signals."""
        decision = fusion.fuse_signals(
            symbol="BTCUSDT",
            tech_signal=bullish_tech_signal,
            whale_signal=neutral_whale_signal,
            sentiment_signal=neutral_sentiment_signal,
        )

        if decision.action != TradingAction.FLAT:
            assert decision.stop_loss_pct > 0
            assert decision.take_profit_pct > 0


class TestInvalidSignals:
    """Tests for handling invalid signals."""

    def test_invalid_tech_signal(self, fusion, neutral_whale_signal, neutral_sentiment_signal):
        """Should handle invalid technical signal."""
        invalid_signal = TechnicalSignal(
            symbol="BTCUSDT",
            tech_score=0, tech_confidence=0, trend=Trend.NEUTRAL,
            atr_value=0, atr_pct=0, rsi=50,
            macd=0, macd_signal=0, macd_histogram=0,
            bb_upper=0, bb_middle=0, bb_lower=0, bb_position=0,
            ema_fast=0, ema_slow=0, current_price=0,
            is_valid=False,
            error_message="Test error",
        )

        decision = fusion.fuse_signals(
            symbol="BTCUSDT",
            tech_signal=invalid_signal,
            whale_signal=neutral_whale_signal,
            sentiment_signal=neutral_sentiment_signal,
        )

        assert decision.action == TradingAction.FLAT
        assert "invalid" in decision.decision_reason.lower()

    def test_market_conditions_not_ok(
        self, fusion, bullish_tech_signal, neutral_whale_signal, neutral_sentiment_signal
    ):
        """Should return FLAT when market conditions not OK."""
        decision = fusion.fuse_signals(
            symbol="BTCUSDT",
            tech_signal=bullish_tech_signal,
            whale_signal=neutral_whale_signal,
            sentiment_signal=neutral_sentiment_signal,
            market_conditions_ok=False,
        )

        assert decision.action == TradingAction.FLAT
        assert "conditions" in decision.decision_reason.lower()
