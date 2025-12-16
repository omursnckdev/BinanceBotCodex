"""
Tests for risk management module.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from risk.risk_manager import RiskManager, RiskStatus, RiskViolation
from config import RiskConfig


@pytest.fixture
def risk_config():
    """Create test risk configuration."""
    return RiskConfig(
        max_daily_loss_pct=5.0,
        max_consecutive_losses=5,
        max_total_exposure_pct=50.0,
        max_open_positions=3,
        default_risk_per_trade_pct=1.0,
        max_leverage=5,
        min_leverage=1,
        sl_atr_multiplier=1.5,
        sl_min_pct=0.3,
        sl_max_pct=0.9,
        tp_default_pct=1.2,
        min_risk_reward_ratio=1.2,
    )


@pytest.fixture
def mock_client():
    """Create mock Binance client."""
    client = Mock()
    client.get_usdt_balance.return_value = 10000.0
    client.get_positions.return_value = []
    return client


@pytest.fixture
def risk_manager(mock_client, risk_config):
    """Create RiskManager instance."""
    return RiskManager(mock_client, risk_config)


class TestRiskStatus:
    """Tests for risk status checking."""

    def test_initial_status_allows_trading(self, risk_manager):
        """Initial status should allow trading."""
        status = risk_manager.check_risk_status()
        assert status.is_trading_allowed
        assert len(status.violations) == 0

    def test_max_daily_loss_violation(self, risk_manager, mock_client):
        """Should detect max daily loss violation."""
        # Simulate 6% loss
        mock_client.get_usdt_balance.return_value = 9400.0  # Started at 10000

        # Force reinitialize daily stats with original balance
        risk_manager._daily_stats.starting_balance = 10000.0

        status = risk_manager.check_risk_status()

        assert not status.is_trading_allowed
        assert RiskViolation.MAX_DAILY_LOSS in status.violations

    def test_max_consecutive_losses_violation(self, risk_manager):
        """Should detect max consecutive losses violation."""
        # Record 5 consecutive losses
        for _ in range(5):
            risk_manager.record_trade_result(-100, is_win=False)

        status = risk_manager.check_risk_status()

        assert not status.is_trading_allowed
        assert RiskViolation.MAX_CONSECUTIVE_LOSSES in status.violations

    def test_consecutive_losses_reset_on_win(self, risk_manager):
        """Consecutive losses should reset on a win."""
        # Record 4 losses then 1 win
        for _ in range(4):
            risk_manager.record_trade_result(-100, is_win=False)
        risk_manager.record_trade_result(200, is_win=True)

        status = risk_manager.check_risk_status()

        assert status.is_trading_allowed
        assert status.consecutive_losses == 0

    def test_max_exposure_violation(self, risk_manager, mock_client):
        """Should detect max exposure violation."""
        # Mock position with 60% exposure
        position = Mock()
        position.notional = 6000.0
        position.unrealized_pnl = 0
        mock_client.get_positions.return_value = [position]

        status = risk_manager.check_risk_status()

        assert not status.is_trading_allowed
        assert RiskViolation.MAX_EXPOSURE in status.violations

    def test_max_positions_violation(self, risk_manager, mock_client):
        """Should detect max positions violation."""
        # Mock 3 positions (at max)
        positions = [Mock(notional=1000, unrealized_pnl=0) for _ in range(3)]
        mock_client.get_positions.return_value = positions

        status = risk_manager.check_risk_status()

        assert not status.is_trading_allowed
        assert RiskViolation.MAX_POSITIONS in status.violations


class TestCanOpenPosition:
    """Tests for position opening checks."""

    def test_can_open_when_allowed(self, risk_manager):
        """Should allow opening when all checks pass."""
        can_open, reason = risk_manager.can_open_position("BTCUSDT", 1000)
        assert can_open
        assert reason == "OK"

    def test_cannot_open_exceeds_exposure(self, risk_manager, mock_client):
        """Should deny when would exceed exposure limit."""
        # Try to open position that would exceed 50% exposure
        can_open, reason = risk_manager.can_open_position("BTCUSDT", 6000)
        assert not can_open
        assert "exposure" in reason.lower()

    def test_cannot_open_at_max_positions(self, risk_manager, mock_client):
        """Should deny when at max positions."""
        positions = [Mock(notional=1000, unrealized_pnl=0) for _ in range(3)]
        mock_client.get_positions.return_value = positions

        can_open, reason = risk_manager.can_open_position("BTCUSDT", 500)
        assert not can_open
        assert "position" in reason.lower()


class TestPositionSizing:
    """Tests for position size calculation."""

    def test_position_size_based_on_risk(self, risk_manager):
        """Position size should be based on risk per trade."""
        # 1% risk of 10000 = 100 USDT risk
        # With 1% stop loss: position = 100 / 0.01 = 10000 notional
        size = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            balance=10000,
            stop_loss_pct=1.0,
            leverage=1,
        )

        # Should be around 10000 or less (limited by exposure)
        assert size > 0
        assert size <= 10000

    def test_position_size_respects_exposure_limit(self, risk_manager, mock_client):
        """Position size should respect remaining exposure."""
        # Already have 4000 exposure
        position = Mock(notional=4000, unrealized_pnl=0)
        mock_client.get_positions.return_value = [position]

        size = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            balance=10000,
            stop_loss_pct=0.5,
            leverage=1,
        )

        # Max remaining exposure is 5000 - 4000 = 1000
        assert size <= 1000

    def test_position_size_zero_for_invalid_stop(self, risk_manager):
        """Position size should be 0 for invalid stop loss."""
        size = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            balance=10000,
            stop_loss_pct=0,  # Invalid
            leverage=1,
        )
        assert size == 0


class TestStopLossCalculation:
    """Tests for stop loss price calculation."""

    def test_long_stop_loss_below_entry(self, risk_manager):
        """Long stop loss should be below entry price."""
        sl = risk_manager.get_stop_loss_price(
            entry_price=50000,
            side="LONG",
            stop_loss_pct=1.0,
        )
        assert sl < 50000
        assert sl == 49500  # 50000 * (1 - 0.01)

    def test_short_stop_loss_above_entry(self, risk_manager):
        """Short stop loss should be above entry price."""
        sl = risk_manager.get_stop_loss_price(
            entry_price=50000,
            side="SHORT",
            stop_loss_pct=1.0,
        )
        assert sl > 50000
        assert sl == 50500  # 50000 * (1 + 0.01)


class TestTakeProfitCalculation:
    """Tests for take profit price calculation."""

    def test_long_take_profit_above_entry(self, risk_manager):
        """Long take profit should be above entry price."""
        tp = risk_manager.get_take_profit_price(
            entry_price=50000,
            side="LONG",
            take_profit_pct=1.5,
        )
        assert tp > 50000
        assert tp == 50750  # 50000 * (1 + 0.015)

    def test_short_take_profit_below_entry(self, risk_manager):
        """Short take profit should be below entry price."""
        tp = risk_manager.get_take_profit_price(
            entry_price=50000,
            side="SHORT",
            take_profit_pct=1.5,
        )
        assert tp < 50000
        assert tp == 49250  # 50000 * (1 - 0.015)


class TestStopTightening:
    """Tests for stop loss tightening."""

    def test_tighten_stop_when_profitable(self, risk_manager):
        """Should tighten stop when position is profitable."""
        should_tighten, new_stop = risk_manager.should_tighten_stop(
            entry_price=50000,
            current_price=50500,  # 1% profit
            side="LONG",
            current_stop_price=49500,  # Original stop
            profit_threshold_pct=0.5,
        )

        assert should_tighten
        assert new_stop > 49500  # Moved up

    def test_no_tighten_when_not_profitable(self, risk_manager):
        """Should not tighten when position not profitable enough."""
        should_tighten, new_stop = risk_manager.should_tighten_stop(
            entry_price=50000,
            current_price=50100,  # Only 0.2% profit
            side="LONG",
            current_stop_price=49500,
            profit_threshold_pct=0.5,
        )

        assert not should_tighten

    def test_tighten_stop_for_short(self, risk_manager):
        """Should tighten stop correctly for short position."""
        should_tighten, new_stop = risk_manager.should_tighten_stop(
            entry_price=50000,
            current_price=49500,  # 1% profit for short
            side="SHORT",
            current_stop_price=50500,  # Original stop
            profit_threshold_pct=0.5,
        )

        assert should_tighten
        assert new_stop < 50500  # Moved down


class TestTradeRecording:
    """Tests for trade result recording."""

    def test_record_winning_trade(self, risk_manager):
        """Should correctly record winning trade."""
        risk_manager.record_trade_result(150, is_win=True)

        assert risk_manager._daily_stats.trades_count == 1
        assert risk_manager._daily_stats.wins == 1
        assert risk_manager._daily_stats.losses == 0
        assert risk_manager._daily_stats.consecutive_losses == 0

    def test_record_losing_trade(self, risk_manager):
        """Should correctly record losing trade."""
        risk_manager.record_trade_result(-50, is_win=False)

        assert risk_manager._daily_stats.trades_count == 1
        assert risk_manager._daily_stats.wins == 0
        assert risk_manager._daily_stats.losses == 1
        assert risk_manager._daily_stats.consecutive_losses == 1

    def test_max_consecutive_losses_tracking(self, risk_manager):
        """Should track maximum consecutive losses."""
        # 3 losses, 1 win, 2 losses
        for _ in range(3):
            risk_manager.record_trade_result(-50, is_win=False)
        risk_manager.record_trade_result(100, is_win=True)
        for _ in range(2):
            risk_manager.record_trade_result(-50, is_win=False)

        assert risk_manager._daily_stats.max_consecutive_losses == 3
        assert risk_manager._daily_stats.consecutive_losses == 2
