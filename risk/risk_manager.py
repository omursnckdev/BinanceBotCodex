"""
Risk management module with kill switches.
Implements mandatory safety controls for trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from enum import Enum

from config import RiskConfig, get_settings
from exchange.binance_client import BinanceClient, Position
from utils.logger import get_logger


logger = get_logger(__name__)


class RiskViolation(Enum):
    """Types of risk violations."""
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_CONSECUTIVE_LOSSES = "max_consecutive_losses"
    MAX_EXPOSURE = "max_total_exposure"
    MAX_POSITIONS = "max_open_positions"


@dataclass
class RiskStatus:
    """Current risk status."""
    timestamp: datetime

    # Account status
    balance: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_exposure: float = 0.0
    exposure_pct: float = 0.0

    # Daily tracking
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0

    # Loss tracking
    consecutive_losses: int = 0
    open_positions: int = 0

    # Kill switch status
    is_trading_allowed: bool = True
    violations: List[RiskViolation] = field(default_factory=list)
    violation_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "balance": round(self.balance, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(self.daily_pnl_pct, 4),
            "consecutive_losses": self.consecutive_losses,
            "open_positions": self.open_positions,
            "exposure_pct": round(self.exposure_pct, 2),
            "is_trading_allowed": self.is_trading_allowed,
            "violations": [v.value for v in self.violations],
        }


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    starting_balance: float = 0.0
    current_balance: float = 0.0
    realized_pnl: float = 0.0
    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0


class RiskManager:
    """
    Risk manager with kill switches.

    Kill switches (mandatory):
    1. max_daily_loss -> stop trading for the day
    2. max_consecutive_losses -> stop opening new trades
    3. max_total_exposure_pct -> cap portfolio exposure
    4. max_open_positions -> cap number of positions
    """

    def __init__(
        self,
        client: BinanceClient,
        config: Optional[RiskConfig] = None,
    ):
        self.client = client
        self.config = config or get_settings().risk

        # Daily tracking
        self._daily_stats: Optional[DailyStats] = None
        self._last_check_date: Optional[date] = None

        # Initialize
        self._initialize_daily_stats()

    def _initialize_daily_stats(self) -> None:
        """Initialize or reset daily statistics."""
        today = date.today()

        if self._daily_stats is None or self._last_check_date != today:
            try:
                balance = self.client.get_usdt_balance()
            except Exception:
                balance = 0.0

            self._daily_stats = DailyStats(
                date=today,
                starting_balance=balance,
                current_balance=balance,
            )
            self._last_check_date = today

            logger.info(
                f"Daily stats initialized",
                extra={"date": str(today), "starting_balance": balance}
            )

    def check_risk_status(self) -> RiskStatus:
        """
        Check current risk status and enforce kill switches.

        Returns:
            RiskStatus with current state and any violations
        """
        self._initialize_daily_stats()
        timestamp = datetime.utcnow()

        status = RiskStatus(timestamp=timestamp)
        violations = []
        messages = []

        try:
            # Get account state
            status.balance = self.client.get_usdt_balance()
            positions = self.client.get_positions()

            # Calculate exposure
            status.open_positions = len(positions)
            status.total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
            status.total_exposure = sum(p.notional for p in positions)
            status.exposure_pct = (
                (status.total_exposure / status.balance * 100)
                if status.balance > 0 else 0
            )

            # Update daily stats
            self._daily_stats.current_balance = status.balance
            status.daily_pnl = status.balance - self._daily_stats.starting_balance
            status.daily_pnl_pct = (
                (status.daily_pnl / self._daily_stats.starting_balance * 100)
                if self._daily_stats.starting_balance > 0 else 0
            )
            status.trades_today = self._daily_stats.trades_count
            status.wins_today = self._daily_stats.wins
            status.losses_today = self._daily_stats.losses
            status.consecutive_losses = self._daily_stats.consecutive_losses

            # === KILL SWITCH CHECKS ===

            # 1. Max daily loss
            if status.daily_pnl_pct <= -self.config.max_daily_loss_pct:
                violations.append(RiskViolation.MAX_DAILY_LOSS)
                messages.append(
                    f"Daily loss {status.daily_pnl_pct:.2f}% exceeds max {self.config.max_daily_loss_pct}%"
                )

            # 2. Max consecutive losses
            if status.consecutive_losses >= self.config.max_consecutive_losses:
                violations.append(RiskViolation.MAX_CONSECUTIVE_LOSSES)
                messages.append(
                    f"Consecutive losses ({status.consecutive_losses}) exceeds max ({self.config.max_consecutive_losses})"
                )

            # 3. Max exposure
            if status.exposure_pct > self.config.max_total_exposure_pct:
                violations.append(RiskViolation.MAX_EXPOSURE)
                messages.append(
                    f"Exposure {status.exposure_pct:.2f}% exceeds max {self.config.max_total_exposure_pct}%"
                )

            # 4. Max positions
            if status.open_positions >= self.config.max_open_positions:
                violations.append(RiskViolation.MAX_POSITIONS)
                messages.append(
                    f"Open positions ({status.open_positions}) at max ({self.config.max_open_positions})"
                )

            status.violations = violations
            status.violation_messages = messages
            status.is_trading_allowed = len(violations) == 0

            if violations:
                for msg in messages:
                    logger.risk_event(msg)

        except Exception as e:
            logger.error(f"Risk check error: {e}")
            status.is_trading_allowed = False
            status.violation_messages = [f"Risk check error: {e}"]

        return status

    def can_open_position(self, symbol: str, notional: float) -> tuple:
        """
        Check if a new position can be opened.

        Args:
            symbol: Trading pair
            notional: Position notional value

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        status = self.check_risk_status()

        if not status.is_trading_allowed:
            return False, f"Trading disabled: {'; '.join(status.violation_messages)}"

        # Check if this position would exceed exposure
        new_exposure_pct = (
            (status.total_exposure + notional) / status.balance * 100
            if status.balance > 0 else 100
        )
        if new_exposure_pct > self.config.max_total_exposure_pct:
            return False, f"Would exceed max exposure ({new_exposure_pct:.1f}% > {self.config.max_total_exposure_pct}%)"

        # Check position count
        if status.open_positions >= self.config.max_open_positions:
            return False, f"Max positions reached ({status.open_positions})"

        return True, "OK"

    def record_trade_result(self, pnl: float, is_win: bool) -> None:
        """
        Record a trade result for daily tracking.

        Args:
            pnl: Trade PnL in USDT
            is_win: Whether trade was profitable
        """
        self._initialize_daily_stats()

        self._daily_stats.trades_count += 1
        self._daily_stats.realized_pnl += pnl

        if is_win:
            self._daily_stats.wins += 1
            self._daily_stats.consecutive_losses = 0
        else:
            self._daily_stats.losses += 1
            self._daily_stats.consecutive_losses += 1
            self._daily_stats.max_consecutive_losses = max(
                self._daily_stats.max_consecutive_losses,
                self._daily_stats.consecutive_losses
            )

        logger.info(
            f"Trade recorded",
            extra={
                "pnl": pnl,
                "is_win": is_win,
                "consecutive_losses": self._daily_stats.consecutive_losses,
                "trades_today": self._daily_stats.trades_count,
            }
        )

    def calculate_position_size(
        self,
        symbol: str,
        balance: float,
        stop_loss_pct: float,
        leverage: int = 1,
    ) -> float:
        """
        Calculate safe position size based on risk parameters.

        Args:
            symbol: Trading pair
            balance: Account balance
            stop_loss_pct: Stop loss percentage
            leverage: Position leverage

        Returns:
            Maximum notional value for position
        """
        if stop_loss_pct <= 0:
            return 0.0

        # Risk amount based on config
        risk_amount = balance * (self.config.default_risk_per_trade_pct / 100)

        # Position size = risk / stop_loss
        max_notional = risk_amount / (stop_loss_pct / 100)

        # Check exposure limits
        status = self.check_risk_status()
        remaining_exposure = balance * (self.config.max_total_exposure_pct / 100) - status.total_exposure
        max_notional = min(max_notional, remaining_exposure)

        return max(0, max_notional)

    def get_stop_loss_price(
        self,
        entry_price: float,
        side: str,
        stop_loss_pct: float,
    ) -> float:
        """
        Calculate stop loss price from entry price.

        IMPORTANT: Use actual fill price, not ticker price.

        Args:
            entry_price: Actual entry fill price
            side: 'LONG' or 'SHORT'
            stop_loss_pct: Stop loss percentage

        Returns:
            Stop loss price
        """
        if side.upper() == "LONG":
            return entry_price * (1 - stop_loss_pct / 100)
        else:
            return entry_price * (1 + stop_loss_pct / 100)

    def get_take_profit_price(
        self,
        entry_price: float,
        side: str,
        take_profit_pct: float,
    ) -> float:
        """
        Calculate take profit price from entry price.

        IMPORTANT: Use actual fill price, not ticker price.

        Args:
            entry_price: Actual entry fill price
            side: 'LONG' or 'SHORT'
            take_profit_pct: Take profit percentage

        Returns:
            Take profit price
        """
        if side.upper() == "LONG":
            return entry_price * (1 + take_profit_pct / 100)
        else:
            return entry_price * (1 - take_profit_pct / 100)

    def should_tighten_stop(
        self,
        entry_price: float,
        current_price: float,
        side: str,
        current_stop_price: float,
        profit_threshold_pct: float = 0.5,
    ) -> tuple:
        """
        Check if stop should be tightened to lock in profits.

        Args:
            entry_price: Entry price
            current_price: Current market price
            side: Position side
            current_stop_price: Current stop loss price
            profit_threshold_pct: Minimum profit to trigger tightening

        Returns:
            Tuple of (should_tighten: bool, new_stop_price: float)
        """
        if side.upper() == "LONG":
            pnl_pct = (current_price - entry_price) / entry_price * 100
            if pnl_pct >= profit_threshold_pct:
                # Move stop to break-even + small buffer
                new_stop = entry_price * 1.001
                if new_stop > current_stop_price:
                    return True, new_stop
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
            if pnl_pct >= profit_threshold_pct:
                new_stop = entry_price * 0.999
                if new_stop < current_stop_price:
                    return True, new_stop

        return False, current_stop_price

    def reset_daily_stats(self) -> None:
        """Force reset of daily statistics (for testing)."""
        self._daily_stats = None
        self._last_check_date = None
        self._initialize_daily_stats()
