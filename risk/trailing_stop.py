"""
Trailing Stop Manager (v3 new feature).
Implements dynamic trailing stops for profit protection in scalping strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

from config import TrailingStopConfig, get_settings
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class TrailingStopState:
    """Tracking state for a single position's trailing stop."""
    symbol: str
    entry_price: float
    side: str  # "LONG" or "SHORT"
    initial_stop: float
    current_stop: float

    # Tracking
    highest_pnl_pct: float = 0.0
    lowest_pnl_pct: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0

    # State flags
    breakeven_reached: bool = False
    trailing_active: bool = False

    last_update: datetime = None

    def __post_init__(self):
        self.last_update = datetime.utcnow()
        if self.side == "LONG":
            self.highest_price = self.entry_price
            self.lowest_price = self.entry_price
        else:
            self.highest_price = self.entry_price
            self.lowest_price = self.entry_price


class TrailingStopManager:
    """
    Manages trailing stops for all active positions.

    Trailing Stop Phases:
    1. Initial: Stop at original SL level
    2. Breakeven: Move stop to entry price + small buffer when profit >= breakeven_at_pct
    3. Trailing: Trail behind price by trail_distance_pct when profit >= activation_pct

    Benefits for scalping:
    - Locks in profits on winning trades
    - Moves to breakeven quickly to eliminate risk
    - Lets winners run while protecting gains
    """

    def __init__(self, config: Optional[TrailingStopConfig] = None):
        self.config = config or get_settings().risk.trailing
        self._states: Dict[str, TrailingStopState] = {}

    def initialize_position(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        initial_stop: float,
    ) -> TrailingStopState:
        """
        Initialize trailing stop tracking for a new position.

        Args:
            symbol: Trading pair
            entry_price: Actual fill price
            side: "LONG" or "SHORT"
            initial_stop: Initial stop loss price

        Returns:
            TrailingStopState for the position
        """
        state = TrailingStopState(
            symbol=symbol,
            entry_price=entry_price,
            side=side.upper(),
            initial_stop=initial_stop,
            current_stop=initial_stop,
        )

        self._states[symbol] = state

        logger.info(
            f"Trailing stop initialized",
            extra={
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "initial_stop": initial_stop,
                "config": {
                    "breakeven_at": self.config.breakeven_at_pct,
                    "activation": self.config.activation_pct,
                    "trail_distance": self.config.trail_distance_pct,
                }
            }
        )

        return state

    def update(
        self,
        symbol: str,
        current_price: float,
    ) -> Tuple[bool, float, str]:
        """
        Update trailing stop based on current price.

        Args:
            symbol: Trading pair
            current_price: Current market price

        Returns:
            Tuple of (should_update: bool, new_stop: float, reason: str)
        """
        if not self.config.enabled:
            return False, 0.0, "Trailing stop disabled"

        state = self._states.get(symbol)
        if not state:
            return False, 0.0, "No trailing stop state for symbol"

        # Calculate current PnL percentage
        if state.side == "LONG":
            pnl_pct = (current_price - state.entry_price) / state.entry_price * 100
            # Track highest price for trailing
            state.highest_price = max(state.highest_price, current_price)
        else:  # SHORT
            pnl_pct = (state.entry_price - current_price) / state.entry_price * 100
            # Track lowest price for trailing
            state.lowest_price = min(state.lowest_price, current_price)

        # Track PnL extremes
        state.highest_pnl_pct = max(state.highest_pnl_pct, pnl_pct)
        state.lowest_pnl_pct = min(state.lowest_pnl_pct, pnl_pct)
        state.last_update = datetime.utcnow()

        # Current stop
        old_stop = state.current_stop

        # Phase 1: Check for breakeven
        if not state.breakeven_reached and pnl_pct >= self.config.breakeven_at_pct:
            new_stop = self._calculate_breakeven_stop(state)
            if self._is_better_stop(new_stop, old_stop, state.side):
                state.current_stop = new_stop
                state.breakeven_reached = True
                logger.info(
                    f"Trailing stop: Moved to breakeven",
                    extra={
                        "symbol": symbol,
                        "pnl_pct": round(pnl_pct, 2),
                        "new_stop": new_stop,
                    }
                )
                return True, new_stop, f"Breakeven reached (PnL: {pnl_pct:.2f}%)"

        # Phase 2: Activate trailing stop
        if pnl_pct >= self.config.activation_pct:
            state.trailing_active = True
            new_stop = self._calculate_trailing_stop(state, current_price)

            if self._is_better_stop(new_stop, old_stop, state.side):
                state.current_stop = new_stop
                logger.debug(
                    f"Trailing stop updated",
                    extra={
                        "symbol": symbol,
                        "pnl_pct": round(pnl_pct, 2),
                        "old_stop": old_stop,
                        "new_stop": new_stop,
                    }
                )
                return True, new_stop, f"Trailing (PnL: {pnl_pct:.2f}%, highest: {state.highest_pnl_pct:.2f}%)"

        return False, state.current_stop, "No update needed"

    def _calculate_breakeven_stop(self, state: TrailingStopState) -> float:
        """Calculate breakeven stop price with small buffer."""
        # Add 0.05% buffer above/below entry to account for fees
        buffer_pct = 0.05 / 100

        if state.side == "LONG":
            return state.entry_price * (1 + buffer_pct)
        else:
            return state.entry_price * (1 - buffer_pct)

    def _calculate_trailing_stop(
        self,
        state: TrailingStopState,
        current_price: float,
    ) -> float:
        """Calculate trailing stop price."""
        trail_distance = self.config.trail_distance_pct / 100

        if state.side == "LONG":
            # Trail below the highest price reached
            reference_price = state.highest_price
            new_stop = reference_price * (1 - trail_distance)
        else:  # SHORT
            # Trail above the lowest price reached
            reference_price = state.lowest_price
            new_stop = reference_price * (1 + trail_distance)

        return new_stop

    def _is_better_stop(
        self,
        new_stop: float,
        current_stop: float,
        side: str,
    ) -> bool:
        """
        Check if new stop is better (more protective) than current.

        For LONG: Higher stop is better
        For SHORT: Lower stop is better
        """
        if side == "LONG":
            return new_stop > current_stop
        else:
            return new_stop < current_stop

    def get_state(self, symbol: str) -> Optional[TrailingStopState]:
        """Get trailing stop state for a symbol."""
        return self._states.get(symbol)

    def get_current_stop(self, symbol: str) -> Optional[float]:
        """Get current trailing stop price."""
        state = self._states.get(symbol)
        return state.current_stop if state else None

    def is_trailing_active(self, symbol: str) -> bool:
        """Check if trailing is actively tracking."""
        state = self._states.get(symbol)
        return state.trailing_active if state else False

    def is_breakeven_reached(self, symbol: str) -> bool:
        """Check if breakeven has been reached."""
        state = self._states.get(symbol)
        return state.breakeven_reached if state else False

    def reset(self, symbol: str) -> None:
        """Reset trailing stop state for a symbol (after position close)."""
        if symbol in self._states:
            logger.info(
                f"Trailing stop reset",
                extra={
                    "symbol": symbol,
                    "final_state": {
                        "highest_pnl": self._states[symbol].highest_pnl_pct,
                        "trailing_was_active": self._states[symbol].trailing_active,
                        "breakeven_reached": self._states[symbol].breakeven_reached,
                    }
                }
            )
            del self._states[symbol]

    def reset_all(self) -> None:
        """Reset all trailing stop states."""
        self._states.clear()

    def get_summary(self, symbol: str) -> Dict:
        """Get summary of trailing stop state for logging."""
        state = self._states.get(symbol)
        if not state:
            return {"symbol": symbol, "active": False}

        return {
            "symbol": symbol,
            "side": state.side,
            "entry_price": state.entry_price,
            "current_stop": state.current_stop,
            "initial_stop": state.initial_stop,
            "breakeven_reached": state.breakeven_reached,
            "trailing_active": state.trailing_active,
            "highest_pnl_pct": round(state.highest_pnl_pct, 2),
            "improvement_pct": round(
                abs(state.current_stop - state.initial_stop) / state.entry_price * 100, 2
            ),
        }
