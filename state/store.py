"""
State store for tracking positions, cooldowns, and trading state.
Provides persistence and recovery for bot state.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

from config import TradingAction, WhaleDefensiveAction
from utils.logger import get_logger


logger = get_logger(__name__)


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


@dataclass
class OrderState:
    """Order state tracking."""
    order_id: int
    client_order_id: str
    symbol: str
    side: str  # BUY or SELL
    order_type: str  # MARKET, LIMIT, STOP_MARKET, TAKE_PROFIT_MARKET
    quantity: float
    price: float
    stop_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_reduce_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['status'] = self.status.value
        d['created_at'] = self.created_at.isoformat()
        d['updated_at'] = self.updated_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderState':
        data['status'] = OrderStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class PositionState:
    """Position state tracking."""
    symbol: str
    side: str  # LONG or SHORT
    entry_price: float  # Actual fill price
    quantity: float
    leverage: int
    notional: float

    # Stop/TP tracking
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    stop_loss_order_id: Optional[int] = None
    take_profit_order_id: Optional[int] = None

    # Timing
    entry_time: datetime = field(default_factory=datetime.utcnow)
    last_update: datetime = field(default_factory=datetime.utcnow)

    # PnL tracking
    unrealized_pnl: float = 0.0
    highest_pnl: float = 0.0
    lowest_pnl: float = 0.0

    # State flags
    is_active: bool = True
    exit_reason: Optional[str] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['entry_time'] = self.entry_time.isoformat()
        d['last_update'] = self.last_update.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionState':
        data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        data['last_update'] = datetime.fromisoformat(data['last_update'])
        return cls(**data)

    def update_pnl(self, current_price: float) -> None:
        """Update unrealized PnL tracking."""
        if self.side == "LONG":
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price

        self.unrealized_pnl = pnl_pct * self.notional
        self.highest_pnl = max(self.highest_pnl, self.unrealized_pnl)
        self.lowest_pnl = min(self.lowest_pnl, self.unrealized_pnl)
        self.last_update = datetime.utcnow()


@dataclass
class SymbolState:
    """Per-symbol trading state."""
    symbol: str

    # Position tracking
    position: Optional[PositionState] = None
    pending_orders: Dict[int, OrderState] = field(default_factory=dict)

    # Cooldown tracking
    last_trade_time: Optional[datetime] = None
    last_trade_action: Optional[TradingAction] = None
    last_exit_reason: Optional[str] = None

    # Daily stats
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    daily_pnl: float = 0.0

    # Anti flip-flop
    cooldown_until: Optional[datetime] = None

    def is_in_cooldown(self) -> bool:
        """Check if symbol is in cooldown period."""
        if self.cooldown_until is None:
            return False
        return datetime.utcnow() < self.cooldown_until

    def set_cooldown(self, seconds: int) -> None:
        """Set cooldown period."""
        self.cooldown_until = datetime.utcnow() + timedelta(seconds=seconds)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'position': self.position.to_dict() if self.position else None,
            'pending_orders': {str(k): v.to_dict() for k, v in self.pending_orders.items()},
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'last_trade_action': self.last_trade_action.value if self.last_trade_action else None,
            'last_exit_reason': self.last_exit_reason,
            'trades_today': self.trades_today,
            'wins_today': self.wins_today,
            'losses_today': self.losses_today,
            'daily_pnl': self.daily_pnl,
            'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SymbolState':
        return cls(
            symbol=data['symbol'],
            position=PositionState.from_dict(data['position']) if data.get('position') else None,
            pending_orders={int(k): OrderState.from_dict(v) for k, v in data.get('pending_orders', {}).items()},
            last_trade_time=datetime.fromisoformat(data['last_trade_time']) if data.get('last_trade_time') else None,
            last_trade_action=TradingAction(data['last_trade_action']) if data.get('last_trade_action') else None,
            last_exit_reason=data.get('last_exit_reason'),
            trades_today=data.get('trades_today', 0),
            wins_today=data.get('wins_today', 0),
            losses_today=data.get('losses_today', 0),
            daily_pnl=data.get('daily_pnl', 0.0),
            cooldown_until=datetime.fromisoformat(data['cooldown_until']) if data.get('cooldown_until') else None,
        )


class StateStore:
    """
    Central state store for the trading bot.
    Tracks positions, orders, cooldowns, and provides persistence.
    """

    def __init__(self, state_file: Optional[str] = None):
        self.state_file = Path(state_file) if state_file else Path("data/bot_state.json")
        self._symbols: Dict[str, SymbolState] = {}
        self._global_state: Dict[str, Any] = {
            'start_time': datetime.utcnow().isoformat(),
            'last_update': datetime.utcnow().isoformat(),
            'total_trades': 0,
            'total_wins': 0,
            'total_losses': 0,
        }

        # Try to load existing state
        self._load_state()

    def _load_state(self) -> None:
        """Load state from file if exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)

                self._global_state = data.get('global', self._global_state)
                for symbol, sym_data in data.get('symbols', {}).items():
                    self._symbols[symbol] = SymbolState.from_dict(sym_data)

                logger.info(f"State loaded from {self.state_file}")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")

    def save_state(self) -> None:
        """Save current state to file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self._global_state['last_update'] = datetime.utcnow().isoformat()

            data = {
                'global': self._global_state,
                'symbols': {s: state.to_dict() for s, state in self._symbols.items()},
            }

            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Could not save state: {e}")

    def get_symbol_state(self, symbol: str) -> SymbolState:
        """Get or create state for a symbol."""
        symbol = symbol.upper()
        if symbol not in self._symbols:
            self._symbols[symbol] = SymbolState(symbol=symbol)
        return self._symbols[symbol]

    def has_position(self, symbol: str) -> bool:
        """Check if symbol has an active position."""
        state = self.get_symbol_state(symbol)
        return state.position is not None and state.position.is_active

    def get_position(self, symbol: str) -> Optional[PositionState]:
        """Get position for symbol if exists."""
        state = self.get_symbol_state(symbol)
        if state.position and state.position.is_active:
            return state.position
        return None

    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        leverage: int,
        stop_loss_price: float,
        take_profit_price: float,
        stop_loss_order_id: Optional[int] = None,
        take_profit_order_id: Optional[int] = None,
    ) -> PositionState:
        """
        Record a new position opening.

        IMPORTANT: entry_price should be actual fill price, not ticker.
        """
        state = self.get_symbol_state(symbol)

        position = PositionState(
            symbol=symbol,
            side=side.upper(),
            entry_price=entry_price,
            quantity=quantity,
            leverage=leverage,
            notional=entry_price * quantity,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            stop_loss_order_id=stop_loss_order_id,
            take_profit_order_id=take_profit_order_id,
        )

        state.position = position
        state.last_trade_time = datetime.utcnow()
        state.last_trade_action = TradingAction.LONG if side == "LONG" else TradingAction.SHORT
        state.trades_today += 1

        self._global_state['total_trades'] = self._global_state.get('total_trades', 0) + 1

        logger.info(
            f"Position opened",
            extra={
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "quantity": quantity,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
            }
        )

        self.save_state()
        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
    ) -> Optional[float]:
        """
        Record position close.

        Args:
            symbol: Trading pair
            exit_price: Actual exit fill price
            exit_reason: Reason for closing

        Returns:
            Realized PnL if position existed, None otherwise
        """
        state = self.get_symbol_state(symbol)

        if not state.position or not state.position.is_active:
            logger.warning(f"No active position to close for {symbol}")
            return None

        position = state.position

        # Calculate PnL
        if position.side == "LONG":
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price

        realized_pnl = pnl_pct * position.notional

        # Update position state
        position.is_active = False
        position.exit_reason = exit_reason
        position.exit_price = exit_price
        position.realized_pnl = realized_pnl

        # Update symbol state
        state.last_exit_reason = exit_reason
        state.daily_pnl += realized_pnl

        is_win = realized_pnl > 0
        if is_win:
            state.wins_today += 1
            self._global_state['total_wins'] = self._global_state.get('total_wins', 0) + 1
        else:
            state.losses_today += 1
            self._global_state['total_losses'] = self._global_state.get('total_losses', 0) + 1

        logger.info(
            f"Position closed",
            extra={
                "symbol": symbol,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "realized_pnl": realized_pnl,
                "is_win": is_win,
            }
        )

        self.save_state()
        return realized_pnl

    def update_stop_loss_order(self, symbol: str, order_id: int, new_price: float) -> None:
        """Update stop loss order ID and price."""
        state = self.get_symbol_state(symbol)
        if state.position:
            state.position.stop_loss_order_id = order_id
            state.position.stop_loss_price = new_price
            state.position.last_update = datetime.utcnow()
            self.save_state()

    def update_take_profit_order(self, symbol: str, order_id: int, new_price: float) -> None:
        """Update take profit order ID and price."""
        state = self.get_symbol_state(symbol)
        if state.position:
            state.position.take_profit_order_id = order_id
            state.position.take_profit_price = new_price
            state.position.last_update = datetime.utcnow()
            self.save_state()

    def add_pending_order(self, symbol: str, order: OrderState) -> None:
        """Add a pending order to track."""
        state = self.get_symbol_state(symbol)
        state.pending_orders[order.order_id] = order
        self.save_state()

    def update_order_status(self, symbol: str, order_id: int, status: OrderStatus, filled_qty: float = 0, avg_price: float = 0) -> None:
        """Update order status."""
        state = self.get_symbol_state(symbol)
        if order_id in state.pending_orders:
            order = state.pending_orders[order_id]
            order.status = status
            order.filled_qty = filled_qty
            order.avg_fill_price = avg_price
            order.updated_at = datetime.utcnow()

            if status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED, OrderStatus.REJECTED]:
                del state.pending_orders[order_id]

            self.save_state()

    def is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown."""
        return self.get_symbol_state(symbol).is_in_cooldown()

    def set_cooldown(self, symbol: str, seconds: int) -> None:
        """Set cooldown for symbol."""
        self.get_symbol_state(symbol).set_cooldown(seconds)
        self.save_state()

    def clear_position(self, symbol: str) -> None:
        """Clear position state (for cleanup/sync)."""
        state = self.get_symbol_state(symbol)
        state.position = None
        self.save_state()

    def get_all_positions(self) -> Dict[str, PositionState]:
        """Get all active positions."""
        return {
            symbol: state.position
            for symbol, state in self._symbols.items()
            if state.position and state.position.is_active
        }

    def reset_daily_stats(self) -> None:
        """Reset daily statistics for all symbols."""
        for state in self._symbols.values():
            state.trades_today = 0
            state.wins_today = 0
            state.losses_today = 0
            state.daily_pnl = 0.0
        self.save_state()
