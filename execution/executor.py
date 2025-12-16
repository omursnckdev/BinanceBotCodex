"""
Order execution module.
Handles order placement, bracket orders, and defensive actions.

IMPORTANT:
- SL/TP computed from actual fill price, not ticker
- Defensive actions are explicit, not triggered by FLAT
- Safe closing flow with proper order management
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from config import (
    Environment, WhaleDefensiveAction, get_settings
)
from exchange.binance_client import (
    BinanceClient, BinanceClientError, OrderResult, SymbolInfo
)
from risk.risk_manager import RiskManager
from state.store import StateStore, PositionState, OrderState, OrderStatus
from strategy.fusion import FusionDecision
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of order execution."""
    success: bool
    message: str
    order_id: Optional[int] = None
    fill_price: Optional[float] = None
    fill_quantity: Optional[float] = None
    sl_order_id: Optional[int] = None
    tp_order_id: Optional[int] = None


class OrderExecutor:
    """
    Order executor with safety controls.

    Features:
    - Dry-run mode support
    - Bracket orders (entry + SL + TP)
    - Defensive action execution
    - Fill price based SL/TP calculation
    - Idempotency handling
    """

    def __init__(
        self,
        client: BinanceClient,
        risk_manager: RiskManager,
        state_store: StateStore,
    ):
        self.client = client
        self.risk_manager = risk_manager
        self.state = state_store
        self.settings = get_settings()

        # Order ID tracking for idempotency
        self._pending_entries: Dict[str, str] = {}  # symbol -> client_order_id

    def _is_live_trading(self) -> bool:
        """Check if live trading is enabled."""
        return self.settings.is_live_trading_enabled()

    def _generate_client_order_id(self, prefix: str = "BOT") -> str:
        """Generate unique client order ID."""
        return f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

    def _get_opposite_side(self, side: str) -> str:
        """Get opposite side for closing orders."""
        return "SELL" if side.upper() == "BUY" else "BUY"

    def _log_dry_run(self, action: str, **kwargs) -> None:
        """Log dry-run order."""
        logger.info(
            f"[DRY-RUN] {action}",
            extra={"dry_run": True, **kwargs}
        )

    def setup_symbol(self, symbol: str, leverage: int) -> bool:
        """
        Setup symbol for trading (leverage, margin type).

        Args:
            symbol: Trading pair
            leverage: Desired leverage

        Returns:
            True if setup successful
        """
        if not self._is_live_trading():
            self._log_dry_run("Symbol setup", symbol=symbol, leverage=leverage)
            return True

        try:
            symbol_config = self.settings.get_symbol_config(symbol)

            # Set leverage
            if not self.client.set_leverage(symbol, leverage):
                logger.warning(f"Failed to set leverage for {symbol}")

            # Set margin type
            if not self.client.set_margin_type(symbol, symbol_config.margin_type):
                logger.warning(f"Failed to set margin type for {symbol}")

            return True

        except Exception as e:
            logger.error(f"Symbol setup failed: {e}", extra={"symbol": symbol})
            return False

    def execute_entry(
        self,
        decision: FusionDecision,
        current_price: float,
    ) -> ExecutionResult:
        """
        Execute entry order with bracket (SL + optional TP).

        IMPORTANT:
        - SL/TP calculated from actual fill price, not ticker
        - Handles partial fills safely

        Args:
            decision: Fusion decision with entry parameters
            current_price: Current market price (for sizing only)

        Returns:
            ExecutionResult with order details
        """
        symbol = decision.symbol

        # Check for existing position
        if self.state.has_position(symbol):
            return ExecutionResult(
                success=False,
                message=f"Position already exists for {symbol}"
            )

        # Check cooldown
        if self.state.is_in_cooldown(symbol):
            return ExecutionResult(
                success=False,
                message=f"{symbol} is in cooldown period"
            )

        # Determine side
        if decision.action.value == "long":
            side = "BUY"
            position_side = "LONG"
        elif decision.action.value == "short":
            side = "SELL"
            position_side = "SHORT"
        else:
            return ExecutionResult(
                success=False,
                message=f"Invalid action for entry: {decision.action.value}"
            )

        # Calculate position size
        balance = self.client.get_usdt_balance() if self._is_live_trading() else 10000.0
        quantity = self.risk_manager.calculate_position_size(
            symbol=symbol,
            balance=balance,
            stop_loss_pct=decision.stop_loss_pct,
            leverage=decision.leverage,
        )

        if quantity <= 0:
            return ExecutionResult(
                success=False,
                message="Position size too small"
            )

        # Convert to base asset quantity
        quantity = quantity / current_price

        # Validate against exchange filters
        symbol_info = self.client.get_symbol_info(symbol)
        if not symbol_info:
            return ExecutionResult(
                success=False,
                message=f"Symbol info not found: {symbol}"
            )

        quantity = float(symbol_info.round_quantity(quantity))
        notional = quantity * current_price

        valid, reason = symbol_info.validate_order(quantity, current_price)
        if not valid:
            return ExecutionResult(
                success=False,
                message=f"Order validation failed: {reason}"
            )

        # Check risk limits
        can_open, reason = self.risk_manager.can_open_position(symbol, notional)
        if not can_open:
            return ExecutionResult(
                success=False,
                message=f"Risk check failed: {reason}"
            )

        # Generate client order ID for idempotency
        client_order_id = self._generate_client_order_id("ENTRY")

        # Check for duplicate entry
        if symbol in self._pending_entries:
            return ExecutionResult(
                success=False,
                message="Entry already pending for this symbol"
            )

        self._pending_entries[symbol] = client_order_id

        try:
            # Setup symbol (leverage, margin)
            self.setup_symbol(symbol, decision.leverage)

            # === EXECUTE ENTRY ===
            if not self._is_live_trading():
                # Dry-run mode
                self._log_dry_run(
                    f"Market {side} order",
                    symbol=symbol,
                    quantity=quantity,
                    notional=notional,
                    leverage=decision.leverage,
                )

                # Simulate fill at current price
                fill_price = current_price
                fill_qty = quantity

            else:
                # Live execution
                result = self.client.place_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    client_order_id=client_order_id,
                )

                if not result.is_filled and not result.is_partially_filled:
                    return ExecutionResult(
                        success=False,
                        message=f"Entry order not filled: {result.status}"
                    )

                # Use actual fill price (CRITICAL)
                fill_price = result.fill_price
                fill_qty = result.executed_qty

                logger.order_event(
                    symbol=symbol,
                    order_type="MARKET",
                    side=side,
                    quantity=fill_qty,
                    price=fill_price,
                    order_id=str(result.order_id),
                    status="FILLED",
                )

            # === CALCULATE SL/TP FROM FILL PRICE (CRITICAL) ===
            stop_loss_price = self.risk_manager.get_stop_loss_price(
                entry_price=fill_price,
                side=position_side,
                stop_loss_pct=decision.stop_loss_pct,
            )

            take_profit_price = self.risk_manager.get_take_profit_price(
                entry_price=fill_price,
                side=position_side,
                take_profit_pct=decision.take_profit_pct,
            )

            # Round to valid prices
            stop_loss_price = float(symbol_info.round_price(stop_loss_price))
            take_profit_price = float(symbol_info.round_price(take_profit_price))

            # === PLACE STOP LOSS ORDER ===
            sl_order_id = None
            sl_side = self._get_opposite_side(side)

            if not self._is_live_trading():
                self._log_dry_run(
                    f"Stop-loss {sl_side} order",
                    symbol=symbol,
                    stop_price=stop_loss_price,
                    quantity=fill_qty,
                )
            else:
                try:
                    sl_result = self.client.place_stop_loss_order(
                        symbol=symbol,
                        side=sl_side,
                        quantity=fill_qty,
                        stop_price=stop_loss_price,
                        reduce_only=True,
                        client_order_id=self._generate_client_order_id("SL"),
                    )
                    sl_order_id = sl_result.order_id
                except BinanceClientError as e:
                    logger.error(f"Failed to place SL order: {e}")
                    # Position is open but no SL - critical!
                    # Try to close immediately
                    self._emergency_close(symbol, sl_side, fill_qty, "SL placement failed")
                    return ExecutionResult(
                        success=False,
                        message=f"Failed to place SL: {e}"
                    )

            # === PLACE TAKE PROFIT ORDER ===
            tp_order_id = None

            if not self._is_live_trading():
                self._log_dry_run(
                    f"Take-profit {sl_side} order",
                    symbol=symbol,
                    stop_price=take_profit_price,
                    quantity=fill_qty,
                )
            else:
                try:
                    tp_result = self.client.place_take_profit_order(
                        symbol=symbol,
                        side=sl_side,
                        quantity=fill_qty,
                        stop_price=take_profit_price,
                        reduce_only=True,
                        client_order_id=self._generate_client_order_id("TP"),
                    )
                    tp_order_id = tp_result.order_id
                except BinanceClientError as e:
                    logger.warning(f"Failed to place TP order: {e}")
                    # Not critical - SL is in place

            # === UPDATE STATE ===
            self.state.open_position(
                symbol=symbol,
                side=position_side,
                entry_price=fill_price,
                quantity=fill_qty,
                leverage=decision.leverage,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                stop_loss_order_id=sl_order_id,
                take_profit_order_id=tp_order_id,
            )

            # Log the trade signal
            logger.trade_signal(
                symbol=symbol,
                action=decision.action.value,
                scores={
                    "tech": decision.tech_score,
                    "whale": decision.whale_score,
                    "sentiment": decision.sentiment_score,
                    "final": decision.final_score,
                },
                reason=decision.decision_reason,
                fill_price=fill_price,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                confidence=decision.confidence,
            )

            return ExecutionResult(
                success=True,
                message=f"Entry executed: {position_side} {symbol}",
                order_id=result.order_id if self._is_live_trading() else 0,
                fill_price=fill_price,
                fill_quantity=fill_qty,
                sl_order_id=sl_order_id,
                tp_order_id=tp_order_id,
            )

        except Exception as e:
            logger.error(f"Entry execution failed: {e}", extra={"symbol": symbol})
            return ExecutionResult(
                success=False,
                message=f"Entry failed: {e}"
            )

        finally:
            # Clean up pending entry tracking
            self._pending_entries.pop(symbol, None)

    def execute_close(
        self,
        symbol: str,
        reason: str,
        cancel_orders: bool = True,
    ) -> ExecutionResult:
        """
        Close an existing position.

        Args:
            symbol: Trading pair
            reason: Reason for closing
            cancel_orders: Whether to cancel existing SL/TP orders first

        Returns:
            ExecutionResult with close details
        """
        position = self.state.get_position(symbol)
        if not position:
            return ExecutionResult(
                success=False,
                message=f"No position to close for {symbol}"
            )

        close_side = "SELL" if position.side == "LONG" else "BUY"

        try:
            # Cancel existing orders first if requested
            if cancel_orders and self._is_live_trading():
                # Note: Be careful here - if close fails after cancel,
                # we need to re-place the SL!
                if position.stop_loss_order_id:
                    self.client.cancel_order(symbol, position.stop_loss_order_id)
                if position.take_profit_order_id:
                    self.client.cancel_order(symbol, position.take_profit_order_id)

            # Execute close
            if not self._is_live_trading():
                self._log_dry_run(
                    f"Close {close_side} order",
                    symbol=symbol,
                    quantity=position.quantity,
                    reason=reason,
                )
                fill_price = position.entry_price  # Simulate break-even for dry-run
            else:
                result = self.client.place_market_order(
                    symbol=symbol,
                    side=close_side,
                    quantity=position.quantity,
                    reduce_only=True,
                    client_order_id=self._generate_client_order_id("CLOSE"),
                )

                if not result.is_filled:
                    # Critical: Failed to close - need to re-place SL!
                    if cancel_orders:
                        self._replace_stop_loss(symbol, position)
                    return ExecutionResult(
                        success=False,
                        message=f"Close order not filled: {result.status}"
                    )

                fill_price = result.fill_price

            # Update state
            pnl = self.state.close_position(symbol, fill_price, reason)

            # Record trade result
            is_win = pnl is not None and pnl > 0
            self.risk_manager.record_trade_result(pnl or 0, is_win)

            # Set cooldown
            symbol_config = self.settings.get_symbol_config(symbol)
            self.state.set_cooldown(symbol, symbol_config.cooldown_seconds)

            logger.info(
                f"Position closed",
                extra={
                    "symbol": symbol,
                    "reason": reason,
                    "exit_price": fill_price,
                    "pnl": pnl,
                }
            )

            return ExecutionResult(
                success=True,
                message=f"Position closed: {symbol} ({reason})",
                fill_price=fill_price,
                fill_quantity=position.quantity,
            )

        except Exception as e:
            logger.error(f"Close execution failed: {e}", extra={"symbol": symbol})
            # Try to re-place SL on failure
            if cancel_orders and self._is_live_trading():
                self._replace_stop_loss(symbol, position)
            return ExecutionResult(
                success=False,
                message=f"Close failed: {e}"
            )

    def execute_defensive_action(
        self,
        symbol: str,
        action: WhaleDefensiveAction,
    ) -> ExecutionResult:
        """
        Execute a defensive action from whale detection.

        IMPORTANT: These are EXPLICIT actions, not triggered by FLAT.

        Args:
            symbol: Trading pair
            action: Defensive action to take

        Returns:
            ExecutionResult
        """
        if action == WhaleDefensiveAction.NONE:
            return ExecutionResult(success=True, message="No action needed")

        position = self.state.get_position(symbol)
        if not position:
            return ExecutionResult(
                success=False,
                message=f"No position for defensive action: {symbol}"
            )

        logger.whale_alert(
            symbol=symbol,
            direction=position.side,
            score=0.0,  # Not available here
            action=action.value,
        )

        if action == WhaleDefensiveAction.CLOSE_MARKET:
            return self.execute_close(symbol, f"Whale defensive: {action.value}")

        elif action == WhaleDefensiveAction.REDUCE_50_PERCENT:
            return self._reduce_position(symbol, 0.5, f"Whale defensive: {action.value}")

        elif action == WhaleDefensiveAction.TIGHTEN_STOP:
            return self._tighten_stop(symbol)

        return ExecutionResult(
            success=False,
            message=f"Unknown defensive action: {action.value}"
        )

    def _reduce_position(
        self,
        symbol: str,
        reduction_pct: float,
        reason: str,
    ) -> ExecutionResult:
        """Reduce position by percentage."""
        position = self.state.get_position(symbol)
        if not position:
            return ExecutionResult(success=False, message="No position to reduce")

        reduce_qty = position.quantity * reduction_pct
        close_side = "SELL" if position.side == "LONG" else "BUY"

        symbol_info = self.client.get_symbol_info(symbol)
        if symbol_info:
            reduce_qty = float(symbol_info.round_quantity(reduce_qty))

        if not self._is_live_trading():
            self._log_dry_run(
                f"Reduce position {close_side}",
                symbol=symbol,
                quantity=reduce_qty,
                reason=reason,
            )
            return ExecutionResult(
                success=True,
                message=f"[DRY-RUN] Position reduced by {reduction_pct*100}%"
            )

        try:
            result = self.client.place_market_order(
                symbol=symbol,
                side=close_side,
                quantity=reduce_qty,
                reduce_only=True,
                client_order_id=self._generate_client_order_id("REDUCE"),
            )

            if result.is_filled:
                # Update position quantity in state
                position.quantity -= result.executed_qty
                self.state.save_state()

                return ExecutionResult(
                    success=True,
                    message=f"Position reduced by {reduction_pct*100}%",
                    fill_price=result.fill_price,
                    fill_quantity=result.executed_qty,
                )

            return ExecutionResult(
                success=False,
                message=f"Reduce order not filled: {result.status}"
            )

        except Exception as e:
            logger.error(f"Position reduction failed: {e}")
            return ExecutionResult(success=False, message=f"Reduce failed: {e}")

    def _tighten_stop(self, symbol: str) -> ExecutionResult:
        """Tighten stop loss to lock in profits."""
        position = self.state.get_position(symbol)
        if not position:
            return ExecutionResult(success=False, message="No position")

        # Get current price
        try:
            current_price = self.client.get_ticker_price(symbol)
        except Exception as e:
            return ExecutionResult(success=False, message=f"Could not get price: {e}")

        # Calculate new stop
        should_tighten, new_stop = self.risk_manager.should_tighten_stop(
            entry_price=position.entry_price,
            current_price=current_price,
            side=position.side,
            current_stop_price=position.stop_loss_price,
        )

        if not should_tighten:
            return ExecutionResult(
                success=True,
                message="Stop tightening not needed"
            )

        # Cancel old stop and place new one
        if not self._is_live_trading():
            self._log_dry_run(
                "Tighten stop loss",
                symbol=symbol,
                old_stop=position.stop_loss_price,
                new_stop=new_stop,
            )
            return ExecutionResult(
                success=True,
                message=f"[DRY-RUN] Stop tightened to {new_stop}"
            )

        try:
            # Cancel old SL
            if position.stop_loss_order_id:
                self.client.cancel_order(symbol, position.stop_loss_order_id)

            # Place new SL
            sl_side = "SELL" if position.side == "LONG" else "BUY"
            symbol_info = self.client.get_symbol_info(symbol)
            if symbol_info:
                new_stop = float(symbol_info.round_price(new_stop))

            result = self.client.place_stop_loss_order(
                symbol=symbol,
                side=sl_side,
                quantity=position.quantity,
                stop_price=new_stop,
                reduce_only=True,
            )

            # Update state
            self.state.update_stop_loss_order(symbol, result.order_id, new_stop)

            return ExecutionResult(
                success=True,
                message=f"Stop tightened to {new_stop}",
                sl_order_id=result.order_id,
            )

        except Exception as e:
            logger.error(f"Stop tightening failed: {e}")
            # Try to re-place original stop
            self._replace_stop_loss(symbol, position)
            return ExecutionResult(success=False, message=f"Tighten failed: {e}")

    def _replace_stop_loss(self, symbol: str, position: PositionState) -> None:
        """Emergency re-place stop loss order."""
        if not self._is_live_trading():
            return

        try:
            sl_side = "SELL" if position.side == "LONG" else "BUY"
            result = self.client.place_stop_loss_order(
                symbol=symbol,
                side=sl_side,
                quantity=position.quantity,
                stop_price=position.stop_loss_price,
                reduce_only=True,
            )
            self.state.update_stop_loss_order(symbol, result.order_id, position.stop_loss_price)
            logger.info(f"Stop-loss re-placed for {symbol}")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to re-place SL for {symbol}: {e}")

    def _emergency_close(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reason: str,
    ) -> None:
        """Emergency close on order failure."""
        if not self._is_live_trading():
            return

        try:
            self.client.place_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                reduce_only=True,
            )
            logger.warning(f"Emergency close for {symbol}: {reason}")
        except Exception as e:
            logger.error(f"CRITICAL: Emergency close failed for {symbol}: {e}")

    def sync_positions(self) -> None:
        """Sync state with exchange positions."""
        if not self._is_live_trading():
            return

        try:
            exchange_positions = self.client.get_positions()
            state_positions = self.state.get_all_positions()

            # Check for positions on exchange not in state
            exchange_symbols = {p.symbol for p in exchange_positions}
            state_symbols = set(state_positions.keys())

            # Positions on exchange but not in state
            for pos in exchange_positions:
                if pos.symbol not in state_symbols:
                    logger.warning(
                        f"Position on exchange not in state: {pos.symbol}",
                        extra={"quantity": pos.quantity, "side": pos.side}
                    )

            # Positions in state but not on exchange
            for symbol in state_symbols:
                if symbol not in exchange_symbols:
                    logger.warning(f"Position in state not on exchange: {symbol}")
                    self.state.clear_position(symbol)

        except Exception as e:
            logger.error(f"Position sync failed: {e}")
