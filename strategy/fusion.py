"""
Fusion strategy module.
Combines technical, whale, and sentiment signals into trading decisions.

IMPORTANT:
- FLAT means "NO NEW ENTRY" - it does NOT mean close position
- Exits must be explicit with EXIT or exit_reason
- Whale defensive actions are handled separately from entry signals
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from config import (
    FusionConfig, SymbolConfig, TradingAction, Trend,
    WhaleDefensiveAction, get_settings
)
from signals.indicators import TechnicalSignal
from signals.whales import WhaleSignal
from signals.sentiment import SentimentSignal
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class FusionDecision:
    """
    Combined trading decision from all signals.

    IMPORTANT:
    - action=FLAT means "do not enter new position" - NOT close position
    - For exits, use exit_reason field
    - Whale defensive actions are in recommended_defensive_action
    """
    symbol: str
    timestamp: datetime

    # Entry decision (FLAT = no new entry, NOT close)
    action: TradingAction  # LONG, SHORT, or FLAT (no entry)
    final_score: float  # -1 to +1
    confidence: float  # 0 to 1

    # Position parameters
    leverage: int = 1
    position_size_pct: float = 0.0  # Percentage of balance
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0

    # Defensive action (EXPLICIT - not FLAT)
    recommended_defensive_action: Optional[WhaleDefensiveAction] = None
    exit_reason: Optional[str] = None  # If set, suggests closing position

    # Signal components
    tech_score: float = 0.0
    tech_confidence: float = 0.0
    whale_score: float = 0.0
    whale_confidence: float = 0.0
    sentiment_score: float = 0.0
    sentiment_confidence: float = 0.0

    # Context
    trend: Trend = Trend.NEUTRAL
    atr_pct: float = 0.0

    # Reasoning
    decision_reason: str = ""
    skip_reasons: List[str] = field(default_factory=list)

    # Validity
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "final_score": round(self.final_score, 4),
            "confidence": round(self.confidence, 4),
            "leverage": self.leverage,
            "stop_loss_pct": round(self.stop_loss_pct, 4),
            "take_profit_pct": round(self.take_profit_pct, 4),
            "defensive_action": self.recommended_defensive_action.value if self.recommended_defensive_action else None,
            "exit_reason": self.exit_reason,
            "tech_score": round(self.tech_score, 4),
            "whale_score": round(self.whale_score, 4),
            "sentiment_score": round(self.sentiment_score, 4),
            "trend": self.trend.value,
            "decision_reason": self.decision_reason,
        }


class FusionStrategy:
    """
    Combines multiple signal sources into trading decisions.

    Key Design Principles:
    1. FLAT means "no new entry" - NEVER means close position
    2. Exits are explicit via exit_reason
    3. Whale defensive actions are explicit via recommended_defensive_action
    4. Scalping profile with tight TP/SL targets
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or get_settings().fusion
        self.settings = get_settings()

    def fuse_signals(
        self,
        symbol: str,
        tech_signal: TechnicalSignal,
        whale_signal: WhaleSignal,
        sentiment_signal: SentimentSignal,
        current_position_side: Optional[str] = None,  # 'LONG', 'SHORT', or None
        market_conditions_ok: bool = True,
    ) -> FusionDecision:
        """
        Combine all signals into a trading decision.

        Args:
            symbol: Trading pair
            tech_signal: Technical analysis signal
            whale_signal: Whale activity signal
            sentiment_signal: News sentiment signal
            current_position_side: Current position (for defensive action context)
            market_conditions_ok: Whether market conditions allow trading

        Returns:
            FusionDecision with action and parameters

        IMPORTANT:
        - action=FLAT means "do not enter" - NOT close
        - Check recommended_defensive_action for position protection
        - Check exit_reason for explicit exit signals
        """
        symbol = symbol.upper()
        timestamp = datetime.utcnow()
        symbol_config = self.settings.get_symbol_config(symbol)

        # Get weights
        tech_weight = symbol_config.tech_weight
        whale_weight = symbol_config.whale_weight
        sentiment_weight = symbol_config.sentiment_weight

        # Normalize weights
        total_weight = tech_weight + whale_weight + sentiment_weight
        if total_weight > 0:
            tech_weight /= total_weight
            whale_weight /= total_weight
            sentiment_weight /= total_weight
        else:
            tech_weight = 1.0
            whale_weight = 0.0
            sentiment_weight = 0.0

        skip_reasons = []

        # Check signal validity
        if not tech_signal.is_valid:
            skip_reasons.append("Technical signal invalid")
        if not whale_signal.is_valid and whale_weight > 0:
            skip_reasons.append("Whale signal invalid")
        if not sentiment_signal.is_valid and sentiment_weight > 0:
            # Sentiment is optional - reduce weight instead
            tech_weight += sentiment_weight / 2
            whale_weight += sentiment_weight / 2
            sentiment_weight = 0

        if not market_conditions_ok:
            skip_reasons.append("Market conditions not suitable")

        # Calculate final score
        final_score = (
            tech_signal.tech_score * tech_weight +
            whale_signal.whale_score * whale_weight +
            sentiment_signal.sentiment_score * sentiment_weight
        )

        # Calculate combined confidence
        confidence = (
            tech_signal.tech_confidence * tech_weight +
            whale_signal.whale_confidence * whale_weight +
            sentiment_signal.confidence * sentiment_weight
        )

        # Handle whale defensive action (EXPLICIT - not converted to FLAT)
        recommended_defensive = None
        exit_reason = None

        if whale_signal.whale_alert_opposite and current_position_side:
            recommended_defensive = whale_signal.recommended_action
            if recommended_defensive == WhaleDefensiveAction.CLOSE_MARKET:
                exit_reason = f"Whale activity opposite to {current_position_side} position"

        # Determine action
        action = TradingAction.FLAT  # Default: no new entry
        decision_reason = ""

        # Check if we should skip
        if skip_reasons:
            decision_reason = f"Skip: {'; '.join(skip_reasons)}"

        # Check minimum score threshold
        elif abs(final_score) < self.config.entry_threshold:
            decision_reason = f"Score {final_score:.3f} below threshold {self.config.entry_threshold}"
            skip_reasons.append("Score below threshold")

        # Check minimum confidence
        elif confidence < self.config.min_confidence_for_entry:
            decision_reason = f"Confidence {confidence:.3f} below minimum {self.config.min_confidence_for_entry}"
            skip_reasons.append("Confidence too low")

        else:
            # Determine direction
            is_high_confidence = confidence >= self.config.high_confidence_threshold

            if final_score > 0:
                # Bullish signal
                if self.config.require_trend_confirmation:
                    if tech_signal.trend == Trend.BULLISH:
                        action = TradingAction.LONG
                        decision_reason = "Bullish signal with trend confirmation"
                    elif tech_signal.trend == Trend.NEUTRAL and is_high_confidence:
                        action = TradingAction.LONG
                        decision_reason = "Bullish signal with high confidence (neutral trend)"
                    elif self.config.allow_counter_trend_high_confidence and is_high_confidence:
                        action = TradingAction.LONG
                        decision_reason = "Bullish signal with very high confidence (counter-trend)"
                    else:
                        skip_reasons.append("No trend confirmation for LONG")
                        decision_reason = f"Bullish signal but trend is {tech_signal.trend.value}"
                else:
                    action = TradingAction.LONG
                    decision_reason = "Bullish signal"

            elif final_score < 0:
                # Bearish signal
                if self.config.require_trend_confirmation:
                    if tech_signal.trend == Trend.BEARISH:
                        action = TradingAction.SHORT
                        decision_reason = "Bearish signal with trend confirmation"
                    elif tech_signal.trend == Trend.NEUTRAL and is_high_confidence:
                        action = TradingAction.SHORT
                        decision_reason = "Bearish signal with high confidence (neutral trend)"
                    elif self.config.allow_counter_trend_high_confidence and is_high_confidence:
                        action = TradingAction.SHORT
                        decision_reason = "Bearish signal with very high confidence (counter-trend)"
                    else:
                        skip_reasons.append("No trend confirmation for SHORT")
                        decision_reason = f"Bearish signal but trend is {tech_signal.trend.value}"
                else:
                    action = TradingAction.SHORT
                    decision_reason = "Bearish signal"

        # Calculate position parameters (only if entry)
        leverage = 1
        stop_loss_pct = symbol_config.stop_loss_pct
        take_profit_pct = symbol_config.take_profit_pct
        position_size_pct = 0.0

        if action != TradingAction.FLAT:
            # Dynamic leverage based on confidence (scalping: conservative)
            risk_config = self.settings.risk
            max_leverage = min(symbol_config.leverage, risk_config.max_leverage)
            min_leverage = risk_config.min_leverage

            leverage = min_leverage + int((max_leverage - min_leverage) * confidence)
            leverage = max(min_leverage, min(leverage, max_leverage))

            # ATR-based stop loss (clamped to scalping bounds)
            atr_stop = tech_signal.atr_pct * risk_config.sl_atr_multiplier
            stop_loss_pct = max(
                risk_config.sl_min_pct,
                min(atr_stop, risk_config.sl_max_pct)
            )

            # Take profit based on risk-reward ratio
            take_profit_pct = max(
                stop_loss_pct * risk_config.min_risk_reward_ratio,
                symbol_config.take_profit_pct
            )

            # Position size based on risk per trade
            position_size_pct = symbol_config.risk_per_trade_pct

        return FusionDecision(
            symbol=symbol,
            timestamp=timestamp,
            action=action,
            final_score=final_score,
            confidence=confidence,
            leverage=leverage,
            position_size_pct=position_size_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            recommended_defensive_action=recommended_defensive,
            exit_reason=exit_reason,
            tech_score=tech_signal.tech_score,
            tech_confidence=tech_signal.tech_confidence,
            whale_score=whale_signal.whale_score,
            whale_confidence=whale_signal.whale_confidence,
            sentiment_score=sentiment_signal.sentiment_score,
            sentiment_confidence=sentiment_signal.confidence,
            trend=tech_signal.trend,
            atr_pct=tech_signal.atr_pct,
            decision_reason=decision_reason,
            skip_reasons=skip_reasons,
            is_valid=len(skip_reasons) == 0 or action != TradingAction.FLAT,
        )

    def should_close_position(
        self,
        decision: FusionDecision,
        position_side: str,
        position_entry_time: datetime,
        position_entry_price: float,
        current_price: float,
    ) -> tuple:
        """
        Check if position should be closed based on decision.

        Returns:
            Tuple of (should_close: bool, reason: str)

        IMPORTANT: This is for explicit exit logic, separate from FLAT.
        """
        # Check explicit exit reason (e.g., from whale alert)
        if decision.exit_reason:
            return True, decision.exit_reason

        # Check defensive action
        if decision.recommended_defensive_action == WhaleDefensiveAction.CLOSE_MARKET:
            return True, "Whale defensive action: CLOSE_MARKET"

        # Check time-stop
        risk_config = self.settings.risk
        position_age_minutes = (datetime.utcnow() - position_entry_time).total_seconds() / 60
        if position_age_minutes >= risk_config.time_stop_minutes:
            return True, f"Time-stop: Position open for {position_age_minutes:.0f} minutes"

        # Check for strong opposite signal
        if position_side == "LONG" and decision.action == TradingAction.SHORT:
            if decision.confidence >= self.config.high_confidence_threshold:
                return True, "Strong opposite signal (SHORT) while in LONG"

        elif position_side == "SHORT" and decision.action == TradingAction.LONG:
            if decision.confidence >= self.config.high_confidence_threshold:
                return True, "Strong opposite signal (LONG) while in SHORT"

        return False, ""

    def calculate_position_size(
        self,
        balance: float,
        risk_per_trade_pct: float,
        stop_loss_pct: float,
        leverage: int,
        current_price: float,
    ) -> float:
        """
        Calculate position size based on risk management.

        Args:
            balance: Account balance in USDT
            risk_per_trade_pct: Percentage of balance to risk
            stop_loss_pct: Stop loss percentage
            leverage: Position leverage
            current_price: Current asset price

        Returns:
            Position size in base asset units
        """
        if stop_loss_pct <= 0 or current_price <= 0:
            return 0.0

        # Risk amount in USDT
        risk_amount = balance * (risk_per_trade_pct / 100)

        # Position size = risk / (stop_loss_pct / 100)
        # With leverage, actual position = position * leverage
        notional = risk_amount / (stop_loss_pct / 100)

        # Cap at max position size from config
        symbol_config = self.settings.symbol_configs.get("BTCUSDT")  # Default
        if symbol_config:
            notional = min(notional, symbol_config.max_position_size_usd)

        # Convert to quantity
        quantity = notional / current_price

        return quantity
