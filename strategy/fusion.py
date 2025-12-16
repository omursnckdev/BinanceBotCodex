"""
Fusion strategy module (v3).
Combines technical, whale, and sentiment signals into trading decisions.

v3 Features:
- Dynamic weight adjustments based on signal strength
- RSI extreme override capability
- Whale entry signal integration
- Trailing stop state tracking

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
    WhaleDefensiveAction, RSIExtremeDirection, get_settings
)
from signals.indicators import TechnicalSignal
from signals.whales import WhaleSignal
from signals.sentiment import SentimentSignal
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class FusionDecision:
    """
    Combined trading decision from all signals (v3).

    IMPORTANT:
    - action=FLAT means "do not enter new position" - NOT close position
    - For exits, use exit_reason field
    - Whale defensive actions are in recommended_defensive_action

    v3 additions:
    - trailing_stop_active: Whether trailing stop is recommended
    - override_reason: If entry was triggered by RSI extreme or whale signal
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

    # v3: Trailing stop
    trailing_stop_active: bool = True  # Default enabled for scalping
    current_trailing_stop: Optional[float] = None

    # Signal components
    tech_score: float = 0.0
    tech_confidence: float = 0.0
    whale_score: float = 0.0
    whale_confidence: float = 0.0
    sentiment_score: float = 0.0
    sentiment_confidence: float = 0.0

    # v3: Override info
    override_reason: Optional[str] = None  # RSI extreme or whale entry override
    rsi_extreme: bool = False
    whale_entry_signal: Optional[str] = None

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
            "trailing_stop_active": self.trailing_stop_active,
            "defensive_action": self.recommended_defensive_action.value if self.recommended_defensive_action else None,
            "exit_reason": self.exit_reason,
            "override_reason": self.override_reason,
            "tech_score": round(self.tech_score, 4),
            "whale_score": round(self.whale_score, 4),
            "sentiment_score": round(self.sentiment_score, 4),
            "rsi_extreme": self.rsi_extreme,
            "whale_entry_signal": self.whale_entry_signal,
            "trend": self.trend.value,
            "decision_reason": self.decision_reason,
        }


class FusionStrategy:
    """
    Combines multiple signal sources into trading decisions (v3).

    Key Design Principles:
    1. FLAT means "no new entry" - NEVER means close position
    2. Exits are explicit via exit_reason
    3. Whale defensive actions are explicit via recommended_defensive_action
    4. Scalping profile with tight TP/SL targets

    v3 Enhancements:
    - Dynamic weight adjustment based on signal strength
    - RSI extreme can override low fusion scores
    - Whale entry signals can trigger entries
    - Trailing stop recommendations included
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or get_settings().fusion
        self.settings = get_settings()

    def _calculate_dynamic_weights(
        self,
        tech_signal: TechnicalSignal,
        whale_signal: WhaleSignal,
        base_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate dynamic weights based on signal strength (v3).

        Adjusts weights when:
        - RSI shows extreme values (boost tech weight)
        - Whale has strong entry signal (boost whale weight)

        Args:
            tech_signal: Technical analysis signal
            whale_signal: Whale activity signal
            base_weights: Base weight configuration

        Returns:
            Dict with adjusted weights
        """
        tech_weight = base_weights["tech"]
        whale_weight = base_weights["whale"]
        sentiment_weight = base_weights["sentiment"]

        # Boost whale weight if strong entry signal
        if whale_signal.whale_entry_confidence >= 0.7:
            whale_weight = self.config.whale_strong_signal_weight_boost
            tech_weight = (1.0 - whale_weight) * 0.7
            sentiment_weight = (1.0 - whale_weight) * 0.3
            logger.debug(
                f"Weight boost: whale entry signal (conf={whale_signal.whale_entry_confidence:.2f})"
            )

        # Boost tech weight if RSI extreme
        if tech_signal.rsi_extreme:
            tech_weight = self.config.rsi_extreme_weight_boost
            whale_weight = (1.0 - tech_weight) * 0.6
            sentiment_weight = (1.0 - tech_weight) * 0.4
            logger.debug(
                f"Weight boost: RSI extreme ({tech_signal.rsi_extreme_direction.value})"
            )

        # Normalize
        total = tech_weight + whale_weight + sentiment_weight
        if total > 0:
            tech_weight /= total
            whale_weight /= total
            sentiment_weight /= total

        return {
            "tech": tech_weight,
            "whale": whale_weight,
            "sentiment": sentiment_weight,
        }

    def _check_override_conditions(
        self,
        tech_signal: TechnicalSignal,
        whale_signal: WhaleSignal,
        final_score: float,
    ) -> tuple[Optional[TradingAction], Optional[str]]:
        """
        Check if RSI extreme or whale entry should override the fusion decision (v3).

        Returns:
            Tuple of (override_action, override_reason) or (None, None)
        """
        indicator_config = self.settings.indicators

        # RSI extreme override (if enabled)
        if indicator_config.rsi_extreme_override_enabled and tech_signal.rsi_extreme:
            if tech_signal.rsi_extreme_confidence >= self.config.rsi_extreme_override_confidence:
                if tech_signal.rsi_extreme_direction == RSIExtremeDirection.OVERSOLD:
                    logger.info(
                        f"RSI extreme override: LONG (RSI={tech_signal.rsi:.1f}, "
                        f"conf={tech_signal.rsi_extreme_confidence:.2f})"
                    )
                    return TradingAction.LONG, f"RSI extreme oversold override (RSI={tech_signal.rsi:.1f})"

                elif tech_signal.rsi_extreme_direction == RSIExtremeDirection.OVERBOUGHT:
                    logger.info(
                        f"RSI extreme override: SHORT (RSI={tech_signal.rsi:.1f}, "
                        f"conf={tech_signal.rsi_extreme_confidence:.2f})"
                    )
                    return TradingAction.SHORT, f"RSI extreme overbought override (RSI={tech_signal.rsi:.1f})"

        # Whale entry signal override
        if whale_signal.whale_entry_signal and whale_signal.whale_entry_confidence >= self.config.whale_entry_override_confidence:
            if whale_signal.whale_entry_signal == "LONG":
                logger.info(
                    f"Whale entry override: LONG (conf={whale_signal.whale_entry_confidence:.2f})"
                )
                return TradingAction.LONG, f"Strong whale buying signal override (conf={whale_signal.whale_entry_confidence:.2f})"

            elif whale_signal.whale_entry_signal == "SHORT":
                logger.info(
                    f"Whale entry override: SHORT (conf={whale_signal.whale_entry_confidence:.2f})"
                )
                return TradingAction.SHORT, f"Strong whale selling signal override (conf={whale_signal.whale_entry_confidence:.2f})"

        return None, None

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
        Combine all signals into a trading decision (v3).

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
        - v3: Check override_reason for RSI/whale overrides
        """
        symbol = symbol.upper()
        timestamp = datetime.utcnow()
        symbol_config = self.settings.get_symbol_config(symbol)

        # Get base weights
        base_weights = {
            "tech": self.config.default_tech_weight,
            "whale": self.config.default_whale_weight,
            "sentiment": self.config.default_sentiment_weight,
        }

        # v3: Calculate dynamic weights
        weights = self._calculate_dynamic_weights(tech_signal, whale_signal, base_weights)

        tech_weight = weights["tech"]
        whale_weight = weights["whale"]
        sentiment_weight = weights["sentiment"]

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
        override_reason = None

        # v3: Check for RSI extreme or whale entry override
        override_action, override_msg = self._check_override_conditions(
            tech_signal, whale_signal, final_score
        )

        if override_action and not skip_reasons:
            action = override_action
            override_reason = override_msg
            decision_reason = override_msg
            # Boost confidence for overrides
            confidence = max(confidence, 0.75)

        # Check if we should skip
        elif skip_reasons:
            decision_reason = f"Skip: {'; '.join(skip_reasons)}"

        # Check minimum score threshold (v3: increased to 0.4)
        elif abs(final_score) < self.config.entry_threshold:
            decision_reason = f"Score {final_score:.3f} below threshold {self.config.entry_threshold}"
            skip_reasons.append("Score below threshold")

        # Check minimum confidence (v3: increased to 0.45)
        elif confidence < self.config.min_confidence_for_entry:
            decision_reason = f"Confidence {confidence:.3f} below minimum {self.config.min_confidence_for_entry}"
            skip_reasons.append("Confidence too low")

        else:
            # Determine direction
            is_high_confidence = confidence >= self.config.high_confidence_threshold

            if final_score > 0:
                # Bullish signal
                if self.config.require_trend_confirmation:
                    # v3: RSI extreme can bypass trend requirement
                    if tech_signal.trend == Trend.BULLISH:
                        action = TradingAction.LONG
                        decision_reason = "Bullish signal with trend confirmation"
                    elif tech_signal.trend == Trend.NEUTRAL and is_high_confidence:
                        action = TradingAction.LONG
                        decision_reason = "Bullish signal with high confidence (neutral trend)"
                    elif tech_signal.rsi_extreme and tech_signal.rsi_extreme_direction == RSIExtremeDirection.OVERSOLD:
                        action = TradingAction.LONG
                        decision_reason = f"Bullish signal with RSI extreme oversold (RSI={tech_signal.rsi:.1f})"
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
                    # v3: RSI extreme can bypass trend requirement
                    if tech_signal.trend == Trend.BEARISH:
                        action = TradingAction.SHORT
                        decision_reason = "Bearish signal with trend confirmation"
                    elif tech_signal.trend == Trend.NEUTRAL and is_high_confidence:
                        action = TradingAction.SHORT
                        decision_reason = "Bearish signal with high confidence (neutral trend)"
                    elif tech_signal.rsi_extreme and tech_signal.rsi_extreme_direction == RSIExtremeDirection.OVERBOUGHT:
                        action = TradingAction.SHORT
                        decision_reason = f"Bearish signal with RSI extreme overbought (RSI={tech_signal.rsi:.1f})"
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
            # v3: Dynamic leverage based on confidence and RSI extreme
            risk_config = self.settings.risk
            max_leverage = min(symbol_config.leverage, risk_config.max_leverage)
            min_leverage = risk_config.min_leverage

            # RSI extreme = higher confidence in direction
            if tech_signal.rsi_extreme and confidence > 0.7:
                leverage = max_leverage
            elif confidence >= 0.85:
                leverage = max_leverage
            elif confidence >= 0.70:
                leverage = min(max_leverage, 4)
            elif confidence >= 0.55:
                leverage = min(max_leverage, 3)
            else:
                leverage = min_leverage

            leverage = max(min_leverage, min(leverage, max_leverage))

            # v3: Tighter ATR-based stop loss for scalping
            atr_stop = tech_signal.atr_pct * risk_config.sl_atr_multiplier
            stop_loss_pct = max(
                risk_config.sl_min_pct,
                min(atr_stop, risk_config.sl_max_pct)
            )

            # v3: Take profit based on tighter risk-reward ratio
            take_profit_pct = max(
                stop_loss_pct * risk_config.min_risk_reward_ratio,
                risk_config.tp_default_pct
            )

            # Cap take profit
            take_profit_pct = min(take_profit_pct, risk_config.tp_max_pct)

            # Position size based on risk per trade
            position_size_pct = symbol_config.risk_per_trade_pct

        # v3: Trailing stop is always active for scalping
        trailing_stop_active = action != TradingAction.FLAT and self.settings.risk.trailing.enabled

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
            trailing_stop_active=trailing_stop_active,
            tech_score=tech_signal.tech_score,
            tech_confidence=tech_signal.tech_confidence,
            whale_score=whale_signal.whale_score,
            whale_confidence=whale_signal.whale_confidence,
            sentiment_score=sentiment_signal.sentiment_score,
            sentiment_confidence=sentiment_signal.confidence,
            override_reason=override_reason,
            rsi_extreme=tech_signal.rsi_extreme,
            whale_entry_signal=whale_signal.whale_entry_signal,
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
        current_pnl_pct: float = 0.0,
    ) -> tuple:
        """
        Check if position should be closed based on decision (v3).

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

        # v3: Graduated time-stop based on PnL
        risk_config = self.settings.risk
        position_age_minutes = (datetime.utcnow() - position_entry_time).total_seconds() / 60

        # If profitable, allow more time
        if current_pnl_pct > 0.3:
            if position_age_minutes >= risk_config.time_stop_profitable_minutes:
                return True, f"Time-stop (profitable): Position open for {position_age_minutes:.0f} minutes"
        # If slightly negative, exit faster
        elif current_pnl_pct < -0.1:
            if position_age_minutes >= risk_config.time_stop_losing_minutes:
                return True, f"Time-stop (losing): Position open for {position_age_minutes:.0f} minutes"
        # Default time stop
        elif position_age_minutes >= risk_config.time_stop_minutes:
            return True, f"Time-stop: Position open for {position_age_minutes:.0f} minutes"

        # Check for strong opposite signal
        if position_side == "LONG" and decision.action == TradingAction.SHORT:
            if decision.confidence >= self.config.high_confidence_threshold:
                return True, "Strong opposite signal (SHORT) while in LONG"

        elif position_side == "SHORT" and decision.action == TradingAction.LONG:
            if decision.confidence >= self.config.high_confidence_threshold:
                return True, "Strong opposite signal (LONG) while in SHORT"

        # v3: Check RSI extreme in opposite direction
        if decision.rsi_extreme:
            if position_side == "LONG" and decision.override_reason and "overbought" in decision.override_reason.lower():
                return True, f"RSI extreme overbought while in LONG (RSI signal)"
            elif position_side == "SHORT" and decision.override_reason and "oversold" in decision.override_reason.lower():
                return True, f"RSI extreme oversold while in SHORT (RSI signal)"

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
        Calculate position size based on risk management (v3).

        Args:
            balance: Account balance in USDT
            risk_per_trade_pct: Percentage of balance to risk
            stop_loss_pct: Stop loss percentage
            leverage: Position leverage
            current_price: Current asset price

        Returns:
            Position size in base asset units

        v3: Updated for tighter scalping parameters
        """
        if stop_loss_pct <= 0 or current_price <= 0:
            return 0.0

        # Risk amount in USDT
        risk_amount = balance * (risk_per_trade_pct / 100)

        # Position size = risk / (stop_loss_pct / 100)
        notional = risk_amount / (stop_loss_pct / 100)

        # Cap at max position size (50% of balance for scalping)
        max_notional = balance * 0.5
        notional = min(notional, max_notional)

        # Cap at symbol-specific max
        symbol_config = self.settings.symbol_configs.get("BTCUSDT")  # Default
        if symbol_config:
            notional = min(notional, symbol_config.max_position_size_usd)

        # Convert to quantity
        quantity = notional / current_price

        return quantity

    def get_fusion_summary(self, decision: FusionDecision) -> str:
        """
        Get a human-readable summary of the fusion decision.

        Returns:
            Summary string for logging/display
        """
        parts = [
            f"[{decision.symbol}]",
            f"Action: {decision.action.value}",
            f"Score: {decision.final_score:.3f}",
            f"Conf: {decision.confidence:.2f}",
        ]

        if decision.override_reason:
            parts.append(f"Override: {decision.override_reason}")

        if decision.action != TradingAction.FLAT:
            parts.extend([
                f"Lev: {decision.leverage}x",
                f"SL: {decision.stop_loss_pct:.2f}%",
                f"TP: {decision.take_profit_pct:.2f}%",
            ])

        if decision.trailing_stop_active:
            parts.append("Trailing: ON")

        if decision.recommended_defensive_action:
            parts.append(f"Defense: {decision.recommended_defensive_action.value}")

        return " | ".join(parts)
