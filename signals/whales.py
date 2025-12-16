"""
Whale activity detection module (v3).
Detects large trades and order book imbalances.
Provides explicit defensive actions (NOT FLAT signals).

v3: Added offensive whale entry signals for momentum trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
import time

import numpy as np

from config import WhaleConfig, WhaleDefensiveAction, get_settings
from data.market_data import MarketDataManager
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class WhaleSignal:
    """
    Whale activity signal output (v3).

    IMPORTANT: This module does NOT output FLAT to trigger closes.
    It outputs explicit defensive actions via recommended_action.

    v3: Added whale_entry_signal for offensive trading based on whale activity.
    """
    symbol: str
    timestamp: datetime

    # Core outputs
    whale_score: float  # -1 (bearish whale activity) to +1 (bullish whale activity)
    whale_confidence: float  # 0 to 1

    # Alert flags - CRITICAL: whale_alert_opposite triggers defensive action
    whale_alert_opposite: bool  # True if whale activity is against your position
    recommended_action: WhaleDefensiveAction  # Explicit action to take

    # v3 NEW: Entry signals from whale activity (offensive trading)
    whale_entry_signal: Optional[str] = None  # "LONG", "SHORT", or None
    whale_entry_confidence: float = 0.0  # 0 to 1 - confidence in entry signal

    # Detection details
    large_buy_count: int = 0
    large_sell_count: int = 0
    large_buy_volume: float = 0.0
    large_sell_volume: float = 0.0
    orderbook_imbalance: float = 0.0  # -1 to +1
    imbalance_spike: bool = False

    # Metadata
    detection_reason: str = ""
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "symbol": self.symbol,
            "whale_score": round(self.whale_score, 4),
            "whale_confidence": round(self.whale_confidence, 4),
            "whale_alert_opposite": self.whale_alert_opposite,
            "recommended_action": self.recommended_action.value,
            "whale_entry_signal": self.whale_entry_signal,
            "whale_entry_confidence": round(self.whale_entry_confidence, 4),
            "large_buy_count": self.large_buy_count,
            "large_sell_count": self.large_sell_count,
            "orderbook_imbalance": round(self.orderbook_imbalance, 4),
            "detection_reason": self.detection_reason,
        }


@dataclass
class WhaleThresholds:
    """Dynamic thresholds for whale detection."""
    large_trade_size: float  # Notional value threshold
    percentile_95: float
    percentile_99: float
    avg_trade_size: float
    std_trade_size: float


class WhaleDetector:
    """
    Whale activity detector using Binance public data.

    Detection methods:
    1. AggTrades: Detect large notional trades vs rolling percentile
    2. Order book imbalance spikes
    3. Optional: OI/funding anomalies (flagged only if robust)

    IMPORTANT: This module outputs explicit defensive actions.
    It does NOT return FLAT to trigger closes.
    """

    def __init__(
        self,
        market_data: MarketDataManager,
        config: Optional[WhaleConfig] = None,
    ):
        self.market_data = market_data
        self.config = config or get_settings().whale

        # Rolling thresholds per symbol
        self._thresholds: Dict[str, WhaleThresholds] = {}

        # Alert cooldowns per symbol
        self._last_alert_time: Dict[str, datetime] = {}

        # Historical imbalance for spike detection
        self._imbalance_history: Dict[str, deque] = {}

    def _calculate_thresholds(self, symbol: str) -> Optional[WhaleThresholds]:
        """
        Calculate dynamic thresholds from recent trade data.
        """
        trades = self.market_data.get_cached_trades(symbol)
        if len(trades) < 100:
            return None

        # Calculate notional values
        notionals = [t['price'] * t['quantity'] for t in trades]

        percentile = self.config.large_trade_percentile
        threshold = WhaleThresholds(
            large_trade_size=np.percentile(notionals, percentile),
            percentile_95=np.percentile(notionals, 95),
            percentile_99=np.percentile(notionals, 99),
            avg_trade_size=np.mean(notionals),
            std_trade_size=np.std(notionals),
        )

        self._thresholds[symbol] = threshold
        return threshold

    def _detect_large_trades(
        self,
        symbol: str,
        lookback_seconds: int = 60,
    ) -> Dict[str, Any]:
        """
        Detect large trades in recent data.

        Returns:
            Dict with large_buys, large_sells, buy_volume, sell_volume
        """
        thresholds = self._thresholds.get(symbol)
        if not thresholds:
            thresholds = self._calculate_thresholds(symbol)
            if not thresholds:
                return {
                    'large_buys': 0, 'large_sells': 0,
                    'buy_volume': 0.0, 'sell_volume': 0.0,
                    'is_valid': False,
                }

        trades = self.market_data.get_cached_trades(symbol)
        if not trades:
            return {
                'large_buys': 0, 'large_sells': 0,
                'buy_volume': 0.0, 'sell_volume': 0.0,
                'is_valid': False,
            }

        # Filter to recent trades
        cutoff_time = int((time.time() - lookback_seconds) * 1000)
        recent_trades = [t for t in trades if t['timestamp'] >= cutoff_time]

        large_buys = 0
        large_sells = 0
        buy_volume = 0.0
        sell_volume = 0.0

        for trade in recent_trades:
            notional = trade['price'] * trade['quantity']

            if notional >= thresholds.large_trade_size:
                if trade['is_buyer_maker']:
                    # Seller was taker (aggressive sell)
                    large_sells += 1
                    sell_volume += notional
                else:
                    # Buyer was taker (aggressive buy)
                    large_buys += 1
                    buy_volume += notional

        return {
            'large_buys': large_buys,
            'large_sells': large_sells,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'threshold': thresholds.large_trade_size,
            'is_valid': True,
        }

    def _detect_imbalance_spike(self, symbol: str, current_imbalance: float) -> bool:
        """
        Detect sudden spike in order book imbalance.
        """
        if symbol not in self._imbalance_history:
            self._imbalance_history[symbol] = deque(maxlen=60)  # 1 minute history at 1s intervals

        history = self._imbalance_history[symbol]
        history.append(current_imbalance)

        if len(history) < 10:
            return False

        # Calculate recent change
        recent_avg = np.mean(list(history)[-10:])
        older_avg = np.mean(list(history)[:-10]) if len(history) > 10 else 0

        change = abs(current_imbalance - older_avg)

        # Spike if change exceeds threshold
        return change >= self.config.imbalance_threshold

    def _detect_whale_entry_signal(
        self,
        symbol: str,
        buy_volume: float,
        sell_volume: float,
        large_buy_count: int,
        large_sell_count: int,
        orderbook_imbalance: float,
    ) -> tuple[Optional[str], float]:
        """
        Detect whale entry signals (v3 offensive trading).

        Use whale activity as ENTRY signal, not just defensive.

        Args:
            symbol: Trading pair
            buy_volume: Total large buy volume
            sell_volume: Total large sell volume
            large_buy_count: Number of large buy trades
            large_sell_count: Number of large sell trades
            orderbook_imbalance: Current orderbook imbalance

        Returns:
            Tuple of (signal: "LONG"/"SHORT"/None, confidence: 0-1)
        """
        if not self.config.entry_signal_enabled:
            return None, 0.0

        total_volume = buy_volume + sell_volume
        total_large_trades = large_buy_count + large_sell_count

        if total_volume == 0:
            return None, 0.0

        # Calculate volume imbalance
        imbalance = (buy_volume - sell_volume) / total_volume

        # Get thresholds from config
        imbalance_threshold = self.config.entry_imbalance_threshold
        strong_imbalance_threshold = self.config.entry_strong_imbalance_threshold
        min_large_trades = self.config.entry_min_large_trades

        # Check for LONG entry signal (whale buying)
        if imbalance > 0:
            # Very strong whale buying - 3:1 buy:sell ratio with high imbalance
            if imbalance >= strong_imbalance_threshold and buy_volume > sell_volume * 3:
                confidence = 0.95
                logger.info(
                    f"Whale LONG entry signal (very strong)",
                    extra={
                        "symbol": symbol,
                        "imbalance": round(imbalance, 4),
                        "buy_volume": buy_volume,
                        "sell_volume": sell_volume,
                        "confidence": confidence,
                    }
                )
                return "LONG", confidence

            # Strong whale buying - significant imbalance with multiple large trades
            if imbalance >= imbalance_threshold and large_buy_count >= min_large_trades:
                confidence = min(0.9, 0.5 + imbalance * 0.5)

                # Boost confidence if orderbook supports
                if orderbook_imbalance > 0.3:
                    confidence = min(0.95, confidence + 0.1)

                if confidence >= self.config.entry_min_confidence:
                    logger.info(
                        f"Whale LONG entry signal",
                        extra={
                            "symbol": symbol,
                            "imbalance": round(imbalance, 4),
                            "large_buys": large_buy_count,
                            "confidence": round(confidence, 4),
                        }
                    )
                    return "LONG", confidence

        # Check for SHORT entry signal (whale selling)
        elif imbalance < 0:
            abs_imbalance = abs(imbalance)

            # Very strong whale selling - 3:1 sell:buy ratio with high imbalance
            if abs_imbalance >= strong_imbalance_threshold and sell_volume > buy_volume * 3:
                confidence = 0.95
                logger.info(
                    f"Whale SHORT entry signal (very strong)",
                    extra={
                        "symbol": symbol,
                        "imbalance": round(imbalance, 4),
                        "buy_volume": buy_volume,
                        "sell_volume": sell_volume,
                        "confidence": confidence,
                    }
                )
                return "SHORT", confidence

            # Strong whale selling - significant imbalance with multiple large trades
            if abs_imbalance >= imbalance_threshold and large_sell_count >= min_large_trades:
                confidence = min(0.9, 0.5 + abs_imbalance * 0.5)

                # Boost confidence if orderbook supports
                if orderbook_imbalance < -0.3:
                    confidence = min(0.95, confidence + 0.1)

                if confidence >= self.config.entry_min_confidence:
                    logger.info(
                        f"Whale SHORT entry signal",
                        extra={
                            "symbol": symbol,
                            "imbalance": round(imbalance, 4),
                            "large_sells": large_sell_count,
                            "confidence": round(confidence, 4),
                        }
                    )
                    return "SHORT", confidence

        return None, 0.0

    def _is_alert_cooldown(self, symbol: str) -> bool:
        """Check if we're in alert cooldown period."""
        if symbol not in self._last_alert_time:
            return False

        elapsed = (datetime.utcnow() - self._last_alert_time[symbol]).total_seconds()
        return elapsed < self.config.alert_cooldown_seconds

    def analyze(
        self,
        symbol: str,
        position_side: Optional[str] = None,  # 'LONG', 'SHORT', or None
    ) -> WhaleSignal:
        """
        Analyze whale activity for a symbol (v3).

        Args:
            symbol: Trading pair
            position_side: Current position side (for determining opposite activity)

        Returns:
            WhaleSignal with explicit defensive action if needed
            v3: Also includes whale_entry_signal for offensive trading

        IMPORTANT: This method outputs recommended_action for defensive measures.
        It does NOT output FLAT to trigger position closes.
        """
        symbol = symbol.upper()
        timestamp = datetime.utcnow()

        # Ensure we have fresh data
        try:
            self.market_data.fetch_recent_trades(symbol)
        except Exception as e:
            logger.warning(f"Could not fetch trades for whale detection: {e}")
            return WhaleSignal(
                symbol=symbol,
                timestamp=timestamp,
                whale_score=0,
                whale_confidence=0,
                whale_alert_opposite=False,
                recommended_action=WhaleDefensiveAction.NONE,
                is_valid=False,
            )

        # 1. Detect large trades
        trade_data = self._detect_large_trades(symbol)

        large_buy_count = trade_data['large_buys']
        large_sell_count = trade_data['large_sells']
        buy_volume = trade_data['buy_volume']
        sell_volume = trade_data['sell_volume']

        # 2. Get order book imbalance
        imbalance = self.market_data.calculate_orderbook_imbalance(symbol)
        imbalance_spike = self._detect_imbalance_spike(symbol, imbalance)

        # 3. Calculate whale score
        # Positive = bullish whale activity, Negative = bearish whale activity
        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            volume_direction = (buy_volume - sell_volume) / total_volume
        else:
            volume_direction = 0

        # Combine trade direction with orderbook imbalance
        whale_score = (volume_direction * 0.7) + (imbalance * 0.3)
        whale_score = np.clip(whale_score, -1, 1)

        # 4. Calculate confidence
        # Higher confidence with more large trades and stronger imbalance
        trade_count_factor = min((large_buy_count + large_sell_count) / 5, 1.0)
        imbalance_factor = abs(imbalance)
        whale_confidence = (trade_count_factor * 0.6 + imbalance_factor * 0.4)
        whale_confidence = np.clip(whale_confidence, 0, 1)

        # 5. v3 NEW: Detect whale entry signals (offensive trading)
        whale_entry_signal, whale_entry_confidence = self._detect_whale_entry_signal(
            symbol=symbol,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            large_buy_count=large_buy_count,
            large_sell_count=large_sell_count,
            orderbook_imbalance=imbalance,
        )

        # 6. Determine if whale activity is opposite to position
        whale_alert_opposite = False
        recommended_action = WhaleDefensiveAction.NONE
        detection_reason = ""

        if position_side and not self._is_alert_cooldown(symbol):
            # Check if whale activity threatens position
            if position_side == "LONG":
                # Threat: Strong bearish whale activity
                if whale_score < -0.4 and whale_confidence > 0.5:
                    whale_alert_opposite = True
                    detection_reason = f"Bearish whale activity detected (score={whale_score:.2f}) against LONG position"
                elif imbalance_spike and imbalance < -0.3:
                    whale_alert_opposite = True
                    detection_reason = f"Order book imbalance spike (sell pressure) against LONG position"

            elif position_side == "SHORT":
                # Threat: Strong bullish whale activity
                if whale_score > 0.4 and whale_confidence > 0.5:
                    whale_alert_opposite = True
                    detection_reason = f"Bullish whale activity detected (score={whale_score:.2f}) against SHORT position"
                elif imbalance_spike and imbalance > 0.3:
                    whale_alert_opposite = True
                    detection_reason = f"Order book imbalance spike (buy pressure) against SHORT position"

            # 7. Determine defensive action (EXPLICIT, NOT FLAT)
            if whale_alert_opposite:
                self._last_alert_time[symbol] = timestamp

                # Severity determines action
                if abs(whale_score) > 0.7 or (imbalance_spike and abs(imbalance) > 0.5):
                    # Severe threat - close immediately
                    recommended_action = self.config.default_defensive_action
                elif abs(whale_score) > 0.5:
                    # Moderate threat - reduce position
                    recommended_action = WhaleDefensiveAction.REDUCE_50_PERCENT
                else:
                    # Light threat - tighten stop
                    recommended_action = WhaleDefensiveAction.TIGHTEN_STOP

                logger.whale_alert(
                    symbol=symbol,
                    direction="BEARISH" if whale_score < 0 else "BULLISH",
                    score=whale_score,
                    action=recommended_action.value,
                    detection_reason=detection_reason,
                    confidence=whale_confidence,
                )

        return WhaleSignal(
            symbol=symbol,
            timestamp=timestamp,
            whale_score=whale_score,
            whale_confidence=whale_confidence,
            whale_alert_opposite=whale_alert_opposite,
            recommended_action=recommended_action,
            whale_entry_signal=whale_entry_signal,  # v3: Entry signal
            whale_entry_confidence=whale_entry_confidence,  # v3: Entry confidence
            large_buy_count=large_buy_count,
            large_sell_count=large_sell_count,
            large_buy_volume=buy_volume,
            large_sell_volume=sell_volume,
            orderbook_imbalance=imbalance,
            imbalance_spike=imbalance_spike,
            detection_reason=detection_reason,
            is_valid=trade_data.get('is_valid', True),
        )

    def get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get overall market sentiment from whale activity.
        Does not include defensive action logic.
        """
        signal = self.analyze(symbol, position_side=None)

        if signal.whale_score > 0.3:
            sentiment = "bullish"
        elif signal.whale_score < -0.3:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        return {
            "symbol": symbol,
            "sentiment": sentiment,
            "whale_score": signal.whale_score,
            "confidence": signal.whale_confidence,
            "large_trade_count": signal.large_buy_count + signal.large_sell_count,
            "orderbook_imbalance": signal.orderbook_imbalance,
        }
