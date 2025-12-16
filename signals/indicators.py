"""
Technical indicators module (v3).
Computes RSI, MACD, Bollinger Bands, ATR, EMA and generates trading signals.
Includes RSI extreme detection for strong standalone entry signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from config import IndicatorConfig, Trend, RSIExtremeDirection, get_settings
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class TechnicalSignal:
    """Technical analysis signal output (v3 with RSI extreme)."""
    symbol: str

    # Core outputs
    tech_score: float  # -1 to +1
    tech_confidence: float  # 0 to 1
    trend: Trend

    # ATR for position sizing
    atr_value: float
    atr_pct: float  # ATR as percentage of price

    # Individual indicator values
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_position: float  # Position within bands (-1 to +1)
    ema_fast: float
    ema_slow: float
    current_price: float

    # RSI Extreme Detection (v3 new)
    rsi_extreme: bool = False
    rsi_extreme_direction: RSIExtremeDirection = RSIExtremeDirection.NONE
    rsi_extreme_score: float = 0.0  # Standalone RSI extreme score
    rsi_extreme_confidence: float = 0.0  # Confidence in RSI extreme signal

    # Validity
    is_valid: bool = True
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "symbol": self.symbol,
            "tech_score": round(self.tech_score, 4),
            "tech_confidence": round(self.tech_confidence, 4),
            "trend": self.trend.value,
            "atr_pct": round(self.atr_pct, 4),
            "rsi": round(self.rsi, 2),
            "rsi_extreme": self.rsi_extreme,
            "rsi_extreme_direction": self.rsi_extreme_direction.value,
            "rsi_extreme_confidence": round(self.rsi_extreme_confidence, 4),
            "macd_histogram": round(self.macd_histogram, 6),
            "bb_position": round(self.bb_position, 4),
            "is_valid": self.is_valid,
        }


class TechnicalIndicators:
    """
    Technical indicator calculator (v3).
    Thread-safe, stateless computation on provided OHLCV data.
    Includes RSI extreme value detection for strong entry signals.
    """

    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or get_settings().indicators

    def calculate_rsi(self, prices: pd.Series, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Relative Strength Index.

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        period = period or self.config.rsi_period

        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)  # Neutral on insufficient data

    def calculate_macd(
        self,
        prices: pd.Series,
        fast: Optional[int] = None,
        slow: Optional[int] = None,
        signal: Optional[int] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        fast = fast or self.config.macd_fast
        slow = slow or self.config.macd_slow
        signal = signal or self.config.macd_signal

        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: Optional[int] = None,
        std_dev: Optional[float] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Returns:
            Tuple of (Upper band, Middle band (SMA), Lower band)
        """
        period = period or self.config.bb_period
        std_dev = std_dev or self.config.bb_std

        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    def calculate_atr(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Average True Range.

        ATR = EMA of True Range
        True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        """
        period = period or self.config.atr_period

        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()

    def _score_rsi_extreme(self, rsi: float) -> Tuple[float, float, RSIExtremeDirection, bool]:
        """
        Score RSI for extreme values (v3 new).

        RSI extreme values generate strong standalone signals that can
        override low fusion scores.

        Args:
            rsi: Current RSI value

        Returns:
            Tuple of (score, confidence, direction, is_extreme)
        """
        very_oversold = self.config.rsi_very_extreme_oversold
        oversold = self.config.rsi_extreme_oversold
        overbought = self.config.rsi_extreme_overbought
        very_overbought = self.config.rsi_very_extreme_overbought

        # VERY EXTREME OVERSOLD - Very strong BUY signal
        if rsi <= very_oversold:
            return 0.95, 0.90, RSIExtremeDirection.OVERSOLD, True

        # STRONG OVERSOLD - Strong BUY signal
        if rsi <= oversold:
            # Scale score between 0.85 and 0.95 based on how close to very_extreme
            score = 0.85 + (oversold - rsi) / (oversold - very_oversold) * 0.10
            confidence = 0.80 + (oversold - rsi) / (oversold - very_oversold) * 0.10
            return score, confidence, RSIExtremeDirection.OVERSOLD, True

        # MODERATE OVERSOLD (between extreme threshold and standard threshold)
        if rsi <= self.config.rsi_oversold:
            score = 0.60 + (self.config.rsi_oversold - rsi) / (self.config.rsi_oversold - oversold) * 0.25
            confidence = 0.65
            return score, confidence, RSIExtremeDirection.OVERSOLD, False

        # VERY EXTREME OVERBOUGHT - Very strong SELL signal
        if rsi >= very_overbought:
            return -0.95, 0.90, RSIExtremeDirection.OVERBOUGHT, True

        # STRONG OVERBOUGHT - Strong SELL signal
        if rsi >= overbought:
            # Scale score between -0.85 and -0.95
            score = -0.85 - (rsi - overbought) / (very_overbought - overbought) * 0.10
            confidence = 0.80 + (rsi - overbought) / (very_overbought - overbought) * 0.10
            return score, confidence, RSIExtremeDirection.OVERBOUGHT, True

        # MODERATE OVERBOUGHT
        if rsi >= self.config.rsi_overbought:
            score = -0.60 - (rsi - self.config.rsi_overbought) / (overbought - self.config.rsi_overbought) * 0.25
            confidence = 0.65
            return score, confidence, RSIExtremeDirection.OVERBOUGHT, False

        # NEUTRAL ZONE - Weak signal
        mid = 50
        if rsi > mid:
            score = -((rsi - mid) / (self.config.rsi_overbought - mid)) * 0.3
        else:
            score = ((mid - rsi) / (mid - self.config.rsi_oversold)) * 0.3

        return score, 0.3, RSIExtremeDirection.NONE, False

    def _score_rsi(self, rsi: float) -> Tuple[float, float]:
        """
        Score RSI indicator (standard scoring for fusion).

        Returns:
            Tuple of (score -1 to +1, confidence 0 to 1)
        """
        overbought = self.config.rsi_overbought
        oversold = self.config.rsi_oversold

        if rsi >= overbought:
            # Overbought - bearish signal
            score = -((rsi - overbought) / (100 - overbought))
            confidence = min((rsi - overbought) / 20, 1.0)
        elif rsi <= oversold:
            # Oversold - bullish signal
            score = (oversold - rsi) / oversold
            confidence = min((oversold - rsi) / 20, 1.0)
        else:
            # Neutral zone
            mid = (overbought + oversold) / 2
            if rsi > mid:
                score = -((rsi - mid) / (overbought - mid)) * 0.3
            else:
                score = ((mid - rsi) / (mid - oversold)) * 0.3
            confidence = 0.3

        return np.clip(score, -1, 1), np.clip(confidence, 0, 1)

    def _score_macd(self, histogram: float, macd: float, prev_histogram: float) -> Tuple[float, float]:
        """
        Score MACD indicator based on histogram and crossovers.

        Returns:
            Tuple of (score -1 to +1, confidence 0 to 1)
        """
        # Direction from histogram
        if histogram > 0:
            direction = 1  # Bullish
        elif histogram < 0:
            direction = -1  # Bearish
        else:
            direction = 0

        # Momentum (histogram change)
        momentum = histogram - prev_histogram if prev_histogram is not None else 0

        # Score based on direction and momentum alignment
        if direction > 0 and momentum > 0:
            score = min(0.5 + abs(momentum) * 100, 1.0)  # Strong bullish
        elif direction < 0 and momentum < 0:
            score = max(-0.5 - abs(momentum) * 100, -1.0)  # Strong bearish
        elif direction > 0:
            score = 0.3  # Weak bullish (histogram positive but falling)
        elif direction < 0:
            score = -0.3  # Weak bearish (histogram negative but rising)
        else:
            score = 0

        # Confidence based on histogram magnitude
        confidence = min(abs(histogram) * 1000, 1.0)

        return np.clip(score, -1, 1), np.clip(confidence, 0.1, 1)

    def _score_bollinger(self, price: float, upper: float, middle: float, lower: float) -> Tuple[float, float]:
        """
        Score Bollinger Bands position.

        Returns:
            Tuple of (score -1 to +1, bb_position -1 to +1)
        """
        band_width = upper - lower
        if band_width == 0:
            return 0, 0

        # Position within bands: -1 at lower, 0 at middle, +1 at upper
        position = (price - middle) / (band_width / 2)
        position = np.clip(position, -1.5, 1.5)

        # Score: mean reversion assumption
        if position > 1:
            score = -(position - 1) / 0.5  # Overbought
        elif position < -1:
            score = (-position - 1) / 0.5  # Oversold
        else:
            score = -position * 0.3  # Weak mean reversion signal

        return np.clip(score, -1, 1), position

    def analyze(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> TechnicalSignal:
        """
        Perform full technical analysis on OHLCV DataFrame (v3).

        Args:
            df: DataFrame with columns: open, high, low, close, volume
            symbol: Symbol name for logging

        Returns:
            TechnicalSignal with computed indicators, score, and RSI extreme detection
        """
        try:
            # Validate input
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                return self._create_invalid_signal(symbol, "Missing required OHLCV columns")

            min_rows = max(self.config.ema_slow, self.config.bb_period, self.config.macd_slow) + 10
            if len(df) < min_rows:
                return self._create_invalid_signal(symbol, f"Insufficient data: {len(df)} rows, need {min_rows}")

            close = df['close']
            current_price = float(close.iloc[-1])

            # Calculate all indicators
            rsi = self.calculate_rsi(close)
            macd_line, macd_signal, macd_hist = self.calculate_macd(close)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            atr = self.calculate_atr(df)
            ema_fast = self.calculate_ema(close, self.config.ema_fast)
            ema_slow = self.calculate_ema(close, self.config.ema_slow)

            # Get current values
            current_rsi = float(rsi.iloc[-1])
            current_macd = float(macd_line.iloc[-1])
            current_macd_signal = float(macd_signal.iloc[-1])
            current_macd_hist = float(macd_hist.iloc[-1])
            prev_macd_hist = float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else 0
            current_bb_upper = float(bb_upper.iloc[-1])
            current_bb_middle = float(bb_middle.iloc[-1])
            current_bb_lower = float(bb_lower.iloc[-1])
            current_atr = float(atr.iloc[-1])
            current_ema_fast = float(ema_fast.iloc[-1])
            current_ema_slow = float(ema_slow.iloc[-1])

            # Check for NaN values
            values = [current_rsi, current_macd, current_atr, current_ema_fast, current_ema_slow]
            if any(pd.isna(v) for v in values):
                return self._create_invalid_signal(symbol, "NaN values in indicators", current_price)

            # ATR as percentage
            atr_pct = (current_atr / current_price * 100) if current_price > 0 else 0

            # Determine trend from EMAs
            if current_ema_fast > current_ema_slow * 1.002:  # 0.2% buffer
                trend = Trend.BULLISH
            elif current_ema_fast < current_ema_slow * 0.998:
                trend = Trend.BEARISH
            else:
                trend = Trend.NEUTRAL

            # RSI Extreme Detection (v3 new)
            rsi_extreme_score, rsi_extreme_conf, rsi_extreme_dir, is_rsi_extreme = self._score_rsi_extreme(current_rsi)

            # Score individual indicators
            rsi_score, rsi_conf = self._score_rsi(current_rsi)
            macd_score, macd_conf = self._score_macd(current_macd_hist, current_macd, prev_macd_hist)
            bb_score, bb_position = self._score_bollinger(
                current_price, current_bb_upper, current_bb_middle, current_bb_lower
            )

            # Trend contribution to score
            if trend == Trend.BULLISH:
                trend_score = 0.3
            elif trend == Trend.BEARISH:
                trend_score = -0.3
            else:
                trend_score = 0

            # Combine scores (weighted average)
            # RSI: 30%, MACD: 30%, BB: 25%, Trend: 15% (v3: slightly more weight on RSI)
            tech_score = (
                rsi_score * 0.30 +
                macd_score * 0.30 +
                bb_score * 0.25 +
                trend_score * 0.15
            )

            # Combine confidence
            tech_confidence = (rsi_conf * 0.35 + macd_conf * 0.35 + 0.3)  # Base confidence

            # Boost confidence if RSI is extreme
            if is_rsi_extreme:
                tech_confidence = max(tech_confidence, rsi_extreme_conf)
                # Also boost tech_score toward RSI extreme direction
                tech_score = tech_score * 0.6 + rsi_extreme_score * 0.4

            # Reduce confidence if indicators disagree
            scores = [rsi_score, macd_score, bb_score]
            if not all(s >= 0 for s in scores) and not all(s <= 0 for s in scores):
                if not is_rsi_extreme:  # Don't reduce if RSI extreme
                    tech_confidence *= 0.7

            return TechnicalSignal(
                symbol=symbol,
                tech_score=np.clip(tech_score, -1, 1),
                tech_confidence=np.clip(tech_confidence, 0, 1),
                trend=trend,
                atr_value=current_atr,
                atr_pct=atr_pct,
                rsi=current_rsi,
                macd=current_macd,
                macd_signal=current_macd_signal,
                macd_histogram=current_macd_hist,
                bb_upper=current_bb_upper,
                bb_middle=current_bb_middle,
                bb_lower=current_bb_lower,
                bb_position=bb_position,
                ema_fast=current_ema_fast,
                ema_slow=current_ema_slow,
                current_price=current_price,
                rsi_extreme=is_rsi_extreme,
                rsi_extreme_direction=rsi_extreme_dir,
                rsi_extreme_score=rsi_extreme_score,
                rsi_extreme_confidence=rsi_extreme_conf,
                is_valid=True,
            )

        except Exception as e:
            logger.error(f"Technical analysis error: {e}", extra={"symbol": symbol})
            return self._create_invalid_signal(symbol, str(e))

    def _create_invalid_signal(
        self,
        symbol: str,
        error_message: str,
        current_price: float = 0.0
    ) -> TechnicalSignal:
        """Create an invalid TechnicalSignal with default values."""
        return TechnicalSignal(
            symbol=symbol,
            tech_score=0,
            tech_confidence=0,
            trend=Trend.NEUTRAL,
            atr_value=0,
            atr_pct=0,
            rsi=50,
            macd=0,
            macd_signal=0,
            macd_histogram=0,
            bb_upper=0,
            bb_middle=0,
            bb_lower=0,
            bb_position=0,
            ema_fast=0,
            ema_slow=0,
            current_price=current_price,
            rsi_extreme=False,
            rsi_extreme_direction=RSIExtremeDirection.NONE,
            rsi_extreme_score=0,
            rsi_extreme_confidence=0,
            is_valid=False,
            error_message=error_message,
        )

    def analyze_multi_timeframe(
        self,
        candles: Dict[str, pd.DataFrame],
        symbol: str = "UNKNOWN",
    ) -> Dict[str, TechnicalSignal]:
        """
        Analyze multiple timeframes.

        Args:
            candles: Dict mapping interval to DataFrame
            symbol: Symbol name

        Returns:
            Dict mapping interval to TechnicalSignal
        """
        results = {}
        for interval, df in candles.items():
            results[interval] = self.analyze(df, f"{symbol}_{interval}")
        return results
