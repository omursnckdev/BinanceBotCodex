"""
Configuration module with Pydantic settings and safety toggles.
Testnet-first design with multiple safety layers for live trading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Trading environment."""
    TESTNET = "testnet"
    MAINNET = "mainnet"


class MarginType(str, Enum):
    """Margin type for futures trading."""
    ISOLATED = "ISOLATED"
    CROSSED = "CROSSED"


class NewsProvider(str, Enum):
    """Supported news/sentiment providers."""
    CRYPTOPANIC = "cryptopanic"
    NEWSAPI = "newsapi"
    GDELT = "gdelt"


class WhaleDefensiveAction(str, Enum):
    """Defensive actions when whale activity detected against position."""
    NONE = "none"
    CLOSE_MARKET = "close_market"
    REDUCE_50_PERCENT = "reduce_50_percent"
    TIGHTEN_STOP = "tighten_stop"


class TradingAction(str, Enum):
    """Trading action - FLAT means NO NEW ENTRY, NOT close position."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"  # NO NEW ENTRY - does NOT mean close position


class Trend(str, Enum):
    """Market trend direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SymbolConfig(BaseModel):
    """Per-symbol configuration."""
    symbol: str
    enabled: bool = True
    leverage: int = Field(default=3, ge=1, le=125)
    margin_type: MarginType = MarginType.ISOLATED

    # Position sizing
    max_position_size_usd: float = Field(default=1000.0, gt=0)
    risk_per_trade_pct: float = Field(default=1.0, ge=0.1, le=5.0)

    # Scalping parameters
    take_profit_pct: float = Field(default=1.2, ge=0.5, le=5.0)
    stop_loss_pct: float = Field(default=0.6, ge=0.2, le=3.0)

    # Signal weights for fusion
    tech_weight: float = Field(default=0.5, ge=0, le=1)
    whale_weight: float = Field(default=0.3, ge=0, le=1)
    sentiment_weight: float = Field(default=0.2, ge=0, le=1)

    # Cooldown after exit (seconds)
    cooldown_seconds: int = Field(default=300, ge=0)

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return v.upper()


class RiskConfig(BaseModel):
    """Risk management configuration with kill switches."""
    # Kill switches - mandatory
    max_daily_loss_pct: float = Field(default=5.0, ge=0.5, le=20.0)
    max_consecutive_losses: int = Field(default=5, ge=1, le=20)
    max_total_exposure_pct: float = Field(default=50.0, ge=10, le=100)
    max_open_positions: int = Field(default=3, ge=1, le=10)

    # Position sizing
    default_risk_per_trade_pct: float = Field(default=1.0, ge=0.1, le=5.0)
    max_leverage: int = Field(default=5, ge=1, le=20)
    min_leverage: int = Field(default=1, ge=1, le=5)

    # Stop-loss bounds (ATR-based, clamped to scalping range)
    sl_atr_multiplier: float = Field(default=1.5, ge=0.5, le=5.0)
    sl_min_pct: float = Field(default=0.3, ge=0.1, le=1.0)
    sl_max_pct: float = Field(default=0.9, ge=0.5, le=3.0)

    # Take-profit
    tp_default_pct: float = Field(default=1.2, ge=0.5, le=5.0)
    min_risk_reward_ratio: float = Field(default=1.2, ge=1.0, le=5.0)

    # Time-stop (minutes)
    time_stop_minutes: int = Field(default=45, ge=5, le=240)


class DataConfig(BaseModel):
    """Market data configuration."""
    # Candle timeframes
    primary_timeframe: str = Field(default="1m")
    confirmation_timeframes: List[str] = Field(default=["5m", "15m"])

    # Data integrity thresholds
    max_candle_staleness_seconds: int = Field(default=120, ge=30, le=600)
    max_time_sync_drift_ms: int = Field(default=2000, ge=500, le=10000)
    max_spread_pct: float = Field(default=0.15, ge=0.01, le=1.0)
    min_liquidity_usd: float = Field(default=50000, ge=1000)

    # Candle history for indicators
    candle_limit: int = Field(default=500, ge=100, le=1500)


class IndicatorConfig(BaseModel):
    """Technical indicator configuration."""
    rsi_period: int = Field(default=14, ge=5, le=50)
    rsi_overbought: float = Field(default=70.0, ge=60, le=90)
    rsi_oversold: float = Field(default=30.0, ge=10, le=40)

    macd_fast: int = Field(default=12, ge=5, le=20)
    macd_slow: int = Field(default=26, ge=15, le=50)
    macd_signal: int = Field(default=9, ge=5, le=20)

    bb_period: int = Field(default=20, ge=10, le=50)
    bb_std: float = Field(default=2.0, ge=1.0, le=3.0)

    atr_period: int = Field(default=14, ge=5, le=50)

    ema_fast: int = Field(default=50, ge=10, le=100)
    ema_slow: int = Field(default=200, ge=50, le=500)


class WhaleConfig(BaseModel):
    """Whale detection configuration."""
    enabled: bool = True

    # AggTrades detection
    large_trade_percentile: float = Field(default=95.0, ge=80, le=99.9)
    rolling_window_trades: int = Field(default=1000, ge=100, le=10000)

    # Order book imbalance
    orderbook_depth_levels: int = Field(default=20, ge=5, le=100)
    imbalance_threshold: float = Field(default=0.3, ge=0.1, le=0.8)

    # Defensive action on testnet
    default_defensive_action: WhaleDefensiveAction = WhaleDefensiveAction.CLOSE_MARKET

    # Alert cooldown
    alert_cooldown_seconds: int = Field(default=60, ge=10, le=300)


class SentimentConfig(BaseModel):
    """Sentiment analysis configuration."""
    enabled: bool = True
    provider: NewsProvider = NewsProvider.CRYPTOPANIC

    # API settings
    cache_ttl_seconds: int = Field(default=300, ge=60, le=1800)
    max_articles_per_request: int = Field(default=50, ge=10, le=200)

    # Scoring
    use_transformer: bool = False  # Default to VADER
    min_confidence_threshold: float = Field(default=0.3, ge=0, le=1)

    # Rate limiting
    rate_limit_requests_per_minute: int = Field(default=10, ge=1, le=60)


class FusionConfig(BaseModel):
    """Signal fusion configuration."""
    # Entry threshold - skip if |final_score| below this
    entry_threshold: float = Field(default=0.3, ge=0.1, le=0.8)

    # Confidence thresholds
    min_confidence_for_entry: float = Field(default=0.4, ge=0.2, le=0.9)
    high_confidence_threshold: float = Field(default=0.7, ge=0.5, le=0.95)

    # Anti flip-flop
    min_cooldown_after_exit_seconds: int = Field(default=300, ge=60, le=3600)
    require_trend_confirmation: bool = True
    allow_counter_trend_high_confidence: bool = True


class Settings(BaseSettings):
    """Main application settings - loads from environment."""

    # === SAFETY TOGGLES (CRITICAL) ===
    env: Environment = Field(default=Environment.TESTNET)
    dry_run: bool = Field(default=True)
    allow_live_trading: bool = Field(default=False)

    # === API KEYS ===
    binance_api_key: str = Field(default="")
    binance_api_secret: str = Field(default="")

    # News API keys (optional)
    cryptopanic_api_key: str = Field(default="")
    newsapi_api_key: str = Field(default="")

    # === TRADING SYMBOLS ===
    trading_symbols: List[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    )

    # === LOGGING ===
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/trading_bot.log")
    log_rotation_mb: int = Field(default=10)
    log_backup_count: int = Field(default=5)

    # === LOOP SETTINGS ===
    main_loop_interval_seconds: float = Field(default=5.0, ge=1.0, le=60.0)

    # === SUB-CONFIGS ===
    risk: RiskConfig = Field(default_factory=RiskConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    indicators: IndicatorConfig = Field(default_factory=IndicatorConfig)
    whale: WhaleConfig = Field(default_factory=WhaleConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)

    # Per-symbol configs (optional override)
    symbol_configs: Dict[str, SymbolConfig] = Field(default_factory=dict)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "env_nested_delimiter": "__",
    }

    @model_validator(mode='after')
    def validate_safety_settings(self) -> 'Settings':
        """Ensure safety settings are valid."""
        # Create default symbol configs if not provided
        for symbol in self.trading_symbols:
            if symbol not in self.symbol_configs:
                self.symbol_configs[symbol] = SymbolConfig(symbol=symbol)

        return self

    def is_live_trading_enabled(self) -> bool:
        """
        Check if live trading is enabled.
        ALL THREE conditions must be true for live trading.
        """
        return (
            self.env == Environment.MAINNET and
            self.allow_live_trading and
            not self.dry_run
        )

    def get_safety_status(self) -> Dict[str, any]:
        """Get current safety status for logging/display."""
        return {
            "env": self.env.value,
            "dry_run": self.dry_run,
            "allow_live_trading": self.allow_live_trading,
            "live_trading_active": self.is_live_trading_enabled(),
        }

    def get_symbol_config(self, symbol: str) -> SymbolConfig:
        """Get config for a specific symbol."""
        symbol = symbol.upper()
        if symbol not in self.symbol_configs:
            self.symbol_configs[symbol] = SymbolConfig(symbol=symbol)
        return self.symbol_configs[symbol]

    def print_safety_banner(self) -> str:
        """Generate safety banner for startup."""
        lines = [
            "=" * 60,
            "TRADING BOT SAFETY STATUS",
            "=" * 60,
            f"Environment:        {self.env.value.upper()}",
            f"Dry Run:            {self.dry_run}",
            f"Allow Live Trading: {self.allow_live_trading}",
            "-" * 60,
        ]

        if self.is_live_trading_enabled():
            lines.extend([
                "⚠️  WARNING: LIVE TRADING IS ENABLED! ⚠️",
                "⚠️  REAL ORDERS WILL BE PLACED! ⚠️",
                "⚠️  REAL MONEY IS AT RISK! ⚠️",
            ])
        else:
            lines.extend([
                "✅ SAFE MODE: No real orders will be placed",
                f"   Reason: {'DRY_RUN=true' if self.dry_run else ''}",
            ])
            if self.env == Environment.TESTNET:
                lines.append("   Reason: ENV=testnet")
            if not self.allow_live_trading:
                lines.append("   Reason: ALLOW_LIVE_TRADING=false")

        lines.append("=" * 60)
        return "\n".join(lines)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings
