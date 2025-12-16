"""
Market data manager for candles, orderbook, and trading conditions.
Implements data integrity checks required for safe trading.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import pandas as pd
import numpy as np

from config import DataConfig, get_settings
from exchange.binance_client import BinanceClient
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class MarketConditions:
    """Current market conditions and data integrity status."""
    symbol: str
    timestamp: datetime

    # Price data
    last_price: float = 0.0
    bid_price: float = 0.0
    ask_price: float = 0.0
    spread_pct: float = 0.0

    # Volume/Liquidity
    bid_liquidity_usd: float = 0.0
    ask_liquidity_usd: float = 0.0
    total_liquidity_usd: float = 0.0

    # Data freshness
    candle_age_seconds: float = 0.0
    time_sync_drift_ms: int = 0

    # Validity flags
    is_candle_fresh: bool = False
    is_time_synced: bool = False
    is_spread_acceptable: bool = False
    is_liquidity_sufficient: bool = False
    has_valid_filters: bool = False

    @property
    def is_tradeable(self) -> bool:
        """Check if all conditions are met for trading."""
        return (
            self.is_candle_fresh and
            self.is_time_synced and
            self.is_spread_acceptable and
            self.is_liquidity_sufficient and
            self.has_valid_filters
        )

    def get_issues(self) -> List[str]:
        """Get list of issues preventing trading."""
        issues = []
        if not self.is_candle_fresh:
            issues.append(f"Candle data stale ({self.candle_age_seconds:.0f}s)")
        if not self.is_time_synced:
            issues.append(f"Time sync drift too high ({self.time_sync_drift_ms}ms)")
        if not self.is_spread_acceptable:
            issues.append(f"Spread too wide ({self.spread_pct:.4f}%)")
        if not self.is_liquidity_sufficient:
            issues.append(f"Liquidity too low (${self.total_liquidity_usd:.0f})")
        if not self.has_valid_filters:
            issues.append("Missing symbol filters/precision")
        return issues


@dataclass
class CandleData:
    """OHLCV candle data container."""
    symbol: str
    interval: str
    df: pd.DataFrame
    last_update: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_empty(self) -> bool:
        return self.df is None or len(self.df) == 0

    @property
    def last_close(self) -> float:
        if self.is_empty:
            return 0.0
        return float(self.df['close'].iloc[-1])

    @property
    def last_candle_time(self) -> Optional[datetime]:
        if self.is_empty:
            return None
        return pd.to_datetime(self.df['close_time'].iloc[-1], unit='ms')


class MarketDataManager:
    """
    Manages market data fetching, caching, and integrity checks.
    """

    def __init__(self, client: BinanceClient, config: Optional[DataConfig] = None):
        self.client = client
        self.config = config or get_settings().data

        # Candle cache: {symbol: {interval: CandleData}}
        self._candles: Dict[str, Dict[str, CandleData]] = {}

        # Order book cache: {symbol: {'bids': [...], 'asks': [...], 'timestamp': datetime}}
        self._orderbook: Dict[str, Dict] = {}

        # Recent trades cache for whale detection: {symbol: deque of trades}
        self._trades: Dict[str, deque] = {}

        # Market conditions cache
        self._conditions: Dict[str, MarketConditions] = {}

    def fetch_candles(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 500,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch and cache candlestick data.

        Args:
            symbol: Trading pair
            interval: Candle interval (1m, 5m, 15m, etc.)
            limit: Number of candles to fetch
            force: Force refresh even if cached

        Returns:
            DataFrame with OHLCV data
        """
        symbol = symbol.upper()

        # Check cache
        if not force and symbol in self._candles and interval in self._candles[symbol]:
            cached = self._candles[symbol][interval]
            age = (datetime.utcnow() - cached.last_update).total_seconds()

            # Refresh if older than interval duration
            interval_seconds = self._interval_to_seconds(interval)
            if age < interval_seconds * 0.5:  # Refresh at half interval
                return cached.df

        try:
            klines = self.client.get_klines(symbol, interval, limit)

            df = pd.DataFrame(klines)
            df = df.astype({
                'open': float, 'high': float, 'low': float, 'close': float,
                'volume': float, 'quote_volume': float,
            })

            # Store in cache
            if symbol not in self._candles:
                self._candles[symbol] = {}

            self._candles[symbol][interval] = CandleData(
                symbol=symbol,
                interval=interval,
                df=df,
                last_update=datetime.utcnow(),
            )

            logger.debug(
                f"Fetched candles",
                extra={"symbol": symbol, "interval": interval, "count": len(df)}
            )

            return df

        except Exception as e:
            logger.error(f"Failed to fetch candles: {e}", extra={"symbol": symbol, "interval": interval})
            raise

    def fetch_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """
        Fetch and cache order book.

        Returns:
            Dict with 'bids', 'asks', 'bid_liquidity', 'ask_liquidity', 'spread_pct'
        """
        symbol = symbol.upper()

        try:
            data = self.client.get_orderbook(symbol, limit=depth)

            bids = data['bids']
            asks = data['asks']

            # Calculate liquidity (sum of value at each level)
            bid_liquidity = sum(price * qty for price, qty in bids)
            ask_liquidity = sum(price * qty for price, qty in asks)

            # Calculate spread
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
            spread_pct = ((best_ask - best_bid) / mid_price * 100) if mid_price > 0 else float('inf')

            result = {
                'bids': bids,
                'asks': asks,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'mid_price': mid_price,
                'spread_pct': spread_pct,
                'bid_liquidity': bid_liquidity,
                'ask_liquidity': ask_liquidity,
                'total_liquidity': bid_liquidity + ask_liquidity,
                'timestamp': datetime.utcnow(),
            }

            self._orderbook[symbol] = result
            return result

        except Exception as e:
            logger.error(f"Failed to fetch orderbook: {e}", extra={"symbol": symbol})
            raise

    def fetch_recent_trades(self, symbol: str, limit: int = 1000) -> List[Dict]:
        """
        Fetch and cache recent aggregate trades.
        """
        symbol = symbol.upper()

        try:
            trades = self.client.get_recent_trades(symbol, limit=limit)

            if symbol not in self._trades:
                self._trades[symbol] = deque(maxlen=limit * 2)

            # Add new trades (avoiding duplicates based on ID)
            existing_ids = {t['id'] for t in self._trades[symbol]}
            for trade in trades:
                if trade['id'] not in existing_ids:
                    self._trades[symbol].append(trade)

            return list(self._trades[symbol])

        except Exception as e:
            logger.error(f"Failed to fetch trades: {e}", extra={"symbol": symbol})
            raise

    def check_conditions(self, symbol: str) -> MarketConditions:
        """
        Check and return current market conditions for a symbol.
        This is the main data integrity check.
        """
        symbol = symbol.upper()
        settings = get_settings()

        conditions = MarketConditions(
            symbol=symbol,
            timestamp=datetime.utcnow(),
        )

        try:
            # 1. Check symbol info/filters
            symbol_info = self.client.get_symbol_info(symbol)
            conditions.has_valid_filters = symbol_info is not None and symbol_info.status == "TRADING"

            # 2. Fetch orderbook and check spread/liquidity
            orderbook = self.fetch_orderbook(symbol, self.config.orderbook_depth_levels if hasattr(self.config, 'orderbook_depth_levels') else 20)
            conditions.bid_price = orderbook['best_bid']
            conditions.ask_price = orderbook['best_ask']
            conditions.last_price = orderbook['mid_price']
            conditions.spread_pct = orderbook['spread_pct']
            conditions.bid_liquidity_usd = orderbook['bid_liquidity']
            conditions.ask_liquidity_usd = orderbook['ask_liquidity']
            conditions.total_liquidity_usd = orderbook['total_liquidity']

            conditions.is_spread_acceptable = conditions.spread_pct <= self.config.max_spread_pct
            conditions.is_liquidity_sufficient = conditions.total_liquidity_usd >= self.config.min_liquidity_usd

            # 3. Check candle freshness
            primary_interval = self.config.primary_timeframe
            candles = self.fetch_candles(symbol, primary_interval)

            if len(candles) > 0:
                last_candle_time = pd.to_datetime(candles['close_time'].iloc[-1], unit='ms')
                candle_age = (datetime.utcnow() - last_candle_time).total_seconds()
                conditions.candle_age_seconds = candle_age
                conditions.is_candle_fresh = candle_age <= self.config.max_candle_staleness_seconds

            # 4. Check time sync
            conditions.time_sync_drift_ms = self.client.get_time_sync_drift()
            conditions.is_time_synced = conditions.time_sync_drift_ms <= self.config.max_time_sync_drift_ms

            self._conditions[symbol] = conditions

        except Exception as e:
            logger.error(f"Error checking conditions: {e}", extra={"symbol": symbol})
            # Return conditions with all flags false on error

        return conditions

    def get_cached_candles(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Get cached candles without fetching."""
        symbol = symbol.upper()
        if symbol in self._candles and interval in self._candles[symbol]:
            return self._candles[symbol][interval].df
        return None

    def get_cached_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get cached orderbook without fetching."""
        return self._orderbook.get(symbol.upper())

    def get_cached_trades(self, symbol: str) -> List[Dict]:
        """Get cached trades without fetching."""
        symbol = symbol.upper()
        if symbol in self._trades:
            return list(self._trades[symbol])
        return []

    def get_multi_timeframe_candles(
        self,
        symbol: str,
        force: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch candles for all configured timeframes.

        Returns:
            Dict mapping interval to DataFrame
        """
        symbol = symbol.upper()
        result = {}

        # Primary timeframe
        result[self.config.primary_timeframe] = self.fetch_candles(
            symbol, self.config.primary_timeframe, self.config.candle_limit, force
        )

        # Confirmation timeframes
        for interval in self.config.confirmation_timeframes:
            result[interval] = self.fetch_candles(
                symbol, interval, self.config.candle_limit, force
            )

        return result

    def _interval_to_seconds(self, interval: str) -> int:
        """Convert interval string to seconds."""
        multipliers = {
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800,
        }
        unit = interval[-1].lower()
        value = int(interval[:-1])
        return value * multipliers.get(unit, 60)

    def calculate_trade_metrics(self, symbol: str) -> Dict[str, float]:
        """
        Calculate trade metrics for whale detection.

        Returns:
            Dict with trade_volume, avg_trade_size, large_trade_threshold, etc.
        """
        trades = self.get_cached_trades(symbol)
        if not trades:
            return {}

        quantities = [t['quantity'] for t in trades]
        prices = [t['price'] for t in trades]
        notionals = [q * p for q, p in zip(quantities, prices)]

        return {
            'trade_count': len(trades),
            'total_volume': sum(quantities),
            'total_notional': sum(notionals),
            'avg_trade_size': np.mean(notionals),
            'median_trade_size': np.median(notionals),
            'std_trade_size': np.std(notionals),
            'p95_trade_size': np.percentile(notionals, 95),
            'p99_trade_size': np.percentile(notionals, 99),
            'max_trade_size': max(notionals),
        }

    def calculate_orderbook_imbalance(self, symbol: str, depth: int = 10) -> float:
        """
        Calculate order book imbalance.

        Returns:
            Value between -1 (sell pressure) and +1 (buy pressure)
        """
        orderbook = self.get_cached_orderbook(symbol)
        if not orderbook:
            return 0.0

        bids = orderbook['bids'][:depth]
        asks = orderbook['asks'][:depth]

        bid_volume = sum(q for _, q in bids)
        ask_volume = sum(q for _, q in asks)

        total = bid_volume + ask_volume
        if total == 0:
            return 0.0

        # Positive = more bids (buying pressure), Negative = more asks (selling pressure)
        return (bid_volume - ask_volume) / total

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate."""
        try:
            data = self.client.get_funding_rate(symbol)
            return data.get('funding_rate')
        except Exception as e:
            logger.debug(f"Could not get funding rate: {e}")
            return None

    def get_open_interest(self, symbol: str) -> Optional[float]:
        """Get current open interest."""
        try:
            return self.client.get_open_interest(symbol)
        except Exception as e:
            logger.debug(f"Could not get open interest: {e}")
            return None
