"""
News sentiment analysis module (v3).
Uses CryptoPanic with 6-hour cache strategy for API efficiency.

v3: Implements 6-hour batch caching for all symbols to minimize API calls
    and provide stable sentiment signals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

from config import NewsProvider, SentimentConfig, get_settings
from utils.logger import get_logger


logger = get_logger(__name__)


# Lazy import for VADER to avoid startup cost
_vader_analyzer = None


def get_vader_analyzer():
    """Lazy load VADER analyzer."""
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader_analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("vaderSentiment not installed. Sentiment analysis will be limited.")
            _vader_analyzer = None
    return _vader_analyzer


@dataclass
class CachedSentiment:
    """
    Cached sentiment data for a single symbol (v3).

    6-hour cache strategy means this data is fetched once and reused
    for all trading decisions until the cache expires.
    """
    symbol: str  # e.g., "BTC", "ETH"
    sentiment: str  # "bullish", "bearish", "neutral"
    sentiment_score: float  # -1 to +1
    bullish_count: int
    bearish_count: int
    total_articles: int
    fetched_at: datetime
    expires_at: datetime


@dataclass
class SentimentSignal:
    """Sentiment analysis signal output (v3 with cache info)."""
    symbol: str
    timestamp: datetime

    # Core outputs
    sentiment_score: float  # -1 (negative) to +1 (positive)
    confidence: float  # 0 to 1

    # Global market sentiment (BTC as proxy)
    global_sentiment_score: float
    global_confidence: float

    # Metadata
    article_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0

    # v3 Cache info
    cache_expires_at: Optional[datetime] = None
    is_from_cache: bool = True

    # Validity
    is_valid: bool = True
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "symbol": self.symbol,
            "sentiment_score": round(self.sentiment_score, 4),
            "confidence": round(self.confidence, 4),
            "global_sentiment": round(self.global_sentiment_score, 4),
            "article_count": self.article_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "is_valid": self.is_valid,
            "cache_expires_at": self.cache_expires_at.isoformat() if self.cache_expires_at else None,
        }


class SentimentAnalyzer:
    """
    News sentiment analyzer with 6-hour cache strategy (v3).

    API Call Strategy:
    - Fetch sentiment data once every 6 hours for all 5 symbols in a single batch
    - Cache the bullish/bearish classification for each symbol
    - Use cached sentiment for all trading decisions until next refresh
    - This minimizes API calls and respects rate limits

    Benefits:
    - API Efficiency: Only 4 API calls per day instead of thousands
    - Rate Limit Safe: Well within CryptoPanic free tier limits
    - Stable Signals: Sentiment doesn't flip-flop on every article
    - Reduced Latency: Instant sentiment lookup from cache
    - Failure Resilient: Uses stale cache if API fails
    """

    # Supported symbols for CryptoPanic
    SUPPORTED_SYMBOLS = ["BTC", "ETH", "BNB", "SOL", "XRP"]

    # Map trading symbol to CryptoPanic currency code
    SYMBOL_MAP = {
        "BTCUSDT": "BTC",
        "ETHUSDT": "ETH",
        "BNBUSDT": "BNB",
        "SOLUSDT": "SOL",
        "XRPUSDT": "XRP",
    }

    def __init__(self, config: Optional[SentimentConfig] = None):
        settings = get_settings()
        self.config = config or settings.sentiment

        # API key
        self._api_key = settings.cryptopanic_api_key

        # v3: 6-hour cache
        self.cache: Dict[str, CachedSentiment] = {}
        self.last_fetch: Optional[datetime] = None

        # HTTP session
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "BinanceTradingBot/3.0",
        })

    def _should_refresh(self) -> bool:
        """
        Check if cache needs refresh (every 6 hours).

        Returns:
            True if refresh is needed
        """
        if self.last_fetch is None:
            return True

        elapsed = datetime.utcnow() - self.last_fetch
        cache_duration_seconds = self.config.cache_duration_hours * 3600
        return elapsed.total_seconds() >= cache_duration_seconds

    def _analyze_text_vader(self, text: str) -> float:
        """
        Analyze text sentiment using VADER.

        Returns:
            Score from -1 (negative) to +1 (positive)
        """
        analyzer = get_vader_analyzer()
        if analyzer is None:
            return 0.0

        try:
            scores = analyzer.polarity_scores(text)
            return scores['compound']  # -1 to +1
        except Exception as e:
            logger.debug(f"VADER analysis error: {e}")
            return 0.0

    def _fetch_from_cryptopanic(self) -> Dict[str, CachedSentiment]:
        """
        Fetch sentiment for all 5 coins in minimal API calls.

        CryptoPanic allows filtering by multiple currencies in one request.

        Returns:
            Dict mapping symbol to CachedSentiment
        """
        results: Dict[str, CachedSentiment] = {}

        if not self._api_key:
            logger.warning("No CryptoPanic API key configured")
            return self._create_default_sentiments()

        try:
            # Single API call with all currencies
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                "auth_token": self._api_key,
                "currencies": ",".join(self.SUPPORTED_SYMBOLS),  # "BTC,ETH,BNB,SOL,XRP"
                "filter": "hot",
                "public": "true",
                "kind": "news",
            }

            response = self._session.get(
                url,
                params=params,
                timeout=self.config.request_timeout_seconds
            )
            response.raise_for_status()
            data = response.json()

            # Process and categorize by symbol
            symbol_articles: Dict[str, List[Dict]] = {s: [] for s in self.SUPPORTED_SYMBOLS}

            for article in data.get("results", []):
                votes = article.get("votes", {})
                title = article.get("title", "")

                # Use VADER for more nuanced sentiment
                vader_score = self._analyze_text_vader(title)

                article_data = {
                    "positive": votes.get("positive", 0),
                    "negative": votes.get("negative", 0),
                    "liked": votes.get("liked", 0),
                    "disliked": votes.get("disliked", 0),
                    "vader_score": vader_score,
                    "title": title,
                }

                # Assign to relevant symbols
                for currency in article.get("currencies", []):
                    code = currency.get("code", "").upper()
                    if code in self.SUPPORTED_SYMBOLS:
                        symbol_articles[code].append(article_data)

            # Calculate sentiment for each symbol
            now = datetime.utcnow()
            expires = now + timedelta(hours=self.config.cache_duration_hours)

            for symbol in self.SUPPORTED_SYMBOLS:
                articles = symbol_articles[symbol]

                if not articles:
                    results[symbol] = CachedSentiment(
                        symbol=symbol,
                        sentiment="neutral",
                        sentiment_score=0.0,
                        bullish_count=0,
                        bearish_count=0,
                        total_articles=0,
                        fetched_at=now,
                        expires_at=expires,
                    )
                    continue

                # Calculate sentiment from votes
                bullish = sum(a["positive"] + a["liked"] for a in articles)
                bearish = sum(a["negative"] + a["disliked"] for a in articles)
                total_votes = bullish + bearish

                # Calculate VADER-based sentiment
                vader_scores = [a["vader_score"] for a in articles if a["vader_score"] != 0]
                vader_avg = sum(vader_scores) / len(vader_scores) if vader_scores else 0.0

                # Combine vote-based and VADER sentiment
                if total_votes > 0:
                    vote_score = (bullish - bearish) / total_votes  # -1 to +1
                else:
                    vote_score = 0.0

                # Weight: 40% votes, 60% VADER (VADER is more nuanced)
                final_score = vote_score * 0.4 + vader_avg * 0.6

                # Classify sentiment
                if final_score > self.config.bullish_threshold:
                    sentiment = "bullish"
                elif final_score < self.config.bearish_threshold:
                    sentiment = "bearish"
                else:
                    sentiment = "neutral"

                results[symbol] = CachedSentiment(
                    symbol=symbol,
                    sentiment=sentiment,
                    sentiment_score=final_score,
                    bullish_count=bullish,
                    bearish_count=bearish,
                    total_articles=len(articles),
                    fetched_at=now,
                    expires_at=expires,
                )

            return results

        except requests.exceptions.Timeout:
            logger.error("CryptoPanic API timeout")
            return self._create_default_sentiments()

        except requests.exceptions.RequestException as e:
            logger.error(f"CryptoPanic API error: {e}")
            return self._create_default_sentiments()

        except Exception as e:
            logger.error(f"Failed to fetch sentiment: {e}")
            return self._create_default_sentiments()

    def _create_default_sentiments(self) -> Dict[str, CachedSentiment]:
        """Create default neutral sentiments for all symbols."""
        now = datetime.utcnow()
        expires = now + timedelta(hours=1)  # Short expiry to retry soon

        return {
            symbol: CachedSentiment(
                symbol=symbol,
                sentiment="neutral",
                sentiment_score=0.0,
                bullish_count=0,
                bearish_count=0,
                total_articles=0,
                fetched_at=now,
                expires_at=expires,
            )
            for symbol in self.SUPPORTED_SYMBOLS
        }

    def refresh_if_needed(self) -> bool:
        """
        Refresh cache if 6 hours have passed.

        Call this at the start of main loop or on a schedule.

        Returns:
            True if refresh was performed
        """
        if not self.config.enabled:
            return False

        if not self._should_refresh():
            return False

        try:
            logger.info(
                f"Refreshing sentiment cache ({self.config.cache_duration_hours}-hour interval)..."
            )
            self.cache = self._fetch_from_cryptopanic()
            self.last_fetch = datetime.utcnow()

            # Log summary
            for symbol, data in self.cache.items():
                logger.info(
                    f"Sentiment cached: {symbol} -> {data.sentiment} "
                    f"(score={data.sentiment_score:.2f}, articles={data.total_articles})"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to refresh sentiment: {e}")
            # Keep using stale cache if refresh fails
            return False

    def get_sentiment(self, symbol: str) -> SentimentSignal:
        """
        Get cached sentiment for a symbol (v3).

        Returns neutral if no data available.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')

        Returns:
            SentimentSignal from cache
        """
        symbol = symbol.upper()
        timestamp = datetime.utcnow()

        if not self.config.enabled:
            return SentimentSignal(
                symbol=symbol,
                timestamp=timestamp,
                sentiment_score=0.0,
                confidence=0.0,
                global_sentiment_score=0.0,
                global_confidence=0.0,
                is_valid=False,
                error_message="Sentiment analysis disabled",
            )

        # Map trading symbol to CryptoPanic symbol
        crypto_symbol = self.SYMBOL_MAP.get(symbol, symbol.replace("USDT", ""))
        cached = self.cache.get(crypto_symbol)

        if cached is None:
            # No cached data - return neutral
            return SentimentSignal(
                symbol=symbol,
                timestamp=timestamp,
                sentiment_score=0.0,
                confidence=0.0,
                global_sentiment_score=self._get_global_sentiment(),
                global_confidence=0.5,
                is_valid=False,
                error_message="No cached sentiment data",
            )

        # Check if cache is expired (warning only, still use data)
        if datetime.utcnow() > cached.expires_at:
            logger.warning(f"Sentiment cache expired for {symbol}, using stale data")

        # Calculate confidence based on article count
        min_articles = self.config.min_articles_for_confidence
        max_articles = self.config.max_confidence_articles
        if cached.total_articles >= max_articles:
            confidence = 1.0
        elif cached.total_articles >= min_articles:
            confidence = cached.total_articles / max_articles
        else:
            confidence = cached.total_articles / max_articles * 0.5  # Reduced confidence

        # Apply minimum confidence threshold
        if confidence < self.config.min_confidence_threshold:
            confidence = 0.0

        # Classify positive/negative counts
        if cached.sentiment_score > 0:
            positive_count = max(cached.bullish_count, 1)
            negative_count = cached.bearish_count
        elif cached.sentiment_score < 0:
            positive_count = cached.bullish_count
            negative_count = max(cached.bearish_count, 1)
        else:
            positive_count = cached.bullish_count
            negative_count = cached.bearish_count

        return SentimentSignal(
            symbol=symbol,
            timestamp=cached.fetched_at,
            sentiment_score=cached.sentiment_score,
            confidence=confidence,
            global_sentiment_score=self._get_global_sentiment(),
            global_confidence=0.5,
            article_count=cached.total_articles,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=0,
            cache_expires_at=cached.expires_at,
            is_from_cache=True,
            is_valid=True,
        )

    def _get_global_sentiment(self) -> float:
        """Calculate overall market sentiment from BTC (market proxy)."""
        btc = self.cache.get("BTC")
        if btc:
            return btc.sentiment_score
        return 0.0

    def analyze(self, symbol: str) -> SentimentSignal:
        """
        Analyze sentiment for a symbol (v3 - wrapper for get_sentiment).

        This method maintains backward compatibility.
        Internally uses the 6-hour cache.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')

        Returns:
            SentimentSignal from cache
        """
        # Ensure cache is populated
        self.refresh_if_needed()
        return self.get_sentiment(symbol)

    def analyze_batch(self, symbols: List[str]) -> Dict[str, SentimentSignal]:
        """
        Analyze sentiment for multiple symbols (v3).

        With 6-hour cache, this is very efficient as data is already cached.

        Args:
            symbols: List of trading pairs

        Returns:
            Dict mapping symbol to SentimentSignal
        """
        # Ensure cache is populated
        self.refresh_if_needed()

        results = {}
        for symbol in symbols:
            results[symbol] = self.get_sentiment(symbol)
        return results

    def get_cache_status(self) -> Dict[str, Any]:
        """
        Get current cache status for monitoring.

        Returns:
            Dict with cache info
        """
        now = datetime.utcnow()
        return {
            "last_fetch": self.last_fetch.isoformat() if self.last_fetch else None,
            "cache_size": len(self.cache),
            "cached_symbols": list(self.cache.keys()),
            "is_stale": self._should_refresh(),
            "next_refresh_in_minutes": (
                (self.config.cache_duration_hours * 60) -
                (now - self.last_fetch).total_seconds() / 60
                if self.last_fetch else 0
            ),
            "sentiments": {
                symbol: {
                    "sentiment": data.sentiment,
                    "score": round(data.sentiment_score, 4),
                    "articles": data.total_articles,
                }
                for symbol, data in self.cache.items()
            }
        }

    def get_market_fear_greed(self) -> Dict[str, Any]:
        """
        Get overall market fear/greed indicator from sentiment.

        Returns:
            Dict with fear_greed_score (0-100), sentiment, and description
        """
        # Ensure cache is populated
        self.refresh_if_needed()

        # Use BTC as market proxy
        btc_sentiment = self._get_global_sentiment()

        # Convert sentiment score to fear/greed scale (0-100)
        # -1 = 0 (Extreme Fear), 0 = 50 (Neutral), +1 = 100 (Extreme Greed)
        fear_greed = int((btc_sentiment + 1) * 50)
        fear_greed = max(0, min(100, fear_greed))

        if fear_greed < 20:
            sentiment = "extreme_fear"
            description = "Extreme Fear - Potential buying opportunity"
        elif fear_greed < 40:
            sentiment = "fear"
            description = "Fear - Market is nervous"
        elif fear_greed < 60:
            sentiment = "neutral"
            description = "Neutral - Market is balanced"
        elif fear_greed < 80:
            sentiment = "greed"
            description = "Greed - Market is getting greedy"
        else:
            sentiment = "extreme_greed"
            description = "Extreme Greed - Potential correction ahead"

        btc_cached = self.cache.get("BTC")

        return {
            "fear_greed_score": fear_greed,
            "sentiment": sentiment,
            "description": description,
            "confidence": 0.5,
            "article_count": btc_cached.total_articles if btc_cached else 0,
            "timestamp": btc_cached.fetched_at.isoformat() if btc_cached else None,
            "cache_expires_at": btc_cached.expires_at.isoformat() if btc_cached else None,
        }
