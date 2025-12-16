"""
News sentiment analysis module.
Uses CryptoPanic, NewsAPI, or GDELT for near real-time sentiment.
Default analyzer is VADER; transformer-based is optional.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from functools import lru_cache

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
class NewsArticle:
    """News article with metadata."""
    title: str
    source: str
    published_at: datetime
    url: str
    sentiment_score: float = 0.0  # -1 to +1
    relevance: float = 1.0  # 0 to 1
    currencies: List[str] = field(default_factory=list)


@dataclass
class SentimentSignal:
    """Sentiment analysis signal output."""
    symbol: str
    timestamp: datetime

    # Core outputs
    sentiment_score: float  # -1 (negative) to +1 (positive)
    confidence: float  # 0 to 1

    # Global market sentiment
    global_sentiment_score: float
    global_confidence: float

    # Metadata
    article_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0

    # Source tracking
    source_breakdown: Dict[str, float] = field(default_factory=dict)

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
        }


class RateLimiter:
    """Simple rate limiter."""

    def __init__(self, requests_per_minute: int):
        self.interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0

    def wait(self) -> None:
        """Wait if necessary to respect rate limit."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_request_time = time.time()


class SentimentCache:
    """Simple TTL cache for sentiment data."""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._cache: Dict[str, tuple] = {}  # key -> (value, timestamp)

    def get(self, key: str) -> Optional[Any]:
        """Get value if not expired."""
        if key in self._cache:
            value, ts = self._cache[key]
            if time.time() - ts < self.ttl:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value with current timestamp."""
        self._cache[key] = (value, time.time())


class SentimentAnalyzer:
    """
    News sentiment analyzer with multiple provider support.
    """

    # Symbol to search term mapping
    SYMBOL_KEYWORDS = {
        "BTCUSDT": ["bitcoin", "btc"],
        "ETHUSDT": ["ethereum", "eth"],
        "BNBUSDT": ["binance coin", "bnb"],
        "SOLUSDT": ["solana", "sol"],
        "XRPUSDT": ["ripple", "xrp"],
    }

    def __init__(self, config: Optional[SentimentConfig] = None):
        settings = get_settings()
        self.config = config or settings.sentiment

        # API keys
        self._cryptopanic_key = settings.cryptopanic_api_key
        self._newsapi_key = settings.newsapi_api_key

        # Rate limiting
        self._rate_limiter = RateLimiter(self.config.rate_limit_requests_per_minute)

        # Caching
        self._cache = SentimentCache(self.config.cache_ttl_seconds)

        # HTTP session
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "BinanceTradingBot/1.0",
        })

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

    def _fetch_cryptopanic(self, symbol: str) -> List[NewsArticle]:
        """
        Fetch news from CryptoPanic API.
        """
        if not self._cryptopanic_key:
            return []

        keywords = self.SYMBOL_KEYWORDS.get(symbol, [symbol.replace("USDT", "").lower()])
        currency = keywords[0] if keywords else "BTC"

        cache_key = f"cryptopanic_{symbol}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            self._rate_limiter.wait()

            url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                "auth_token": self._cryptopanic_key,
                "currencies": currency.upper(),
                "kind": "news",
                "public": "true",
            }

            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get("results", [])[:self.config.max_articles_per_request]:
                # Parse published time
                pub_str = item.get("published_at", "")
                try:
                    published = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                except:
                    published = datetime.utcnow()

                title = item.get("title", "")
                sentiment = self._analyze_text_vader(title)

                articles.append(NewsArticle(
                    title=title,
                    source=item.get("source", {}).get("title", "CryptoPanic"),
                    published_at=published,
                    url=item.get("url", ""),
                    sentiment_score=sentiment,
                    currencies=[c.get("code", "") for c in item.get("currencies", [])],
                ))

            self._cache.set(cache_key, articles)
            return articles

        except Exception as e:
            logger.warning(f"CryptoPanic fetch error: {e}")
            return []

    def _fetch_newsapi(self, symbol: str) -> List[NewsArticle]:
        """
        Fetch news from NewsAPI.
        """
        if not self._newsapi_key:
            return []

        keywords = self.SYMBOL_KEYWORDS.get(symbol, [symbol.replace("USDT", "").lower()])
        query = " OR ".join(keywords)

        cache_key = f"newsapi_{symbol}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            self._rate_limiter.wait()

            url = "https://newsapi.org/v2/everything"
            params = {
                "apiKey": self._newsapi_key,
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(self.config.max_articles_per_request, 100),
            }

            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get("articles", []):
                # Parse published time
                pub_str = item.get("publishedAt", "")
                try:
                    published = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                except:
                    published = datetime.utcnow()

                # Combine title and description for sentiment
                text = f"{item.get('title', '')} {item.get('description', '')}"
                sentiment = self._analyze_text_vader(text)

                articles.append(NewsArticle(
                    title=item.get("title", ""),
                    source=item.get("source", {}).get("name", "NewsAPI"),
                    published_at=published,
                    url=item.get("url", ""),
                    sentiment_score=sentiment,
                ))

            self._cache.set(cache_key, articles)
            return articles

        except Exception as e:
            logger.warning(f"NewsAPI fetch error: {e}")
            return []

    def _calculate_time_weighted_sentiment(
        self,
        articles: List[NewsArticle],
        decay_hours: float = 24.0,
    ) -> tuple:
        """
        Calculate time-weighted sentiment score.

        More recent articles have higher weight.

        Returns:
            Tuple of (score, confidence, positive_count, negative_count, neutral_count)
        """
        if not articles:
            return 0.0, 0.0, 0, 0, 0

        now = datetime.utcnow()
        weighted_sum = 0.0
        weight_total = 0.0
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for article in articles:
            # Time decay weight
            try:
                age_hours = (now - article.published_at.replace(tzinfo=None)).total_seconds() / 3600
            except:
                age_hours = decay_hours / 2  # Default to middle weight

            time_weight = max(0.1, 1.0 - (age_hours / decay_hours))

            # Relevance weight
            relevance_weight = article.relevance

            weight = time_weight * relevance_weight
            weighted_sum += article.sentiment_score * weight
            weight_total += weight

            # Count sentiment categories
            if article.sentiment_score > 0.1:
                positive_count += 1
            elif article.sentiment_score < -0.1:
                negative_count += 1
            else:
                neutral_count += 1

        if weight_total == 0:
            return 0.0, 0.0, 0, 0, 0

        score = weighted_sum / weight_total

        # Confidence based on article count and sentiment consistency
        count_factor = min(len(articles) / 10, 1.0)
        consistency = 1.0 - min(abs(positive_count - negative_count) / (len(articles) + 1), 1.0)
        confidence = count_factor * 0.7 + (1 - consistency) * 0.3

        return score, confidence, positive_count, negative_count, neutral_count

    def analyze(self, symbol: str) -> SentimentSignal:
        """
        Analyze sentiment for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')

        Returns:
            SentimentSignal with sentiment scores and confidence
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

        all_articles = []
        source_breakdown = {}

        # Fetch from configured provider
        if self.config.provider == NewsProvider.CRYPTOPANIC:
            articles = self._fetch_cryptopanic(symbol)
            if articles:
                all_articles.extend(articles)
                source_breakdown["cryptopanic"] = len(articles)

        elif self.config.provider == NewsProvider.NEWSAPI:
            articles = self._fetch_newsapi(symbol)
            if articles:
                all_articles.extend(articles)
                source_breakdown["newsapi"] = len(articles)

        # Calculate symbol-specific sentiment
        score, confidence, pos, neg, neu = self._calculate_time_weighted_sentiment(all_articles)

        # For global sentiment, fetch BTC as proxy (if not already BTC)
        global_score = score
        global_confidence = confidence
        if symbol != "BTCUSDT":
            # Try to get cached BTC sentiment
            btc_signal = self._cache.get("global_sentiment")
            if btc_signal:
                global_score, global_confidence = btc_signal
            else:
                # Quick BTC sentiment check
                btc_articles = self._fetch_cryptopanic("BTCUSDT") if self._cryptopanic_key else []
                if btc_articles:
                    global_score, global_confidence, _, _, _ = self._calculate_time_weighted_sentiment(btc_articles)
                    self._cache.set("global_sentiment", (global_score, global_confidence))

        # Apply minimum confidence threshold
        if confidence < self.config.min_confidence_threshold:
            confidence = 0.0
            score = 0.0  # Don't use low-confidence signals

        return SentimentSignal(
            symbol=symbol,
            timestamp=timestamp,
            sentiment_score=score,
            confidence=confidence,
            global_sentiment_score=global_score,
            global_confidence=global_confidence,
            article_count=len(all_articles),
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            source_breakdown=source_breakdown,
            is_valid=True,
        )

    def analyze_batch(self, symbols: List[str]) -> Dict[str, SentimentSignal]:
        """
        Analyze sentiment for multiple symbols.

        Args:
            symbols: List of trading pairs

        Returns:
            Dict mapping symbol to SentimentSignal
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.analyze(symbol)
        return results

    def get_market_fear_greed(self) -> Dict[str, Any]:
        """
        Get overall market fear/greed indicator from sentiment.

        Returns:
            Dict with fear_greed_score (0-100), sentiment, and description
        """
        # Analyze BTC as market proxy
        signal = self.analyze("BTCUSDT")

        # Convert sentiment score to fear/greed scale (0-100)
        # -1 = 0 (Extreme Fear), 0 = 50 (Neutral), +1 = 100 (Extreme Greed)
        fear_greed = int((signal.sentiment_score + 1) * 50)
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

        return {
            "fear_greed_score": fear_greed,
            "sentiment": sentiment,
            "description": description,
            "confidence": signal.confidence,
            "article_count": signal.article_count,
            "timestamp": signal.timestamp.isoformat(),
        }
