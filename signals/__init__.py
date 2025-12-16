"""Signals module."""
from signals.indicators import TechnicalIndicators, TechnicalSignal
from signals.whales import WhaleDetector, WhaleSignal
from signals.sentiment import SentimentAnalyzer, SentimentSignal

__all__ = [
    "TechnicalIndicators", "TechnicalSignal",
    "WhaleDetector", "WhaleSignal",
    "SentimentAnalyzer", "SentimentSignal",
]
