"""
Secure structured logging with JSON format and rotating file handler.
NEVER logs secrets (API keys, passwords, etc.).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


# Patterns to detect and mask sensitive data
SENSITIVE_PATTERNS = [
    (re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9]+)', re.I), 'api_key=***REDACTED***'),
    (re.compile(r'api[_-]?secret["\']?\s*[:=]\s*["\']?([a-zA-Z0-9]+)', re.I), 'api_secret=***REDACTED***'),
    (re.compile(r'password["\']?\s*[:=]\s*["\']?([^\s"\']+)', re.I), 'password=***REDACTED***'),
    (re.compile(r'secret["\']?\s*[:=]\s*["\']?([a-zA-Z0-9]+)', re.I), 'secret=***REDACTED***'),
    (re.compile(r'token["\']?\s*[:=]\s*["\']?([a-zA-Z0-9._-]+)', re.I), 'token=***REDACTED***'),
    (re.compile(r'"signature"\s*:\s*"[a-fA-F0-9]+"', re.I), '"signature":"***REDACTED***"'),
]

# Keys to mask in dictionaries
SENSITIVE_KEYS = {
    'api_key', 'apikey', 'api-key',
    'api_secret', 'apisecret', 'api-secret',
    'secret', 'password', 'token',
    'signature', 'sign', 'auth',
    'binance_api_key', 'binance_api_secret',
    'cryptopanic_api_key', 'newsapi_api_key',
}


def mask_sensitive_string(text: str) -> str:
    """Mask sensitive data in a string."""
    result = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def mask_sensitive_dict(data: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
    """Recursively mask sensitive data in a dictionary."""
    if depth > 10:  # Prevent infinite recursion
        return {"__truncated__": "max depth reached"}

    masked = {}
    for key, value in data.items():
        key_lower = key.lower()
        if key_lower in SENSITIVE_KEYS:
            masked[key] = "***REDACTED***"
        elif isinstance(value, dict):
            masked[key] = mask_sensitive_dict(value, depth + 1)
        elif isinstance(value, list):
            masked[key] = [
                mask_sensitive_dict(v, depth + 1) if isinstance(v, dict)
                else "***REDACTED***" if isinstance(v, str) and any(k in v.lower() for k in SENSITIVE_KEYS)
                else v
                for v in value
            ]
        elif isinstance(value, str):
            masked[key] = mask_sensitive_string(value)
        else:
            masked[key] = value
    return masked


class SecureJsonFormatter(logging.Formatter):
    """JSON formatter that masks sensitive data."""

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": mask_sensitive_string(str(record.getMessage())),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Add extra fields (masked)
        if self.include_extra and hasattr(record, '__dict__'):
            extra = {}
            skip_keys = {
                'name', 'msg', 'args', 'created', 'filename',
                'funcName', 'levelname', 'levelno', 'lineno',
                'module', 'msecs', 'pathname', 'process',
                'processName', 'relativeCreated', 'stack_info',
                'exc_info', 'exc_text', 'thread', 'threadName',
                'message', 'taskName'
            }
            for key, value in record.__dict__.items():
                if key not in skip_keys and not key.startswith('_'):
                    if isinstance(value, dict):
                        extra[key] = mask_sensitive_dict(value)
                    elif isinstance(value, str):
                        extra[key] = mask_sensitive_string(value)
                    else:
                        extra[key] = value
            if extra:
                log_obj["extra"] = extra

        return json.dumps(log_obj, default=str)


class SecureConsoleFormatter(logging.Formatter):
    """Console formatter that masks sensitive data with colors."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        message = mask_sensitive_string(str(record.getMessage()))

        formatted = f"{color}{timestamp} [{record.levelname:8}]{self.RESET} {record.name}: {message}"

        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


class TradingLogger(logging.Logger):
    """Extended logger with trading-specific methods."""

    def trade_signal(
        self,
        symbol: str,
        action: str,
        scores: Dict[str, float],
        reason: str,
        **kwargs
    ) -> None:
        """Log a trade signal decision."""
        extra = {
            "event_type": "trade_signal",
            "symbol": symbol,
            "action": action,
            "scores": mask_sensitive_dict(scores),
            "reason": reason,
            **mask_sensitive_dict(kwargs)
        }
        self.info(f"Signal: {symbol} -> {action} | {reason}", extra=extra)

    def order_event(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_id: Optional[str] = None,
        status: str = "NEW",
        **kwargs
    ) -> None:
        """Log an order event."""
        extra = {
            "event_type": "order",
            "symbol": symbol,
            "order_type": order_type,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_id": order_id,
            "status": status,
            **mask_sensitive_dict(kwargs)
        }
        self.info(
            f"Order: {symbol} {side} {quantity} @ {price or 'MARKET'} [{status}]",
            extra=extra
        )

    def risk_event(
        self,
        event: str,
        symbol: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log a risk management event."""
        extra = {
            "event_type": "risk",
            "risk_event": event,
            "symbol": symbol,
            **mask_sensitive_dict(kwargs)
        }
        self.warning(f"Risk Event: {event}" + (f" ({symbol})" if symbol else ""), extra=extra)

    def whale_alert(
        self,
        symbol: str,
        direction: str,
        score: float,
        action: str,
        **kwargs
    ) -> None:
        """Log a whale activity alert."""
        extra = {
            "event_type": "whale_alert",
            "symbol": symbol,
            "direction": direction,
            "whale_score": score,
            "defensive_action": action,
            **mask_sensitive_dict(kwargs)
        }
        self.warning(f"Whale Alert: {symbol} {direction} (score={score:.2f}) -> {action}", extra=extra)


# Set custom logger class
logging.setLoggerClass(TradingLogger)

# Module-level logger cache
_loggers: Dict[str, TradingLogger] = {}


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation_mb: int = 10,
    backup_count: int = 5,
    json_console: bool = False,
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        rotation_mb: Max file size in MB before rotation
        backup_count: Number of backup files to keep
        json_console: If True, use JSON format for console output
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    if json_console:
        console_handler.setFormatter(SecureJsonFormatter())
    else:
        console_handler.setFormatter(SecureConsoleFormatter())
    root_logger.addHandler(console_handler)

    # File handler (JSON format)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=rotation_mb * 1024 * 1024,
            backupCount=backup_count,
        )
        file_handler.setLevel(logging.DEBUG)  # File gets all logs
        file_handler.setFormatter(SecureJsonFormatter())
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> TradingLogger:
    """
    Get or create a logger with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        TradingLogger instance
    """
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)  # type: ignore
    return _loggers[name]
