"""
Binance USDT-M Futures client with testnet/mainnet support.
Handles time sync, symbol precision, filters, leverage, and margin type.
Uses binance-futures-connector library.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

from config import Environment, MarginType, get_settings
from utils.logger import get_logger


logger = get_logger(__name__)


# Binance endpoints
ENDPOINTS = {
    Environment.TESTNET: {
        "base_url": "https://testnet.binancefuture.com",
        "ws_url": "wss://stream.binancefuture.com",
    },
    Environment.MAINNET: {
        "base_url": "https://fapi.binance.com",
        "ws_url": "wss://fstream.binance.com",
    },
}


@dataclass
class SymbolInfo:
    """Symbol trading information and filters."""
    symbol: str
    base_asset: str
    quote_asset: str
    price_precision: int
    quantity_precision: int
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    max_qty: Decimal
    min_notional: Decimal
    status: str

    def round_price(self, price: float) -> Decimal:
        """Round price to valid tick size."""
        d_price = Decimal(str(price))
        return (d_price / self.tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * self.tick_size

    def round_quantity(self, quantity: float) -> Decimal:
        """Round quantity to valid step size."""
        d_qty = Decimal(str(quantity))
        return (d_qty / self.step_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * self.step_size

    def validate_order(self, quantity: float, price: float) -> Tuple[bool, str]:
        """Validate order parameters against symbol filters."""
        d_qty = self.round_quantity(quantity)
        d_price = self.round_price(price)
        notional = d_qty * d_price

        if d_qty < self.min_qty:
            return False, f"Quantity {d_qty} below minimum {self.min_qty}"
        if d_qty > self.max_qty:
            return False, f"Quantity {d_qty} above maximum {self.max_qty}"
        if notional < self.min_notional:
            return False, f"Notional {notional} below minimum {self.min_notional}"

        return True, "OK"


@dataclass
class AccountBalance:
    """Account balance information."""
    asset: str
    wallet_balance: float
    unrealized_pnl: float
    margin_balance: float
    available_balance: float
    cross_wallet_balance: float
    cross_unrealized_pnl: float


@dataclass
class Position:
    """Open position information."""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    quantity: float
    entry_price: float
    unrealized_pnl: float
    leverage: int
    margin_type: str
    liquidation_price: float
    mark_price: float
    notional: float
    isolated_margin: float = 0.0


@dataclass
class OrderResult:
    """Order execution result."""
    order_id: int
    client_order_id: str
    symbol: str
    status: str
    type: str
    side: str
    price: float
    avg_price: float
    orig_qty: float
    executed_qty: float
    cum_quote: float  # Total quote quantity (cost)
    reduce_only: bool
    time_in_force: str
    update_time: int

    @property
    def is_filled(self) -> bool:
        return self.status == "FILLED"

    @property
    def is_partially_filled(self) -> bool:
        return self.status == "PARTIALLY_FILLED"

    @property
    def fill_price(self) -> float:
        """Get actual fill price (avg_price if filled, else price)."""
        if self.executed_qty > 0 and self.avg_price > 0:
            return self.avg_price
        return self.price


class BinanceClientError(Exception):
    """Binance client error."""
    def __init__(self, message: str, code: Optional[int] = None, response: Optional[Dict] = None):
        self.code = code
        self.response = response
        super().__init__(message)


class BinanceClient:
    """
    Binance USDT-M Futures client.
    Supports testnet and mainnet with safety checks.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        environment: Environment = Environment.TESTNET,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.environment = environment

        endpoints = ENDPOINTS[environment]
        self.base_url = endpoints["base_url"]
        self.ws_url = endpoints["ws_url"]

        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": api_key,
            "Content-Type": "application/json",
        })

        # Cache
        self._symbol_info: Dict[str, SymbolInfo] = {}
        self._exchange_info_time: Optional[datetime] = None
        self._time_offset_ms: int = 0
        self._last_time_sync: Optional[datetime] = None

        logger.info(
            f"BinanceClient initialized",
            extra={
                "environment": environment.value,
                "base_url": self.base_url,
            }
        )

    def _sign(self, params: Dict[str, Any]) -> str:
        """Sign request parameters."""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _get_timestamp(self) -> int:
        """Get server-adjusted timestamp."""
        return int(time.time() * 1000) + self._time_offset_ms

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        """Make API request."""
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        if signed:
            params["timestamp"] = self._get_timestamp()
            params["signature"] = self._sign(params)

        try:
            if method == "GET":
                response = self.session.get(url, params=params, timeout=10)
            elif method == "POST":
                response = self.session.post(url, params=params, timeout=10)
            elif method == "DELETE":
                response = self.session.delete(url, params=params, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            error_data = {}
            try:
                error_data = e.response.json()
            except:
                pass
            code = error_data.get("code")
            msg = error_data.get("msg", str(e))
            logger.error(f"Binance API error: {msg}", extra={"code": code, "endpoint": endpoint})
            raise BinanceClientError(msg, code=code, response=error_data)

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}", extra={"endpoint": endpoint})
            raise BinanceClientError(str(e))

    def sync_time(self) -> int:
        """
        Synchronize time with Binance server.
        Returns time offset in milliseconds.
        """
        try:
            local_time = int(time.time() * 1000)
            server_time = self._request("GET", "/fapi/v1/time")["serverTime"]
            self._time_offset_ms = server_time - local_time
            self._last_time_sync = datetime.utcnow()

            logger.info(
                f"Time synchronized",
                extra={"offset_ms": self._time_offset_ms}
            )
            return self._time_offset_ms

        except Exception as e:
            logger.error(f"Time sync failed: {e}")
            raise

    def get_time_sync_drift(self) -> int:
        """Get current time synchronization drift in ms."""
        return abs(self._time_offset_ms)

    def is_time_sync_required(self, max_age_minutes: int = 5) -> bool:
        """Check if time sync is required."""
        if self._last_time_sync is None:
            return True
        age = datetime.utcnow() - self._last_time_sync
        return age > timedelta(minutes=max_age_minutes)

    def load_exchange_info(self, force: bool = False) -> None:
        """
        Load exchange information including symbol filters.
        Caches for 1 hour unless forced.
        """
        if not force and self._exchange_info_time:
            age = datetime.utcnow() - self._exchange_info_time
            if age < timedelta(hours=1):
                return

        try:
            info = self._request("GET", "/fapi/v1/exchangeInfo")

            for symbol_data in info.get("symbols", []):
                symbol = symbol_data["symbol"]
                filters = {f["filterType"]: f for f in symbol_data.get("filters", [])}

                price_filter = filters.get("PRICE_FILTER", {})
                lot_filter = filters.get("LOT_SIZE", {})
                min_notional = filters.get("MIN_NOTIONAL", {})

                self._symbol_info[symbol] = SymbolInfo(
                    symbol=symbol,
                    base_asset=symbol_data.get("baseAsset", ""),
                    quote_asset=symbol_data.get("quoteAsset", ""),
                    price_precision=symbol_data.get("pricePrecision", 8),
                    quantity_precision=symbol_data.get("quantityPrecision", 8),
                    tick_size=Decimal(price_filter.get("tickSize", "0.00000001")),
                    step_size=Decimal(lot_filter.get("stepSize", "0.00000001")),
                    min_qty=Decimal(lot_filter.get("minQty", "0")),
                    max_qty=Decimal(lot_filter.get("maxQty", "999999")),
                    min_notional=Decimal(min_notional.get("notional", "5")),
                    status=symbol_data.get("status", "UNKNOWN"),
                )

            self._exchange_info_time = datetime.utcnow()
            logger.info(f"Exchange info loaded", extra={"symbols_count": len(self._symbol_info)})

        except Exception as e:
            logger.error(f"Failed to load exchange info: {e}")
            raise

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get symbol information."""
        if not self._symbol_info:
            self.load_exchange_info()
        return self._symbol_info.get(symbol.upper())

    def get_account_balance(self) -> Dict[str, AccountBalance]:
        """Get account balance."""
        data = self._request("GET", "/fapi/v2/balance", signed=True)
        balances = {}

        for item in data:
            balances[item["asset"]] = AccountBalance(
                asset=item["asset"],
                wallet_balance=float(item["balance"]),
                unrealized_pnl=float(item.get("crossUnPnl", 0)),
                margin_balance=float(item.get("marginBalance", item["balance"])),
                available_balance=float(item.get("availableBalance", item["balance"])),
                cross_wallet_balance=float(item.get("crossWalletBalance", 0)),
                cross_unrealized_pnl=float(item.get("crossUnPnl", 0)),
            )

        return balances

    def get_usdt_balance(self) -> float:
        """Get USDT wallet balance."""
        balances = self.get_account_balance()
        if "USDT" in balances:
            return balances["USDT"].wallet_balance
        return 0.0

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get current positions."""
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()

        data = self._request("GET", "/fapi/v2/positionRisk", params=params, signed=True)
        positions = []

        for item in data:
            qty = float(item["positionAmt"])
            if qty == 0:
                continue

            positions.append(Position(
                symbol=item["symbol"],
                side="LONG" if qty > 0 else "SHORT",
                quantity=abs(qty),
                entry_price=float(item["entryPrice"]),
                unrealized_pnl=float(item["unRealizedProfit"]),
                leverage=int(item["leverage"]),
                margin_type=item["marginType"],
                liquidation_price=float(item["liquidationPrice"]),
                mark_price=float(item["markPrice"]),
                notional=abs(float(item["notional"])),
                isolated_margin=float(item.get("isolatedMargin", 0)),
            ))

        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        positions = self.get_positions(symbol)
        return positions[0] if positions else None

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        try:
            self._request(
                "POST",
                "/fapi/v1/leverage",
                params={"symbol": symbol.upper(), "leverage": leverage},
                signed=True,
            )
            logger.info(f"Leverage set", extra={"symbol": symbol, "leverage": leverage})
            return True
        except BinanceClientError as e:
            # Code -4028: Leverage not changed (already set)
            if e.code == -4028:
                return True
            logger.error(f"Failed to set leverage: {e}")
            return False

    def set_margin_type(self, symbol: str, margin_type: MarginType) -> bool:
        """Set margin type for a symbol."""
        try:
            self._request(
                "POST",
                "/fapi/v1/marginType",
                params={"symbol": symbol.upper(), "marginType": margin_type.value},
                signed=True,
            )
            logger.info(f"Margin type set", extra={"symbol": symbol, "margin_type": margin_type.value})
            return True
        except BinanceClientError as e:
            # Code -4046: No need to change margin type
            if e.code == -4046:
                return True
            logger.error(f"Failed to set margin type: {e}")
            return False

    def get_ticker_price(self, symbol: str) -> float:
        """Get current ticker price."""
        data = self._request("GET", "/fapi/v1/ticker/price", params={"symbol": symbol.upper()})
        return float(data["price"])

    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book."""
        data = self._request(
            "GET",
            "/fapi/v1/depth",
            params={"symbol": symbol.upper(), "limit": limit}
        )
        return {
            "bids": [(float(p), float(q)) for p, q in data.get("bids", [])],
            "asks": [(float(p), float(q)) for p, q in data.get("asks", [])],
            "last_update_id": data.get("lastUpdateId"),
        }

    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """Get recent aggregate trades."""
        data = self._request(
            "GET",
            "/fapi/v1/aggTrades",
            params={"symbol": symbol.upper(), "limit": limit}
        )
        return [
            {
                "id": t["a"],
                "price": float(t["p"]),
                "quantity": float(t["q"]),
                "timestamp": t["T"],
                "is_buyer_maker": t["m"],
            }
            for t in data
        ]

    def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict]:
        """
        Get klines/candlesticks.

        Returns list of dicts with:
            open_time, open, high, low, close, volume, close_time,
            quote_volume, trades, taker_buy_volume, taker_buy_quote_volume
        """
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = self._request("GET", "/fapi/v1/klines", params=params)

        return [
            {
                "open_time": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": k[6],
                "quote_volume": float(k[7]),
                "trades": int(k[8]),
                "taker_buy_volume": float(k[9]),
                "taker_buy_quote_volume": float(k[10]),
            }
            for k in data
        ]

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()

        return self._request("GET", "/fapi/v1/openOrders", params=params, signed=True)

    def cancel_order(self, symbol: str, order_id: int) -> bool:
        """Cancel an order."""
        try:
            self._request(
                "DELETE",
                "/fapi/v1/order",
                params={"symbol": symbol.upper(), "orderId": order_id},
                signed=True,
            )
            logger.info(f"Order cancelled", extra={"symbol": symbol, "order_id": order_id})
            return True
        except BinanceClientError as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all open orders for a symbol."""
        try:
            self._request(
                "DELETE",
                "/fapi/v1/allOpenOrders",
                params={"symbol": symbol.upper()},
                signed=True,
            )
            logger.info(f"All orders cancelled", extra={"symbol": symbol})
            return True
        except BinanceClientError as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """
        Place a market order.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            reduce_only: If True, only reduces position
            client_order_id: Custom order ID for idempotency
        """
        info = self.get_symbol_info(symbol)
        if not info:
            raise BinanceClientError(f"Symbol info not found: {symbol}")

        rounded_qty = info.round_quantity(quantity)

        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "MARKET",
            "quantity": str(rounded_qty),
        }

        if reduce_only:
            params["reduceOnly"] = "true"

        if client_order_id:
            params["newClientOrderId"] = client_order_id

        data = self._request("POST", "/fapi/v1/order", params=params, signed=True)

        result = OrderResult(
            order_id=data["orderId"],
            client_order_id=data.get("clientOrderId", ""),
            symbol=data["symbol"],
            status=data["status"],
            type=data["type"],
            side=data["side"],
            price=float(data.get("price", 0)),
            avg_price=float(data.get("avgPrice", 0)),
            orig_qty=float(data["origQty"]),
            executed_qty=float(data["executedQty"]),
            cum_quote=float(data.get("cumQuote", 0)),
            reduce_only=data.get("reduceOnly", False),
            time_in_force=data.get("timeInForce", "GTC"),
            update_time=data.get("updateTime", 0),
        )

        logger.order_event(
            symbol=symbol,
            order_type="MARKET",
            side=side,
            quantity=float(rounded_qty),
            order_id=str(result.order_id),
            status=result.status,
            avg_price=result.avg_price,
        )

        return result

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """Place a limit order."""
        info = self.get_symbol_info(symbol)
        if not info:
            raise BinanceClientError(f"Symbol info not found: {symbol}")

        rounded_qty = info.round_quantity(quantity)
        rounded_price = info.round_price(price)

        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "LIMIT",
            "quantity": str(rounded_qty),
            "price": str(rounded_price),
            "timeInForce": time_in_force,
        }

        if reduce_only:
            params["reduceOnly"] = "true"

        if client_order_id:
            params["newClientOrderId"] = client_order_id

        data = self._request("POST", "/fapi/v1/order", params=params, signed=True)

        result = OrderResult(
            order_id=data["orderId"],
            client_order_id=data.get("clientOrderId", ""),
            symbol=data["symbol"],
            status=data["status"],
            type=data["type"],
            side=data["side"],
            price=float(data.get("price", 0)),
            avg_price=float(data.get("avgPrice", 0)),
            orig_qty=float(data["origQty"]),
            executed_qty=float(data["executedQty"]),
            cum_quote=float(data.get("cumQuote", 0)),
            reduce_only=data.get("reduceOnly", False),
            time_in_force=data.get("timeInForce", "GTC"),
            update_time=data.get("updateTime", 0),
        )

        logger.order_event(
            symbol=symbol,
            order_type="LIMIT",
            side=side,
            quantity=float(rounded_qty),
            price=float(rounded_price),
            order_id=str(result.order_id),
            status=result.status,
        )

        return result

    def place_stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """
        Place a stop market order (stop-loss).
        """
        info = self.get_symbol_info(symbol)
        if not info:
            raise BinanceClientError(f"Symbol info not found: {symbol}")

        rounded_qty = info.round_quantity(quantity)
        rounded_stop = info.round_price(stop_price)

        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "STOP_MARKET",
            "quantity": str(rounded_qty),
            "stopPrice": str(rounded_stop),
            "reduceOnly": "true" if reduce_only else "false",
        }

        if client_order_id:
            params["newClientOrderId"] = client_order_id

        data = self._request("POST", "/fapi/v1/order", params=params, signed=True)

        result = OrderResult(
            order_id=data["orderId"],
            client_order_id=data.get("clientOrderId", ""),
            symbol=data["symbol"],
            status=data["status"],
            type=data["type"],
            side=data["side"],
            price=float(data.get("stopPrice", stop_price)),
            avg_price=float(data.get("avgPrice", 0)),
            orig_qty=float(data["origQty"]),
            executed_qty=float(data["executedQty"]),
            cum_quote=float(data.get("cumQuote", 0)),
            reduce_only=data.get("reduceOnly", True),
            time_in_force=data.get("timeInForce", "GTC"),
            update_time=data.get("updateTime", 0),
        )

        logger.order_event(
            symbol=symbol,
            order_type="STOP_MARKET",
            side=side,
            quantity=float(rounded_qty),
            price=float(rounded_stop),
            order_id=str(result.order_id),
            status=result.status,
        )

        return result

    def place_take_profit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """
        Place a take profit market order.
        """
        info = self.get_symbol_info(symbol)
        if not info:
            raise BinanceClientError(f"Symbol info not found: {symbol}")

        rounded_qty = info.round_quantity(quantity)
        rounded_stop = info.round_price(stop_price)

        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "TAKE_PROFIT_MARKET",
            "quantity": str(rounded_qty),
            "stopPrice": str(rounded_stop),
            "reduceOnly": "true" if reduce_only else "false",
        }

        if client_order_id:
            params["newClientOrderId"] = client_order_id

        data = self._request("POST", "/fapi/v1/order", params=params, signed=True)

        result = OrderResult(
            order_id=data["orderId"],
            client_order_id=data.get("clientOrderId", ""),
            symbol=data["symbol"],
            status=data["status"],
            type=data["type"],
            side=data["side"],
            price=float(data.get("stopPrice", stop_price)),
            avg_price=float(data.get("avgPrice", 0)),
            orig_qty=float(data["origQty"]),
            executed_qty=float(data["executedQty"]),
            cum_quote=float(data.get("cumQuote", 0)),
            reduce_only=data.get("reduceOnly", True),
            time_in_force=data.get("timeInForce", "GTC"),
            update_time=data.get("updateTime", 0),
        )

        logger.order_event(
            symbol=symbol,
            order_type="TAKE_PROFIT_MARKET",
            side=side,
            quantity=float(rounded_qty),
            price=float(rounded_stop),
            order_id=str(result.order_id),
            status=result.status,
        )

        return result

    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Get current funding rate."""
        data = self._request(
            "GET",
            "/fapi/v1/premiumIndex",
            params={"symbol": symbol.upper()}
        )
        return {
            "symbol": data["symbol"],
            "mark_price": float(data["markPrice"]),
            "index_price": float(data["indexPrice"]),
            "funding_rate": float(data["lastFundingRate"]),
            "next_funding_time": data["nextFundingTime"],
        }

    def get_open_interest(self, symbol: str) -> float:
        """Get open interest."""
        data = self._request(
            "GET",
            "/fapi/v1/openInterest",
            params={"symbol": symbol.upper()}
        )
        return float(data["openInterest"])

    def test_connectivity(self) -> bool:
        """Test API connectivity."""
        try:
            self._request("GET", "/fapi/v1/ping")
            return True
        except Exception:
            return False


def create_client() -> BinanceClient:
    """Create client from settings."""
    settings = get_settings()
    return BinanceClient(
        api_key=settings.binance_api_key,
        api_secret=settings.binance_api_secret,
        environment=settings.env,
    )
