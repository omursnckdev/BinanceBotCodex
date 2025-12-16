# Binance Futures Trading Bot

A production-grade Python trading bot for Binance USDT-M Futures with testnet-first design, comprehensive safety controls, and a scalping strategy targeting 1-2% profits per trade.

## ⚠️ IMPORTANT SAFETY WARNINGS

**This bot can lose real money. Use at your own risk.**

- **Default configuration is SAFE**: Testnet + Dry Run enabled
- **Live trading requires explicit opt-in** via THREE separate toggles
- **Never share your API keys** or commit `.env` files
- **Test thoroughly on testnet** before considering mainnet
- **Start with minimal capital** if you decide to use real money

## Features

- **Testnet-First Design**: Safe by default, requires explicit opt-in for live trading
- **Multiple Signal Sources**:
  - Technical indicators (RSI, MACD, Bollinger Bands, ATR, EMA)
  - Whale activity detection (large trades, orderbook imbalance)
  - News sentiment analysis (CryptoPanic, NewsAPI)
- **Risk Management**:
  - Kill switches for daily loss, consecutive losses, exposure
  - Position limits and per-trade risk controls
  - ATR-based stop losses clamped to scalping bounds
- **Scalping Strategy**:
  - Targets 1.0-2.0% take profit
  - Stop losses 0.3-0.9% (ATR-bounded)
  - Time-stops for stale positions
  - Anti flip-flop cooldowns
- **Safety Features**:
  - FLAT means "no entry" - never triggers closes
  - Whale defensive actions are explicit (CLOSE_MARKET, REDUCE, TIGHTEN_STOP)
  - SL/TP computed from actual fill price
  - Graceful shutdown with state persistence

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Get Testnet API Keys

1. Go to [Binance Futures Testnet](https://testnet.binancefuture.com/)
2. Login with GitHub or email
3. Go to API Management
4. Create a new API key
5. **Enable Futures trading** in the API restrictions

### 3. Configure the Bot

```bash
# Copy example config
cp .env.example .env

# Edit with your API keys
nano .env  # or your preferred editor
```

Set these values in `.env`:
```
BINANCE_API_KEY=your_testnet_api_key
BINANCE_API_SECRET=your_testnet_api_secret
ENV=testnet
DRY_RUN=true
ALLOW_LIVE_TRADING=false
```

### 4. Run in Dry-Run Mode (Safest)

```bash
python main.py
```

This mode:
- Connects to testnet
- Fetches real market data
- Generates signals
- **Does NOT place any orders**
- Logs what it would do

### 5. Run on Testnet with Orders

Edit `.env`:
```
ENV=testnet
DRY_RUN=false
ALLOW_LIVE_TRADING=true
```

Then run:
```bash
python main.py
```

This mode:
- Connects to testnet
- **Places real orders on testnet** (fake money)
- Good for testing order execution

### 6. Switch to Mainnet (REAL MONEY - DANGER!)

⚠️ **WARNING: This uses REAL MONEY. You can lose everything.**

1. Get mainnet API keys from [Binance](https://www.binance.com/en/my/settings/api-management)
2. Enable Futures trading on API key
3. Edit `.env`:

```
BINANCE_API_KEY=your_mainnet_api_key
BINANCE_API_SECRET=your_mainnet_api_secret
ENV=mainnet
DRY_RUN=false
ALLOW_LIVE_TRADING=true
```

All THREE settings must be set correctly for live trading.

## Project Structure

```
BinanceBotCodex/
├── config.py              # Configuration with Pydantic
├── main.py                # Main entry point and event loop
├── exchange/
│   └── binance_client.py  # Binance API client
├── data/
│   └── market_data.py     # Market data manager
├── signals/
│   ├── indicators.py      # Technical indicators
│   ├── whales.py          # Whale detection
│   └── sentiment.py       # News sentiment
├── strategy/
│   └── fusion.py          # Signal fusion
├── risk/
│   └── risk_manager.py    # Risk management
├── execution/
│   └── executor.py        # Order execution
├── state/
│   └── store.py           # State persistence
├── utils/
│   └── logger.py          # Secure logging
├── tests/
│   ├── test_indicators.py
│   ├── test_fusion.py
│   └── test_risk.py
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

### Safety Toggles

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `testnet` | `testnet` or `mainnet` |
| `DRY_RUN` | `true` | No orders when true |
| `ALLOW_LIVE_TRADING` | `false` | Extra safety lock |

**Live trading requires ALL THREE:**
- `ENV=mainnet`
- `DRY_RUN=false`
- `ALLOW_LIVE_TRADING=true`

### Risk Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `RISK__MAX_DAILY_LOSS_PCT` | 5.0 | Stop trading after X% daily loss |
| `RISK__MAX_CONSECUTIVE_LOSSES` | 5 | Stop after X consecutive losses |
| `RISK__MAX_TOTAL_EXPOSURE_PCT` | 50.0 | Max % of balance exposed |
| `RISK__MAX_OPEN_POSITIONS` | 3 | Max concurrent positions |
| `RISK__DEFAULT_RISK_PER_TRADE_PCT` | 1.0 | Risk per trade (% of balance) |

### Scalping Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `RISK__TP_DEFAULT_PCT` | 1.2 | Take profit percentage |
| `RISK__SL_MIN_PCT` | 0.3 | Minimum stop loss |
| `RISK__SL_MAX_PCT` | 0.9 | Maximum stop loss |
| `RISK__TIME_STOP_MINUTES` | 45 | Exit after X minutes |

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_indicators.py -v
```

## Trading Logic

### Signal Generation

1. **Technical Indicators** (50% weight default):
   - RSI for overbought/oversold
   - MACD for momentum
   - Bollinger Bands for mean reversion
   - EMA crossover for trend

2. **Whale Detection** (30% weight default):
   - Large trade detection via aggTrades
   - Orderbook imbalance analysis
   - Triggers defensive actions when against position

3. **Sentiment Analysis** (20% weight default):
   - News from CryptoPanic/NewsAPI
   - VADER sentiment scoring
   - Cached with rate limiting

### Decision Making

- `LONG`: Enter long when bullish with trend confirmation
- `SHORT`: Enter short when bearish with trend confirmation
- `FLAT`: **No new entry** (does NOT close positions)
- Exit: Explicit via stop-loss, take-profit, time-stop, or whale alert

### Important Design Decisions

1. **FLAT ≠ Close**: FLAT means "do not enter" only
2. **Whale Defensive Actions**: Explicit (CLOSE_MARKET, REDUCE_50_PERCENT, TIGHTEN_STOP)
3. **Fill Price Based**: SL/TP calculated from actual fill, not ticker
4. **Cooldowns**: Prevent immediate re-entry after exit

## Logs

Logs are written to:
- Console (colored output)
- File: `logs/trading_bot.log` (JSON format, rotating)

Sensitive data (API keys) is automatically redacted.

## Troubleshooting

### "Time sync drift too high"

The bot syncs time with Binance. If this error persists:
- Check your system clock
- Check internet connectivity

### "Market conditions not tradeable"

Could be due to:
- Stale candle data
- High spread
- Low liquidity
- Missing exchange info

### "Risk check failed"

You've hit a kill switch:
- Daily loss limit
- Consecutive losses
- Max exposure
- Max positions

Wait until next day or adjust risk parameters.

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses. Always:

1. Test thoroughly on testnet first
2. Understand the code before running
3. Start with small amounts
4. Never invest more than you can afford to lose

## License

MIT License - See LICENSE file for details.
