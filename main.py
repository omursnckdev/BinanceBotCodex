#!/usr/bin/env python3
"""
Binance Futures Trading Bot - Main Entry Point

A production-grade scalping bot for Binance USDT-M Futures.
Testnet-first design with multiple safety layers.

IMPORTANT SAFETY NOTES:
- Default config is TESTNET + DRY_RUN + ALLOW_LIVE_TRADING=false
- Live trading requires ALL THREE: ENV=mainnet, DRY_RUN=false, ALLOW_LIVE_TRADING=true
- Never place real orders unless all safety toggles are satisfied
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time
from datetime import datetime, date
from typing import Dict, Optional

from config import (
    get_settings, TradingAction, WhaleDefensiveAction, Environment
)
from exchange.binance_client import BinanceClient, create_client
from data.market_data import MarketDataManager
from signals.indicators import TechnicalIndicators
from signals.whales import WhaleDetector
from signals.sentiment import SentimentAnalyzer
from strategy.fusion import FusionStrategy
from risk.risk_manager import RiskManager
from execution.executor import OrderExecutor
from state.store import StateStore
from utils.logger import setup_logging, get_logger


logger = get_logger(__name__)


class TradingBot:
    """
    Main trading bot orchestrator.

    Coordinates all modules and runs the main event loop.
    """

    def __init__(self):
        self.settings = get_settings()
        self._running = False
        self._shutdown_requested = False
        self._last_daily_reset: Optional[date] = None

        # Initialize components (lazy)
        self._client: Optional[BinanceClient] = None
        self._market_data: Optional[MarketDataManager] = None
        self._tech_indicators: Optional[TechnicalIndicators] = None
        self._whale_detector: Optional[WhaleDetector] = None
        self._sentiment: Optional[SentimentAnalyzer] = None
        self._fusion: Optional[FusionStrategy] = None
        self._risk_manager: Optional[RiskManager] = None
        self._executor: Optional[OrderExecutor] = None
        self._state: Optional[StateStore] = None

    def _initialize_components(self) -> None:
        """Initialize all trading components."""
        logger.info("Initializing trading components...")

        # Create Binance client
        self._client = create_client()

        # Initialize time sync
        try:
            self._client.sync_time()
        except Exception as e:
            logger.warning(f"Time sync failed: {e}")

        # Load exchange info
        try:
            self._client.load_exchange_info()
        except Exception as e:
            logger.error(f"Failed to load exchange info: {e}")
            raise

        # Initialize other components
        self._market_data = MarketDataManager(self._client, self.settings.data)
        self._tech_indicators = TechnicalIndicators(self.settings.indicators)
        self._whale_detector = WhaleDetector(self._market_data, self.settings.whale)
        self._sentiment = SentimentAnalyzer(self.settings.sentiment)
        self._fusion = FusionStrategy(self.settings.fusion)
        self._risk_manager = RiskManager(self._client, self.settings.risk)
        self._state = StateStore()
        self._executor = OrderExecutor(
            self._client,
            self._risk_manager,
            self._state,
        )

        logger.info("All components initialized successfully")

    def _print_startup_banner(self) -> None:
        """Print startup banner with safety status."""
        print("\n" + "=" * 60)
        print("BINANCE FUTURES TRADING BOT")
        print("=" * 60)
        print(self.settings.print_safety_banner())
        print(f"\nTrading Symbols: {', '.join(self.settings.trading_symbols)}")
        print(f"Main Loop Interval: {self.settings.main_loop_interval_seconds}s")
        print("=" * 60 + "\n")

        if self.settings.is_live_trading_enabled():
            print("\n" + "!" * 60)
            print("!!! WARNING: LIVE TRADING MODE ACTIVE !!!")
            print("!!! REAL MONEY IS AT RISK !!!")
            print("!" * 60 + "\n")

            # Countdown before starting
            for i in range(5, 0, -1):
                print(f"Starting in {i}...")
                time.sleep(1)

    def _check_daily_reset(self) -> None:
        """Check and perform daily stat reset if needed."""
        today = date.today()
        if self._last_daily_reset != today:
            logger.info("Performing daily reset...")
            self._state.reset_daily_stats()
            self._risk_manager.reset_daily_stats()
            self._last_daily_reset = today

    def _process_symbol(self, symbol: str) -> None:
        """
        Process a single symbol through the trading pipeline.

        Args:
            symbol: Trading pair to process
        """
        try:
            # Check if symbol is enabled
            symbol_config = self.settings.get_symbol_config(symbol)
            if not symbol_config.enabled:
                return

            # Check market conditions
            conditions = self._market_data.check_conditions(symbol)
            if not conditions.is_tradeable:
                issues = conditions.get_issues()
                logger.debug(
                    f"Market conditions not tradeable for {symbol}",
                    extra={"issues": issues}
                )
                return

            # Get current position state
            position = self._state.get_position(symbol)
            position_side = position.side if position else None

            # Fetch multi-timeframe candles
            candles = self._market_data.get_multi_timeframe_candles(symbol)

            # Generate technical signal (primary timeframe)
            primary_tf = self.settings.data.primary_timeframe
            tech_signal = self._tech_indicators.analyze(
                candles.get(primary_tf),
                symbol=symbol
            )

            if not tech_signal.is_valid:
                logger.debug(f"Invalid tech signal for {symbol}: {tech_signal.error_message}")
                return

            # Generate whale signal
            whale_signal = self._whale_detector.analyze(symbol, position_side)

            # Generate sentiment signal
            sentiment_signal = self._sentiment.analyze(symbol)

            # Fuse signals into decision
            decision = self._fusion.fuse_signals(
                symbol=symbol,
                tech_signal=tech_signal,
                whale_signal=whale_signal,
                sentiment_signal=sentiment_signal,
                current_position_side=position_side,
                market_conditions_ok=conditions.is_tradeable,
            )

            # Log the decision
            logger.trade_signal(
                symbol=symbol,
                action=decision.action.value,
                scores={
                    "tech": decision.tech_score,
                    "whale": decision.whale_score,
                    "sentiment": decision.sentiment_score,
                    "final": decision.final_score,
                },
                reason=decision.decision_reason,
                confidence=decision.confidence,
                trend=decision.trend.value,
            )

            # Handle existing position
            if position:
                # Check for defensive action (EXPLICIT - not FLAT)
                if decision.recommended_defensive_action and decision.recommended_defensive_action != WhaleDefensiveAction.NONE:
                    logger.info(
                        f"Executing defensive action for {symbol}",
                        extra={"action": decision.recommended_defensive_action.value}
                    )
                    self._executor.execute_defensive_action(
                        symbol,
                        decision.recommended_defensive_action
                    )
                    return

                # Check for explicit exit
                if decision.exit_reason:
                    logger.info(f"Closing position: {decision.exit_reason}")
                    self._executor.execute_close(symbol, decision.exit_reason)
                    return

                # Check time-based exit and other conditions
                should_close, reason = self._fusion.should_close_position(
                    decision=decision,
                    position_side=position.side,
                    position_entry_time=position.entry_time,
                    position_entry_price=position.entry_price,
                    current_price=tech_signal.current_price,
                )
                if should_close:
                    logger.info(f"Closing position: {reason}")
                    self._executor.execute_close(symbol, reason)
                    return

            # No position - check for entry
            else:
                # FLAT means no entry (NOT close)
                if decision.action == TradingAction.FLAT:
                    return

                # Check cooldown
                if self._state.is_in_cooldown(symbol):
                    logger.debug(f"{symbol} in cooldown, skipping entry")
                    return

                # Execute entry
                if decision.action in [TradingAction.LONG, TradingAction.SHORT]:
                    result = self._executor.execute_entry(
                        decision=decision,
                        current_price=tech_signal.current_price,
                    )
                    if result.success:
                        logger.info(f"Entry executed: {result.message}")
                    else:
                        logger.warning(f"Entry failed: {result.message}")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    def _run_cycle(self) -> None:
        """Run one cycle of the trading loop."""
        # Check daily reset
        self._check_daily_reset()

        # Check risk status
        risk_status = self._risk_manager.check_risk_status()
        if not risk_status.is_trading_allowed:
            logger.warning(
                "Trading disabled by risk manager",
                extra={"violations": [v.value for v in risk_status.violations]}
            )
            # Still sync positions even if trading disabled
            self._executor.sync_positions()
            return

        # Sync time periodically
        if self._client.is_time_sync_required():
            try:
                self._client.sync_time()
            except Exception as e:
                logger.warning(f"Time sync failed: {e}")

        # Process each symbol
        for symbol in self.settings.trading_symbols:
            if self._shutdown_requested:
                break
            self._process_symbol(symbol)

        # Sync positions with exchange
        self._executor.sync_positions()

    def run(self) -> None:
        """Run the trading bot main loop."""
        try:
            # Setup logging
            setup_logging(
                level=self.settings.log_level,
                log_file=self.settings.log_file,
                rotation_mb=self.settings.log_rotation_mb,
                backup_count=self.settings.log_backup_count,
            )

            # Print startup banner
            self._print_startup_banner()

            # Initialize components
            self._initialize_components()

            # Setup signal handlers
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)

            logger.info("Trading bot started")
            self._running = True

            # Main loop
            while self._running and not self._shutdown_requested:
                cycle_start = time.time()

                try:
                    self._run_cycle()
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)

                # Sleep for remaining interval
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.settings.main_loop_interval_seconds - elapsed)
                if sleep_time > 0 and not self._shutdown_requested:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            raise
        finally:
            self._shutdown()

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signal."""
        logger.info(f"Shutdown signal received ({signum})")
        self._shutdown_requested = True

    def _shutdown(self) -> None:
        """Perform graceful shutdown."""
        logger.info("Shutting down trading bot...")
        self._running = False

        # Save state
        if self._state:
            self._state.save_state()

        logger.info("Trading bot stopped")


def main():
    """Main entry point."""
    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()
