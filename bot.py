"""
Main Trading Bot - Orchestrates the crypto perpetual futures trading system
"""
import asyncio
import signal
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from config import TradingConfig, DEFAULT_CONFIG
from utils.logger import get_trading_logger

logger = get_trading_logger()


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    size: float
    leverage: int
    stop_loss: float
    take_profit: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    trailing_stop: Optional[float] = None


@dataclass
class Signal:
    """Trading signal from strategy"""
    symbol: str
    action: str  # 'long', 'short', 'close', 'none'
    confidence: float  # 0-1
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


class TradingBot:
    """
    Main trading bot that orchestrates:
    - Data fetching
    - Strategy signal generation
    - Risk management
    - Order execution
    """

    def __init__(self, config: TradingConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.positions: Dict[str, Position] = {}
        self.pending_signals: List[Signal] = []
        self.is_running = False
        self.portfolio_value = 0.0
        self.initial_capital = 10000.0

        # Components (initialized in setup)
        self.exchange = None
        self.strategy = None
        self.risk_manager = None

        # Performance tracking
        self.trades_history: List[dict] = []
        self.equity_curve: List[float] = []

    async def setup(self, exchange_id: str = "binance", api_key: str = None, secret: str = None):
        """Initialize all components"""
        logger.info("Setting up trading bot...")

        # Import components
        from exchange.binance_futures import BinanceFuturesExchange
        from strategies.trend_momentum_strategy import TrendMomentumStrategy
        from risk.risk_manager import RiskManager

        # Initialize exchange
        self.exchange = BinanceFuturesExchange(api_key=api_key, secret=secret)
        await self.exchange.connect()

        # Initialize strategy
        self.strategy = TrendMomentumStrategy(self.config)

        # Initialize risk manager
        self.risk_manager = RiskManager(
            initial_capital=self.initial_capital,
            max_position_pct=self.config.max_position_size_pct,
            max_drawdown_pct=self.config.max_drawdown_pct,
            max_total_exposure=self.config.max_total_exposure
        )

        # Get initial portfolio value
        balance = await self.exchange.get_balance()
        self.portfolio_value = balance.get('total', self.initial_capital)
        self.initial_capital = self.portfolio_value

        logger.info(f"Bot initialized with ${self.portfolio_value:.2f} capital")

    async def fetch_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol"""
        try:
            # Fetch primary timeframe
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe=self.config.primary_timeframe,
                limit=200
            )
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()

    async def generate_signals(self) -> List[Signal]:
        """Generate trading signals for all symbols"""
        signals = []

        for symbol in self.config.symbols:
            try:
                # Fetch market data
                df = await self.fetch_market_data(symbol)
                if df.empty:
                    continue

                # Generate signal from strategy
                signal_data = self.strategy.generate_signal(df, symbol)

                if signal_data['action'] != 'none':
                    signal = Signal(
                        symbol=symbol,
                        action=signal_data['action'],
                        confidence=signal_data['confidence'],
                        entry_price=signal_data['entry_price'],
                        stop_loss=signal_data['stop_loss'],
                        take_profit=signal_data['take_profit'],
                        reason=signal_data['reason']
                    )
                    signals.append(signal)
                    logger.info(f"Signal: {symbol} {signal.action.upper()} "
                               f"(confidence: {signal.confidence:.2f}) - {signal.reason}")

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")

        return signals

    async def execute_signal(self, signal: Signal) -> bool:
        """Execute a trading signal"""
        try:
            # Check if we already have a position
            if signal.symbol in self.positions:
                if signal.action == 'close':
                    return await self.close_position(signal.symbol, signal.reason)
                else:
                    logger.info(f"Already have position in {signal.symbol}, skipping")
                    return False

            # Check risk limits
            if not self.risk_manager.can_open_position(
                portfolio_value=self.portfolio_value,
                current_positions=self.positions,
                new_position_value=signal.entry_price
            ):
                logger.warning(f"Risk limits prevent opening position in {signal.symbol}")
                return False

            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                portfolio_value=self.portfolio_value,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                leverage=self.config.default_leverage
            )

            if position_size <= 0:
                logger.warning(f"Calculated position size too small for {signal.symbol}")
                return False

            # Place order
            side = 'buy' if signal.action == 'long' else 'sell'
            order = await self.exchange.create_order(
                symbol=signal.symbol,
                type='market',
                side=side,
                amount=position_size,
                params={'leverage': self.config.default_leverage}
            )

            if order:
                # Create position record
                self.positions[signal.symbol] = Position(
                    symbol=signal.symbol,
                    side=signal.action,
                    entry_price=order['price'] or signal.entry_price,
                    size=position_size,
                    leverage=self.config.default_leverage,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    entry_time=datetime.now()
                )

                # Place stop loss and take profit orders
                await self._place_exit_orders(signal.symbol)

                logger.info(f"Opened {signal.action.upper()} position: {signal.symbol} "
                           f"size={position_size:.4f} @ ${signal.entry_price:.2f}")
                return True

        except Exception as e:
            logger.error(f"Failed to execute signal for {signal.symbol}: {e}")

        return False

    async def _place_exit_orders(self, symbol: str):
        """Place stop loss and take profit orders for a position"""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        try:
            # Stop loss
            sl_side = 'sell' if pos.side == 'long' else 'buy'
            await self.exchange.create_order(
                symbol=symbol,
                type='stop_market',
                side=sl_side,
                amount=pos.size,
                params={'stopPrice': pos.stop_loss, 'reduceOnly': True}
            )

            # Take profit
            tp_side = 'sell' if pos.side == 'long' else 'buy'
            await self.exchange.create_order(
                symbol=symbol,
                type='take_profit_market',
                side=tp_side,
                amount=pos.size,
                params={'stopPrice': pos.take_profit, 'reduceOnly': True}
            )

            logger.info(f"Exit orders placed for {symbol}: SL=${pos.stop_loss:.2f}, TP=${pos.take_profit:.2f}")

        except Exception as e:
            logger.error(f"Failed to place exit orders for {symbol}: {e}")

    async def close_position(self, symbol: str, reason: str = "Signal") -> bool:
        """Close an open position"""
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]

        try:
            # Cancel existing orders
            await self.exchange.cancel_all_orders(symbol)

            # Close position
            side = 'sell' if pos.side == 'long' else 'buy'
            order = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=pos.size,
                params={'reduceOnly': True}
            )

            if order:
                exit_price = order['price'] or await self._get_current_price(symbol)

                # Calculate PnL
                if pos.side == 'long':
                    pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.size * pos.leverage
                else:
                    pnl = (pos.entry_price - exit_price) / pos.entry_price * pos.size * pos.leverage

                # Record trade
                self.trades_history.append({
                    'symbol': symbol,
                    'side': pos.side,
                    'entry_price': pos.entry_price,
                    'exit_price': exit_price,
                    'size': pos.size,
                    'pnl': pnl,
                    'entry_time': pos.entry_time,
                    'exit_time': datetime.now(),
                    'reason': reason
                })

                # Remove position
                del self.positions[symbol]

                logger.info(f"Closed {pos.side.upper()} position: {symbol} "
                           f"PnL: ${pnl:.2f} ({reason})")
                return True

        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")

        return False

    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        ticker = await self.exchange.fetch_ticker(symbol)
        return ticker.get('last', 0.0)

    async def update_positions(self):
        """Update position status and check for exits"""
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]

            try:
                current_price = await self._get_current_price(symbol)

                # Update unrealized PnL
                if pos.side == 'long':
                    pos.unrealized_pnl = (current_price - pos.entry_price) / pos.entry_price * 100
                else:
                    pos.unrealized_pnl = (pos.entry_price - current_price) / pos.entry_price * 100

                # Update trailing stop if in profit
                if self.config.stop_loss_atr_mult > 0 and pos.unrealized_pnl > 2.0:
                    self._update_trailing_stop(pos, current_price)

            except Exception as e:
                logger.error(f"Error updating position {symbol}: {e}")

    def _update_trailing_stop(self, pos: Position, current_price: float):
        """Update trailing stop for a position"""
        if pos.side == 'long':
            new_stop = current_price * 0.98  # 2% trailing
            if pos.trailing_stop is None or new_stop > pos.trailing_stop:
                pos.trailing_stop = new_stop
                pos.stop_loss = max(pos.stop_loss, new_stop)
        else:
            new_stop = current_price * 1.02
            if pos.trailing_stop is None or new_stop < pos.trailing_stop:
                pos.trailing_stop = new_stop
                pos.stop_loss = min(pos.stop_loss, new_stop)

    async def run_cycle(self):
        """Run one trading cycle"""
        logger.info("Running trading cycle...")

        # Update existing positions
        await self.update_positions()

        # Generate new signals
        signals = await self.generate_signals()

        # Execute signals
        for signal in signals:
            if signal.confidence >= 0.6:  # Only trade high confidence signals
                await self.execute_signal(signal)

        # Update portfolio value
        balance = await self.exchange.get_balance()
        self.portfolio_value = balance.get('total', self.portfolio_value)
        self.equity_curve.append(self.portfolio_value)

        # Check for max drawdown
        if self.risk_manager.check_max_drawdown(self.equity_curve):
            logger.warning("Max drawdown reached! Pausing trading...")
            await self.close_all_positions("Max drawdown reached")

    async def close_all_positions(self, reason: str = "Manual"):
        """Close all open positions"""
        for symbol in list(self.positions.keys()):
            await self.close_position(symbol, reason)

    async def run(self, interval_seconds: int = 300):
        """Main bot loop"""
        self.is_running = True
        logger.info(f"Starting trading bot (interval: {interval_seconds}s)...")

        # Setup signal handlers
        def handle_shutdown(sig, frame):
            logger.info("Shutdown signal received...")
            self.is_running = False

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        while self.is_running:
            try:
                await self.run_cycle()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)  # Wait a minute on error

        # Cleanup
        logger.info("Shutting down bot...")
        await self.close_all_positions("Shutdown")
        if self.exchange:
            await self.exchange.close()

    def get_performance_summary(self) -> dict:
        """Get trading performance summary"""
        if not self.trades_history:
            return {"message": "No trades yet"}

        trades_df = pd.DataFrame(self.trades_history)
        total_pnl = trades_df['pnl'].sum()
        win_rate = (trades_df['pnl'] > 0).mean() * 100
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (trades_df['pnl'] < 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        return {
            'total_trades': len(self.trades_history),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'current_positions': len(self.positions),
            'portfolio_value': self.portfolio_value
        }


async def main():
    """Entry point for the trading bot"""
    import os

    config = TradingConfig(
        symbols=["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
        primary_timeframe="4h",
        default_leverage=5
    )

    bot = TradingBot(config)

    # Setup with API keys from environment
    await bot.setup(
        api_key=os.getenv("BINANCE_API_KEY"),
        secret=os.getenv("BINANCE_SECRET")
    )

    # Run the bot
    await bot.run(interval_seconds=300)  # 5 minute cycles


if __name__ == "__main__":
    asyncio.run(main())
