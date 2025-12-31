#!/usr/bin/env python3
"""
Backtest runner for the crypto perpetual futures strategy
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, Dict
from config import TradingConfig, BacktestConfig
from backtest.engine import (
    BacktestEngine, MultiTimeframeDataHandler, Timeframe,
    MarketDataEvent, SignalEvent, EventType, PositionSide
)
from strategies.trend_momentum_strategy import TrendMomentumStrategy, StrategyConfig
from strategies.base_strategy import SignalType
from exchange.binance_futures import BinanceFutures
from exchange.data_fetcher import DataFetcher


class BacktestStrategyAdapter:
    """Adapter to bridge TrendMomentumStrategy with BacktestEngine interface."""

    def __init__(self, strategy: TrendMomentumStrategy, data_handler: MultiTimeframeDataHandler, leverage: float = 5.0):
        self.strategy = strategy
        self.data_handler = data_handler
        self.leverage = leverage
        self._last_signal: Dict[str, SignalType] = {}
        self._positions: Dict[str, str] = {}  # symbol -> position_type

    def on_data(self, event: MarketDataEvent) -> Optional[SignalEvent]:
        """Process market data and generate signals."""
        # Get enough data for indicators
        bars = self.data_handler.get_latest_bars(
            symbol=event.symbol,
            timeframe=event.timeframe,
            n=100,
            current_time=event.timestamp,
        )

        if len(bars) < 50:
            return None

        # Calculate indicators using strategy
        df = self.strategy.calculate_indicators(bars.copy(), event.timeframe.value)

        # Get current position
        current_position = self._positions.get(event.symbol)

        # Generate signal
        signal = self.strategy.generate_signal(df, event.symbol, event.timeframe.value, current_position)

        # Map signal type to backtest engine format
        if signal.signal_type == SignalType.LONG and signal.confidence >= 0.6:
            signal_type = PositionSide.LONG
            self._positions[event.symbol] = 'long'
        elif signal.signal_type == SignalType.SHORT and signal.confidence >= 0.6:
            signal_type = PositionSide.SHORT
            self._positions[event.symbol] = 'short'
        elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
            signal_type = PositionSide.FLAT
            self._positions[event.symbol] = None
        else:
            return None

        # Avoid duplicate signals
        last = self._last_signal.get(event.symbol)
        if signal.signal_type == last:
            return None

        self._last_signal[event.symbol] = signal.signal_type

        return SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=event.timestamp,
            symbol=event.symbol,
            signal_type=signal_type,
            strength=signal.confidence,
            target_leverage=self.leverage,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

    def on_fill(self, event) -> None:
        """Handle fill events."""
        pass


async def fetch_historical_data(symbols: list, start_date: str, end_date: str, timeframe: str = "4h"):
    """Fetch historical data for backtesting"""
    # Create exchange instance (no API keys needed for public data)
    exchange = BinanceFutures(api_key="", api_secret="", testnet=False)
    await exchange.connect()

    fetcher = DataFetcher(exchange)
    data = {}

    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        try:
            df = await fetcher.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )
            if df is not None and not df.empty:
                data[symbol] = df
                print(f"  -> {len(df)} candles loaded")
        except Exception as e:
            print(f"  -> Error: {e}")

    await exchange.disconnect()
    return data


def run_backtest(data: dict, config: TradingConfig, backtest_config: BacktestConfig):
    """Run the backtest"""
    # Convert data to format expected by MultiTimeframeDataHandler
    # Expected: {symbol: {timeframe: df}}
    formatted_data = {}
    for symbol, df in data.items():
        formatted_data[symbol] = {config.primary_timeframe: df}

    # Create data handler
    timeframe_map = {'1h': Timeframe.H1, '4h': Timeframe.H4, '1d': Timeframe.D1}
    primary_tf = timeframe_map.get(config.primary_timeframe, Timeframe.H4)
    data_handler = MultiTimeframeDataHandler(formatted_data, primary_timeframe=primary_tf)

    # Create strategy
    strategy_config = StrategyConfig(
        ema_fast_period=config.ema_fast,
        ema_slow_period=config.ema_slow,
        rsi_period=config.rsi_period,
        atr_period=config.atr_period,
        atr_stop_loss_mult=config.stop_loss_atr_mult,
        atr_take_profit_mult=config.take_profit_atr_mult
    )
    base_strategy = TrendMomentumStrategy(strategy_config)

    # Wrap in adapter for backtest engine
    min_conf = getattr(config, 'min_confidence', 0.55)
    strategy = BacktestStrategyAdapter(base_strategy, data_handler, leverage=float(config.default_leverage))
    strategy.min_confidence = min_conf

    # Create engine
    engine = BacktestEngine(
        data_handler=data_handler,
        strategy=strategy,
        initial_capital=backtest_config.initial_capital,
        max_leverage=float(config.default_leverage),
        taker_fee=backtest_config.commission_pct
    )

    print("\nRunning backtest...")
    results = engine.run()

    return results, engine


def plot_results(engine, results):
    """Plot backtest results"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # 1. Equity curve
    ax1 = axes[0, 0]
    if hasattr(engine, 'equity_curve') and engine.equity_curve:
        equity = pd.Series(engine.equity_curve)
        equity.plot(ax=ax1, color='blue', linewidth=1.5)
    ax1.set_title('Equity Curve')
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = axes[0, 1]
    if hasattr(engine, 'drawdown_curve') and engine.drawdown_curve:
        drawdown = pd.Series(engine.drawdown_curve)
        drawdown.plot(ax=ax2, color='red', linewidth=1.5)
        ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    ax2.set_title('Drawdown')
    ax2.set_xlabel('Trade #')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)

    # 3. Monthly returns
    ax3 = axes[1, 0]
    trades_df = engine.get_trades() if hasattr(engine, 'get_trades') else pd.DataFrame()
    if trades_df is not None and not trades_df.empty and 'exit_time' in trades_df.columns:
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        trades_df['month'] = trades_df['exit_time'].dt.to_period('M')
        pnl_col = 'realized_pnl' if 'realized_pnl' in trades_df.columns else 'pnl'
        if pnl_col in trades_df.columns:
            monthly_returns = trades_df.groupby('month')[pnl_col].sum()
            if len(monthly_returns) > 1:
                monthly_returns.plot(kind='bar', ax=ax3, color=['green' if x > 0 else 'red' for x in monthly_returns])
                ax3.tick_params(axis='x', rotation=45)
    ax3.set_title('Monthly PnL')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('PnL ($)')
    ax3.grid(True, alpha=0.3)

    # 4. Win/Loss distribution
    ax4 = axes[1, 1]
    if trades_df is not None and not trades_df.empty:
        pnl_col = 'realized_pnl' if 'realized_pnl' in trades_df.columns else 'pnl'
        if pnl_col in trades_df.columns:
            pnls = trades_df[trades_df[pnl_col] != 0][pnl_col].tolist()
            if pnls:
                wins = [p for p in pnls if p > 0]
                losses = [p for p in pnls if p <= 0]
                if wins or losses:
                    ax4.hist([wins, losses], bins=20, label=['Wins', 'Losses'], color=['green', 'red'], alpha=0.7)
                    ax4.legend()
    ax4.set_title('PnL Distribution')
    ax4.set_xlabel('PnL ($)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)

    # 5. Cumulative PnL by symbol
    ax5 = axes[2, 0]
    if trades_df is not None and not trades_df.empty and 'symbol' in trades_df.columns:
        pnl_col = 'realized_pnl' if 'realized_pnl' in trades_df.columns else 'pnl'
        if pnl_col in trades_df.columns:
            symbol_pnl = trades_df.groupby('symbol')[pnl_col].sum().sort_values()
            colors = ['green' if x > 0 else 'red' for x in symbol_pnl]
            symbol_pnl.plot(kind='barh', ax=ax5, color=colors)
    ax5.set_title('PnL by Symbol')
    ax5.set_xlabel('Total PnL ($)')
    ax5.grid(True, alpha=0.3)

    # 6. Performance metrics table
    ax6 = axes[2, 1]
    ax6.axis('off')
    metrics_text = f"""
    BACKTEST RESULTS
    ================

    Total Return:      {results.total_return * 100:.2f}%
    Sharpe Ratio:      {results.sharpe_ratio:.2f}
    Sortino Ratio:     {results.sortino_ratio:.2f}
    Max Drawdown:      {results.max_drawdown * 100:.2f}%

    Total Trades:      {results.total_trades}
    Win Rate:          {results.win_rate * 100:.1f}%
    Profit Factor:     {results.profit_factor:.2f}

    Avg Win:           ${results.avg_win:.2f}
    Avg Loss:          ${results.avg_loss:.2f}
    Largest Win:       ${results.largest_win:.2f}
    Largest Loss:      ${results.largest_loss:.2f}

    Net Profit:        ${results.net_profit:.2f}
    """
    ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nResults saved to backtest_results.png")


def print_results(results, initial_capital=10000):
    """Print backtest results to console"""
    final_capital = initial_capital * (1 + results.total_return)
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Return:      {results.total_return * 100:.2f}%")
    print(f"Sharpe Ratio:      {results.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:     {results.sortino_ratio:.2f}")
    print(f"Max Drawdown:      {results.max_drawdown * 100:.2f}%")
    print("-" * 60)
    print(f"Total Trades:      {results.total_trades}")
    print(f"Win Rate:          {results.win_rate * 100:.1f}%")
    print(f"Profit Factor:     {results.profit_factor:.2f}")
    print("-" * 60)
    print(f"Avg Win:           ${results.avg_win:.2f}")
    print(f"Avg Loss:          ${results.avg_loss:.2f}")
    print(f"Largest Win:       ${results.largest_win:.2f}")
    print(f"Largest Loss:      ${results.largest_loss:.2f}")
    print("-" * 60)
    print(f"Net Profit:        ${results.net_profit:.2f}")
    print(f"Initial Capital:   ${initial_capital:.2f}")
    print(f"Final Capital:     ${final_capital:.2f}")
    print("=" * 60)


async def main():
    """Main entry point"""
    # Use OPTIMIZED configuration from config.py
    trading_config = TradingConfig()  # Uses optimized defaults

    backtest_config = BacktestConfig(
        initial_capital=10000.0,
        start_date="2024-01-01",
        end_date="2024-12-31",
        commission_pct=0.0004
    )

    print("=" * 60)
    print("CRYPTO PERPETUAL FUTURES BACKTEST")
    print("=" * 60)
    print(f"Symbols: {trading_config.symbols}")
    print(f"Timeframe: {trading_config.primary_timeframe}")
    print(f"Period: {backtest_config.start_date} to {backtest_config.end_date}")
    print(f"Leverage: {trading_config.default_leverage}x")
    print("=" * 60)

    # Fetch data
    data = await fetch_historical_data(
        symbols=trading_config.symbols,
        start_date=backtest_config.start_date,
        end_date=backtest_config.end_date,
        timeframe=trading_config.primary_timeframe
    )

    if not data:
        print("No data fetched. Exiting.")
        return

    # Run backtest
    results, engine = run_backtest(data, trading_config, backtest_config)

    # Print results
    print_results(results, initial_capital=backtest_config.initial_capital)

    # Plot results
    try:
        plot_results(engine, results)
    except Exception as e:
        print(f"Could not plot results: {e}")

    # Export trades
    trades_df = engine.get_trades()
    if trades_df is not None and not trades_df.empty:
        trades_df.to_csv('backtest_trades.csv', index=False)
        print(f"\n{len(trades_df)} trades exported to backtest_trades.csv")


if __name__ == "__main__":
    asyncio.run(main())
