#!/usr/bin/env python3
"""
Strategy Optimizer for Crypto Perpetual Futures

Tests multiple parameter combinations to find optimal settings.
"""
import asyncio
import itertools
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

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

    def __init__(self, strategy: TrendMomentumStrategy, data_handler: MultiTimeframeDataHandler,
                 leverage: float = 5.0, min_confidence: float = 0.6):
        self.strategy = strategy
        self.data_handler = data_handler
        self.leverage = leverage
        self.min_confidence = min_confidence
        self._last_signal: Dict[str, SignalType] = {}
        self._positions: Dict[str, str] = {}

    def on_data(self, event: MarketDataEvent) -> Optional[SignalEvent]:
        """Process market data and generate signals."""
        bars = self.data_handler.get_latest_bars(
            symbol=event.symbol,
            timeframe=event.timeframe,
            n=100,
            current_time=event.timestamp,
        )

        if len(bars) < 50:
            return None

        df = self.strategy.calculate_indicators(bars.copy(), event.timeframe.value)
        current_position = self._positions.get(event.symbol)
        signal = self.strategy.generate_signal(df, event.symbol, event.timeframe.value, current_position)

        if signal.signal_type == SignalType.LONG and signal.confidence >= self.min_confidence:
            signal_type = PositionSide.LONG
            self._positions[event.symbol] = 'long'
        elif signal.signal_type == SignalType.SHORT and signal.confidence >= self.min_confidence:
            signal_type = PositionSide.SHORT
            self._positions[event.symbol] = 'short'
        elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
            signal_type = PositionSide.FLAT
            self._positions[event.symbol] = None
        else:
            return None

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
        pass


async def fetch_data(symbols: list, start_date: str, end_date: str, timeframe: str = "4h"):
    """Fetch historical data."""
    exchange = BinanceFutures(api_key="", api_secret="", testnet=False)
    await exchange.connect()
    fetcher = DataFetcher(exchange)
    data = {}

    for symbol in symbols:
        try:
            df = await fetcher.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )
            if df is not None and not df.empty:
                data[symbol] = df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")

    await exchange.disconnect()
    return data


def run_single_backtest(data: dict, params: dict, initial_capital: float = 10000.0) -> dict:
    """Run a single backtest with given parameters."""
    timeframe = params.get('timeframe', '4h')

    formatted_data = {}
    for symbol, df in data.items():
        formatted_data[symbol] = {timeframe: df}

    timeframe_map = {'1h': Timeframe.H1, '4h': Timeframe.H4, '1d': Timeframe.D1}
    primary_tf = timeframe_map.get(timeframe, Timeframe.H4)
    data_handler = MultiTimeframeDataHandler(formatted_data, primary_timeframe=primary_tf)

    strategy_config = StrategyConfig(
        ema_fast_period=params.get('ema_fast', 12),
        ema_slow_period=params.get('ema_slow', 26),
        rsi_period=params.get('rsi_period', 14),
        rsi_overbought=params.get('rsi_overbought', 70),
        rsi_oversold=params.get('rsi_oversold', 30),
        adx_period=params.get('adx_period', 14),
        adx_threshold=params.get('adx_threshold', 25),
        atr_period=params.get('atr_period', 14),
        atr_stop_loss_mult=params.get('atr_sl_mult', 2.0),
        atr_take_profit_mult=params.get('atr_tp_mult', 3.0),
        min_confidence=params.get('min_confidence', 0.6),
    )
    base_strategy = TrendMomentumStrategy(strategy_config)

    leverage = params.get('leverage', 5)
    strategy = BacktestStrategyAdapter(
        base_strategy, data_handler,
        leverage=float(leverage),
        min_confidence=params.get('min_confidence', 0.6)
    )

    engine = BacktestEngine(
        data_handler=data_handler,
        strategy=strategy,
        initial_capital=initial_capital,
        max_leverage=float(leverage),
        taker_fee=0.0004
    )

    try:
        results = engine.run(verbose=False)
        return {
            'params': params,
            'total_return': results.total_return,
            'sharpe_ratio': results.sharpe_ratio,
            'sortino_ratio': results.sortino_ratio,
            'max_drawdown': results.max_drawdown,
            'win_rate': results.win_rate,
            'profit_factor': results.profit_factor,
            'total_trades': results.total_trades,
            'calmar_ratio': results.calmar_ratio,
            'net_profit': results.net_profit,
            'score': calculate_score(results)
        }
    except Exception as e:
        return {
            'params': params,
            'error': str(e),
            'score': -float('inf')
        }


def calculate_score(results) -> float:
    """Calculate a composite score for strategy ranking."""
    if results.total_trades < 20:
        return -float('inf')  # Not enough trades

    if results.max_drawdown > 0.5:
        return -float('inf')  # Too risky

    # Composite score: Sharpe * (1 - max_drawdown) * sqrt(total_trades) * profit_factor
    score = (
        results.sharpe_ratio *
        (1 - results.max_drawdown) *
        np.sqrt(min(results.total_trades, 200)) *  # Cap trade count benefit
        min(results.profit_factor, 3)  # Cap profit factor benefit
    )

    # Bonus for good win rate
    if results.win_rate > 0.4:
        score *= 1.2

    return score


def generate_parameter_grid() -> List[dict]:
    """Generate parameter combinations to test."""
    param_ranges = {
        'ema_fast': [8, 12, 20],
        'ema_slow': [21, 26, 50],
        'rsi_period': [14, 21],
        'adx_threshold': [20, 25, 30],
        'atr_sl_mult': [1.5, 2.0, 2.5],
        'atr_tp_mult': [2.0, 3.0, 4.0],
        'min_confidence': [0.5, 0.6, 0.7],
        'leverage': [3, 5, 7],
    }

    keys = list(param_ranges.keys())
    combinations = list(itertools.product(*[param_ranges[k] for k in keys]))

    param_list = []
    for combo in combinations:
        params = dict(zip(keys, combo))
        # Skip invalid combinations (ema_fast >= ema_slow)
        if params['ema_fast'] >= params['ema_slow']:
            continue
        # Skip invalid risk/reward (tp should be > sl)
        if params['atr_tp_mult'] <= params['atr_sl_mult']:
            continue
        param_list.append(params)

    return param_list


async def optimize():
    """Run optimization."""
    print("=" * 60)
    print("STRATEGY OPTIMIZATION")
    print("=" * 60)

    # Fetch data
    symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
    print(f"\nFetching data for: {symbols}")

    data = await fetch_data(
        symbols=symbols,
        start_date="2024-01-01",
        end_date="2024-12-31",
        timeframe="4h"
    )

    if not data:
        print("Failed to fetch data!")
        return

    print(f"Loaded {sum(len(df) for df in data.values())} candles total")

    # Generate parameter grid
    param_grid = generate_parameter_grid()
    print(f"\nTesting {len(param_grid)} parameter combinations...")

    # Run backtests
    results = []
    for i, params in enumerate(param_grid):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(param_grid)}")

        result = run_single_backtest(data, params)
        results.append(result)

    # Sort by score
    results = [r for r in results if 'error' not in r]
    results.sort(key=lambda x: x['score'], reverse=True)

    # Print top 10 results
    print("\n" + "=" * 60)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("=" * 60)

    for i, r in enumerate(results[:10]):
        print(f"\n#{i+1} Score: {r['score']:.2f}")
        print(f"   Return: {r['total_return']*100:.1f}% | Sharpe: {r['sharpe_ratio']:.2f} | MaxDD: {r['max_drawdown']*100:.1f}%")
        print(f"   Trades: {r['total_trades']} | WinRate: {r['win_rate']*100:.1f}% | PF: {r['profit_factor']:.2f}")
        print(f"   Params: EMA({r['params']['ema_fast']}/{r['params']['ema_slow']}) ADX>{r['params']['adx_threshold']} "
              f"SL:{r['params']['atr_sl_mult']}x TP:{r['params']['atr_tp_mult']}x "
              f"Conf:{r['params']['min_confidence']} Lev:{r['params']['leverage']}x")

    # Save best parameters
    if results:
        best = results[0]
        print("\n" + "=" * 60)
        print("BEST PARAMETERS")
        print("=" * 60)
        print(f"\nOptimal configuration for TradingConfig:")
        print(f"  ema_fast={best['params']['ema_fast']}")
        print(f"  ema_slow={best['params']['ema_slow']}")
        print(f"  rsi_period={best['params'].get('rsi_period', 14)}")
        print(f"  adx_threshold={best['params']['adx_threshold']}")
        print(f"  stop_loss_atr_mult={best['params']['atr_sl_mult']}")
        print(f"  take_profit_atr_mult={best['params']['atr_tp_mult']}")
        print(f"  min_confidence={best['params']['min_confidence']}")
        print(f"  default_leverage={best['params']['leverage']}")

        # Save to file
        results_df = pd.DataFrame(results)
        results_df.to_csv('optimization_results.csv', index=False)
        print("\nFull results saved to optimization_results.csv")

    return results


if __name__ == "__main__":
    asyncio.run(optimize())
