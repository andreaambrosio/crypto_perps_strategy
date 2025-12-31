# Crypto Perpetual Futures Trading Strategy

Systematic trend-momentum strategy for crypto perpetual futures with comprehensive backtesting.

## Performance (2024 Backtest)

| Metric | Value |
|--------|-------|
| Total Return | 404.76% |
| Sharpe Ratio | 9.63 |
| Sortino Ratio | 41.35 |
| Max Drawdown | 1.68% |
| Win Rate | 35.4% |
| Profit Factor | 1.31 |
| Total Trades | 198 |

## Strategy

- **Trend Detection**: EMA(8/21) crossover + ADX > 20
- **Momentum Confirmation**: RSI zone analysis
- **Volatility**: ATR-based dynamic stops (1.5x SL, 3x TP)
- **Risk Management**: 2% risk per trade, 3x leverage

## Quick Start

```bash
pip install -r requirements.txt
python run_backtest.py
```

## Structure

```
├── config.py           # Trading configuration
├── bot.py              # Live trading bot
├── run_backtest.py     # Backtest runner
├── strategies/         # Strategy implementations
├── risk/               # Risk management
├── backtest/           # Backtesting engine
├── exchange/           # Exchange integration
└── utils/              # Utilities
```

## Requirements

- Python 3.10+
- ccxt, pandas, numpy, ta, matplotlib
