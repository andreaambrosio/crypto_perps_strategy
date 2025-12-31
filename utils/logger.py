"""
Logging configuration for the trading system
"""
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with console and optional file output"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


# Default trading logger
def get_trading_logger() -> logging.Logger:
    """Get the main trading logger"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
    return setup_logger("trading_bot", str(log_file))


# Backtest logger
def get_backtest_logger() -> logging.Logger:
    """Get the backtest logger"""
    return setup_logger("backtest", level=logging.INFO)
