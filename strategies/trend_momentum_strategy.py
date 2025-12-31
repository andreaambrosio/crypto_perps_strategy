"""
Trend Momentum Strategy for Crypto Perpetual Futures

This strategy combines multiple technical indicators for high-probability trend trading:
- EMA crossover (12/26) for trend direction
- RSI for momentum confirmation (avoid overbought/oversold entries)
- ADX for trend strength filtering
- ATR for dynamic stop loss and take profit
- Volume confirmation (above average volume)
- Supertrend as additional trend filter

Designed for perpetual futures with support for both long and short positions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass

from .base_strategy import (
    BaseStrategy,
    Signal,
    SignalType,
    TimeFrame,
    IndicatorValues
)


@dataclass
class StrategyConfig:
    """Configuration parameters for the Trend Momentum Strategy."""
    # EMA parameters
    ema_fast_period: int = 12
    ema_slow_period: int = 26

    # RSI parameters
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_neutral_upper: float = 60.0
    rsi_neutral_lower: float = 40.0

    # ADX parameters
    adx_period: int = 14
    adx_threshold: float = 25.0  # Minimum ADX for strong trend
    adx_strong_trend: float = 40.0  # ADX value indicating very strong trend

    # ATR parameters
    atr_period: int = 14
    atr_stop_loss_mult: float = 2.0
    atr_take_profit_mult: float = 3.0

    # Volume parameters
    volume_ma_period: int = 20
    volume_threshold: float = 1.2  # Volume must be 1.2x average

    # Supertrend parameters
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0

    # Signal thresholds
    min_confidence: float = 0.6  # Minimum confidence to generate signal
    strong_signal_confidence: float = 0.8

    # Timeframes
    primary_timeframe: TimeFrame = TimeFrame.H4
    secondary_timeframe: TimeFrame = TimeFrame.H1


class TrendMomentumStrategy(BaseStrategy):
    """
    A comprehensive trend-following strategy with momentum confirmation.

    This strategy is designed for crypto perpetual futures trading, combining
    multiple technical indicators to identify high-probability trade setups
    in the direction of the dominant trend.

    Entry Conditions:
    - EMA crossover confirms trend direction
    - ADX above threshold confirms trend strength
    - RSI in favorable zone (not overbought for longs, not oversold for shorts)
    - Volume above average confirms participation
    - Supertrend aligns with trade direction

    Exit Conditions:
    - Opposite EMA crossover
    - RSI reaches extreme levels
    - Supertrend reversal
    - Stop loss or take profit hit
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        name: str = "TrendMomentum"
    ):
        """
        Initialize the Trend Momentum Strategy.

        Args:
            config: Strategy configuration parameters
            name: Strategy identifier
        """
        super().__init__(name=name)
        self.strategy_config = config or StrategyConfig()
        self._position_state: Dict[str, Dict] = {}  # Track position state per symbol

    def calculate_indicators(self, df: pd.DataFrame, timeframe: TimeFrame) -> pd.DataFrame:
        """
        Calculate all required indicators for the strategy.

        Args:
            df: OHLCV DataFrame
            timeframe: The timeframe of the data

        Returns:
            DataFrame with additional indicator columns
        """
        self.validate_dataframe(df)
        df = df.copy()

        # EMA calculations
        df['ema_fast'] = self._calculate_ema(df['close'], self.strategy_config.ema_fast_period)
        df['ema_slow'] = self._calculate_ema(df['close'], self.strategy_config.ema_slow_period)
        df['ema_diff'] = df['ema_fast'] - df['ema_slow']
        df['ema_cross'] = np.sign(df['ema_diff']).diff()

        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.strategy_config.rsi_period)

        # ATR
        df['atr'] = self._calculate_atr(df, self.strategy_config.atr_period)

        # ADX with +DI and -DI
        adx_data = self._calculate_adx(df, self.strategy_config.adx_period)
        df['adx'] = adx_data['adx']
        df['plus_di'] = adx_data['plus_di']
        df['minus_di'] = adx_data['minus_di']

        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=self.strategy_config.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Supertrend
        supertrend_data = self._calculate_supertrend(
            df,
            self.strategy_config.supertrend_period,
            self.strategy_config.supertrend_multiplier
        )
        df['supertrend'] = supertrend_data['supertrend']
        df['supertrend_direction'] = supertrend_data['direction']

        # Trend classification
        df['trend'] = self._classify_trend(df)

        # Momentum score
        df['momentum_score'] = self._calculate_momentum_score(df)

        return df

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: TimeFrame,
        current_position: Optional[str] = None
    ) -> Signal:
        """
        Generate a trading signal based on current market conditions.

        Args:
            df: DataFrame with OHLCV data and calculated indicators
            symbol: Trading pair symbol
            timeframe: The timeframe of the analysis
            current_position: Current position state ("LONG", "SHORT", or None)

        Returns:
            Signal object with trading recommendation
        """
        if df.empty or len(df) < self.strategy_config.ema_slow_period:
            return Signal(
                signal_type=SignalType.NO_SIGNAL,
                symbol=symbol,
                timestamp=datetime.now(),
                timeframe=timeframe,
                metadata={"reason": "Insufficient data"}
            )

        # Get latest indicator values
        indicators = self.get_latest_indicators(df)
        current_price = indicators.close

        # Check for exit conditions first if in position
        if current_position:
            entry_price = self._position_state.get(symbol, {}).get('entry_price', current_price)
            should_exit, exit_reason = self.get_exit_conditions(
                indicators, current_position, entry_price
            )
            if should_exit:
                signal_type = (SignalType.CLOSE_LONG if current_position == "LONG"
                              else SignalType.CLOSE_SHORT)
                return Signal(
                    signal_type=signal_type,
                    symbol=symbol,
                    timestamp=indicators.timestamp,
                    confidence=0.9,
                    entry_price=current_price,
                    timeframe=timeframe,
                    metadata={"exit_reason": exit_reason}
                )

        # Check for entry conditions
        long_condition, short_condition, confidence = self.get_entry_conditions(indicators)

        # Determine signal type
        if long_condition and confidence >= self.strategy_config.min_confidence:
            signal_type = SignalType.LONG
        elif short_condition and confidence >= self.strategy_config.min_confidence:
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.HOLD if current_position else SignalType.NO_SIGNAL

        # Calculate stop loss and take profit for entry signals
        stop_loss = None
        take_profit = None

        if signal_type in (SignalType.LONG, SignalType.SHORT):
            atr = indicators.atr or self._estimate_atr(df)
            stop_loss = self.calculate_stop_loss(
                current_price, atr, signal_type,
                self.strategy_config.atr_stop_loss_mult
            )
            take_profit = self.calculate_take_profit(
                current_price, atr, signal_type,
                self.strategy_config.atr_take_profit_mult
            )

            # Track position state
            self._position_state[symbol] = {
                'entry_price': current_price,
                'entry_time': indicators.timestamp,
                'signal_type': signal_type.value
            }

        # Build metadata
        metadata = self._build_signal_metadata(indicators, df)

        return Signal(
            signal_type=signal_type,
            symbol=symbol,
            timestamp=indicators.timestamp,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe=timeframe,
            metadata=metadata
        )

    def get_entry_conditions(self, indicators: IndicatorValues) -> Tuple[bool, bool, float]:
        """
        Evaluate entry conditions based on indicator values.

        Args:
            indicators: Current indicator values

        Returns:
            Tuple of (long_condition, short_condition, confidence_score)
        """
        # Initialize conditions
        long_conditions = {}
        short_conditions = {}

        # 1. EMA Trend Direction (weight: 0.25)
        if indicators.ema_fast and indicators.ema_slow:
            ema_bullish = indicators.ema_fast > indicators.ema_slow
            ema_bearish = indicators.ema_fast < indicators.ema_slow
            long_conditions['ema_trend'] = (ema_bullish, 0.25)
            short_conditions['ema_trend'] = (ema_bearish, 0.25)
        else:
            long_conditions['ema_trend'] = (False, 0.25)
            short_conditions['ema_trend'] = (False, 0.25)

        # 2. ADX Trend Strength (weight: 0.20)
        if indicators.adx:
            trend_strong = indicators.adx >= self.strategy_config.adx_threshold
            long_conditions['adx_strength'] = (trend_strong, 0.20)
            short_conditions['adx_strength'] = (trend_strong, 0.20)
        else:
            long_conditions['adx_strength'] = (False, 0.20)
            short_conditions['adx_strength'] = (False, 0.20)

        # 3. DI Confirmation (weight: 0.10)
        if indicators.plus_di and indicators.minus_di:
            di_bullish = indicators.plus_di > indicators.minus_di
            di_bearish = indicators.minus_di > indicators.plus_di
            long_conditions['di_direction'] = (di_bullish, 0.10)
            short_conditions['di_direction'] = (di_bearish, 0.10)
        else:
            long_conditions['di_direction'] = (False, 0.10)
            short_conditions['di_direction'] = (False, 0.10)

        # 4. RSI Momentum (weight: 0.15)
        if indicators.rsi:
            rsi = indicators.rsi
            # For longs: RSI should not be overbought, ideal if coming from oversold
            rsi_favorable_long = (self.strategy_config.rsi_neutral_lower <= rsi <=
                                  self.strategy_config.rsi_overbought)
            # For shorts: RSI should not be oversold, ideal if coming from overbought
            rsi_favorable_short = (self.strategy_config.rsi_oversold <= rsi <=
                                   self.strategy_config.rsi_neutral_upper)
            long_conditions['rsi_momentum'] = (rsi_favorable_long, 0.15)
            short_conditions['rsi_momentum'] = (rsi_favorable_short, 0.15)
        else:
            long_conditions['rsi_momentum'] = (False, 0.15)
            short_conditions['rsi_momentum'] = (False, 0.15)

        # 5. Volume Confirmation (weight: 0.15)
        if indicators.volume_ratio:
            volume_confirmed = indicators.volume_ratio >= self.strategy_config.volume_threshold
            long_conditions['volume'] = (volume_confirmed, 0.15)
            short_conditions['volume'] = (volume_confirmed, 0.15)
        else:
            long_conditions['volume'] = (False, 0.15)
            short_conditions['volume'] = (False, 0.15)

        # 6. Supertrend Alignment (weight: 0.15)
        if indicators.supertrend_direction is not None:
            st_bullish = indicators.supertrend_direction == 1
            st_bearish = indicators.supertrend_direction == -1
            long_conditions['supertrend'] = (st_bullish, 0.15)
            short_conditions['supertrend'] = (st_bearish, 0.15)
        else:
            long_conditions['supertrend'] = (False, 0.15)
            short_conditions['supertrend'] = (False, 0.15)

        # Calculate confidence scores
        long_confidence = self.calculate_confidence_score(long_conditions)
        short_confidence = self.calculate_confidence_score(short_conditions)

        # Determine primary direction based on higher confidence
        long_entry = (long_confidence >= self.strategy_config.min_confidence and
                      long_confidence > short_confidence and
                      all(cond for cond, _ in [long_conditions['ema_trend'],
                                               long_conditions['adx_strength']]))

        short_entry = (short_confidence >= self.strategy_config.min_confidence and
                       short_confidence > long_confidence and
                       all(cond for cond, _ in [short_conditions['ema_trend'],
                                                short_conditions['adx_strength']]))

        primary_confidence = long_confidence if long_entry else (
            short_confidence if short_entry else max(long_confidence, short_confidence)
        )

        return long_entry, short_entry, primary_confidence

    def get_exit_conditions(
        self,
        indicators: IndicatorValues,
        position_type: str,
        entry_price: float
    ) -> Tuple[bool, str]:
        """
        Evaluate exit conditions for an existing position.

        Args:
            indicators: Current indicator values
            position_type: "LONG" or "SHORT"
            entry_price: The entry price of the position

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        current_price = indicators.close or entry_price

        # 1. EMA Crossover Exit
        if indicators.ema_fast and indicators.ema_slow:
            if position_type == "LONG" and indicators.ema_fast < indicators.ema_slow:
                return True, "EMA bearish crossover"
            elif position_type == "SHORT" and indicators.ema_fast > indicators.ema_slow:
                return True, "EMA bullish crossover"

        # 2. Supertrend Reversal
        if indicators.supertrend_direction is not None:
            if position_type == "LONG" and indicators.supertrend_direction == -1:
                return True, "Supertrend bearish reversal"
            elif position_type == "SHORT" and indicators.supertrend_direction == 1:
                return True, "Supertrend bullish reversal"

        # 3. RSI Extreme Levels
        if indicators.rsi:
            if position_type == "LONG" and indicators.rsi >= self.strategy_config.rsi_overbought:
                return True, "RSI overbought"
            elif position_type == "SHORT" and indicators.rsi <= self.strategy_config.rsi_oversold:
                return True, "RSI oversold"

        # 4. ADX Trend Weakening (optional exit)
        if indicators.adx and indicators.adx < self.strategy_config.adx_threshold * 0.8:
            # Only exit if also losing money
            if position_type == "LONG" and current_price < entry_price:
                return True, "Trend weakening with loss"
            elif position_type == "SHORT" and current_price > entry_price:
                return True, "Trend weakening with loss"

        return False, ""

    def analyze_multi_timeframe(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        symbol: str,
        current_position: Optional[str] = None
    ) -> Signal:
        """
        Perform multi-timeframe analysis for more accurate signals.

        Uses primary timeframe (4H) for trend direction and
        secondary timeframe (1H) for entry timing.

        Args:
            primary_df: DataFrame for primary (higher) timeframe
            secondary_df: DataFrame for secondary (lower) timeframe
            symbol: Trading pair symbol
            current_position: Current position state

        Returns:
            Combined Signal from multi-timeframe analysis
        """
        # Calculate indicators for both timeframes
        primary_with_indicators = self.calculate_indicators(
            primary_df, self.strategy_config.primary_timeframe
        )
        secondary_with_indicators = self.calculate_indicators(
            secondary_df, self.strategy_config.secondary_timeframe
        )

        # Get signals from both timeframes
        primary_signal = self.generate_signal(
            primary_with_indicators,
            symbol,
            self.strategy_config.primary_timeframe,
            current_position
        )
        secondary_signal = self.generate_signal(
            secondary_with_indicators,
            symbol,
            self.strategy_config.secondary_timeframe,
            current_position
        )

        # Combine signals with primary taking precedence
        return self._combine_signals(primary_signal, secondary_signal)

    def get_signal_summary(self, signal: Signal) -> Dict[str, Any]:
        """
        Get a human-readable summary of the signal.

        Args:
            signal: The trading signal

        Returns:
            Dictionary with formatted signal information
        """
        summary = {
            "action": signal.signal_type.value,
            "symbol": signal.symbol,
            "confidence": f"{signal.confidence:.1%}",
            "confidence_level": self._get_confidence_level(signal.confidence),
            "timestamp": signal.timestamp.isoformat(),
            "timeframe": signal.timeframe.value
        }

        if signal.is_entry_signal:
            summary.update({
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "risk_reward": f"{signal.risk_reward_ratio:.2f}" if signal.risk_reward_ratio else "N/A"
            })

        if signal.metadata:
            summary["analysis"] = {
                k: v for k, v in signal.metadata.items()
                if k not in ['raw_indicators']
            }

        return summary

    # ========================
    # Private Helper Methods
    # ========================

    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> Dict[str, pd.Series]:
        """Calculate Average Directional Index with +DI and -DI."""
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Calculate True Range
        tr = self._calculate_atr(df, 1) * 1  # Get single period TR

        # Smooth the values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
        adx = dx.rolling(window=period).mean()

        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }

    def _calculate_supertrend(
        self,
        df: pd.DataFrame,
        period: int,
        multiplier: float
    ) -> Dict[str, pd.Series]:
        """Calculate Supertrend indicator."""
        hl2 = (df['high'] + df['low']) / 2
        atr = self._calculate_atr(df, period)

        # Calculate basic upper and lower bands
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        # Initialize supertrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)

        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = -1

        for i in range(1, len(df)):
            if df['close'].iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif df['close'].iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]

                if direction.iloc[i] == 1 and lower_band.iloc[i] > supertrend.iloc[i]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                elif direction.iloc[i] == -1 and upper_band.iloc[i] < supertrend.iloc[i]:
                    supertrend.iloc[i] = upper_band.iloc[i]

        return {
            'supertrend': supertrend,
            'direction': direction
        }

    def _classify_trend(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify the current trend state.

        Returns:
            Series with trend classification: 'strong_up', 'up', 'neutral', 'down', 'strong_down'
        """
        trend = pd.Series(index=df.index, dtype=str)

        for i in range(len(df)):
            ema_diff = df['ema_fast'].iloc[i] - df['ema_slow'].iloc[i] if not pd.isna(df['ema_fast'].iloc[i]) else 0
            adx = df['adx'].iloc[i] if not pd.isna(df['adx'].iloc[i]) else 0
            plus_di = df['plus_di'].iloc[i] if not pd.isna(df['plus_di'].iloc[i]) else 0
            minus_di = df['minus_di'].iloc[i] if not pd.isna(df['minus_di'].iloc[i]) else 0

            if adx >= self.strategy_config.adx_strong_trend:
                if ema_diff > 0 and plus_di > minus_di:
                    trend.iloc[i] = 'strong_up'
                elif ema_diff < 0 and minus_di > plus_di:
                    trend.iloc[i] = 'strong_down'
                else:
                    trend.iloc[i] = 'neutral'
            elif adx >= self.strategy_config.adx_threshold:
                if ema_diff > 0:
                    trend.iloc[i] = 'up'
                elif ema_diff < 0:
                    trend.iloc[i] = 'down'
                else:
                    trend.iloc[i] = 'neutral'
            else:
                trend.iloc[i] = 'neutral'

        return trend

    def _calculate_momentum_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate a composite momentum score.

        Returns:
            Series with momentum score from -100 to 100
        """
        score = pd.Series(index=df.index, dtype=float)

        for i in range(len(df)):
            components = []

            # RSI component (-50 to 50)
            rsi = df['rsi'].iloc[i] if not pd.isna(df['rsi'].iloc[i]) else 50
            rsi_score = (rsi - 50)
            components.append(rsi_score)

            # EMA trend component (-30 to 30)
            if not pd.isna(df['ema_fast'].iloc[i]) and not pd.isna(df['ema_slow'].iloc[i]):
                ema_pct = ((df['ema_fast'].iloc[i] - df['ema_slow'].iloc[i]) /
                          df['ema_slow'].iloc[i] * 100)
                ema_score = max(min(ema_pct * 10, 30), -30)
                components.append(ema_score)

            # Supertrend component (-20 to 20)
            if not pd.isna(df['supertrend_direction'].iloc[i]):
                st_score = df['supertrend_direction'].iloc[i] * 20
                components.append(st_score)

            score.iloc[i] = sum(components)

        return score

    def _estimate_atr(self, df: pd.DataFrame) -> float:
        """Estimate ATR from recent price action if not calculated."""
        if len(df) < 2:
            return df['close'].iloc[-1] * 0.02  # Default 2% of price
        recent_range = (df['high'].iloc[-14:] - df['low'].iloc[-14:]).mean()
        return recent_range

    def _build_signal_metadata(
        self,
        indicators: IndicatorValues,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Build comprehensive metadata for the signal."""
        return {
            "trend": df['trend'].iloc[-1] if 'trend' in df.columns else None,
            "momentum_score": df['momentum_score'].iloc[-1] if 'momentum_score' in df.columns else None,
            "indicators": {
                "ema_fast": round(indicators.ema_fast, 2) if indicators.ema_fast else None,
                "ema_slow": round(indicators.ema_slow, 2) if indicators.ema_slow else None,
                "rsi": round(indicators.rsi, 2) if indicators.rsi else None,
                "adx": round(indicators.adx, 2) if indicators.adx else None,
                "atr": round(indicators.atr, 4) if indicators.atr else None,
                "volume_ratio": round(indicators.volume_ratio, 2) if indicators.volume_ratio else None,
                "supertrend_direction": indicators.supertrend_direction
            },
            "conditions": {
                "trend_confirmed": (indicators.adx or 0) >= self.strategy_config.adx_threshold,
                "volume_confirmed": (indicators.volume_ratio or 0) >= self.strategy_config.volume_threshold,
                "rsi_favorable": (self.strategy_config.rsi_oversold <
                                 (indicators.rsi or 50) <
                                 self.strategy_config.rsi_overbought)
            }
        }

    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to descriptive level."""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.8:
            return "High"
        elif confidence >= 0.7:
            return "Moderate"
        elif confidence >= 0.6:
            return "Low"
        else:
            return "Very Low"

    def __repr__(self) -> str:
        return (f"TrendMomentumStrategy(name='{self.name}', "
                f"ema={self.strategy_config.ema_fast_period}/{self.strategy_config.ema_slow_period}, "
                f"adx_threshold={self.strategy_config.adx_threshold})")
