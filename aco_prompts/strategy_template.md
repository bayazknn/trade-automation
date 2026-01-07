# Freqtrade Strategy Generation Template

This template is used by the ACO optimizer to generate trading strategies via Claude Code.

## Template Variables

The following variables are replaced dynamically:
- `{strategy_name}` - Unique strategy class name (e.g., ACO_1_5)
- `{entry_indicators_json}` - JSON definitions for entry indicators
- `{exit_indicators_json}` - JSON definitions for exit indicators
- `{operators_json}` - Operator reference (lt, gt, crosses_above, etc.)

## Strategy Structure

All generated strategies must follow this structure:

```python
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import qtpylib.indicators as qtpylib

class {strategy_name}(IStrategy):
    """
    ACO-optimized strategy with selected indicators.
    Entry indicators: {entry_count}
    Exit indicators: {exit_count}
    """

    timeframe = '1h'
    stoploss = -0.05

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    # ROI
    minimal_roi = {
        "0": 0.03,
        "60": 0.02,
        "180": 0.01
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate all required indicators
        # ... indicator calculations ...
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # All entry conditions with AND logic
        dataframe.loc[
            (condition1) &
            (condition2) &
            # ... more conditions ...
            (dataframe['volume'] > 0),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # All exit conditions with AND logic
        dataframe.loc[
            (condition1) &
            (condition2) &
            # ... more conditions ...
            (dataframe['volume'] > 0),
            'exit_long'] = 1
        return dataframe
```

## Indicator Implementation Guide

### Threshold Signals (RSI, STOCH, CCI, etc.)

Entry example (RSI < 30):
```python
dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
# Condition: dataframe['rsi'] < 30
```

Exit example (RSI > 70):
```python
# Condition: dataframe['rsi'] > 70
```

### Crossover Signals (MACD, EMA, etc.)

Entry example (MACD crosses above signal):
```python
macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
dataframe['macd'] = macd['macd']
dataframe['macd_signal'] = macd['macdsignal']
# Condition: qtpylib.crossed_above(dataframe['macd'], dataframe['macd_signal'])
```

### Dual Moving Average Crossover

Entry example (EMA 9 crosses above EMA 21):
```python
dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
# Condition: qtpylib.crossed_above(dataframe['ema_9'], dataframe['ema_21'])
```

### Bollinger Bands with Factor

Entry example (close < lower band * 1.02):
```python
bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
dataframe['bb_lower'] = bollinger['lowerband']
dataframe['bb_upper'] = bollinger['upperband']
# Entry: dataframe['close'] < dataframe['bb_lower'] * 1.02
# Exit: dataframe['close'] > dataframe['bb_upper'] * 0.98
```

### Volume Indicators with SMA

Entry example (OBV crosses above its SMA):
```python
dataframe['obv'] = ta.OBV(dataframe)
dataframe['obv_sma'] = ta.SMA(dataframe['obv'], timeperiod=20)
# Condition: qtpylib.crossed_above(dataframe['obv'], dataframe['obv_sma'])
```

## Expected Output Format

After generating the strategy and running backtest, return JSON:

```json
{
    "strategy_name": "ACO_1_5",
    "total_profit": 123.45,
    "total_trades": 42,
    "win_rate": 0.65,
    "max_drawdown": 15.5,
    "sharpe_ratio": 1.23,
    "status": "success"
}
```

If error occurs:
```json
{
    "strategy_name": "ACO_1_5",
    "total_profit": 0,
    "total_trades": 0,
    "status": "error",
    "error_message": "Description of what went wrong"
}
```

## Important Rules

1. **Always use AND logic** - All conditions must be true for entry/exit
2. **Include volume check** - `dataframe['volume'] > 0` in all conditions
3. **Handle NaN values** - Indicators have NaN at start, use `.fillna(0)` or handle in conditions
4. **Use exact parameter values** from the JSON definitions
5. **Follow signal_type exactly** - threshold vs crossover
6. **Run backtest** with: `freqtrade backtesting --strategy {name} --config user_data/config.json --timerange 20251001- --export none`
7. **Return only JSON** - No additional text or explanation
