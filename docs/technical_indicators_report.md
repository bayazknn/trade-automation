# Technical Indicators Report for Crypto Trading

## Overview

This report documents all TA-Lib technical indicators and their interpretation for crypto trading signal generation. The corresponding JSON configuration file (`technical_indicators_config.json`) can be used to programmatically create freqtrade strategy classes.

## Indicator Categories

### 1. Momentum Indicators (28 indicators)

| Indicator | Description | Entry Signal (Long) | Exit Signal (Long) |
|-----------|-------------|---------------------|-------------------|
| **RSI** | Relative Strength Index | RSI < 30 (oversold) | RSI > 70 (overbought) |
| **MACD** | Moving Avg Convergence/Divergence | MACD crosses above Signal | MACD crosses below Signal |
| **STOCH** | Stochastic Oscillator | %K < 20 & crosses %D | %K > 80 |
| **STOCHRSI** | Stochastic RSI | StochRSI < 20 | StochRSI > 80 |
| **CCI** | Commodity Channel Index | CCI < -100 | CCI > 100 |
| **MFI** | Money Flow Index | MFI < 20 | MFI > 80 |
| **WILLR** | Williams %R | %R < -80 | %R > -20 |
| **CMO** | Chande Momentum Oscillator | CMO < -50 | CMO > 50 |
| **ADX** | Average Directional Index | ADX > 25 (with +DI > -DI) | ADX < 20 |
| **PLUS_DI** | Plus Directional Indicator | +DI crosses above -DI | +DI crosses below -DI |
| **MINUS_DI** | Minus Directional Indicator | -DI crosses above +DI (short) | -DI crosses below +DI |
| **MOM** | Momentum | MOM crosses above 0 | MOM crosses below 0 |
| **ROC** | Rate of Change | ROC crosses above 0 | ROC crosses below 0 |
| **TRIX** | Triple Exponential Average | TRIX crosses above 0 | TRIX crosses below 0 |
| **ULTOSC** | Ultimate Oscillator | ULTOSC < 30 | ULTOSC > 70 |
| **APO** | Absolute Price Oscillator | APO crosses above 0 | APO crosses below 0 |
| **PPO** | Percentage Price Oscillator | PPO crosses above 0 | PPO crosses below 0 |
| **BOP** | Balance of Power | BOP crosses above 0 | BOP crosses below 0 |
| **AROON** | Aroon Indicator | AroonUp crosses above AroonDown | AroonDown crosses above AroonUp |
| **AROONOSC** | Aroon Oscillator | AroonOsc crosses above 0 | AroonOsc crosses below 0 |

### 2. Overlap Studies (15 indicators)

| Indicator | Description | Entry Signal (Long) | Exit Signal (Long) |
|-----------|-------------|---------------------|-------------------|
| **SMA** | Simple Moving Average | Price crosses above SMA | Price crosses below SMA |
| **EMA** | Exponential Moving Average | Fast EMA crosses above Slow EMA | Fast EMA crosses below Slow EMA |
| **DEMA** | Double EMA | Price crosses above DEMA | Price crosses below DEMA |
| **TEMA** | Triple EMA | Price crosses above TEMA | Price crosses below TEMA |
| **KAMA** | Kaufman Adaptive MA | Price crosses above KAMA | Price crosses below KAMA |
| **BBANDS** | Bollinger Bands | Price < Lower Band | Price > Upper Band |
| **SAR** | Parabolic SAR | Price crosses above SAR | Price crosses below SAR |

### 3. Volume Indicators (3 indicators)

| Indicator | Description | Entry Signal (Long) | Exit Signal (Long) |
|-----------|-------------|---------------------|-------------------|
| **OBV** | On Balance Volume | OBV crosses above its SMA | OBV crosses below its SMA |
| **AD** | Chaikin A/D Line | A/D rising | A/D falling |
| **ADOSC** | Chaikin A/D Oscillator | ADOSC crosses above 0 | ADOSC crosses below 0 |

### 4. Volatility Indicators (3 indicators)

| Indicator | Description | Usage |
|-----------|-------------|-------|
| **ATR** | Average True Range | Stop loss = Entry - (ATR * 2.0-2.5) |
| **NATR** | Normalized ATR | Filter: Trade when NATR > 2-3% |
| **TRANGE** | True Range | Volatility measurement |

## Signal Types

### 1. Threshold Signals
Compare indicator value against a constant:
- RSI < 30 (oversold)
- RSI > 70 (overbought)
- ADX > 25 (strong trend)

### 2. Crossover Signals
Indicator crosses above/below another value:
- MACD crosses above Signal line
- Price crosses above EMA
- Fast EMA crosses above Slow EMA

### 3. Divergence Signals
Price and indicator moving in opposite directions:
- Bullish: Price makes lower low, indicator makes higher low
- Bearish: Price makes higher high, indicator makes lower high

## Recommended Constants for Crypto Trading

### Overbought/Oversold Thresholds

| Indicator | Oversold (Buy) | Overbought (Sell) |
|-----------|----------------|-------------------|
| RSI | 30 | 70 |
| Stochastic | 20 | 80 |
| StochRSI | 20 | 80 |
| MFI | 20 | 80 |
| Williams %R | -80 | -20 |
| CCI | -100 | 100 |
| CMO | -50 | 50 |
| Ultimate Osc | 30 | 70 |

### Moving Average Periods

| Type | Fast | Slow | Signal |
|------|------|------|--------|
| EMA Crossover | 9-12 | 21-26 | - |
| MACD | 12 | 26 | 9 |
| Bollinger Bands | - | 20 | - |
| RSI | 14 | - | - |

### ADX Trend Strength

| ADX Value | Interpretation |
|-----------|----------------|
| 0-20 | No trend / Weak |
| 20-25 | Developing trend |
| 25-50 | Strong trend |
| 50-75 | Very strong trend |
| 75-100 | Extremely strong |

## Candlestick Patterns

### High Reliability Bullish Patterns
- Hammer
- Morning Star
- Morning Doji Star
- Bullish Engulfing
- Piercing Pattern
- Three White Soldiers

### High Reliability Bearish Patterns
- Shooting Star
- Evening Star
- Evening Doji Star
- Bearish Engulfing
- Dark Cloud Cover
- Three Black Crows

### Pattern Signal Values
- `200`: Strong bullish
- `100`: Bullish
- `0`: Neutral/No pattern
- `-100`: Bearish
- `-200`: Strong bearish

## JSON Configuration Structure

```json
{
  "indicator_name": {
    "name": "Full Name",
    "category": "momentum|overlap_studies|volume|volatility",
    "talib_function": "ta.INDICATOR",
    "params": {
      "param_name": {
        "default": value,
        "range": [min, max],
        "type": "int|float"
      }
    },
    "outputs": ["output1", "output2"],
    "value_range": [min, max] or null,
    "entry": {
      "long": {
        "signal_type": "threshold|crossover|divergence",
        "rules": [{
          "name": "rule_name",
          "condition": {
            "left": "indicator_output",
            "operator": "lt|gt|crosses_above|crosses_below",
            "right": "constant|other_indicator"
          },
          "constant": {
            "value": number,
            "crypto_recommended": number,
            "range": [min, max]
          }
        }]
      }
    },
    "exit": { ... }
  }
}
```

## Best Practices for Crypto Trading

### 1. Combine Multiple Indicators
- Use trend indicator (EMA, ADX) + momentum oscillator (RSI, MACD)
- Confirm with volume (OBV, MFI)

### 2. Adjust for Crypto Volatility
- Crypto markets are more volatile than traditional markets
- Consider wider stops (ATR * 2.5 instead of 2.0)
- RSI may stay overbought/oversold longer

### 3. Timeframe Considerations
- 1h timeframe: Good for swing trading
- 4h timeframe: Better signal quality
- 1d timeframe: Trend following

### 4. Avoid Common Pitfalls
- Don't use too many indicators (3-4 max)
- Avoid conflicting signals
- Test with backtesting before live trading

## Sources

- [TA-Lib Functions](https://ta-lib.github.io/ta-lib-python/funcs.html)
- [StockCharts ChartSchool](https://chartschool.stockcharts.com/)
- [Gate.io Crypto Wiki](https://web3.gate.com/en/crypto-wiki/)
- [Morpher Trading Guides](https://www.morpher.com/blog/)
- [TradingView Indicators](https://www.tradingview.com/scripts/)
