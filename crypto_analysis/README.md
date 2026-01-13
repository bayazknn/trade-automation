# Crypto Analysis Package

A Python package for cryptocurrency data analysis and signal generation, designed for use with Freqtrade.

## Features

- **Signal Population**: Generate entry/exit signals based on price percentage changes
- **Indicator Optimization**: Optimize technical indicator parameters using grid search and hyperopt
- **LSTM Signal Prediction**: PyTorch LSTM model for predicting entry/hold/exit signals
- **33 Technical Indicators**: RSI, MACD, STOCH, BBANDS, EMA, SMA, and more
- **Parallel Processing**: Multi-threaded optimization with configurable worker count
- **High Performance**: NumPy-vectorized operations for fast backtesting

## Installation

The package is part of the Freqtrade user_data directory. Ensure dependencies are installed:

```bash
pip install pandas numpy ta-lib optuna torch scikit-learn
```

## Quick Start

```python
from crypto_analysis import SignalPopulator, DatasetBuilder

# 1. Generate target signals based on price changes
populator = SignalPopulator(
    data_dir="path/to/data/binance",
    period_hours=4
)
df = populator.populate_signals(symbol="BTC", threshold_pct=3.0)

# 2. Build optimized dataset with indicator signals
builder = DatasetBuilder(
    data_dir="path/to/data/binance",
    n_workers=4  # Parallel processing
)

result_df = builder.build(
    symbol="BTC",
    threshold_pct=3.0,
    indicators=["RSI", "MACD", "BBANDS"],  # or None for all
    grid_search=True,
    hyperopt=False,
    verbose=True
)

# 3. Train LSTM model to predict signals
from crypto_analysis import Trainer, TrainingConfig

trainer = Trainer(result_df, config=TrainingConfig(epochs=100))
history = trainer.train()

# Evaluate model performance
metrics = trainer.evaluate(trainer.test_dataset)
```

## Modules

### SignalPopulator

Scans OHLCV data in configurable periods and tags entry/exit signals based on price threshold.

```python
from crypto_analysis import SignalPopulator

populator = SignalPopulator(data_dir="data/binance", period_hours=4)
df = populator.populate_signals("ETH", threshold_pct=5.0)

# Output columns: date, open, high, low, close, volume, signal, signal_pct_change, period_id
```

### GridSearchOptimizer

Finds indicator parameters that best match SignalPopulator signals.

```python
from crypto_analysis import GridSearchOptimizer, SignalMatcher
from crypto_analysis.indicator_optimizer.indicators import get_indicator

optimizer = GridSearchOptimizer(
    step_sizes={"int": 1, "float": 0.1},
)

indicator = get_indicator("RSI")()
result = optimizer.optimize(
    indicator=indicator,
    df=ohlcv_df,
    target_df=signals_df,
    max_combinations=3000,
    verbose=True
)

print(f"Best params: {result.best_params}")
print(f"Score: {result.score}")
```

### HyperoptOptimizer

Freqtrade-style optimization using Optuna to maximize trading performance.

```python
from crypto_analysis import HyperoptOptimizer

optimizer = HyperoptOptimizer(loss_function="profit")  # or "sharpe", "win_rate"
result = optimizer.optimize(
    indicator=indicator,
    df=ohlcv_df,
    n_trials=100,
    verbose=True
)
```

### DatasetBuilder

Facade that orchestrates all components to build a complete dataset.

```python
from crypto_analysis import DatasetBuilder

builder = DatasetBuilder(
    data_dir="data/binance",
    period_hours=4,
    n_workers=4,
    signal_shift=4  # Shift indicator signals to predict t+4 (default: 4)
)

# Build with all indicators
df = builder.build(
    symbol="BTC",
    threshold_pct=3.0,
    grid_search=True,
    hyperopt=True,
    hyperopt_trials=100,
    verbose=True
)

# Get optimization results
results = builder.get_optimization_results("BTC")
summary = builder.get_best_params_summary("BTC")
```

#### Signal Shift

The `signal_shift` parameter shifts indicator signals forward by N steps, making them predictive of future targets. This aligns with the LSTM model's `target_shift` for consistent future prediction.

- `signal_shift=4` (default): Indicator signal at time t predicts target at t+4
- `signal_shift=0`: No shifting (indicator matches current target)

**Note**: The first `signal_shift` rows are dropped from the output DataFrame since they don't have valid shifted indicator signals. For example, with `signal_shift=4`, if the input has 10,000 rows, the output will have 9,996 rows.

### LSTM Signal Prediction

PyTorch-based LSTM model for predicting trading signals from DatasetBuilder output.

#### Training

```python
from crypto_analysis import Trainer, TrainingConfig

# Load DatasetBuilder output
df = builder.build(symbol="BTC", threshold_pct=3.0)

# Configure training
config = TrainingConfig(
    epochs=100,
    batch_size=64,
    learning_rate=1e-3,
    early_stopping_patience=20,
    checkpoint_dir="checkpoints/"
)

# Train model
trainer = Trainer(df, config=config)
history = trainer.train()

# Evaluate on test set
metrics = trainer.evaluate(trainer.test_dataset)
# Output: accuracy, precision, recall, F1 for entry/hold/exit classes
```

#### Inference

```python
from crypto_analysis import Predictor

# Load trained model
predictor = Predictor.from_checkpoint(
    checkpoint_path="checkpoints/best_model.pt",
    preprocessor_path="checkpoints/preprocessor.pkl"
)

# Batch predictions
result = predictor.predict(df)
print(result.labels[:5])  # [['hold', 'hold', 'entry', 'hold'], ...]
print(result.confidence)  # Prediction confidence scores

# Real-time prediction (predict next 4 signals from recent 12 rows)
recent_data = df.tail(16)  # Need 12 + target_shift rows
next_signals = predictor.predict_next(recent_data)
print(next_signals['signals'])  # ['entry', 'hold', 'hold', 'exit']
```

#### Model Architecture

| Component | Value |
|-----------|-------|
| Input sequence | 12 timesteps |
| Output sequence | 4 timesteps |
| Classes | entry (0), hold (1), exit (2) |
| Target shift | 4 steps (features at t predict t+4) |
| Hidden size | 128 |
| LSTM layers | 2 |
| Dropout | 0.2 |

#### Custom Loss Function

The model uses a weighted loss that prioritizes:
1. **Highest penalty**: Wrong predictions during hold periods (false signals)
2. **Second penalty**: Wrong entry/exit at sequence boundaries
3. **Lower penalty**: Wrong hold predictions in middle positions

```python
from crypto_analysis import WeightedSignalLoss

loss_fn = WeightedSignalLoss(
    all_hold_weight=3.0,      # Penalty for false signals in hold sequences
    entry_exit_weight=2.0,    # Penalty for wrong entry/exit
    middle_hold_weight=1.0    # Penalty for wrong middle holds
)
```

#### Sequence Validation

Only valid trading patterns are used for training:
- **ALL_HOLD**: `[hold, hold, hold, hold]` - No trade period
- **ENTRY_EXIT**: `[entry, hold*, hold*, exit]` - Valid trade sequence

Invalid sequences (e.g., multiple entries, exits without entry) are filtered out.

## Available Indicators

### Momentum (18)
RSI, MACD, STOCH, STOCHRSI, CCI, MFI, WILLR, CMO, ADX, MOM, ROC, TRIX, ULTOSC, APO, PPO, BOP, AROON, AROONOSC

### Overlap/Moving Averages (10)
SMA, EMA, DEMA, TEMA, KAMA, WMA, TRIMA, T3, SAR, BBANDS

### Volume (3)
OBV, AD, ADOSC

### Volatility (2)
ATR, NATR

## Creating Custom Indicators

Inherit from `BaseIndicatorOptimizer`:

```python
from crypto_analysis import BaseIndicatorOptimizer
import talib.abstract as ta

class MyIndicator(BaseIndicatorOptimizer):
    indicator_name = "MY_IND"
    category = "momentum"

    def get_optimizable_params(self):
        return {
            "timeperiod": {"default": 14, "range": [5, 30], "type": "int"},
            "threshold": {"default": 30, "range": [20, 40], "type": "int"},
        }

    def calculate_indicator(self, df, **params):
        df["my_ind"] = ta.RSI(df, timeperiod=params["timeperiod"])
        return df

    def generate_entry_signal(self, df, **params):
        return df["my_ind"] < params["threshold"]

    def generate_exit_signal(self, df, **params):
        return df["my_ind"] > (100 - params["threshold"])
```

## Output DataFrame

The final dataset includes:

| Column Pattern | Description |
|----------------|-------------|
| `date`, `open`, `high`, `low`, `close`, `volume` | OHLCV data |
| `signal` | SignalPopulator signal ("entry"/"exit") |
| `{INDICATOR}_gs_entry` | Grid search optimized entry signal |
| `{INDICATOR}_gs_exit` | Grid search optimized exit signal |
| `{INDICATOR}_ho_entry` | Hyperopt optimized entry signal |
| `{INDICATOR}_ho_exit` | Hyperopt optimized exit signal |

## Performance

The package uses NumPy-vectorized operations for high performance:

| Operation | Time |
|-----------|------|
| Single indicator grid search (3000 combos) | ~2-5s |
| Full optimization (33 indicators, 4 workers) | ~2-5 min |

## Data Format

OHLCV data should be in Feather format with columns:
- `date`: datetime
- `open`, `high`, `low`, `close`: float
- `volume`: float

File naming: `{SYMBOL}USDT-4h.feather` (e.g., `BTCUSDT-4h.feather`)

## License

Part of the Freqtrade ecosystem.
