# Crypto Analysis Package

A Python package for cryptocurrency data analysis and signal generation, designed for use with Freqtrade.

## Features

- **Signal Population**: Generate entry/exit signals based on price percentage changes
- **Indicator Optimization**: Optimize technical indicator parameters using grid search and hyperopt
- **LSTM Signal Prediction**: PyTorch LSTM model for predicting entry/hold/exit signals
- **LSTM Hyperparameter Optimization**: APO (Artificial Protozoa Optimizer) for joint feature selection and hyperparameter tuning
- **Optimization Log Analysis**: Analyze optimization runs to extract insights and recommend configurations
- **57 Technical Indicators**: RSI, MACD, STOCH, BBANDS, EMA, SMA, Hilbert Transform, and more
- **Parallel Processing**: Multi-threaded optimization with configurable worker count
- **High Performance**: NumPy-vectorized operations for fast backtesting

## Installation

The package is part of the Freqtrade user_data directory. Ensure dependencies are installed:

```bash
pip install pandas numpy ta-lib optuna torch scikit-learn scipy matplotlib seaborn
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

### LSTMMetaheuristicOptimizer

APO (Artificial Protozoa Optimizer) for jointly optimizing LSTM hyperparameters and feature selection. Based on the bio-inspired metaheuristic algorithm modeling protozoa survival mechanisms.

#### Basic Usage

```python
from crypto_analysis import LSTMMetaheuristicOptimizer

# Load DatasetBuilder output
df = builder.build(symbol="BTC", threshold_pct=3.0)

# Create optimizer
optimizer = LSTMMetaheuristicOptimizer(
    df=df,
    pop_size=10,              # Population size
    iterations=50,            # Number of optimization iterations
    n_workers=4,              # Parallel workers for evaluation
    epochs_per_eval=100,      # Training epochs per fitness evaluation
    verbose=True,
    enable_logging=True,      # Enable CSV logging
)

# Run optimization
result = optimizer.optimize()

print(f"Best fitness: {result.best_fitness}")
print(f"Selected features: {result.n_features_selected}")
print(f"Best parameters: {result.best_params}")
```

#### Training from Optimization Result

```python
# Train a full model using best parameters
trainer = optimizer.train_from_result(result, epochs=200)

# Access the trained model
model = trainer.model
preprocessor = trainer.preprocessor
```

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | 10 | Population size for APO algorithm |
| `iterations` | 50 | Number of optimization iterations |
| `n_workers` | 4 | Parallel workers for evaluation |
| `min_features` | 5 | Minimum features that must be selected |
| `epochs_per_eval` | 100 | Training epochs per fitness evaluation |
| `np_neighbors` | 1 | Number of neighbor pairs for APO foraging |
| `pf_max` | 0.1 | Maximum proportion fraction for dormancy/reproduction |
| `elitist_selection` | False | Enable correlation-based feature selection |
| `elitist_constant` | 0.25 | Scaling constant for elitist threshold |
| `enable_logging` | False | Enable CSV logging of optimization progress |
| `checkpoint_interval` | 5 | Save checkpoint every N iterations |

#### APO Algorithm

The optimizer uses the Artificial Protozoa Optimizer (APO) algorithm with four behaviors:

1. **Autotrophic Foraging (Exploration)**: Photosynthesis-driven movement toward random solutions
2. **Heterotrophic Foraging (Exploitation)**: Nutrient absorption behavior refining current solutions
3. **Dormancy (Exploration)**: Random regeneration under stress conditions
4. **Reproduction (Exploitation)**: Binary fission with perturbation for local search

#### Fitness Function

For binary classification (hold=0, trade=1), the fitness combines:

```
fitness = -(trade_score * hold_f1 * trade_recall_factor)
```

Where:
- `trade_score`: Recall-biased geometric mean of trade precision/recall
- `hold_f1`: F1 score for hold class
- `trade_recall_factor`: Penalty for low trade recall (< 30%)

#### Hyperparameter Search Space

| Parameter | Range | Type | Description |
|-----------|-------|------|-------------|
| `class_weight_power` | 0.25-0.40 | float | Class imbalance handling |
| `focal_gamma` | 1.0-4.0 | float | Focal loss gamma |
| `learning_rate` | 0.0008-0.003 | float | Training learning rate |
| `dropout` | 0.18-0.45 | float | Dropout rate |
| `hidden_size` | 140-280 | int | LSTM hidden size |
| `num_layers` | 2-3 | int | Number of LSTM layers |
| `weight_decay` | 0.001-0.018 | float | L2 regularization |
| `label_smoothing` | 0.06-0.20 | float | Label smoothing factor |
| `batch_size` | 32-160 | int | Training batch size |
| `scheduler_patience` | 6-12 | int | LR scheduler patience |
| `input_seq_length` | 16 | int | Input sequence length |

#### Resuming from Checkpoint

```python
# Resume optimization from checkpoint
optimizer, start_iter = LSTMMetaheuristicOptimizer.from_checkpoint(
    df=df,
    checkpoint_path="lstm_optimization_checkpoints/checkpoint_iter_25.pkl",
    pop_size=10,
    iterations=50,
)

# Continue optimization
result = optimizer.optimize(start_iteration=start_iter)
```

#### CSV Log Format

When `enable_logging=True`, logs are saved to `optimization_logs/run_{run_id}.csv` with columns:
- `run_id`, `iteration`, `individual_idx`
- `feat_0`, `feat_1`, ... (binary feature selection)
- Hyperparameter values and bounds
- Optimizer settings
- `fitness`

### LSTMLogAnalyzer

Analyzes optimization CSV logs to provide insights for configuring the APO optimizer, including feature importance, parameter value analysis, evolution tracking, and configuration recommendations.

#### Basic Usage

```python
from crypto_analysis import LSTMLogAnalyzer

# Initialize analyzer
analyzer = LSTMLogAnalyzer(
    log_dir="optimization_logs",
    top_percentile=10.0  # Top 10% performers
)

# Run complete analysis
report = analyzer.analyze_all(run_ids=['97618be3'])

# Generate markdown report
analyzer.generate_report(report, "analysis_report.md", format='markdown')

# Generate JSON report
analyzer.generate_report(report, "analysis_report.json", format='json')
```

#### Feature Importance Analysis

```python
# Load logs
df = analyzer.load_logs(run_ids=['97618be3'])

# Analyze feature importance
feature_result = analyzer.analyze_feature_importance(df, top_k=20)

print("Top features by correlation:")
print(feature_result.feature_correlations.head(10))

# Visualize
analyzer.plot_feature_importance(feature_result)
analyzer.plot_cooccurrence(feature_result)
```

#### Parameter Analysis

```python
# Analyze hyperparameter distributions
param_results = analyzer.analyze_parameters(df)

for param_name, result in param_results.items():
    print(f"{param_name}:")
    print(f"  Correlation: {result.fitness_correlation:.3f}")
    print(f"  Utilization: {result.utilization:.2%}")
    print(f"  Proximity: {result.bound_proximity}")
    if result.recommended_bounds:
        print(f"  Recommended: {result.recommended_bounds}")

# Visualize
analyzer.plot_parameter_distributions(param_results, df)
analyzer.plot_bound_utilization(param_results)
```

#### Evolution Analysis

```python
# Analyze fitness progression over iterations
evolution_result = analyzer.analyze_evolution(df)

print(f"Converged parameters: {evolution_result.converged_params}")
print(f"Still exploring: {evolution_result.diverse_params}")

# Visualize
analyzer.plot_fitness_evolution(evolution_result)
analyzer.plot_parameter_convergence(evolution_result)
```

#### Elite Individual Analysis

Analyze what distinguishes top-performing individuals using statistical tests (Cohen's d, KS test, Mann-Whitney U).

```python
# Analyze elite individuals (fitness <= threshold)
elite_result = analyzer.analyze_elite_individuals(
    df,
    fitness_threshold=-0.48,  # Negative because fitness is negated
    use_absolute=True
)

print(f"Elite count: {elite_result.n_elite} ({elite_result.elite_fraction:.1%})")
print(f"Best fitness: {-elite_result.best_fitness:.4f}")

# Top influential parameters
for param in elite_result.ranked_parameters[:5]:
    influence = elite_result.parameter_influences[param]
    print(f"{param}: influence={influence.influence_score:.3f}, "
          f"effect={influence.effect_size:+.3f}")

# Print detailed summary
analyzer.print_elite_summary(elite_result)

# Visualize
analyzer.plot_elite_analysis(elite_result)
```

#### Generate Configuration Recommendations

```python
# Get synthesized recommendations
recommendations = analyzer.generate_recommendations(
    feature_result, param_results, evolution_result, df
)

print(f"Confidence: {recommendations.confidence:.2f}")
print(f"APO settings: {recommendations.apo_settings}")

# Export as Python code
analyzer.export_config_code(recommendations, "recommended_config.py")
```

#### Analysis Results Dataclasses

| Class | Description |
|-------|-------------|
| `FeatureImportanceResult` | Feature correlations, top features, selection frequency, co-occurrence |
| `ParameterAnalysisResult` | Per-parameter correlation, optimal range, bound proximity, utilization |
| `EvolutionAnalysisResult` | Fitness progression, convergence scores, converged/diverse params |
| `EliteAnalysisResult` | Elite individuals, parameter influences, ranked parameters |
| `EliteParameterInfluence` | Effect size, KS/Mann-Whitney stats, influence score, interpretation |
| `ConfigurationRecommendation` | Recommended bounds, APO settings, feature suggestions |
| `AnalysisReport` | Complete analysis combining all results |

#### Visualization Methods

| Method | Description |
|--------|-------------|
| `plot_feature_importance()` | Feature correlation and selection frequency |
| `plot_cooccurrence()` | Feature co-occurrence heatmap |
| `plot_parameter_distributions()` | Parameter histograms with bounds |
| `plot_bound_utilization()` | Bound utilization bar chart |
| `plot_fitness_evolution()` | Fitness progression over iterations |
| `plot_parameter_convergence()` | Parameter convergence scores |
| `plot_elite_analysis()` | Multi-panel elite analysis visualization |

## Available Indicators

### Momentum (30)
RSI, MACD, STOCH, STOCHRSI, CCI, MFI, WILLR, CMO, ADX, MOM, ROC, TRIX, ULTOSC, APO, PPO, BOP, AROON, AROONOSC, ADXR, DX, MACDEXT, MACDFIX, MINUS_DI, MINUS_DM, PLUS_DI, PLUS_DM, ROCP, ROCR, ROCR100, STOCHF

### Overlap/Moving Averages (16)
SMA, EMA, DEMA, TEMA, KAMA, WMA, TRIMA, T3, SAR, BBANDS, HT_TRENDLINE, MA, MAMA, MIDPOINT, MIDPRICE, SAREXT

### Volume (3)
OBV, AD, ADOSC

### Volatility (3)
ATR, NATR, TRANGE

### Cycle (5)
HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE

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
| Full optimization (57 indicators, 4 workers) | ~3-8 min |

## Data Format

OHLCV data should be in Feather format with columns:
- `date`: datetime
- `open`, `high`, `low`, `close`: float
- `volume`: float

File naming: `{SYMBOL}USDT-4h.feather` (e.g., `BTCUSDT-4h.feather`)

## License

Part of the Freqtrade ecosystem.
