"""
APO (Artificial Protozoa Optimizer) for LSTM/CNN-LSTM Hyperparameters and Feature Selection

Uses APO algorithm to jointly optimize:
1. Feature selection from DatasetBuilder output
2. Model training hyperparameters (supports LSTM and CNN-LSTM architectures)

Supported model architectures:
- 'lstm': Standard LSTM with input projection (LSTMSignalPredictor)
- 'cnn_lstm': CNN feature extractor + LSTM encoder (CNNLSTMSignalPredictor)

Objective: Maximize trade_F1 * hold_F1 on test dataset (binary classification)

For binary classification (hold=0, trade=1):
- Input: 16 timesteps of features
- Output: Single prediction per sequence (0=hold, 1=trade)

Based on: Wang, X., et al. (2024). "Artificial Protozoa Optimizer (APO):
A novel bio-inspired metaheuristic algorithm for engineering optimization"
Knowledge-Based Systems, 111737.
DOI: https://doi.org/10.1016/j.knosys.2024.111737
"""

import csv
import pickle
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import matthews_corrcoef


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Parameters
    ----------
    seed : int
        Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class LSTMOptimizationResult:
    """Result from LSTM/CNN-LSTM metaheuristic optimization."""
    best_fitness: float
    best_individual: np.ndarray
    selected_features: List[str]
    best_params: Dict[str, Any]
    n_features_selected: int
    test_metrics: Dict[str, float]
    history: List[float]  # Best fitness per iteration
    seed: int = 42  # Seed used for the best individual (for reproducibility)
    model_type: str = 'lstm'  # Model architecture used ('lstm' or 'cnn_lstm')


@dataclass
class HyperparamConfig:
    """Hyperparameter bounds configuration."""
    name: str
    min_val: float
    max_val: float
    param_type: str  # 'float' or 'int'
    config_field: str  # Field name in TrainingConfig/ModelConfig


@dataclass
class OptimizationCheckpoint:
    """Checkpoint for resumable optimization."""
    iteration: int
    population: np.ndarray
    fitness_pop: np.ndarray
    cols_dict: Dict[int, List[str]]
    metrics_dict: Dict[int, Dict]
    best_fitness_history: List[float]
    feature_columns: List[str]
    n_features: int
    n_params: int


class LSTMMetaheuristicOptimizer:
    """
    APO-based optimizer for LSTM hyperparameters and feature selection.

    Uses Artificial Protozoa Optimizer (APO) algorithm to jointly optimize
    feature selection and training hyperparameters. APO is a bio-inspired
    metaheuristic modeling protozoa survival mechanisms:

    1. Autotrophic Foraging (Exploration): Photosynthesis-driven movement
    2. Heterotrophic Foraging (Exploitation): Nutrient absorption behavior
    3. Dormancy (Exploration): Random regeneration under stress
    4. Reproduction (Exploitation): Binary fission with perturbation

    Parameters
    ----------
    df : pd.DataFrame
        DatasetBuilder output with OHLCV + indicator signal columns
    pop_size : int
        Population size (default: 10)
    iterations : int
        Number of optimization iterations (default: 50)
    n_workers : int
        Number of parallel workers for evaluation (default: 4)
    min_features : int
        Minimum number of features that must be selected (default: 5)
    epochs_per_eval : int
        Training epochs per fitness evaluation (default: 100)
    checkpoint_interval : int
        Save checkpoint every N iterations (default: 5)
    checkpoint_dir : str
        Directory for saving checkpoints (default: 'lstm_optimization_checkpoints')
    verbose : bool
        Print progress (default: True)
    np_neighbors : int
        Number of neighbor pairs for foraging calculations (default: 1)
    pf_max : float
        Maximum proportion fraction for dormancy/reproduction (default: 0.1)
    elitist_selection : bool
        Enable correlation-based elitist feature selection (default: False).
        When enabled, features with higher correlation to the target signal
        have a lower selection threshold, making them more likely to be selected.
    elitist_constant : float
        Scaling constant for elitist selection threshold (default: 0.25).
        Threshold = feature_lower * correlation * elitist_constant.
        Higher values make selection more strict for low-correlation features.
    enable_logging : bool
        Enable CSV logging of optimization progress (default: False).
        Logs run_id, iteration, individual features (binary), hyperparameters,
        parameter bounds, optimizer settings, and fitness values.
    log_dir : str
        Directory for saving log CSV files (default: 'optimization_logs').
        Each run creates a file: run_{run_id}.csv

    Examples
    --------
    >>> from crypto_analysis import DatasetBuilder, LSTMMetaheuristicOptimizer
    >>> builder = DatasetBuilder(data_dir="data/binance")
    >>> df = builder.build(symbol="BTC", threshold_pct=3.0)
    >>> optimizer = LSTMMetaheuristicOptimizer(df, pop_size=10, iterations=50)
    >>> result = optimizer.optimize()
    >>> print(f"Best fitness: {result.best_fitness}")
    """

    # Base hyperparameter configurations shared by all models
    # For binary classification (hold=0, trade=1)
    # Updated based on run 7caf7327 analysis (elite individuals at 5th percentile)
    BASE_HYPERPARAM_CONFIGS = [
        # Class imbalance handling
        HyperparamConfig('class_weight_power', 0.15, 0.35, 'float', 'class_weight_power'),
        HyperparamConfig('focal_gamma', 1.0, 4.0, 'float', 'focal_gamma'),
        # Training parameters (narrowed based on elite analysis)
        HyperparamConfig('learning_rate', 0.0003, 0.003, 'float', 'learning_rate'),
        HyperparamConfig('dropout', 0.01, 0.10, 'float', 'dropout'),
        HyperparamConfig('hidden_size', 96, 256, 'int', 'hidden_size'),
        HyperparamConfig('num_layers', 1, 3, 'int', 'num_layers'),
        HyperparamConfig('weight_decay', 0.0005, 0.02, 'float', 'weight_decay'),
        HyperparamConfig('label_smoothing', 0.05, 0.12, 'float', 'label_smoothing'),
        HyperparamConfig('batch_size', 64, 128, 'int', 'batch_size'),
        HyperparamConfig('scheduler_patience', 5, 10, 'int', 'scheduler_patience'),
        HyperparamConfig('input_seq_length', 12, 21, 'int', 'input_seq_length'),
    ]

    # CNN-LSTM specific hyperparameters
    CNN_HYPERPARAM_CONFIGS = [
        # CNN kernel size for Conv1d layers (3, 5, or 7)
        HyperparamConfig('kernel_size', 3, 9, 'int', 'kernel_size'),
        # Number of CNN conv blocks (1-3)
        HyperparamConfig('num_conv_layers', 1, 3, 'int', 'num_conv_layers'),
    ]

    # Default config uses base (for backward compatibility)
    HYPERPARAM_CONFIGS = BASE_HYPERPARAM_CONFIGS



    # Columns to exclude from feature selection
    EXCLUDED_COLUMNS = ['date', 'signal', 'signal_pct_change', 'period_id', 'tradeable']

    def __init__(
        self,
        df: pd.DataFrame,
        pop_size: int = 10,
        iterations: int = 50,
        n_workers: int = 4,
        min_features: int = 1,
        epochs_per_eval: int = 100,
        checkpoint_interval: int = 5,
        checkpoint_dir: str = 'lstm_optimization_checkpoints',
        verbose: bool = True,
        np_neighbors: int = 2,
        pf_max: float = 0.18,
        elitist_selection: bool = False,
        elitist_constant: float = 0.25,
        enable_logging: bool = False,
        log_dir: str = 'optimization_logs',
        seed: int = 42,
        model_type: str = 'lstm',
    ):
        self.df = df
        self.pop_size = pop_size
        self.iterations = iterations
        self.n_workers = n_workers
        self.min_features = min_features
        self.epochs_per_eval = epochs_per_eval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.verbose = verbose
        self.np_neighbors = np_neighbors
        self.pf_max = pf_max
        self.elitist_selection = elitist_selection
        self.elitist_constant = elitist_constant
        self.enable_logging = enable_logging
        self.log_dir = Path(log_dir)
        self.seed = seed
        self.run_id: Optional[str] = None  # Generated when optimize() starts

        # Model type selection
        if model_type not in ('lstm', 'cnn_lstm'):
            raise ValueError(f"model_type must be 'lstm' or 'cnn_lstm', got '{model_type}'")
        self.model_type = model_type

        # Set hyperparameter configs based on model type
        if model_type == 'cnn_lstm':
            self.HYPERPARAM_CONFIGS = self.BASE_HYPERPARAM_CONFIGS + self.CNN_HYPERPARAM_CONFIGS
        else:
            self.HYPERPARAM_CONFIGS = self.BASE_HYPERPARAM_CONFIGS

        # Identify selectable feature columns (all columns except excluded ones)
        self.feature_columns = [
            col for col in df.columns
            if col not in self.EXCLUDED_COLUMNS
        ]
        self.n_features = len(self.feature_columns)
        self.n_params = len(self.HYPERPARAM_CONFIGS)
        self.dimension = self.n_features + self.n_params

        # Build bounds arrays
        self._build_bounds()

        # Calculate correlation vector for elitist selection
        self.correlation_vector: Optional[np.ndarray] = None
        if self.elitist_selection:
            self.correlation_vector = self._calculate_correlation_vector()

        # State for optimization (initialized in optimize())
        self.population: Optional[np.ndarray] = None
        self.fitness_pop: Optional[np.ndarray] = None
        self.cols_dict: Dict[int, List[str]] = {}
        self.metrics_dict: Dict[int, Dict] = {}
        self.best_fitness_history: List[float] = []

        # Detect DataFrame mode (binary vs raw indicators)
        self.dataframe_mode = self._detect_dataframe_mode()

        if self.verbose:
            print(f"LSTMMetaheuristicOptimizer (APO) initialized:")
            print(f"  - Model type: {self.model_type}")
            print(f"  - DataFrame mode: {self.dataframe_mode}")
            print(f"  - Feature columns: {self.n_features}")
            print(f"  - Hyperparameters: {self.n_params}")
            print(f"  - Total dimension: {self.dimension}")
            print(f"  - Population size: {self.pop_size}")
            print(f"  - Iterations: {self.iterations}")
            print(f"  - Workers: {self.n_workers}")
            print(f"  - APO np_neighbors: {self.np_neighbors}")
            print(f"  - APO pf_max: {self.pf_max}")
            print(f"  - Elitist selection: {self.elitist_selection}")
            if self.elitist_selection:
                print(f"  - Elitist constant: {self.elitist_constant}")
                corr_stats = self.correlation_vector
                print(f"  - Correlation vector: min={corr_stats.min():.4f}, "
                      f"max={corr_stats.max():.4f}, mean={corr_stats.mean():.4f}")

    def _build_bounds(self):
        """Build lower and upper bound arrays for the search space."""
        # Feature bounds: [-100, 100]
        feature_lower = np.full(self.n_features, -100.0)
        feature_upper = np.full(self.n_features, 100.0)

        # Hyperparameter bounds
        param_lower = np.array([cfg.min_val for cfg in self.HYPERPARAM_CONFIGS])
        param_upper = np.array([cfg.max_val for cfg in self.HYPERPARAM_CONFIGS])

        self.lower_bound = np.concatenate([feature_lower, param_lower])
        self.upper_bound = np.concatenate([feature_upper, param_upper])

    def _detect_dataframe_mode(self) -> str:
        """
        Detect if DataFrame uses binary or raw indicator format.

        Binary mode: columns end with _entry, _exit (e.g., RSI_gs_entry)
        Raw mode: columns like RSI_gs_rsi, MACD_gs_macd (indicator output names)

        Returns
        -------
        str
            'binary' or 'raw'
        """
        # Count binary-style columns (ending with _entry or _exit)
        binary_cols = [
            c for c in self.df.columns
            if c.endswith(('_entry', '_exit'))
        ]

        # Count raw-style columns (containing _gs_ or _ho_ but not ending with _entry/_exit)
        raw_cols = [
            c for c in self.df.columns
            if ('_gs_' in c or '_ho_' in c)
            and not c.endswith(('_entry', '_exit'))
        ]

        if len(binary_cols) > len(raw_cols):
            return 'binary'
        elif len(raw_cols) > 0:
            return 'raw'
        else:
            # Default to binary if no indicator columns found
            return 'binary'

    def _calculate_correlation_vector(self) -> np.ndarray:
        """
        Calculate correlation vector between features and target (tradeable).

        Uses different correlation measures based on feature type:
        - Continuous features (open, high, low, close, volume): eta-squared
        - Binary features: Phi coefficient (Matthews correlation)

        For binary target (hold/trade), calculates correlation with trade class.

        Returns
        -------
        np.ndarray
            Correlation vector of shape (n_features,) with values in [0, 1]
        """
        # Continuous columns (OHLCV)
        continuous_cols = {'open', 'high', 'low', 'close', 'volume'}

        # Target classes (hold/trade -> 0/1)
        target = self.df['tradeable']
        classes = target.unique()

        correlation_vector = np.zeros(self.n_features)

        for i, col in enumerate(self.feature_columns):
            feature = self.df[col]

            if col.lower() in continuous_cols:
                # Continuous feature: use eta-squared (ANOVA-based)
                # η² = SS_between / SS_total
                try:
                    groups = [feature[target == cls].dropna() for cls in classes]
                    # Filter out empty groups
                    groups = [g for g in groups if len(g) > 0]

                    if len(groups) >= 2:
                        ss_between = sum(
                            len(g) * (g.mean() - feature.mean())**2
                            for g in groups
                        )
                        ss_total = ((feature - feature.mean())**2).sum()

                        if ss_total > 0:
                            eta_sq = ss_between / ss_total
                            correlation_vector[i] = min(eta_sq, 1.0)  # Clamp to [0, 1]
                        else:
                            correlation_vector[i] = 0.0
                    else:
                        correlation_vector[i] = 0.0
                except Exception:
                    correlation_vector[i] = 0.0
            else:
                # Binary feature: use Phi coefficient (Matthews correlation)
                # Calculate per-class and take max absolute
                max_phi = 0.0
                try:
                    feature_binary = feature.astype(int).values

                    for cls in classes:
                        target_binary = (target == cls).astype(int).values

                        # Matthews correlation coefficient
                        phi = matthews_corrcoef(target_binary, feature_binary)
                        max_phi = max(max_phi, abs(phi))

                    correlation_vector[i] = max_phi
                except Exception:
                    correlation_vector[i] = 0.0

        return correlation_vector

    def _log_iteration(
        self,
        iteration: int,
        population: np.ndarray,
        fitness_values: np.ndarray,
    ) -> None:
        """
        Log iteration data to CSV file.

        Logs run metadata, individual representations, and fitness values.
        Features are represented as binary (0=not selected, 1=selected).

        Parameters
        ----------
        iteration : int
            Current iteration number (-1 for initial population)
        population : np.ndarray
            Population array of shape (pop_size, dimension)
        fitness_values : np.ndarray
            Fitness values for each individual
        """
        if not self.enable_logging or self.run_id is None:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / f"run_{self.run_id}.csv"

        # Check if file exists to determine if we need to write header
        write_header = not log_file.exists()

        # Build header columns
        # Format: run_id, iteration, individual_idx, feature_0, feature_1, ..., param_0, param_1, ..., fitness
        feature_cols = [f"feat_{i}" for i in range(self.n_features)]
        param_cols = [cfg.name for cfg in self.HYPERPARAM_CONFIGS]
        param_bound_cols = [f"{cfg.name}_lower" for cfg in self.HYPERPARAM_CONFIGS] + \
                          [f"{cfg.name}_upper" for cfg in self.HYPERPARAM_CONFIGS]

        # Optimizer hyperparameters to log (only in first row of each iteration)
        optimizer_params = {
            'pop_size': self.pop_size,
            'iterations': self.iterations,
            'n_workers': self.n_workers,
            'min_features': self.min_features,
            'epochs_per_eval': self.epochs_per_eval,
            'np_neighbors': self.np_neighbors,
            'pf_max': self.pf_max,
            'elitist_selection': self.elitist_selection,
            'elitist_constant': self.elitist_constant,
        }

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if write_header:
                header = (
                    ['run_id', 'iteration', 'individual_idx'] +
                    feature_cols +
                    param_cols +
                    param_bound_cols +
                    list(optimizer_params.keys()) +
                    ['fitness']
                )
                writer.writerow(header)

            # Get parameter bounds
            param_lower = [cfg.min_val for cfg in self.HYPERPARAM_CONFIGS]
            param_upper = [cfg.max_val for cfg in self.HYPERPARAM_CONFIGS]

            for idx, (individual, fitness) in enumerate(zip(population, fitness_values)):
                # Convert features to binary (0/1) based on selection threshold
                feature_values = individual[:self.n_features]
                if self.elitist_selection and self.correlation_vector is not None:
                    feature_lower = self.lower_bound[:self.n_features]
                    threshold = feature_lower * self.correlation_vector * self.elitist_constant
                    feature_binary = (feature_values > threshold).astype(int).tolist()
                else:
                    feature_binary = (feature_values >= 0).astype(int).tolist()

                # Get hyperparameter values
                param_values = individual[self.n_features:].tolist()

                # Build row
                row = (
                    [self.run_id, iteration, idx] +
                    feature_binary +
                    param_values +
                    param_lower +
                    param_upper +
                    list(optimizer_params.values()) +
                    [fitness]
                )
                writer.writerow(row)

    def _decode_individual(self, individual: np.ndarray) -> Tuple[List[str], Dict[str, Any]]:
        """
        Decode individual vector into selected features and hyperparameters.

        Parameters
        ----------
        individual : np.ndarray
            Individual vector of shape (dimension,)

        Returns
        -------
        tuple
            (selected_features, config_params)

        Notes
        -----
        When elitist_selection is enabled, features with higher correlation to
        the target have a lower selection threshold, making them more likely
        to be selected. The threshold is calculated as:
            threshold = feature_lower * correlation * elitist_constant

        For example, with feature_lower=-100, correlation=0.8, constant=0.25:
            threshold = -100 * 0.8 * 0.25 = -20
        A feature value > -20 will be selected.

        For low correlation=0.1:
            threshold = -100 * 0.1 * 0.25 = -2.5
        A feature value > -2.5 will be selected (harder threshold).
        """
        feature_values = individual[:self.n_features]

        if self.elitist_selection and self.correlation_vector is not None:
            # Elitist selection: threshold based on correlation
            # Higher correlation = lower (more negative) threshold = easier to select
            feature_lower = self.lower_bound[:self.n_features]


            # Min-max normalization to [0, 1]
            corr_min = self.correlation_vector.min()
            corr_max = self.correlation_vector.max()
            if corr_max > corr_min:
                correlation_vector = (self.correlation_vector - corr_min) / (corr_max - corr_min)
            # If all values are the same, leave as-is (already 0 or all equal)

            conditions = [
                correlation_vector < 0.01,
                correlation_vector < 0.05,
                correlation_vector > 0.80,
                correlation_vector > 0.90,
            ]
            choices = [
                -self.elitist_constant * 1.8,
                -self.elitist_constant * 1.6,
                self.elitist_constant * 1.1,
                self.elitist_constant * 1.3,
            ]
            # default is used when no condition is met
            constant = np.select(conditions, choices, default=self.elitist_constant)


            threshold = feature_lower * correlation_vector * constant
            feature_mask = feature_values > threshold
        else:
            # Standard selection: >= 0 = selected
            feature_mask = feature_values >= -30 # Experimental: select most features

        selected_features = [
            col for col, sel in zip(self.feature_columns, feature_mask) if sel
        ]

        # Extract and convert hyperparameters
        params = individual[self.n_features:]
        config_params = {}
        for i, cfg in enumerate(self.HYPERPARAM_CONFIGS):
            val = params[i]
            if cfg.param_type == 'int':
                val = int(round(val))
            config_params[cfg.config_field] = val

        return selected_features, config_params

    def _evaluate_individual(
        self,
        individual: np.ndarray,
        idx: int
    ) -> Tuple[float, List[str], Dict]:
        """
        Evaluate a single individual's fitness for binary classification.

        Parameters
        ----------
        individual : np.ndarray
            Individual vector
        idx : int
            Individual index (for logging)

        Returns
        -------
        tuple
            (fitness, selected_features, metrics)
            fitness is inf for invalid individuals
        """
        # Import LSTM modules inside worker to avoid threading issues
        from .lstm import (
            DataPreprocessor, create_sequences,
            SignalDataset, TrainingConfig, ModelConfig,
            LSTMSignalPredictor, CNNLSTMSignalPredictor, Trainer
        )

        selected_features, config_params = self._decode_individual(individual)

        # Ensure minimum features selected
        if len(selected_features) < self.min_features:
            return float('inf'), [], {}

        # Set deterministic seed for reproducibility
        # Use base seed + individual index to ensure each individual is reproducible
        # but different individuals explore different random paths
        individual_seed = self.seed + idx
        set_seed(individual_seed)

        try:
            # Build selected DataFrame (keep tradeable + selected features)
            df_selected = self.df[['tradeable'] + selected_features].copy()

            # 1. Preprocess (uses 'tradeable' column for binary classification)
            preprocessor = DataPreprocessor(target_shift=4)
            features, targets = preprocessor.fit_transform(df_selected)

            # 2. Create sequences (use input_seq_length from params)
            # output_seq_length=1 for binary classification
            input_seq_length = config_params['input_seq_length']
            feat_seqs, tgt_seqs = create_sequences(
                features, targets,
                input_seq_length=input_seq_length,
                output_seq_length=1
            )

            # Check we have enough sequences
            if len(feat_seqs) < 100:
                return float('inf'), selected_features, {}

            # 3. Create dataset (no sequence validation needed for binary)
            dataset = SignalDataset(feat_seqs, tgt_seqs)

            # 4. Build configs - force CPU for thread safety
            training_config = TrainingConfig(
                epochs=self.epochs_per_eval,
                batch_size=config_params['batch_size'],
                learning_rate=config_params['learning_rate'],
                hidden_size=config_params['hidden_size'],
                num_layers=config_params['num_layers'],
                dropout=config_params['dropout'],
                weight_decay=config_params['weight_decay'],
                auto_class_weights=True,
                class_weight_power=config_params['class_weight_power'],
                focal_loss=False,
                focal_gamma=config_params['focal_gamma'],
                label_smoothing=config_params['label_smoothing'],
                scheduler_patience=config_params['scheduler_patience'],
                patience=10,
                verbose=False,
                device='cuda',  # Force CPU for thread safety
                # bidirectional=True, # Experimentally commented out
            )

            model_config = ModelConfig(
                input_size=preprocessor.get_num_features(),
                hidden_size=config_params['hidden_size'],
                num_layers=config_params['num_layers'],
                dropout=config_params['dropout'],
                kernel_size=config_params['kernel_size'],
                input_seq_length=input_seq_length,
            )

            # 5. Create model based on model_type
            if self.model_type == 'cnn_lstm':
                model = CNNLSTMSignalPredictor(model_config)
            else:
                model = LSTMSignalPredictor(model_config)
            trainer = Trainer(model, training_config, preprocessor=preprocessor)
            trainer.train(dataset)

            # 6. Evaluate on TEST set only
            metrics = trainer.evaluate(trainer.test_dataset, verbose=False)

            # 7. Compute fitness for binary classification (hold=0, trade=1)
            # Optimize for trade class detection while maintaining hold accuracy
            trade_precision = metrics.get('trade_precision', 0)
            trade_recall = metrics.get('trade_recall', 0)
            trade_f1 = metrics.get('trade_f1', 0)
            hold_f1 = metrics.get('hold_f1', 0)

            eps = 1e-10

            # Recall-biased geometric mean for trade class
            # β > 1 emphasizes recall (catching trade opportunities) over precision
            beta = 2.0
            p_exp = 1.0 / (1.0 + beta)  # 0.333
            r_exp = beta / (1.0 + beta)  # 0.667

            trade_score = ((trade_precision + eps) ** p_exp) * ((trade_recall + eps) ** r_exp)

            # Recall floor penalty: penalize if trade recall < min_recall
            # Prevents models that rarely predict "trade"
            min_recall = 0.30
            trade_recall_factor = min(trade_recall / min_recall, 1.0) if min_recall > 0 else 1.0

            # Combined fitness: trade detection quality * hold accuracy * recall penalty
            fitness = -(trade_score * hold_f1 * trade_recall_factor)

            return fitness, selected_features, metrics

        except Exception as e:
            if self.verbose:
                print(f"  Individual {idx} failed: {e}")
            return float('inf'), [], {}

    def _evaluate_population_parallel(
        self,
        population: np.ndarray,
        iteration: int = -1
    ) -> Tuple[np.ndarray, Dict[int, List[str]], Dict[int, Dict]]:
        """
        Evaluate population in parallel using ThreadPoolExecutor.

        Parameters
        ----------
        population : np.ndarray
            Population array of shape (pop_size, dimension)
        iteration : int
            Current iteration number for logging

        Returns
        -------
        tuple
            (fitness_values, cols_dict, metrics_dict)
        """
        fitness_values = np.full(len(population), float('inf'))
        cols_dict: Dict[int, List[str]] = {}
        metrics_dict: Dict[int, Dict] = {}

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self._evaluate_individual, ind, idx): idx
                for idx, ind in enumerate(population)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    fitness, cols, metrics = future.result()
                    fitness_values[idx] = fitness
                    cols_dict[idx] = cols
                    metrics_dict[idx] = metrics

                    if self.verbose:
                        if fitness != float('inf'):
                            print(f"iter:{iteration} indv:{idx} fitness:{-fitness:.4f} "
                                  f"features:{len(cols)}")
                        else:
                            print(f"iter:{iteration} indv:{idx} fitness:invalid")

                except Exception as e:
                    print(f"iter:{iteration} indv:{idx} error: {e}")

        return fitness_values, cols_dict, metrics_dict

    def _print_best_metrics(
        self,
        fitness_values: np.ndarray,
        metrics_dict: Dict[int, Dict],
        iteration: int = -1,
    ) -> None:
        """
        Print per-class metrics table for the best individual in the population.

        For binary classification (hold=0, trade=1).

        Parameters
        ----------
        fitness_values : np.ndarray
            Fitness values for each individual
        metrics_dict : dict
            Metrics dictionary keyed by individual index
        iteration : int
            Current iteration number
        """
        if not self.verbose:
            return

        # Find best individual
        best_idx = int(np.argmin(fitness_values))
        best_fitness = fitness_values[best_idx]

        if best_fitness == float('inf') or best_idx not in metrics_dict:
            return

        metrics = metrics_dict[best_idx]
        if not metrics:
            return

        # Print header
        iter_label = "INIT" if iteration == -1 else f"ITER {iteration}"
        print(f"\n{'-' * 72}")
        print(f"{iter_label} - Best Individual Per-Class Metrics (fitness: {-best_fitness:.6f})")
        print(f"{'-' * 72}")
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print(f"{'-' * 56}")

        # Hold metrics (class 0)
        print(f"{'hold':<10} "
              f"{metrics.get('hold_precision', 0):<12.4f} "
              f"{metrics.get('hold_recall', 0):<12.4f} "
              f"{metrics.get('hold_f1', 0):<12.4f} "
              f"{metrics.get('hold_support', 0):<10}")

        # Trade metrics (class 1)
        print(f"{'trade':<10} "
              f"{metrics.get('trade_precision', 0):<12.4f} "
              f"{metrics.get('trade_recall', 0):<12.4f} "
              f"{metrics.get('trade_f1', 0):<12.4f} "
              f"{metrics.get('trade_support', 0):<10}")

    def save_checkpoint(self, path: Optional[str] = None, iteration: int = 0):
        """
        Save current optimization state to checkpoint file.

        Parameters
        ----------
        path : str, optional
            Path to save checkpoint. If None, uses default naming.
        iteration : int
            Current iteration number
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if path is None:
            path = self.checkpoint_dir / f"checkpoint_iter_{iteration}.pkl"
        else:
            path = Path(path)

        checkpoint = OptimizationCheckpoint(
            iteration=iteration,
            population=self.population,
            fitness_pop=self.fitness_pop,
            cols_dict=self.cols_dict,
            metrics_dict=self.metrics_dict,
            best_fitness_history=self.best_fitness_history,
            feature_columns=self.feature_columns,
            n_features=self.n_features,
            n_params=self.n_params,
        )

        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

        if self.verbose:
            print(f"Checkpoint saved to {path}")

    @classmethod
    def from_checkpoint(
        cls,
        df: pd.DataFrame,
        checkpoint_path: str,
        **kwargs
    ) -> Tuple['LSTMMetaheuristicOptimizer', int]:
        """
        Resume optimization from checkpoint.

        Parameters
        ----------
        df : pd.DataFrame
            DatasetBuilder output (must be same as original)
        checkpoint_path : str
            Path to checkpoint file
        **kwargs
            Additional arguments passed to constructor

        Returns
        -------
        tuple
            (optimizer, start_iteration)
        """
        with open(checkpoint_path, 'rb') as f:
            checkpoint: OptimizationCheckpoint = pickle.load(f)

        # Create optimizer with same settings
        optimizer = cls(df, **kwargs)

        # Verify feature columns match
        if optimizer.feature_columns != checkpoint.feature_columns:
            raise ValueError(
                "Feature columns in checkpoint don't match current DataFrame. "
                "Make sure to use the same DataFrame."
            )

        # Restore state
        optimizer.population = checkpoint.population
        optimizer.fitness_pop = checkpoint.fitness_pop
        optimizer.cols_dict = checkpoint.cols_dict
        optimizer.metrics_dict = checkpoint.metrics_dict
        optimizer.best_fitness_history = checkpoint.best_fitness_history

        return optimizer, checkpoint.iteration

    def optimize(self, start_iteration: int = 0) -> LSTMOptimizationResult:
        """
        Run APO (Artificial Protozoa Optimizer) algorithm.

        The APO algorithm is inspired by the survival mechanisms of protozoa (euglena),
        modeling their foraging, dormancy, and reproductive behaviors for optimization.

        Parameters
        ----------
        start_iteration : int
            Starting iteration (used when resuming from checkpoint)

        Returns
        -------
        LSTMOptimizationResult
            Optimization result with best solution and history
        """
        # Generate run_id for logging
        if self.enable_logging:
            self.run_id = uuid.uuid4().hex[:8]  # Short 8-char UUID

        if self.verbose:
            print("\n" + "=" * 60)
            print("Starting APO Metaheuristic Optimization")
            if self.enable_logging:
                print(f"Run ID: {self.run_id}")
            print("=" * 60)

        ps = self.pop_size
        dim = self.dimension
        np_pairs = self.np_neighbors
        iter_max = self.iterations
        Xmin = self.lower_bound
        Xmax = self.upper_bound

        # Initialize population if not resuming
        if self.population is None:
            self.population = np.random.uniform(
                low=Xmin,
                high=Xmax,
                size=(ps, dim)
            )
            self.fitness_pop = np.zeros(ps)
            self.cols_dict = {}
            self.metrics_dict = {}
            self.best_fitness_history = []

            # Evaluate initial population
            if self.verbose:
                print("\nEvaluating initial population...")

            self.fitness_pop, self.cols_dict, self.metrics_dict = \
                self._evaluate_population_parallel(self.population, iteration=-1)

            # Print best metrics for initial population
            self._print_best_metrics(self.fitness_pop, self.metrics_dict, iteration=-1)

            # Log initial population
            self._log_iteration(-1, self.population, self.fitness_pop)

        # Pre-allocate arrays for APO
        new_population = np.zeros((ps, dim))
        epn = np.zeros((np_pairs, dim))  # effect of paired neighbors

        # Main APO optimization loop
        for iter_num in range(start_iteration, iter_max):
            if self.verbose:
                print(f"\n--- Iteration {iter_num + 1}/{iter_max} ---")

            # Sort population by fitness (ascending) - APO requires sorted population
            sort_indices = np.argsort(self.fitness_pop)
            self.fitness_pop = self.fitness_pop[sort_indices]
            self.population = self.population[sort_indices]

            # Reindex cols_dict and metrics_dict after sorting
            new_cols_dict: Dict[int, List[str]] = {}
            new_metrics_dict: Dict[int, Dict] = {}
            for new_idx, old_idx in enumerate(sort_indices):
                if old_idx in self.cols_dict:
                    new_cols_dict[new_idx] = self.cols_dict[old_idx]
                if old_idx in self.metrics_dict:
                    new_metrics_dict[new_idx] = self.metrics_dict[old_idx]
            self.cols_dict = new_cols_dict
            self.metrics_dict = new_metrics_dict

            # Track best fitness (index 0 after sorting is the best)
            best_fitness_iter = self.fitness_pop[0]
            self.best_fitness_history.append(-best_fitness_iter)  # Convert to positive

            if self.verbose:
                print(f"Current best fitness: {-best_fitness_iter:.4f}")

            # Proportion fraction: pf = pf_max * rand
            pf = self.pf_max * np.random.random()

            # Rank indices for dormancy/reproduction: ri = randperm(ps, ceil(ps*pf))
            num_dr = int(np.ceil(ps * pf))
            if num_dr > 0:
                ri = set(np.random.permutation(ps)[:num_dr])
            else:
                ri = set()

            # Process each protozoan
            for i in range(ps):
                i_matlab = i + 1  # Convert to 1-based for formulas

                if i in ri:
                    # Dormancy or Reproduction form
                    # pdr = 1/2 * (1 + cos((1 - i/ps) * pi))
                    pdr = 0.5 * (1 + np.cos((1 - i_matlab / ps) * np.pi))

                    if np.random.random() < pdr:
                        # DORMANCY: Random regeneration
                        # newprotozoa(i,:) = Xmin + rand(1,dim).*(Xmax-Xmin)
                        new_population[i] = Xmin + np.random.random(dim) * (Xmax - Xmin)
                    else:
                        # REPRODUCTION: Binary fission with perturbation
                        # Flag = [1,-1](ceil(2*rand)) -> random +1 or -1
                        Flag = 1 if np.random.random() < 0.5 else -1

                        # Mr = zeros, then Mr(randperm(dim, ceil(rand*dim))) = 1
                        Mr = np.zeros(dim)
                        num_mr = int(np.ceil(np.random.random() * dim))
                        if num_mr > 0:
                            Mr[np.random.permutation(dim)[:num_mr]] = 1

                        # newprotozoa(i,:) = protozoa(i,:) + Flag*rand*(Xmin+rand(1,dim).*(Xmax-Xmin)).*Mr
                        rand_scalar = np.random.random()
                        rand_vec = np.random.random(dim)
                        new_population[i] = (
                            self.population[i] +
                            Flag * rand_scalar * (Xmin + rand_vec * (Xmax - Xmin)) * Mr
                        )
                else:
                    # FORAGING form
                    # f = rand * (1 + cos(iter/iter_max * pi))
                    f = np.random.random() * (1 + np.cos((iter_num + 1) / iter_max * np.pi))

                    # Mf = zeros, then Mf(randperm(dim, ceil(dim*i/ps))) = 1
                    Mf = np.zeros(dim)
                    num_mf = int(np.ceil(dim * i_matlab / ps))
                    Mf[np.random.permutation(dim)[:num_mf]] = 1

                    # pah = 1/2 * (1 + cos(iter/iter_max * pi))
                    pah = 0.5 * (1 + np.cos((iter_num + 1) / iter_max * np.pi))

                    if np.random.random() < pah:
                        # AUTOTROPHIC FORAGING (exploration)
                        # j = randperm(ps, 1) -> random index
                        j = np.random.randint(0, ps)

                        # Calculate neighbor effect
                        epn.fill(0)
                        for k in range(1, np_pairs + 1):
                            # MATLAB logic converted to 0-based:
                            if i == 0:  # i_matlab == 1
                                km = 0
                                kp = 1 + np.random.randint(0, ps - 1) if ps > 1 else 0
                            elif i == ps - 1:  # i_matlab == ps
                                km = np.random.randint(0, ps - 1) if ps > 1 else 0
                                kp = ps - 1
                            else:
                                km = np.random.randint(0, i)
                                kp = i + 1 + np.random.randint(0, ps - i - 1)

                            # wa = exp(-abs(protozoa_Fit(km) / (protozoa_Fit(kp) + eps)))
                            wa = np.exp(-np.abs(
                                self.fitness_pop[km] /
                                (self.fitness_pop[kp] + np.finfo(float).eps)
                            ))

                            # epn(k,:) = wa * (protozoa(km,:) - protozoa(kp,:))
                            epn[k-1] = wa * (self.population[km] - self.population[kp])

                        # newprotozoa(i,:) = protozoa(i,:) + f*(protozoa(j,:)-protozoa(i,:)+1/np*sum(epn,1)).*Mf
                        neighbor_sum = np.sum(epn, axis=0) / np_pairs
                        new_population[i] = (
                            self.population[i] +
                            f * (self.population[j] - self.population[i] + neighbor_sum) * Mf
                        )

                    else:
                        # HETEROTROPHIC FORAGING (exploitation)
                        epn.fill(0)
                        for k in range(1, np_pairs + 1):
                            # MATLAB logic converted to 0-based:
                            if i == 0:  # i_matlab == 1
                                imk = 0
                                ipk = k
                            elif i == ps - 1:  # i_matlab == ps
                                imk = ps - 1 - k
                                ipk = ps - 1
                            else:
                                imk = i - k
                                ipk = i + k

                            # Clamp to valid range [0, ps-1]
                            imk = max(0, min(imk, ps - 1))
                            ipk = max(0, min(ipk, ps - 1))

                            # wh = exp(-abs(protozoa_Fit(imk) / (protozoa_Fit(ipk) + eps)))
                            wh = np.exp(-np.abs(
                                self.fitness_pop[imk] /
                                (self.fitness_pop[ipk] + np.finfo(float).eps)
                            ))

                            # epn(k,:) = wh * (protozoa(imk,:) - protozoa(ipk,:))
                            epn[k-1] = wh * (self.population[imk] - self.population[ipk])

                        # Flag = [1,-1](ceil(2*rand)) -> scalar +1 or -1
                        Flag = 1 if np.random.random() < 0.5 else -1

                        # Xnear = (1 + Flag*rand(1,dim)*(1-iter/iter_max)) .* protozoa(i,:)
                        rand_vec = np.random.random(dim)
                        Xnear = (1 + Flag * rand_vec * (1 - (iter_num + 1) / iter_max)) * self.population[i]

                        # newprotozoa(i,:) = protozoa(i,:) + f*(Xnear-protozoa(i,:)+1/np*sum(epn,1)).*Mf
                        neighbor_sum = np.sum(epn, axis=0) / np_pairs
                        new_population[i] = (
                            self.population[i] +
                            f * (Xnear - self.population[i] + neighbor_sum) * Mf
                        )

            # Boundary check - clip to bounds
            new_population = np.clip(new_population, Xmin, Xmax)

            # Evaluate new population
            if self.verbose:
                print("Evaluating new candidates...")

            new_fitness, new_cols, new_metrics = self._evaluate_population_parallel(
                new_population, iteration=iter_num
            )

            # Greedy selection: replace only if new solution is better
            for idx in range(ps):
                if new_fitness[idx] < self.fitness_pop[idx]:
                    self.fitness_pop[idx] = new_fitness[idx]
                    self.population[idx] = new_population[idx].copy()
                    self.cols_dict[idx] = new_cols[idx]
                    self.metrics_dict[idx] = new_metrics[idx]

            # Print best metrics for this iteration
            self._print_best_metrics(self.fitness_pop, self.metrics_dict, iteration=iter_num)

            # Log iteration
            self._log_iteration(iter_num, self.population, self.fitness_pop)

            # Save checkpoint periodically
            if (iter_num + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(iteration=iter_num + 1)

        # Final best solution
        best_idx = np.argmin(self.fitness_pop)
        best_fitness = -self.fitness_pop[best_idx]  # Convert back to positive
        best_individual = self.population[best_idx]
        selected_features, best_params = self._decode_individual(best_individual)

        if self.verbose:
            print("\n" + "=" * 60)
            print("Optimization Complete (Binary Classification)")
            print("=" * 60)
            print(f"Best fitness: {best_fitness:.4f}")
            print(f"Selected features: {len(selected_features)}")
            print(f"Best parameters: {best_params}")
            if best_idx in self.metrics_dict and self.metrics_dict[best_idx]:
                metrics = self.metrics_dict[best_idx]
                print(f"Test metrics:")
                print(f"  Hold F1: {metrics.get('hold_f1', 0):.4f}")
                print(f"  Trade F1: {metrics.get('trade_f1', 0):.4f}")

        # Calculate the seed used for the best individual
        best_individual_seed = self.seed + best_idx

        return LSTMOptimizationResult(
            best_fitness=best_fitness,
            best_individual=best_individual,
            selected_features=self.cols_dict.get(best_idx, selected_features),
            best_params=best_params,
            n_features_selected=len(self.cols_dict.get(best_idx, selected_features)),
            test_metrics=self.metrics_dict.get(best_idx, {}),
            history=self.best_fitness_history,
            seed=best_individual_seed,
            model_type=self.model_type,
        )

    def train_from_result(
        self,
        result: LSTMOptimizationResult,
        epochs: Optional[int] = None,
        checkpoint_dir: str = 'checkpoints/',
        verbose: bool = True
    ) -> 'Trainer':
        """
        Train a model using the best parameters from an optimization result.

        For binary classification (hold=0, trade=1).

        Parameters
        ----------
        result : LSTMOptimizationResult
            Result from optimize() containing best parameters and features
        epochs : int, optional
            Number of training epochs. If None, uses epochs_per_eval.
        checkpoint_dir : str
            Directory for saving model checkpoints
        verbose : bool
            Print training progress and evaluation report

        Returns
        -------
        Trainer
            Trained Trainer object with model, preprocessor, and datasets
        """
        from .lstm import (
            DataPreprocessor, create_sequences,
            SignalDataset, TrainingConfig, ModelConfig,
            LSTMSignalPredictor, CNNLSTMSignalPredictor, Trainer
        )

        # Set seed for reproducibility (same seed used during optimization)
        set_seed(result.seed)

        # Get parameters from result
        params = result.best_params
        selected_features = result.selected_features
        model_type = result.model_type if hasattr(result, 'model_type') else self.model_type

        if verbose:
            print("=" * 60)
            print("Training Model from Optimization Result (Binary Classification)")
            print("=" * 60)
            print(f"Model type: {model_type}")
            print(f"Seed: {result.seed}")
            print(f"Selected features: {len(selected_features)}")
            print(f"Parameters: {params}")
            print()

        # Build selected DataFrame (use tradeable for binary classification)
        df_selected = self.df[['tradeable'] + selected_features].copy()

        # 1. Preprocess (uses 'tradeable' column)
        preprocessor = DataPreprocessor(target_shift=4)
        features, targets = preprocessor.fit_transform(df_selected)

        # 2. Create sequences (output_seq_length=1 for binary classification)
        input_seq_length = params.get('input_seq_length', 12)
        feat_seqs, tgt_seqs = create_sequences(
            features, targets,
            input_seq_length=input_seq_length,
            output_seq_length=1
        )

        # 3. Create dataset (no sequence validation for binary)
        dataset = SignalDataset(feat_seqs, tgt_seqs)

        # 4. Build training config
        training_epochs = epochs if epochs is not None else self.epochs_per_eval
        config = TrainingConfig(
            epochs=training_epochs,
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            weight_decay=params['weight_decay'],
            auto_class_weights=True,
            class_weight_power=params['class_weight_power'],
            focal_loss=True,
            focal_gamma=params['focal_gamma'],
            label_smoothing=params['label_smoothing'],
            scheduler_patience=params['scheduler_patience'],
            patience=10,
            checkpoint_dir=checkpoint_dir,
            verbose=verbose,
            bidirectional=True,
        )

        # 5. Build model config
        model_config = ModelConfig(
            input_size=preprocessor.get_num_features(),
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            input_seq_length=input_seq_length,
        )

        # 6. Create and train model based on model_type
        if model_type == 'cnn_lstm':
            model = CNNLSTMSignalPredictor(model_config)
        else:
            model = LSTMSignalPredictor(model_config)
        trainer = Trainer(model, config, preprocessor=preprocessor)
        trainer.train(dataset)

        # 7. Evaluate and print report
        if verbose:
            metrics = trainer.evaluate_all()
            trainer.print_evaluation_report(metrics)

        return trainer

    def __repr__(self) -> str:
        return (
            f"LSTMMetaheuristicOptimizer("
            f"model_type='{self.model_type}', "
            f"n_features={self.n_features}, "
            f"n_params={self.n_params}, "
            f"pop_size={self.pop_size}, "
            f"iterations={self.iterations})"
        )
