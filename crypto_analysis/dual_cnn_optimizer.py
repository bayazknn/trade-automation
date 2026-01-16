"""
Dual-CNN LSTM Optimizer using APO (Artificial Protozoa Optimizer)

Jointly optimizes:
1. Binary feature selection from df_binary (114 binary indicator columns)
2. Technical feature selection from df_technical (76 technical + 5 OHLCV columns)
3. Dual-CNN LSTM model hyperparameters

Architecture:
- CNN1: Processes binary indicator signals (pure 0/1 entry/exit)
- CNN2: Processes technical indicators + OHLCV (scaled continuous)
- LSTM: Captures temporal dependencies from combined CNN outputs
- Classifier: Binary prediction (hold=0, trade=1)

Based on: Wang, X., et al. (2024). "Artificial Protozoa Optimizer (APO):
A novel bio-inspired metaheuristic algorithm for engineering optimization"
Knowledge-Based Systems, 111737.
"""

import json
import pickle
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class DualCNNOptimizationResult:
    """Result from Dual-CNN LSTM metaheuristic optimization."""
    best_fitness: float
    best_individual: np.ndarray
    selected_binary_features: List[str]
    selected_technical_features: List[str]
    best_params: Dict[str, Any]
    n_binary_selected: int
    n_technical_selected: int
    test_metrics: Dict[str, float]
    history: List[float]  # Best fitness per iteration
    seed: int = 42
    iteration: int = 0


@dataclass
class HyperparamConfig:
    """Hyperparameter bounds configuration."""
    name: str
    min_val: float
    max_val: float
    param_type: str  # 'float' or 'int'
    config_field: str  # Field name in model config


class DualCNNMetaheuristicOptimizer:
    """
    APO-based optimizer for Dual-CNN LSTM hyperparameters and dual feature selection.

    Uses Artificial Protozoa Optimizer (APO) algorithm to jointly optimize:
    - Binary feature selection (114 columns)
    - Technical + OHLCV feature selection (81 columns)
    - Dual-CNN LSTM architecture hyperparameters
    - Training hyperparameters

    Individual Vector Structure (Total: 213 dimensions):
    - [0:114]    Binary feature selection (threshold >= 0)
    - [114:195]  Technical + OHLCV selection (threshold >= 0)
    - [195:213]  Hyperparameters (18 params)

    Parameters
    ----------
    df_binary : pd.DataFrame
        DataFrame with binary indicator signals (114 columns)
    df_technical : pd.DataFrame
        DataFrame with technical indicators + OHLCV (81 columns)
    pop_size : int
        Population size (default: 10)
    iterations : int
        Number of optimization iterations (default: 50)
    n_workers : int
        Number of parallel workers for evaluation (default: 4)
    min_binary_features : int
        Minimum number of binary features to select (default: 3)
    min_technical_features : int
        Minimum number of technical features to select (default: 3)
    epochs_per_eval : int
        Training epochs per fitness evaluation (default: 100)
    verbose : bool
        Print progress (default: True)
    seed : int
        Random seed (default: 42)

    Examples
    --------
    >>> df_binary = pd.read_csv('doge.csv')
    >>> df_technical = pd.read_csv('doge_ti.csv')
    >>> optimizer = DualCNNMetaheuristicOptimizer(
    ...     df_binary, df_technical,
    ...     pop_size=10, iterations=50
    ... )
    >>> result = optimizer.optimize()
    >>> print(f"Best fitness: {result.best_fitness}")
    """

    # Columns to exclude from feature selection
    EXCLUDED_COLUMNS = ['date', 'signal', 'signal_pct_change', 'period_id', 'tradeable']

    # Hyperparameter configurations (Total: 20 params)
    # Note: kernel sizes use range [1, 4] which maps to odd values [3, 5, 7, 9]
    HYPERPARAM_CONFIGS = [
        # CNN1 (Binary branch) - kernel_size maps to odd: 1->3, 2->5, 3->7, 4->9
        HyperparamConfig('cnn1_kernel_size', 5, 9, 'int', 'cnn1_kernel_size'),
        HyperparamConfig('cnn1_num_channels', 64, 128, 'int', 'cnn1_num_channels'),
        HyperparamConfig('cnn1_num_layers', 2, 4, 'int', 'cnn1_num_layers'),
        # CNN2 (Technical branch) - kernel_size maps to odd: 1->3, 2->5, 3->7, 4->9
        HyperparamConfig('cnn2_kernel_size', 5, 9, 'int', 'cnn2_kernel_size'),
        HyperparamConfig('cnn2_num_channels', 64, 128, 'int', 'cnn2_num_channels'),
        HyperparamConfig('cnn2_num_layers', 2, 4, 'int', 'cnn2_num_layers'),
        # Fusion layer (between CNN concat and LSTM)
        HyperparamConfig('fusion_hidden_size', 64, 256, 'int', 'fusion_hidden_size'),
        HyperparamConfig('fusion_dropout', 0.05, 0.10, 'float', 'fusion_dropout'),
        # LSTM
        HyperparamConfig('lstm_hidden_size', 64, 256, 'int', 'lstm_hidden_size'),
        HyperparamConfig('lstm_num_layers', 1, 3, 'int', 'lstm_num_layers'),
        HyperparamConfig('lstm_dropout', 0.01, 0.15, 'float', 'lstm_dropout'),
        # Classifier
        HyperparamConfig('classifier_hidden_size', 0, 128, 'int', 'classifier_hidden_size'),
        HyperparamConfig('classifier_dropout', 0.01, 0.15, 'float', 'classifier_dropout'),
        # Training
        HyperparamConfig('learning_rate', 0.0005, 0.009, 'float', 'learning_rate'),
        HyperparamConfig('batch_size', 16, 64, 'int', 'batch_size'),
        HyperparamConfig('weight_decay', 0.0005, 0.02, 'float', 'weight_decay'),
        # Focal loss gamma: focusing parameter for hard example mining (1.5-3.0)
        HyperparamConfig('focal_gamma', 1.5, 3.0, 'float', 'focal_gamma'),
        HyperparamConfig('label_smoothing', 0.02, 0.08, 'float', 'label_smoothing'),
        HyperparamConfig('input_seq_length', 12, 21, 'int', 'input_seq_length'),
        HyperparamConfig('scheduler_patience', 5, 10, 'int', 'scheduler_patience'),
    ]

    def __init__(
        self,
        df_binary: pd.DataFrame,
        df_technical: pd.DataFrame,
        pop_size: int = 10,
        iterations: int = 50,
        n_workers: int = 4,
        min_binary_features: int = 3,
        min_technical_features: int = 3,
        epochs_per_eval: int = 100,
        checkpoint_interval: int = 5,
        checkpoint_dir: str = 'dual_cnn_checkpoints',
        verbose: bool = True,
        np_neighbors: int = 2,
        pf_max: float = 0.18,
        seed: int = 42,
    ):
        self.df_binary = df_binary
        self.df_technical = df_technical
        self.pop_size = pop_size
        self.iterations = iterations
        self.n_workers = n_workers
        self.min_binary_features = min_binary_features
        self.min_technical_features = min_technical_features
        self.epochs_per_eval = epochs_per_eval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.verbose = verbose
        self.np_neighbors = np_neighbors
        self.pf_max = pf_max
        self.seed = seed
        self.run_id: Optional[str] = None

        # Identify feature columns
        self.binary_columns = [
            col for col in df_binary.columns
            if col not in self.EXCLUDED_COLUMNS
        ]
        self.technical_columns = [
            col for col in df_technical.columns
            if col not in self.EXCLUDED_COLUMNS
        ]

        self.n_binary = len(self.binary_columns)
        self.n_technical = len(self.technical_columns)
        self.n_params = len(self.HYPERPARAM_CONFIGS)

        # Total dimension: binary + technical + hyperparams
        self.dimension = self.n_binary + self.n_technical + self.n_params

        # Build bounds arrays
        self._build_bounds()

        # State for optimization
        self.population: Optional[np.ndarray] = None
        self.fitness_pop: Optional[np.ndarray] = None
        self.binary_cols_dict: Dict[int, List[str]] = {}
        self.technical_cols_dict: Dict[int, List[str]] = {}
        self.metrics_dict: Dict[int, Dict] = {}
        self.best_fitness_history: List[float] = []

        if self.verbose:
            print(f"DualCNNMetaheuristicOptimizer (APO) initialized:")
            print(f"  - Binary features: {self.n_binary}")
            print(f"  - Technical features: {self.n_technical}")
            print(f"  - Hyperparameters: {self.n_params}")
            print(f"  - Total dimension: {self.dimension}")
            print(f"  - Population size: {self.pop_size}")
            print(f"  - Iterations: {self.iterations}")
            print(f"  - Workers: {self.n_workers}")

    def _build_bounds(self):
        """Build lower and upper bound arrays for the search space."""
        # Binary feature bounds: [-100, 100]
        binary_lower = np.full(self.n_binary, -100.0)
        binary_upper = np.full(self.n_binary, 100.0)

        # Technical feature bounds: [-100, 100]
        technical_lower = np.full(self.n_technical, -100.0)
        technical_upper = np.full(self.n_technical, 100.0)

        # Hyperparameter bounds
        param_lower = np.array([cfg.min_val for cfg in self.HYPERPARAM_CONFIGS])
        param_upper = np.array([cfg.max_val for cfg in self.HYPERPARAM_CONFIGS])

        self.lower_bound = np.concatenate([binary_lower, technical_lower, param_lower])
        self.upper_bound = np.concatenate([binary_upper, technical_upper, param_upper])

    def _decode_individual(
        self,
        individual: np.ndarray
    ) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        Decode individual vector into selected features and hyperparameters.

        Parameters
        ----------
        individual : np.ndarray
            Individual vector of shape (dimension,)

        Returns
        -------
        tuple
            (selected_binary, selected_technical, config_params)
        """
        # Extract feature selection values
        binary_values = individual[:self.n_binary]
        technical_values = individual[self.n_binary:self.n_binary + self.n_technical]

        # Simple threshold >= 0 for selection
        binary_mask = binary_values >= 0
        technical_mask = technical_values >= 0

        selected_binary = [
            col for col, sel in zip(self.binary_columns, binary_mask) if sel
        ]
        selected_technical = [
            col for col, sel in zip(self.technical_columns, technical_mask) if sel
        ]

        # Extract and convert hyperparameters
        params = individual[self.n_binary + self.n_technical:]
        config_params = {}
        for i, cfg in enumerate(self.HYPERPARAM_CONFIGS):
            val = params[i]
            if cfg.param_type == 'int':
                val = int(round(val))
            config_params[cfg.config_field] = val

        # Map kernel sizes from [1,4] to odd values [3,5,7,9]
        # Formula: odd_kernel = 2*val + 1 where val in [1,2,3,4] gives [3,5,7,9]
        config_params['cnn1_kernel_size'] = 2 * config_params['cnn1_kernel_size'] + 1
        config_params['cnn2_kernel_size'] = 2 * config_params['cnn2_kernel_size'] + 1

        return selected_binary, selected_technical, config_params

    def _evaluate_individual(
        self,
        individual: np.ndarray,
        idx: int
    ) -> Tuple[float, List[str], List[str], Dict]:
        """
        Evaluate a single individual's fitness.

        Returns
        -------
        tuple
            (fitness, selected_binary, selected_technical, metrics)
        """
        from .lstm.dual_model import DualModelConfig, DualCNNLSTMPredictor
        from .lstm.dual_preprocessor import (
            DualDataPreprocessor, create_dual_sequences, DualSignalDataset
        )
        from .lstm.loss import FocalBinaryLoss
        from torch.utils.data import DataLoader, Subset
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        from sklearn.metrics import precision_recall_fscore_support

        selected_binary, selected_technical, config_params = self._decode_individual(individual)

        # Check minimum features
        if len(selected_binary) < self.min_binary_features:
            return float('inf'), [], [], {}
        if len(selected_technical) < self.min_technical_features:
            return float('inf'), [], [], {}

        # Set seed for reproducibility
        individual_seed = self.seed + idx
        set_seed(individual_seed)

        try:
            # Build selected DataFrames
            df_binary_selected = self.df_binary[['tradeable'] + selected_binary].copy()
            df_technical_selected = self.df_technical[['tradeable'] + selected_technical].copy()

            # Preprocess
            preprocessor = DualDataPreprocessor(target_shift=4)
            binary_feat, technical_feat, targets = preprocessor.fit_transform(
                df_binary_selected, df_technical_selected
            )

            # Create sequences
            input_seq_length = config_params['input_seq_length']
            binary_seqs, technical_seqs, target_seqs = create_dual_sequences(
                binary_feat, technical_feat, targets,
                input_seq_length=input_seq_length,
                output_seq_length=1
            )

            if len(binary_seqs) < 100:
                return float('inf'), selected_binary, selected_technical, {}

            # Create dataset
            dataset = DualSignalDataset(binary_seqs, technical_seqs, target_seqs)

            # Split dataset (60% train, 20% val, 20% test)
            n = len(dataset)
            test_size = int(n * 0.2)
            val_size = int(n * 0.2)
            train_size = n - val_size - test_size

            train_indices = list(range(train_size))
            val_indices = list(range(train_size, train_size + val_size))
            test_indices = list(range(train_size + val_size, n))

            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            test_dataset = Subset(dataset, test_indices)

            # Create model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model_config = DualModelConfig(
                cnn1_input_features=len(selected_binary),
                cnn2_input_features=len(selected_technical),
                cnn1_kernel_size=config_params['cnn1_kernel_size'],
                cnn1_num_channels=config_params['cnn1_num_channels'],
                cnn1_num_layers=config_params['cnn1_num_layers'],
                cnn2_kernel_size=config_params['cnn2_kernel_size'],
                cnn2_num_channels=config_params['cnn2_num_channels'],
                cnn2_num_layers=config_params['cnn2_num_layers'],
                fusion_hidden_size=config_params['fusion_hidden_size'],
                fusion_dropout=config_params['fusion_dropout'],
                lstm_hidden_size=config_params['lstm_hidden_size'],
                lstm_num_layers=config_params['lstm_num_layers'],
                lstm_dropout=config_params['lstm_dropout'],
                classifier_hidden_size=config_params['classifier_hidden_size'],
                classifier_dropout=config_params['classifier_dropout'],
                input_seq_length=input_seq_length,
            )

            model = DualCNNLSTMPredictor(model_config).to(device)

            # Compute class weights (direct inverse frequency - no power dampening)
            targets_flat = dataset.targets.flatten()
            class_counts = torch.bincount(targets_flat, minlength=2).float()
            class_counts = torch.clamp(class_counts, min=1.0)
            # Direct inverse frequency: trade weight will be ~10x hold weight for 10:1 imbalance
            class_weights = class_counts.sum() / (2.0 * class_counts)
            class_weights = class_weights.to(device)

            # Initialize classifier bias to favor minority class
            n_trade = (targets_flat == 1).sum().item()
            class_prior = max(n_trade / len(targets_flat), 0.05)
            model._init_classifier_bias(class_prior)

            # Loss function: FocalBinaryLoss for hard example mining
            loss_fn = FocalBinaryLoss(
                gamma=config_params['focal_gamma'],
                class_weights=class_weights,
                label_smoothing=config_params['label_smoothing']
            ).to(device)

            # Optimizer and scheduler
            optimizer = AdamW(
                model.parameters(),
                lr=config_params['learning_rate'],
                weight_decay=config_params['weight_decay']
            )
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5,
                patience=config_params['scheduler_patience']
            )

            # Create weighted sampler for oversampling minority class
            from torch.utils.data import WeightedRandomSampler

            # Get targets from training subset
            train_targets = []
            for idx in train_indices:
                train_targets.append(dataset.targets[idx].item())
            train_targets = torch.tensor(train_targets)

            # Compute sample weights (inverse class frequency)
            class_counts = torch.bincount(train_targets, minlength=2).float()
            class_counts = torch.clamp(class_counts, min=1.0)
            sample_weights = 1.0 / class_counts[train_targets]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_targets),
                replacement=True
            )

            # Data loaders with weighted sampling for training
            train_loader = DataLoader(
                train_dataset,
                batch_size=config_params['batch_size'],
                sampler=sampler,  # Use weighted sampler instead of shuffle
                collate_fn=DualSignalDataset.collate_fn
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config_params['batch_size'],
                shuffle=False,
                collate_fn=DualSignalDataset.collate_fn
            )

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 25  # Increased from 10 to allow more time to learn minority class

            for epoch in range(self.epochs_per_eval):
                # Train
                model.train()
                for batch in train_loader:
                    binary_feat_batch = batch['binary_features'].to(device)
                    technical_feat_batch = batch['technical_features'].to(device)
                    targets_batch = batch['targets'].to(device)

                    optimizer.zero_grad()
                    logits = model(binary_feat_batch, technical_feat_batch)
                    loss = loss_fn(logits, targets_batch)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # Validate
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        binary_feat_batch = batch['binary_features'].to(device)
                        technical_feat_batch = batch['technical_features'].to(device)
                        targets_batch = batch['targets'].to(device)

                        logits = model(binary_feat_batch, technical_feat_batch)
                        loss = loss_fn(logits, targets_batch)
                        val_loss += loss.item() * binary_feat_batch.size(0)

                val_loss /= len(val_dataset)
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            # Evaluate on test set
            test_loader = DataLoader(
                test_dataset,
                batch_size=config_params['batch_size'],
                shuffle=False,
                collate_fn=DualSignalDataset.collate_fn
            )

            all_predictions = []
            all_targets = []

            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    binary_feat_batch = batch['binary_features'].to(device)
                    technical_feat_batch = batch['technical_features'].to(device)
                    targets_batch = batch['targets'].to(device)

                    logits = model(binary_feat_batch, technical_feat_batch)
                    predictions = torch.argmax(logits, dim=-1)

                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(targets_batch.cpu().numpy())

            y_pred = np.concatenate(all_predictions).flatten()
            y_true = np.concatenate(all_targets).flatten()

            # Calculate metrics
            precision_per_class, recall_per_class, f1_per_class, support_per_class = \
                precision_recall_fscore_support(
                    y_true, y_pred, average=None, labels=[0, 1], zero_division=0
                )

            metrics = {
                'hold_precision': float(precision_per_class[0]),
                'hold_recall': float(recall_per_class[0]),
                'hold_f1': float(f1_per_class[0]),
                'hold_support': int(support_per_class[0]) if support_per_class is not None else 0,
                'trade_precision': float(precision_per_class[1]),
                'trade_recall': float(recall_per_class[1]),
                'trade_f1': float(f1_per_class[1]),
                'trade_support': int(support_per_class[1]) if support_per_class is not None else 0,
            }

            # Calculate fitness (same as lstm_optimizer.py)
            trade_precision = metrics['trade_precision']
            trade_recall = metrics['trade_recall']
            hold_f1 = metrics['hold_f1']

            eps = 1e-10

            # Recall-biased geometric mean for trade class
            beta = 2.0
            p_exp = 1.0 / (1.0 + beta)
            r_exp = beta / (1.0 + beta)

            trade_score = ((trade_precision + eps) ** p_exp) * ((trade_recall + eps) ** r_exp)

            # Recall floor penalty
            min_recall = 0.30
            trade_recall_factor = min(trade_recall / min_recall, 1.0) if min_recall > 0 else 1.0

            # Combined fitness
            fitness = -(trade_score * hold_f1 * trade_recall_factor)

            return fitness, selected_binary, selected_technical, metrics

        except Exception as e:
            if self.verbose:
                print(f"  Individual {idx} failed: {e}")
            return float('inf'), [], [], {}

    def _evaluate_population_parallel(
        self,
        population: np.ndarray,
        iteration: int = -1
    ) -> Tuple[np.ndarray, Dict[int, List[str]], Dict[int, List[str]], Dict[int, Dict]]:
        """Evaluate population in parallel."""
        fitness_values = np.full(len(population), float('inf'))
        binary_cols_dict: Dict[int, List[str]] = {}
        technical_cols_dict: Dict[int, List[str]] = {}
        metrics_dict: Dict[int, Dict] = {}

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self._evaluate_individual, ind, idx): idx
                for idx, ind in enumerate(population)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    fitness, binary_cols, technical_cols, metrics = future.result()
                    fitness_values[idx] = fitness
                    binary_cols_dict[idx] = binary_cols
                    technical_cols_dict[idx] = technical_cols
                    metrics_dict[idx] = metrics

                    if self.verbose:
                        if fitness != float('inf'):
                            print(
                                f"iter:{iteration} indv:{idx} fitness:{-fitness:.4f} "
                                f"binary:{len(binary_cols)} technical:{len(technical_cols)}"
                            )
                        else:
                            print(f"iter:{iteration} indv:{idx} fitness:invalid")

                except Exception as e:
                    print(f"iter:{iteration} indv:{idx} error: {e}")

        return fitness_values, binary_cols_dict, technical_cols_dict, metrics_dict

    def _print_best_metrics(
        self,
        fitness_values: np.ndarray,
        metrics_dict: Dict[int, Dict],
        iteration: int = -1,
    ) -> None:
        """Print per-class metrics table for the best individual."""
        if not self.verbose:
            return

        best_idx = int(np.argmin(fitness_values))
        best_fitness = fitness_values[best_idx]

        if best_fitness == float('inf') or best_idx not in metrics_dict:
            return

        metrics = metrics_dict[best_idx]
        if not metrics:
            return

        iter_label = "INIT" if iteration == -1 else f"ITER {iteration}"
        print(f"\n{'-' * 72}")
        print(f"{iter_label} - Best Individual Per-Class Metrics (fitness: {-best_fitness:.6f})")
        print(f"{'-' * 72}")
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print(f"{'-' * 56}")

        print(f"{'hold':<10} "
              f"{metrics.get('hold_precision', 0):<12.4f} "
              f"{metrics.get('hold_recall', 0):<12.4f} "
              f"{metrics.get('hold_f1', 0):<12.4f} "
              f"{metrics.get('hold_support', 0):<10}")

        print(f"{'trade':<10} "
              f"{metrics.get('trade_precision', 0):<12.4f} "
              f"{metrics.get('trade_recall', 0):<12.4f} "
              f"{metrics.get('trade_f1', 0):<12.4f} "
              f"{metrics.get('trade_support', 0):<10}")

    def save_checkpoint(self, path: Optional[str] = None, iteration: int = 0):
        """Save current optimization state to checkpoint file."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if path is None:
            path = self.checkpoint_dir / f"checkpoint_iter_{iteration}.pkl"
        else:
            path = Path(path)

        checkpoint = {
            'iteration': iteration,
            'population': self.population,
            'fitness_pop': self.fitness_pop,
            'binary_cols_dict': self.binary_cols_dict,
            'technical_cols_dict': self.technical_cols_dict,
            'metrics_dict': self.metrics_dict,
            'best_fitness_history': self.best_fitness_history,
            'binary_columns': self.binary_columns,
            'technical_columns': self.technical_columns,
            'n_binary': self.n_binary,
            'n_technical': self.n_technical,
            'n_params': self.n_params,
        }

        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

        if self.verbose:
            print(f"Checkpoint saved to {path}")

    def optimize(self, start_iteration: int = 0) -> DualCNNOptimizationResult:
        """
        Run APO (Artificial Protozoa Optimizer) algorithm.

        Parameters
        ----------
        start_iteration : int
            Starting iteration (used when resuming from checkpoint)

        Returns
        -------
        DualCNNOptimizationResult
            Optimization result with best solution and history
        """
        self.run_id = uuid.uuid4().hex[:8]

        if self.verbose:
            print("\n" + "=" * 60)
            print("Starting Dual-CNN LSTM APO Optimization")
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
                low=Xmin, high=Xmax, size=(ps, dim)
            )
            self.fitness_pop = np.zeros(ps)
            self.binary_cols_dict = {}
            self.technical_cols_dict = {}
            self.metrics_dict = {}
            self.best_fitness_history = []

            if self.verbose:
                print("\nEvaluating initial population...")

            (self.fitness_pop, self.binary_cols_dict,
             self.technical_cols_dict, self.metrics_dict) = \
                self._evaluate_population_parallel(self.population, iteration=-1)

            self._print_best_metrics(self.fitness_pop, self.metrics_dict, iteration=-1)

        # Pre-allocate arrays for APO
        new_population = np.zeros((ps, dim))
        epn = np.zeros((np_pairs, dim))

        # Main APO optimization loop
        for iter_num in range(start_iteration, iter_max):
            if self.verbose:
                print(f"\n--- Iteration {iter_num + 1}/{iter_max} ---")

            # Sort population by fitness
            sort_indices = np.argsort(self.fitness_pop)
            self.fitness_pop = self.fitness_pop[sort_indices]
            self.population = self.population[sort_indices]

            # Reindex dictionaries after sorting
            new_binary_cols: Dict[int, List[str]] = {}
            new_technical_cols: Dict[int, List[str]] = {}
            new_metrics: Dict[int, Dict] = {}
            for new_idx, old_idx in enumerate(sort_indices):
                if old_idx in self.binary_cols_dict:
                    new_binary_cols[new_idx] = self.binary_cols_dict[old_idx]
                if old_idx in self.technical_cols_dict:
                    new_technical_cols[new_idx] = self.technical_cols_dict[old_idx]
                if old_idx in self.metrics_dict:
                    new_metrics[new_idx] = self.metrics_dict[old_idx]
            self.binary_cols_dict = new_binary_cols
            self.technical_cols_dict = new_technical_cols
            self.metrics_dict = new_metrics

            best_fitness_iter = self.fitness_pop[0]
            self.best_fitness_history.append(-best_fitness_iter)

            if self.verbose:
                print(f"Current best fitness: {-best_fitness_iter:.4f}")

            # APO proportion fraction
            pf = self.pf_max * np.random.random()
            num_dr = int(np.ceil(ps * pf))
            ri = set(np.random.permutation(ps)[:num_dr]) if num_dr > 0 else set()

            # Process each protozoan
            for i in range(ps):
                i_matlab = i + 1

                if i in ri:
                    # Dormancy or Reproduction
                    pdr = 0.5 * (1 + np.cos((1 - i_matlab / ps) * np.pi))

                    if np.random.random() < pdr:
                        # Dormancy
                        new_population[i] = Xmin + np.random.random(dim) * (Xmax - Xmin)
                    else:
                        # Reproduction
                        Flag = 1 if np.random.random() < 0.5 else -1
                        Mr = np.zeros(dim)
                        num_mr = int(np.ceil(np.random.random() * dim))
                        if num_mr > 0:
                            Mr[np.random.permutation(dim)[:num_mr]] = 1
                        rand_scalar = np.random.random()
                        rand_vec = np.random.random(dim)
                        new_population[i] = (
                            self.population[i] +
                            Flag * rand_scalar * (Xmin + rand_vec * (Xmax - Xmin)) * Mr
                        )
                else:
                    # Foraging
                    f = np.random.random() * (1 + np.cos((iter_num + 1) / iter_max * np.pi))
                    Mf = np.zeros(dim)
                    num_mf = int(np.ceil(dim * i_matlab / ps))
                    Mf[np.random.permutation(dim)[:num_mf]] = 1

                    pah = 0.5 * (1 + np.cos((iter_num + 1) / iter_max * np.pi))

                    if np.random.random() < pah:
                        # Autotrophic foraging
                        j = np.random.randint(0, ps)
                        epn.fill(0)

                        for k in range(1, np_pairs + 1):
                            if i == 0:
                                km = 0
                                kp = 1 + np.random.randint(0, ps - 1) if ps > 1 else 0
                            elif i == ps - 1:
                                km = np.random.randint(0, ps - 1) if ps > 1 else 0
                                kp = ps - 1
                            else:
                                km = np.random.randint(0, i)
                                kp = i + 1 + np.random.randint(0, ps - i - 1)

                            wa = np.exp(-np.abs(
                                self.fitness_pop[km] /
                                (self.fitness_pop[kp] + np.finfo(float).eps)
                            ))
                            epn[k-1] = wa * (self.population[km] - self.population[kp])

                        neighbor_sum = np.sum(epn, axis=0) / np_pairs
                        new_population[i] = (
                            self.population[i] +
                            f * (self.population[j] - self.population[i] + neighbor_sum) * Mf
                        )
                    else:
                        # Heterotrophic foraging
                        epn.fill(0)

                        for k in range(1, np_pairs + 1):
                            if i == 0:
                                imk = 0
                                ipk = k
                            elif i == ps - 1:
                                imk = ps - 1 - k
                                ipk = ps - 1
                            else:
                                imk = i - k
                                ipk = i + k

                            imk = max(0, min(imk, ps - 1))
                            ipk = max(0, min(ipk, ps - 1))

                            wh = np.exp(-np.abs(
                                self.fitness_pop[imk] /
                                (self.fitness_pop[ipk] + np.finfo(float).eps)
                            ))
                            epn[k-1] = wh * (self.population[imk] - self.population[ipk])

                        Flag = 1 if np.random.random() < 0.5 else -1
                        rand_vec = np.random.random(dim)
                        Xnear = (1 + Flag * rand_vec * (1 - (iter_num + 1) / iter_max)) * self.population[i]

                        neighbor_sum = np.sum(epn, axis=0) / np_pairs
                        new_population[i] = (
                            self.population[i] +
                            f * (Xnear - self.population[i] + neighbor_sum) * Mf
                        )

            # Boundary check
            new_population = np.clip(new_population, Xmin, Xmax)

            # Evaluate new population
            if self.verbose:
                print("Evaluating new candidates...")

            (new_fitness, new_binary_cols, new_technical_cols, new_metrics) = \
                self._evaluate_population_parallel(new_population, iteration=iter_num)

            # Greedy selection
            for idx in range(ps):
                if new_fitness[idx] < self.fitness_pop[idx]:
                    self.fitness_pop[idx] = new_fitness[idx]
                    self.population[idx] = new_population[idx].copy()
                    self.binary_cols_dict[idx] = new_binary_cols[idx]
                    self.technical_cols_dict[idx] = new_technical_cols[idx]
                    self.metrics_dict[idx] = new_metrics[idx]

            self._print_best_metrics(self.fitness_pop, self.metrics_dict, iteration=iter_num)

            # Save checkpoint periodically
            if (iter_num + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(iteration=iter_num + 1)

        # Final best solution
        best_idx = np.argmin(self.fitness_pop)
        best_fitness = -self.fitness_pop[best_idx]
        best_individual = self.population[best_idx]
        selected_binary, selected_technical, best_params = self._decode_individual(best_individual)

        if self.verbose:
            print("\n" + "=" * 60)
            print("Optimization Complete (Dual-CNN LSTM)")
            print("=" * 60)
            print(f"Best fitness: {best_fitness:.4f}")
            print(f"Binary features: {len(self.binary_cols_dict.get(best_idx, selected_binary))}")
            print(f"Technical features: {len(self.technical_cols_dict.get(best_idx, selected_technical))}")
            print(f"Best parameters: {best_params}")

        best_individual_seed = self.seed + best_idx

        return DualCNNOptimizationResult(
            best_fitness=best_fitness,
            best_individual=best_individual,
            selected_binary_features=self.binary_cols_dict.get(best_idx, selected_binary),
            selected_technical_features=self.technical_cols_dict.get(best_idx, selected_technical),
            best_params=best_params,
            n_binary_selected=len(self.binary_cols_dict.get(best_idx, selected_binary)),
            n_technical_selected=len(self.technical_cols_dict.get(best_idx, selected_technical)),
            test_metrics=self.metrics_dict.get(best_idx, {}),
            history=self.best_fitness_history,
            seed=best_individual_seed,
            iteration=len(self.best_fitness_history),
        )

    def save_artifacts(
        self,
        result: DualCNNOptimizationResult,
        output_dir: str,
    ) -> Dict[str, str]:
        """
        Save optimization artifacts (model, preprocessor, metadata).

        Parameters
        ----------
        result : DualCNNOptimizationResult
            Result from optimize()
        output_dir : str
            Directory to save artifacts

        Returns
        -------
        dict
            Paths to saved artifacts: {'model': path, 'preprocessor': path, 'metadata': path}
        """
        from .lstm.dual_model import DualModelConfig, DualCNNLSTMPredictor
        from .lstm.dual_preprocessor import (
            DualDataPreprocessor, create_dual_sequences, DualSignalDataset
        )
        from .lstm.loss import FocalBinaryLoss
        from torch.utils.data import DataLoader, Subset
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Set seed for reproducibility
        set_seed(result.seed)

        params = result.best_params
        selected_binary = result.selected_binary_features
        selected_technical = result.selected_technical_features

        if self.verbose:
            print(f"\nTraining final model for artifact saving...")

        # Build selected DataFrames
        df_binary_selected = self.df_binary[['tradeable'] + selected_binary].copy()
        df_technical_selected = self.df_technical[['tradeable'] + selected_technical].copy()

        # Preprocess
        preprocessor = DualDataPreprocessor(target_shift=4)
        binary_feat, technical_feat, targets = preprocessor.fit_transform(
            df_binary_selected, df_technical_selected
        )

        # Create sequences
        input_seq_length = params['input_seq_length']
        binary_seqs, technical_seqs, target_seqs = create_dual_sequences(
            binary_feat, technical_feat, targets,
            input_seq_length=input_seq_length,
            output_seq_length=1
        )

        # Create dataset and split
        dataset = DualSignalDataset(binary_seqs, technical_seqs, target_seqs)
        n = len(dataset)
        test_size = int(n * 0.2)
        val_size = int(n * 0.2)
        train_size = n - val_size - test_size

        train_indices = list(range(train_size))
        val_indices = list(range(train_size, train_size + val_size))

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_config = DualModelConfig(
            cnn1_input_features=len(selected_binary),
            cnn2_input_features=len(selected_technical),
            cnn1_kernel_size=params['cnn1_kernel_size'],
            cnn1_num_channels=params['cnn1_num_channels'],
            cnn1_num_layers=params['cnn1_num_layers'],
            cnn2_kernel_size=params['cnn2_kernel_size'],
            cnn2_num_channels=params['cnn2_num_channels'],
            cnn2_num_layers=params['cnn2_num_layers'],
            fusion_hidden_size=params['fusion_hidden_size'],
            fusion_dropout=params['fusion_dropout'],
            lstm_hidden_size=params['lstm_hidden_size'],
            lstm_num_layers=params['lstm_num_layers'],
            lstm_dropout=params['lstm_dropout'],
            classifier_hidden_size=params['classifier_hidden_size'],
            classifier_dropout=params['classifier_dropout'],
            input_seq_length=input_seq_length,
        )

        model = DualCNNLSTMPredictor(model_config).to(device)

        # Compute class weights (direct inverse frequency - no power dampening)
        targets_flat = dataset.targets.flatten()
        class_counts = torch.bincount(targets_flat, minlength=2).float()
        class_counts = torch.clamp(class_counts, min=1.0)
        # Direct inverse frequency: trade weight will be ~10x hold weight for 10:1 imbalance
        class_weights = class_counts.sum() / (2.0 * class_counts)
        class_weights = class_weights.to(device)

        # Initialize classifier bias to favor minority class
        n_trade = (targets_flat == 1).sum().item()
        class_prior = max(n_trade / len(targets_flat), 0.05)
        model._init_classifier_bias(class_prior)

        # Loss function: FocalBinaryLoss for hard example mining
        loss_fn = FocalBinaryLoss(
            gamma=params['focal_gamma'],
            class_weights=class_weights,
            label_smoothing=params['label_smoothing']
        ).to(device)

        optimizer = AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=params['scheduler_patience']
        )

        # Create weighted sampler for oversampling minority class
        from torch.utils.data import WeightedRandomSampler

        train_targets = []
        for idx in train_indices:
            train_targets.append(dataset.targets[idx].item())
        train_targets_tensor = torch.tensor(train_targets)

        class_counts_train = torch.bincount(train_targets_tensor, minlength=2).float()
        class_counts_train = torch.clamp(class_counts_train, min=1.0)
        sample_weights = 1.0 / class_counts_train[train_targets_tensor]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_targets),
            replacement=True
        )

        # Data loaders with weighted sampling
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            sampler=sampler,
            collate_fn=DualSignalDataset.collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            collate_fn=DualSignalDataset.collate_fn
        )

        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 25  # Increased from 10 to allow more time to learn minority class

        for epoch in range(self.epochs_per_eval):
            model.train()
            for batch in train_loader:
                binary_feat_batch = batch['binary_features'].to(device)
                technical_feat_batch = batch['technical_features'].to(device)
                targets_batch = batch['targets'].to(device)

                optimizer.zero_grad()
                logits = model(binary_feat_batch, technical_feat_batch)
                loss = loss_fn(logits, targets_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    binary_feat_batch = batch['binary_features'].to(device)
                    technical_feat_batch = batch['technical_features'].to(device)
                    targets_batch = batch['targets'].to(device)

                    logits = model(binary_feat_batch, technical_feat_batch)
                    loss = loss_fn(logits, targets_batch)
                    val_loss += loss.item() * binary_feat_batch.size(0)

            val_loss /= len(val_dataset)
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save artifacts
        model_path = output_path / 'best_model.pt'
        preprocessor_path = output_path / 'preprocessor.pkl'
        metadata_path = output_path / 'metadata.json'

        # Save model checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
        }
        torch.save(checkpoint, model_path)

        # Save preprocessor
        preprocessor.save(preprocessor_path)

        # Save metadata
        metadata = {
            'model_type': 'dual_cnn_lstm',
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'optimization_run_id': self.run_id,
            'feature_selection': {
                'binary_features': selected_binary,
                'technical_features': selected_technical,
                'n_binary_selected': len(selected_binary),
                'n_technical_selected': len(selected_technical),
            },
            'model_architecture': {
                'cnn1': {
                    'input_features': len(selected_binary),
                    'kernel_size': params['cnn1_kernel_size'],
                    'num_channels': params['cnn1_num_channels'],
                    'num_layers': params['cnn1_num_layers'],
                },
                'cnn2': {
                    'input_features': len(selected_technical),
                    'kernel_size': params['cnn2_kernel_size'],
                    'num_channels': params['cnn2_num_channels'],
                    'num_layers': params['cnn2_num_layers'],
                },
                'fusion': {
                    'hidden_size': params['fusion_hidden_size'],
                    'dropout': params['fusion_dropout'],
                },
                'lstm': {
                    'hidden_size': params['lstm_hidden_size'],
                    'num_layers': params['lstm_num_layers'],
                    'dropout': params['lstm_dropout'],
                    'bidirectional': False,
                },
                'classifier': {
                    'hidden_size': params['classifier_hidden_size'],
                    'dropout': params['classifier_dropout'],
                    'num_classes': 2,
                },
            },
            'training_config': {
                'epochs': self.epochs_per_eval,
                'batch_size': params['batch_size'],
                'learning_rate': params['learning_rate'],
                'weight_decay': params['weight_decay'],
                'focal_gamma': params['focal_gamma'],
                'label_smoothing': params['label_smoothing'],
                'input_seq_length': params['input_seq_length'],
                'scheduler_patience': params['scheduler_patience'],
            },
            'optimization_results': {
                'best_fitness': result.best_fitness,
                'seed': result.seed,
                'iteration': result.iteration,
                'test_metrics': result.test_metrics,
            },
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        if self.verbose:
            print(f"Artifacts saved to {output_path}:")
            print(f"  - Model: {model_path}")
            print(f"  - Preprocessor: {preprocessor_path}")
            print(f"  - Metadata: {metadata_path}")

        return {
            'model': str(model_path),
            'preprocessor': str(preprocessor_path),
            'metadata': str(metadata_path),
        }

    def __repr__(self) -> str:
        return (
            f"DualCNNMetaheuristicOptimizer("
            f"n_binary={self.n_binary}, "
            f"n_technical={self.n_technical}, "
            f"n_params={self.n_params}, "
            f"pop_size={self.pop_size}, "
            f"iterations={self.iterations})"
        )
