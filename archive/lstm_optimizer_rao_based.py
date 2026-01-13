"""
RAO3 Metaheuristic Optimizer for LSTM Hyperparameters and Feature Selection

Uses RAO3 algorithm to jointly optimize:
1. Feature selection from DatasetBuilder output
2. LSTM training hyperparameters

Objective: Maximize F1(entry) * F1(hold) * F1(exit) on test dataset

Based on rao3_firefly_optimizer.py implementation.
"""

import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class LSTMOptimizationResult:
    """Result from LSTM metaheuristic optimization."""
    best_fitness: float
    best_individual: np.ndarray
    selected_features: List[str]
    best_params: Dict[str, Any]
    n_features_selected: int
    test_metrics: Dict[str, float]
    history: List[float]  # Best fitness per iteration


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
    RAO3-based optimizer for LSTM hyperparameters and feature selection.

    Uses RAO3 algorithm (Rao Teaching-Learning Based Optimization variant 3)
    to jointly optimize feature selection and training hyperparameters.

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

    Examples
    --------
    >>> from crypto_analysis import DatasetBuilder, LSTMMetaheuristicOptimizer
    >>> builder = DatasetBuilder(data_dir="data/binance")
    >>> df = builder.build(symbol="BTC", threshold_pct=3.0)
    >>> optimizer = LSTMMetaheuristicOptimizer(df, pop_size=10, iterations=50)
    >>> result = optimizer.optimize()
    >>> print(f"Best fitness: {result.best_fitness}")
    """

    # Hyperparameter configurations with bounds and types
    HYPERPARAM_CONFIGS = [
        HyperparamConfig('class_weight_power', 0.5, 2.0, 'float', 'class_weight_power'),
        HyperparamConfig('focal_gamma', 1.0, 5.0, 'float', 'focal_gamma'),
        HyperparamConfig('learning_rate', 1e-4, 1e-2, 'float', 'learning_rate'),
        HyperparamConfig('dropout', 0.2, 0.5, 'float', 'dropout'),
        HyperparamConfig('hidden_size', 32, 256, 'int', 'hidden_size'),
        HyperparamConfig('num_layers', 1, 3, 'int', 'num_layers'),
        HyperparamConfig('weight_decay', 1e-4, 0.01, 'float', 'weight_decay'),
        HyperparamConfig('label_smoothing', 0.0, 0.2, 'float', 'label_smoothing'),
        HyperparamConfig('batch_size', 32, 128, 'int', 'batch_size'),
        HyperparamConfig('all_hold_weight', 1.0, 5.0, 'float', 'all_hold_weight'),
        HyperparamConfig('entry_exit_weight', 1.0, 3.0, 'float', 'entry_exit_weight'),
        HyperparamConfig('scheduler_patience', 5, 15, 'int', 'scheduler_patience'),
        HyperparamConfig('input_seq_length', 12, 12, 'int', 'input_seq_length'),
    ]

    # Columns to exclude from feature selection
    EXCLUDED_COLUMNS = ['date', 'signal', 'signal_pct_change', 'period_id']

    def __init__(
        self,
        df: pd.DataFrame,
        pop_size: int = 10,
        iterations: int = 50,
        n_workers: int = 4,
        min_features: int = 5,
        epochs_per_eval: int = 100,
        checkpoint_interval: int = 5,
        checkpoint_dir: str = 'lstm_optimization_checkpoints',
        verbose: bool = True
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

        # State for optimization (initialized in optimize())
        self.population: Optional[np.ndarray] = None
        self.fitness_pop: Optional[np.ndarray] = None
        self.cols_dict: Dict[int, List[str]] = {}
        self.metrics_dict: Dict[int, Dict] = {}
        self.best_fitness_history: List[float] = []

        if self.verbose:
            print(f"LSTMMetaheuristicOptimizer initialized:")
            print(f"  - Feature columns: {self.n_features}")
            print(f"  - Hyperparameters: {self.n_params}")
            print(f"  - Total dimension: {self.dimension}")
            print(f"  - Population size: {self.pop_size}")
            print(f"  - Iterations: {self.iterations}")
            print(f"  - Workers: {self.n_workers}")

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
        """
        # Feature selection: < 0 = not selected, >= 0 = selected
        feature_mask = individual[:self.n_features] >= 0
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
        Evaluate a single individual's fitness.

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
            DataPreprocessor, create_sequences, SequenceValidator,
            SignalDataset, TrainingConfig, ModelConfig,
            LSTMSignalPredictor, Trainer
        )

        selected_features, config_params = self._decode_individual(individual)

        # Ensure minimum features selected
        if len(selected_features) < self.min_features:
            return float('inf'), [], {}

        try:
            # Build selected DataFrame (keep signal + selected features)
            df_selected = self.df[['signal'] + selected_features].copy()

            # 1. Preprocess
            preprocessor = DataPreprocessor(target_shift=4)
            features, targets = preprocessor.fit_transform(df_selected)

            # 2. Create sequences (use input_seq_length from params)
            input_seq_length = config_params['input_seq_length']
            feat_seqs, tgt_seqs = create_sequences(features, targets, input_seq_length, 4)

            # 3. Validate and filter sequences
            validator = SequenceValidator()
            feat_seqs, tgt_seqs, seq_types = validator.filter_valid_sequences(
                feat_seqs, tgt_seqs
            )

            # Check we have enough sequences
            if len(feat_seqs) < 100:
                return float('inf'), selected_features, {}

            # 4. Create dataset
            dataset = SignalDataset(feat_seqs, tgt_seqs, seq_types)

            # 5. Build configs - force CPU for thread safety
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
                focal_loss=True,
                focal_gamma=config_params['focal_gamma'],
                label_smoothing=config_params['label_smoothing'],
                all_hold_weight=config_params['all_hold_weight'],
                entry_exit_weight=config_params['entry_exit_weight'],
                scheduler_patience=config_params['scheduler_patience'],
                patience=10,
                verbose=False,
                device='cpu',  # Force CPU for thread safety
            )

            model_config = ModelConfig(
                input_size=preprocessor.get_num_features(),
                hidden_size=config_params['hidden_size'],
                num_layers=config_params['num_layers'],
                dropout=config_params['dropout'],
            )

            # 6. Train
            model = LSTMSignalPredictor(model_config)
            trainer = Trainer(model, training_config, preprocessor=preprocessor)
            trainer.train(dataset)

            # 7. Evaluate on TEST set only
            metrics = trainer.evaluate(trainer.test_dataset, verbose=False)

            # 8. Compute fitness: product of F1 scores (minimize negative = maximize)
            entry_f1 = metrics['entry_f1']
            hold_f1 = metrics['hold_f1']
            exit_f1 = metrics['exit_f1']
            fitness = -(entry_f1 * hold_f1 * exit_f1)  # Negative for minimization

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
        Run RAO3 optimization algorithm.

        Parameters
        ----------
        start_iteration : int
            Starting iteration (used when resuming from checkpoint)

        Returns
        -------
        LSTMOptimizationResult
            Optimization result with best solution and history
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("Starting RAO3 Metaheuristic Optimization")
            print("=" * 60)

        # Initialize population if not resuming
        if self.population is None:
            self.population = np.random.uniform(
                low=self.lower_bound,
                high=self.upper_bound,
                size=(self.pop_size, self.dimension)
            )
            self.fitness_pop = np.zeros(self.pop_size)
            self.cols_dict = {}
            self.metrics_dict = {}
            self.best_fitness_history = []

            # Evaluate initial population
            if self.verbose:
                print("\nEvaluating initial population...")

            self.fitness_pop, self.cols_dict, self.metrics_dict = \
                self._evaluate_population_parallel(self.population, iteration=-1)

        # Track best/worst solutions across iterations
        best_solution_history = np.zeros((self.iterations + 1, self.dimension))
        best_fitness_iter_history = np.zeros(self.iterations + 1)

        # Main optimization loop
        for iter_num in range(start_iteration, self.iterations):
            if self.verbose:
                print(f"\n--- Iteration {iter_num + 1}/{self.iterations} ---")

            # # Track best and worst in current population
            # # ORIGINAL: Calculate best from current population
            # best_fitness_iter = np.min(self.fitness_pop)
            # best_idx_iter = np.argmin(self.fitness_pop)
            # best_solution_iter = self.population[best_idx_iter].copy()

            # EXPERIMENT: Use global best from history instead of current population best
            assert self.fitness_pop is not None and self.population is not None

            # Get current population's best
            current_best_fitness = np.min(self.fitness_pop)
            current_best_idx = np.argmin(self.fitness_pop)
            current_best_solution = self.population[current_best_idx].copy()

            if iter_num > 0:
                # Find best from all previous iterations
                best_hist_idx = np.argmin(best_fitness_iter_history[:iter_num])
                history_best_fitness = best_fitness_iter_history[best_hist_idx]

                # Compare history best with current population best (lower is better since negative)
                if history_best_fitness < current_best_fitness:
                    best_fitness_iter = history_best_fitness
                    best_solution_iter = best_solution_history[best_hist_idx].copy()
                else:
                    best_fitness_iter = current_best_fitness
                    best_solution_iter = current_best_solution
            else:
                # First iteration: use current population best
                best_fitness_iter = current_best_fitness
                best_solution_iter = current_best_solution

            worst_fitness_iter = np.max(self.fitness_pop[self.fitness_pop != float('inf')])
            worst_idx_iter = np.argmax(
                np.where(self.fitness_pop != float('inf'), self.fitness_pop, -np.inf)
            )
            worst_solution_iter = self.population[worst_idx_iter].copy()

            best_solution_history[iter_num] = best_solution_iter
            best_fitness_iter_history[iter_num] = best_fitness_iter
            self.best_fitness_history.append(-best_fitness_iter)  # Convert to positive

            if self.verbose:
                print(f"Current best fitness: {-best_fitness_iter:.4f}")

            # Generate new population using RAO3 update equations
            new_population = np.zeros_like(self.population)

            for pop_index, solution in enumerate(self.population):
                # Select random candidate (different from current)
                r = np.random.randint(self.pop_size)
                while pop_index == r:
                    r = np.random.randint(self.pop_size)
                random_candidate = self.population[r]

                fitness_solution = self.fitness_pop[pop_index]
                fitness_candidate = self.fitness_pop[r]

                # Compute weighted average of good solutions (better than current)
                # and bad solutions (worse than current)
                valid_mask = self.fitness_pop != float('inf')

                good_mask = (self.fitness_pop <= fitness_solution) & valid_mask
                bad_mask = (self.fitness_pop >= fitness_solution) & valid_mask

                if good_mask.sum() > 0 and self.fitness_pop[good_mask].sum() != 0:
                    good_weights = self.fitness_pop[good_mask]
                    good_sol = np.sum(
                        self.population[good_mask] * good_weights[:, np.newaxis],
                        axis=0
                    ) / good_weights.sum()
                else:
                    good_sol = solution.copy()

                if bad_mask.sum() > 0 and self.fitness_pop[bad_mask].sum() != 0:
                    bad_weights = self.fitness_pop[bad_mask]
                    bad_sol = np.sum(
                        self.population[bad_mask] * bad_weights[:, np.newaxis],
                        axis=0
                    ) / bad_weights.sum()
                else:
                    bad_sol = solution.copy()

                # Generate three random factors in [-0.5, 0.5], normalize to [-1, 1]
                r1 = np.random.rand() - 0.5
                r2 = np.random.rand() - 0.5
                r3 = np.random.rand() - 0.5

                # r_list = [r1, r2, r3]
                # r_min, r_max = np.min(r_list), np.max(r_list)
                # if r_max != r_min:
                #     r_list = [(2 * (r - r_min) / (r_max - r_min) - 1)/2 for r in r_list]
                # r1, r2, r3 = r_list

                # RAO3 update equation
                if fitness_solution > fitness_candidate:
                    # Current is worse than random candidate
                    new_sol = (
                        solution +
                        r1 * (best_solution_iter - worst_solution_iter) +
                        r2 * (solution - random_candidate) +
                        r3 * (good_sol - bad_sol)
                    )
                else:
                    # Current is better than or equal to random candidate
                    new_sol = (
                        solution +
                        r1 * (best_solution_iter - worst_solution_iter) +
                        r2 * (random_candidate - solution) +
                        r3 * (good_sol - bad_sol)
                    )

                # Clip to bounds
                new_sol = np.clip(new_sol, self.lower_bound, self.upper_bound)
                new_population[pop_index] = new_sol

            # Evaluate new population
            if self.verbose:
                print("Evaluating new candidates...")

            new_fitness, new_cols, new_metrics = self._evaluate_population_parallel(
                new_population, iteration=iter_num
            )

            # Greedy selection: replace only if new solution is better
            for idx in range(self.pop_size):
                if new_fitness[idx] < self.fitness_pop[idx]:
                    self.fitness_pop[idx] = new_fitness[idx]
                    self.population[idx] = new_population[idx].copy()
                    self.cols_dict[idx] = new_cols[idx]
                    self.metrics_dict[idx] = new_metrics[idx]

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
            print("Optimization Complete")
            print("=" * 60)
            print(f"Best fitness (F1 product): {best_fitness:.4f}")
            print(f"Selected features: {len(selected_features)}")
            print(f"Best parameters: {best_params}")
            if best_idx in self.metrics_dict and self.metrics_dict[best_idx]:
                metrics = self.metrics_dict[best_idx]
                print(f"Test metrics:")
                print(f"  Entry F1: {metrics.get('entry_f1', 0):.4f}")
                print(f"  Hold F1: {metrics.get('hold_f1', 0):.4f}")
                print(f"  Exit F1: {metrics.get('exit_f1', 0):.4f}")

        return LSTMOptimizationResult(
            best_fitness=best_fitness,
            best_individual=best_individual,
            selected_features=self.cols_dict.get(best_idx, selected_features),
            best_params=best_params,
            n_features_selected=len(self.cols_dict.get(best_idx, selected_features)),
            test_metrics=self.metrics_dict.get(best_idx, {}),
            history=self.best_fitness_history,
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
            DataPreprocessor, create_sequences, SequenceValidator,
            SignalDataset, TrainingConfig, ModelConfig,
            LSTMSignalPredictor, Trainer
        )

        # Get parameters from result
        params = result.best_params
        selected_features = result.selected_features

        if verbose:
            print("=" * 60)
            print("Training Model from Optimization Result")
            print("=" * 60)
            print(f"Selected features: {len(selected_features)}")
            print(f"Parameters: {params}")
            print()

        # Build selected DataFrame
        df_selected = self.df[['signal'] + selected_features].copy()

        # 1. Preprocess
        preprocessor = DataPreprocessor(target_shift=4)
        features, targets = preprocessor.fit_transform(df_selected)

        # 2. Create sequences
        input_seq_length = params.get('input_seq_length', 12)
        feat_seqs, tgt_seqs = create_sequences(
            features, targets,
            input_seq_length=input_seq_length,
            output_seq_length=4
        )

        # 3. Validate and filter
        validator = SequenceValidator()
        feat_seqs, tgt_seqs, seq_types = validator.filter_valid_sequences(feat_seqs, tgt_seqs)

        # 4. Create dataset
        dataset = SignalDataset(feat_seqs, tgt_seqs, seq_types)

        # 5. Build training config
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
            all_hold_weight=params['all_hold_weight'],
            entry_exit_weight=params['entry_exit_weight'],
            scheduler_patience=params['scheduler_patience'],
            patience=10,
            checkpoint_dir=checkpoint_dir,
            verbose=verbose,
        )

        # 6. Build model config
        model_config = ModelConfig(
            input_size=preprocessor.get_num_features(),
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            input_seq_length=input_seq_length,
        )

        # 7. Create and train model
        model = LSTMSignalPredictor(model_config)
        trainer = Trainer(model, config, preprocessor=preprocessor)
        history = trainer.train(dataset)

        # 8. Evaluate and print report
        if verbose:
            metrics = trainer.evaluate_all()
            trainer.print_evaluation_report(metrics)

        return trainer

    def __repr__(self) -> str:
        return (
            f"LSTMMetaheuristicOptimizer("
            f"n_features={self.n_features}, "
            f"n_params={self.n_params}, "
            f"pop_size={self.pop_size}, "
            f"iterations={self.iterations})"
        )
