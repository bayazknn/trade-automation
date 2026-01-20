"""
Optuna-based indicator optimization with automatic Grid/TPE sampler selection.
"""
import logging
from functools import reduce
from operator import mul
from typing import Any, Callable, Dict, Optional

import optuna
import pandas as pd

from .config import OptimizationConfig, OptimizationResult
from .fitness import VectorbtFitness

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """
    Optuna-based optimizer with automatic Grid/TPE sampler selection.

    Selects GridSampler for small parameter spaces (<500 combinations),
    TPESampler for larger spaces.
    """

    def __init__(self, config: OptimizationConfig = None):
        """
        Initialize OptunaOptimizer.

        Parameters
        ----------
        config : OptimizationConfig, optional
            Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.fitness_calculator = VectorbtFitness(self.config)

    def optimize(
        self,
        indicator,
        df: pd.DataFrame,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> OptimizationResult:
        """
        Optimize indicator parameters using Optuna.

        Parameters
        ----------
        indicator : BaseIndicatorOptimizer
            Indicator instance with get_optimizable_params() method
        df : pd.DataFrame
            OHLCV DataFrame for backtesting
        callback : callable, optional
            Progress callback(trial_number, best_value)

        Returns
        -------
        OptimizationResult
            Optimization results including best params and score
        """
        param_space = indicator.get_optimizable_params()
        total_combos = self._count_combinations(param_space)

        logger.info(
            f"Optimizing {indicator.indicator_name}: "
            f"{total_combos} combinations"
        )

        # Select sampler based on parameter space size
        if total_combos <= self.config.max_grid_combinations:
            sampler, n_trials = self._create_grid_sampler(param_space)
            sampler_type = "grid"
        else:
            sampler = optuna.samplers.TPESampler(seed=42)
            n_trials = self.config.max_tpe_trials
            sampler_type = "tpe"

        logger.info(f"Using {sampler_type} sampler with {n_trials} trials")

        # Create objective function
        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial, param_space)

            try:
                # Calculate indicator values first, then generate signals
                df_with_indicator = indicator.calculate_indicator(df.copy(), **params)
                entries = indicator.generate_entry_signal(df_with_indicator, **params)
                exits = indicator.generate_exit_signal(df_with_indicator, **params)

                # Calculate fitness
                score = self.fitness_calculator.quick_score(
                    close=df["close"],
                    entries=entries,
                    exits=exits,
                )

                if callback:
                    callback(trial.number, study.best_value if study.best_value else score)

                return score

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return float("-inf")

        # Create and run study
        study = optuna.create_study(
            direction=self.config.direction,
            sampler=sampler,
            study_name=self.config.study_name,
            storage=self.config.storage,
            load_if_exists=True if self.config.storage else False,
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=self.config.n_jobs,
            show_progress_bar=False,
        )

        # Get detailed fitness for best params
        best_params = study.best_params
        df_with_indicator = indicator.calculate_indicator(df.copy(), **best_params)
        entries = indicator.generate_entry_signal(df_with_indicator, **best_params)
        exits = indicator.generate_exit_signal(df_with_indicator, **best_params)
        fitness_details = self.fitness_calculator.calculate(
            close=df["close"],
            entries=entries,
            exits=exits,
        )

        return OptimizationResult(
            indicator_name=indicator.indicator_name,
            best_params=best_params,
            score=study.best_value,
            sampler_type=sampler_type,
            n_trials=len(study.trials),
            fitness_details=fitness_details,
        )

    def _count_combinations(self, param_space: Dict[str, Dict]) -> int:
        """
        Count total parameter combinations.

        Parameters
        ----------
        param_space : dict
            Parameter space from indicator.get_optimizable_params()

        Returns
        -------
        int
            Total number of combinations
        """
        counts = []
        for param_name, param_def in param_space.items():
            param_range = param_def["range"]
            param_type = param_def.get("type", "int")

            if param_type == "int":
                # Discrete integer range
                count = param_range[1] - param_range[0] + 1
            elif param_type == "float":
                # Discretize float range (assume step of 1.0 for counting)
                count = int(param_range[1] - param_range[0] + 1)
            elif param_type == "categorical":
                count = len(param_range)
            else:
                count = param_range[1] - param_range[0] + 1

            counts.append(count)

        return reduce(mul, counts, 1) if counts else 1

    def _create_grid_sampler(
        self, param_space: Dict[str, Dict]
    ) -> tuple:
        """
        Create GridSampler with exhaustive search space.

        Parameters
        ----------
        param_space : dict
            Parameter space definition

        Returns
        -------
        tuple
            (GridSampler, n_trials)
        """
        search_space = {}

        for param_name, param_def in param_space.items():
            param_range = param_def["range"]
            param_type = param_def.get("type", "int")

            if param_type == "int":
                values = list(range(param_range[0], param_range[1] + 1))
            elif param_type == "float":
                # Create discrete steps for float
                step = param_def.get("step", 1.0)
                values = []
                v = param_range[0]
                while v <= param_range[1]:
                    values.append(v)
                    v += step
            elif param_type == "categorical":
                values = list(param_range)
            else:
                values = list(range(param_range[0], param_range[1] + 1))

            search_space[param_name] = values

        n_trials = reduce(mul, [len(v) for v in search_space.values()], 1)
        sampler = optuna.samplers.GridSampler(search_space)

        return sampler, n_trials

    def _suggest_params(
        self, trial: optuna.Trial, param_space: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Suggest parameters for trial.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object
        param_space : dict
            Parameter space definition

        Returns
        -------
        dict
            Suggested parameters
        """
        params = {}

        for param_name, param_def in param_space.items():
            param_range = param_def["range"]
            param_type = param_def.get("type", "int")

            if param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name, param_range[0], param_range[1]
                )
            elif param_type == "float":
                step = param_def.get("step")
                params[param_name] = trial.suggest_float(
                    param_name, param_range[0], param_range[1], step=step
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_range
                )
            else:
                # Default to int
                params[param_name] = trial.suggest_int(
                    param_name, param_range[0], param_range[1]
                )

        return params


def optimize_indicator_params(
    indicator,
    df: pd.DataFrame,
    max_grid_combinations: int = 500,
    max_tpe_trials: int = 1000,
    n_jobs: int = 4,
    init_cash: float = 100.0,
    fees: float = 0.001,
) -> OptimizationResult:
    """
    Convenience function to optimize indicator parameters.

    Parameters
    ----------
    indicator : BaseIndicatorOptimizer
        Indicator instance
    df : pd.DataFrame
        OHLCV DataFrame
    max_grid_combinations : int
        Threshold for Grid vs TPE
    max_tpe_trials : int
        Max TPE trials
    n_jobs : int
        Optuna parallel jobs
    init_cash : float
        Initial capital
    fees : float
        Trading fees

    Returns
    -------
    OptimizationResult
        Optimization results
    """
    config = OptimizationConfig(
        max_grid_combinations=max_grid_combinations,
        max_tpe_trials=max_tpe_trials,
        n_jobs=n_jobs,
        init_cash=init_cash,
        fees=fees,
    )
    optimizer = OptunaOptimizer(config)
    return optimizer.optimize(indicator, df)
