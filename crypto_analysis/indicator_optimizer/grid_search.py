"""
Grid Search Optimizer Module

Performs grid search over indicator parameter space to find
parameters that best match SignalPopulator's entry/exit signals.

Optimized for performance with:
- Smart grid reduction before generation
- Random sampling for large parameter spaces
- Early stopping when perfect score found
- Memory-efficient iteration
"""

import itertools
import random
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseIndicatorOptimizer, OptimizationResult
from .signal_matcher import SignalMatcher, MatchResult


class GridSearchOptimizer:
    """
    Grid search optimizer for indicator parameters.

    Searches parameter space to find configuration that produces
    signals best matching the target SignalPopulator signals.
    """

    def __init__(
        self,
        signal_matcher: Optional[SignalMatcher] = None,
        step_sizes: Optional[Dict[str, int]] = None
    ):
        """
        Initialize grid search optimizer.

        Parameters
        ----------
        signal_matcher : SignalMatcher, optional
            Signal matcher instance (creates default if not provided)
        step_sizes : dict, optional
            Step sizes for each parameter type.
            Default: {"int": 1, "float": 0.1}
        """
        self.signal_matcher = signal_matcher or SignalMatcher()
        self.step_sizes = step_sizes or {"int": 1, "float": 0.1}

    def optimize(
        self,
        indicator: BaseIndicatorOptimizer,
        df: pd.DataFrame,
        target_df: pd.DataFrame,
        param_overrides: Optional[Dict[str, Dict]] = None,
        max_combinations: int = 3000,
        early_stop_score: Optional[float] = None,
        store_all_results: bool = False,
        verbose: bool = False
    ) -> OptimizationResult:
        """
        Perform grid search optimization.

        Parameters
        ----------
        indicator : BaseIndicatorOptimizer
            Indicator to optimize
        df : pd.DataFrame
            OHLCV DataFrame
        target_df : pd.DataFrame
            DataFrame with target signals from SignalPopulator
        param_overrides : dict, optional
            Override default parameter ranges
        max_combinations : int
            Maximum number of parameter combinations to try (default: 3000)
        early_stop_score : float, optional
            Stop early if this score is reached
        store_all_results : bool
            Whether to store all results (default: False for memory efficiency)
        verbose : bool
            Print progress information

        Returns
        -------
        OptimizationResult
            Best parameters and score
        """
        # Get optimizable parameters
        params = indicator.get_optimizable_params()

        if not params:
            # No parameters to optimize, use defaults
            return OptimizationResult(
                indicator_name=indicator.indicator_name,
                best_params={},
                score=0,
                optimization_type="grid_search",
                all_results=[],
                metadata={"total_combinations": 0, "evaluated": 0}
            )

        # Apply overrides if provided
        if param_overrides:
            for name, override in param_overrides.items():
                if name in params:
                    params[name].update(override)

        # Generate parameter values with smart reduction
        param_values = self._generate_param_values(params, max_combinations)

        # Calculate actual combinations
        total_combinations = 1
        for values in param_values.values():
            total_combinations *= len(values)

        # Use random sampling if still too many
        use_random = total_combinations > max_combinations

        if verbose:
            method = "random sampling" if use_random else "full grid"
            print(f"Grid search ({method}): {min(total_combinations, max_combinations)} "
                  f"combinations for {indicator.indicator_name}")

        # Evaluate combinations
        best_score = -1
        best_params = {}
        all_results = [] if store_all_results else None
        evaluated = 0

        # Get iterator (random or full grid)
        if use_random:
            param_iter = self._random_sample_iterator(param_values, max_combinations)
        else:
            param_iter = self._grid_iterator(param_values)

        for param_combo in param_iter:
            if evaluated >= max_combinations:
                break

            try:
                # Generate signals with these parameters
                signals_df = indicator.generate_signals(df, **param_combo)

                # Calculate match score
                match_result = self.signal_matcher.calculate_match_score(
                    signals_df, target_df
                )

                score = match_result.combined_score
                evaluated += 1

                if store_all_results:
                    all_results.append({
                        "params": param_combo.copy(),
                        "score": score
                    })

                if score > best_score:
                    best_score = score
                    best_params = param_combo.copy()

                    # Early stopping
                    if early_stop_score and score >= early_stop_score:
                        if verbose:
                            print(f"  Early stop at score {score}")
                        break

                if verbose and evaluated % 200 == 0:
                    print(f"  [{indicator.indicator_name}] {evaluated}/{max_combinations}, best: {best_score}")

            except Exception as e:
                if verbose:
                    print(f"  Error with params {param_combo}: {e}")
                continue

        if verbose:
            print(f"  [{indicator.indicator_name}] Done - score: {best_score}")

        return OptimizationResult(
            indicator_name=indicator.indicator_name,
            best_params=best_params,
            score=best_score,
            optimization_type="grid_search",
            all_results=all_results or [],
            metadata={
                "total_combinations": total_combinations,
                "evaluated": evaluated,
                "method": "random" if use_random else "grid"
            }
        )

    def _generate_param_values(
        self,
        params: Dict[str, Dict],
        max_combinations: int
    ) -> Dict[str, List]:
        """
        Generate parameter values with smart step sizing.

        Calculates steps to keep total combinations under max_combinations.
        """
        # First pass: calculate with current step sizes
        param_values = {}
        for name, config in params.items():
            param_range = config.get("range", [config["default"], config["default"]])
            param_type = config.get("type", "int")

            if param_type == "int":
                step = self.step_sizes.get("int", 3)
                values = list(range(int(param_range[0]), int(param_range[1]) + 1, step))
            else:
                step = self.step_sizes.get("float", 0.5)
                values = list(np.arange(param_range[0], param_range[1] + step, step))
                values = [round(v, 3) for v in values]

            # Ensure at least 2 values
            if len(values) < 2:
                values = [param_range[0], param_range[1]]

            param_values[name] = values

        # Calculate total
        total = 1
        for v in param_values.values():
            total *= len(v)

        # If too many, reduce proportionally
        if total > max_combinations * 2:
            n_params = len(param_values)
            # Target values per param to get under max_combinations
            target_per_param = int(max_combinations ** (1 / n_params)) + 1

            for name, values in param_values.items():
                if len(values) > target_per_param:
                    # Take evenly spaced values
                    indices = np.linspace(0, len(values) - 1, target_per_param, dtype=int)
                    param_values[name] = [values[i] for i in indices]

        return param_values

    def _grid_iterator(
        self,
        param_values: Dict[str, List]
    ) -> Iterator[Dict[str, Any]]:
        """Memory-efficient grid iterator."""
        keys = list(param_values.keys())
        value_lists = [param_values[k] for k in keys]

        for combo in itertools.product(*value_lists):
            yield dict(zip(keys, combo))

    def _random_sample_iterator(
        self,
        param_values: Dict[str, List],
        n_samples: int
    ) -> Iterator[Dict[str, Any]]:
        """Random sampling iterator for large spaces."""
        keys = list(param_values.keys())
        seen = set()

        for _ in range(n_samples * 2):  # Over-sample to handle duplicates
            if len(seen) >= n_samples:
                break

            # Random selection
            combo = tuple(random.choice(param_values[k]) for k in keys)

            if combo not in seen:
                seen.add(combo)
                yield dict(zip(keys, combo))

    def optimize_single_param(
        self,
        indicator: BaseIndicatorOptimizer,
        df: pd.DataFrame,
        target_df: pd.DataFrame,
        param_name: str,
        other_params: Optional[Dict] = None,
        verbose: bool = False
    ) -> Tuple[Any, float]:
        """
        Optimize a single parameter while keeping others fixed.

        Useful for understanding parameter sensitivity.

        Parameters
        ----------
        param_name : str
            Name of parameter to optimize
        other_params : dict, optional
            Fixed values for other parameters

        Returns
        -------
        tuple
            (best_value, best_score)
        """
        params = indicator.get_optimizable_params()

        if param_name not in params:
            raise ValueError(f"Unknown parameter: {param_name}")

        # Get values for the parameter to optimize
        config = params[param_name]
        param_range = config.get("range", [config["default"], config["default"]])
        param_type = config.get("type", "int")

        if param_type == "int":
            step = self.step_sizes.get("int", 3)
            values = list(range(int(param_range[0]), int(param_range[1]) + 1, step))
        else:
            step = self.step_sizes.get("float", 0.5)
            values = list(np.arange(param_range[0], param_range[1] + step, step))

        # Set base parameters
        base_params = other_params or indicator.get_default_params()

        best_value = values[0]
        best_score = -1

        for value in values:
            try:
                test_params = base_params.copy()
                test_params[param_name] = value

                signals_df = indicator.generate_signals(df, **test_params)
                match_result = self.signal_matcher.calculate_match_score(
                    signals_df, target_df
                )

                if match_result.combined_score > best_score:
                    best_score = match_result.combined_score
                    best_value = value

            except Exception:
                continue

        if verbose:
            print(f"Best {param_name}: {best_value} (score: {best_score})")

        return best_value, best_score
