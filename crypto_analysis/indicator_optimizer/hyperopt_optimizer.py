"""
Hyperopt Optimizer Module

Freqtrade-style optimization using Optuna to find parameters
that maximize trading performance (Total Profit).
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .base import BaseIndicatorOptimizer, OptimizationResult


class HyperoptOptimizer:
    """
    Freqtrade-style hyperparameter optimization using Optuna.

    Optimizes indicator parameters to maximize trading performance.
    Primary loss function: Total Profit (user selected).
    """

    LOSS_FUNCTIONS = ["profit", "sharpe", "win_rate", "sortino"]

    def __init__(
        self,
        loss_function: str = "profit",
        n_jobs: int = 1,
        seed: Optional[int] = None
    ):
        """
        Initialize hyperopt optimizer.

        Parameters
        ----------
        loss_function : str
            Loss function to optimize. Options:
            - 'profit': Total profit (default, user selected)
            - 'sharpe': Sharpe ratio
            - 'win_rate': Win rate percentage
            - 'sortino': Sortino ratio
        n_jobs : int
            Number of parallel jobs for optimization
        seed : int, optional
            Random seed for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperopt optimization. "
                "Install with: pip install optuna"
            )

        if loss_function not in self.LOSS_FUNCTIONS:
            raise ValueError(
                f"Unknown loss function: {loss_function}. "
                f"Available: {self.LOSS_FUNCTIONS}"
            )

        self.loss_function = loss_function
        self.n_jobs = n_jobs
        self.seed = seed

    def optimize(
        self,
        indicator: BaseIndicatorOptimizer,
        df: pd.DataFrame,
        n_trials: int = 100,
        initial_capital: float = 10000.0,
        timeout: Optional[int] = None,
        verbose: bool = False
    ) -> OptimizationResult:
        """
        Perform hyperparameter optimization using Optuna.

        Parameters
        ----------
        indicator : BaseIndicatorOptimizer
            Indicator to optimize
        df : pd.DataFrame
            OHLCV DataFrame
        n_trials : int
            Number of optimization trials
        initial_capital : float
            Starting capital for trade simulation
        timeout : int, optional
            Maximum time in seconds for optimization
        verbose : bool
            Print progress information

        Returns
        -------
        OptimizationResult
            Best parameters and score
        """
        params = indicator.get_optimizable_params()
        all_results = []

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            # Suggest parameters
            suggested_params = {}
            for name, config in params.items():
                param_range = config.get("range", [config["default"], config["default"]])
                param_type = config.get("type", "int")

                if param_type == "int":
                    suggested_params[name] = trial.suggest_int(
                        name, int(param_range[0]), int(param_range[1])
                    )
                else:  # float
                    suggested_params[name] = trial.suggest_float(
                        name, float(param_range[0]), float(param_range[1])
                    )

            try:
                # Generate signals
                signals_df = indicator.generate_signals(df.copy(), **suggested_params)

                # Simulate trades
                trade_stats = indicator.simulate_trades(signals_df, initial_capital)

                # Calculate loss (negative because Optuna minimizes)
                loss = self._calculate_loss(trade_stats)

                # Store result
                all_results.append({
                    "params": suggested_params.copy(),
                    "trade_stats": trade_stats,
                    "loss": loss
                })

                return loss

            except Exception as e:
                if verbose:
                    print(f"Trial failed: {e}")
                return float("inf")  # Return worst possible value

        # Create study
        sampler = TPESampler(seed=self.seed)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler
        )

        # Optimize
        if verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=verbose
        )

        # Get best result
        best_params = study.best_params
        best_score = -study.best_value  # Negate to get positive score

        if verbose:
            print(f"Best score ({self.loss_function}): {best_score:.4f}")
            print(f"Best params: {best_params}")

        return OptimizationResult(
            indicator_name=indicator.indicator_name,
            best_params=best_params,
            score=best_score,
            optimization_type="hyperopt",
            all_results=all_results,
            metadata={
                "loss_function": self.loss_function,
                "n_trials": n_trials,
                "study": study
            }
        )

    def _calculate_loss(self, trade_stats: Dict[str, Any]) -> float:
        """
        Calculate loss value based on selected loss function.

        Returns negative value because Optuna minimizes.
        """
        if self.loss_function == "profit":
            # Maximize total profit
            return -trade_stats.get("total_profit", 0)

        elif self.loss_function == "sharpe":
            # Maximize Sharpe ratio
            return -trade_stats.get("sharpe_ratio", 0)

        elif self.loss_function == "win_rate":
            # Maximize win rate
            return -trade_stats.get("win_rate", 0)

        elif self.loss_function == "sortino":
            # Calculate Sortino ratio
            trades = trade_stats.get("trades", [])
            if not trades:
                return float("inf")

            profits = [t["profit_pct"] for t in trades]
            mean_return = np.mean(profits)
            downside = [p for p in profits if p < 0]

            if not downside:
                # No losses, perfect score
                return -100.0

            downside_std = np.std(downside)
            if downside_std == 0:
                return -mean_return

            sortino = mean_return / downside_std
            return -sortino

        return 0.0

    def optimize_with_callback(
        self,
        indicator: BaseIndicatorOptimizer,
        df: pd.DataFrame,
        callback: Callable[[Dict], None],
        n_trials: int = 100,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize with a callback function called after each trial.

        Useful for progress tracking or early stopping based on
        custom criteria.

        Parameters
        ----------
        callback : callable
            Function called with trial results dict
        """
        # Store original results
        results = []

        def wrapped_callback(study, trial):
            result = {
                "trial_number": trial.number,
                "params": trial.params,
                "value": trial.value,
                "best_value": study.best_value
            }
            results.append(result)
            callback(result)

        # Run optimization with callback
        params = indicator.get_optimizable_params()

        def objective(trial):
            suggested_params = {}
            for name, config in params.items():
                param_range = config.get("range", [config["default"], config["default"]])
                param_type = config.get("type", "int")

                if param_type == "int":
                    suggested_params[name] = trial.suggest_int(
                        name, int(param_range[0]), int(param_range[1])
                    )
                else:
                    suggested_params[name] = trial.suggest_float(
                        name, float(param_range[0]), float(param_range[1])
                    )

            signals_df = indicator.generate_signals(df.copy(), **suggested_params)
            trade_stats = indicator.simulate_trades(
                signals_df, kwargs.get("initial_capital", 10000.0)
            )
            return self._calculate_loss(trade_stats)

        sampler = TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[wrapped_callback],
            n_jobs=self.n_jobs
        )

        return OptimizationResult(
            indicator_name=indicator.indicator_name,
            best_params=study.best_params,
            score=-study.best_value,
            optimization_type="hyperopt",
            all_results=results,
            metadata={"loss_function": self.loss_function}
        )


def calculate_profit_loss(trade_stats: Dict) -> float:
    """Standalone profit loss function."""
    return -trade_stats.get("total_profit", 0)


def calculate_sharpe_loss(trade_stats: Dict) -> float:
    """Standalone Sharpe ratio loss function."""
    return -trade_stats.get("sharpe_ratio", 0)


def calculate_win_rate_loss(trade_stats: Dict) -> float:
    """Standalone win rate loss function."""
    return -trade_stats.get("win_rate", 0)
