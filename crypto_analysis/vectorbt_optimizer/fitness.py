"""
Vectorbt-based fitness calculation for indicator optimization.

Uses vectorbt Portfolio.from_signals for high-performance backtesting.
"""
import numpy as np
import pandas as pd
import vectorbt as vbt

from .config import FitnessResult, OptimizationConfig


class VectorbtFitness:
    """Calculate fitness using vectorbt Portfolio simulation."""

    def __init__(self, config: OptimizationConfig = None):
        """
        Initialize VectorbtFitness.

        Parameters
        ----------
        config : OptimizationConfig, optional
            Optimization configuration with init_cash and fees
        """
        self.config = config or OptimizationConfig()

    def calculate(
        self,
        close: pd.Series,
        entries: pd.Series,
        exits: pd.Series,
    ) -> FitnessResult:
        """
        Calculate fitness metrics using vectorbt backtesting.

        Parameters
        ----------
        close : pd.Series
            Close prices
        entries : pd.Series
            Boolean series for entry signals
        exits : pd.Series
            Boolean series for exit signals

        Returns
        -------
        FitnessResult
            Fitness metrics including total return, trades, win rate, sharpe
        """
        # Ensure boolean arrays
        entries = entries.astype(bool)
        exits = exits.astype(bool)

        # Create portfolio from signals
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=self.config.init_cash,
            fees=self.config.fees,
            freq="1h",  # Assuming hourly data
        )

        # Extract metrics with safe defaults
        try:
            total_return = float(pf.total_return())
        except Exception:
            total_return = 0.0

        try:
            num_trades = int(pf.trades.count())
        except Exception:
            num_trades = 0

        try:
            win_rate = float(pf.trades.win_rate())
            if np.isnan(win_rate):
                win_rate = 0.0
        except Exception:
            win_rate = 0.0

        try:
            sharpe_ratio = float(pf.sharpe_ratio())
            if np.isnan(sharpe_ratio):
                sharpe_ratio = 0.0
        except Exception:
            sharpe_ratio = 0.0

        try:
            max_drawdown = float(pf.max_drawdown())
            if np.isnan(max_drawdown):
                max_drawdown = 0.0
        except Exception:
            max_drawdown = 0.0

        return FitnessResult(
            total_return=total_return,
            num_trades=num_trades,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
        )

    def calculate_from_df(
        self,
        df: pd.DataFrame,
        entry_col: str = "entry",
        exit_col: str = "exit",
        close_col: str = "close",
    ) -> FitnessResult:
        """
        Calculate fitness from DataFrame with signal columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with close prices and signal columns
        entry_col : str
            Name of entry signal column
        exit_col : str
            Name of exit signal column
        close_col : str
            Name of close price column

        Returns
        -------
        FitnessResult
            Fitness metrics
        """
        return self.calculate(
            close=df[close_col],
            entries=df[entry_col],
            exits=df[exit_col],
        )

    def quick_score(
        self,
        close: pd.Series,
        entries: pd.Series,
        exits: pd.Series,
    ) -> float:
        """
        Quick fitness score calculation (total return only).

        Useful for optimization where only the score matters.

        Parameters
        ----------
        close : pd.Series
            Close prices
        entries : pd.Series
            Boolean series for entry signals
        exits : pd.Series
            Boolean series for exit signals

        Returns
        -------
        float
            Total return as fitness score
        """
        entries = entries.astype(bool)
        exits = exits.astype(bool)

        try:
            pf = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                init_cash=self.config.init_cash,
                fees=self.config.fees,
                freq="1h",
            )
            return float(pf.total_return())
        except Exception:
            return -np.inf  # Return worst possible score on error


def calculate_fitness(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    init_cash: float = 100.0,
    fees: float = 0.001,
) -> FitnessResult:
    """
    Convenience function to calculate fitness.

    Parameters
    ----------
    close : pd.Series
        Close prices
    entries : pd.Series
        Entry signals
    exits : pd.Series
        Exit signals
    init_cash : float
        Initial capital
    fees : float
        Trading fees

    Returns
    -------
    FitnessResult
        Fitness metrics
    """
    config = OptimizationConfig(init_cash=init_cash, fees=fees)
    fitness = VectorbtFitness(config)
    return fitness.calculate(close, entries, exits)
