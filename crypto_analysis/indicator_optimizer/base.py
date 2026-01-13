"""
Base Module

Abstract base class for indicator optimization, following
the IStrategy pattern from Freqtrade.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    indicator_name: str
    best_params: Dict[str, Any]
    score: float
    optimization_type: str  # 'grid_search' or 'hyperopt'
    all_results: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (
            f"OptimizationResult({self.indicator_name}, "
            f"type={self.optimization_type}, score={self.score:.4f})"
        )


class BaseIndicatorOptimizer(ABC):
    """
    Abstract base class for indicator optimization.

    Similar to IStrategy pattern from Freqtrade, each indicator
    inherits from this class and implements its specific logic.
    """

    # Class attributes (to be overridden by subclasses)
    indicator_name: str = ""
    talib_function: str = ""
    category: str = ""
    outputs: List[str] = []

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize indicator optimizer.

        Parameters
        ----------
        config : dict, optional
            Indicator configuration from config_loader
        """
        self.config = config or {}
        self._default_params = self.get_default_params()

    @abstractmethod
    def get_optimizable_params(self) -> Dict[str, Dict]:
        """
        Return dict of optimizable parameters.

        Returns
        -------
        dict
            param_name -> {default, range, type}
        """
        pass

    def get_default_params(self) -> Dict[str, Any]:
        """Get default values for all parameters."""
        params = self.get_optimizable_params()
        return {name: p["default"] for name, p in params.items()}

    @abstractmethod
    def calculate_indicator(
        self,
        df: pd.DataFrame,
        **params
    ) -> pd.DataFrame:
        """
        Calculate indicator values with given parameters.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame
        **params
            Indicator parameters

        Returns
        -------
        pd.DataFrame
            DataFrame with indicator columns added
        """
        pass

    @abstractmethod
    def generate_entry_signal(
        self,
        df: pd.DataFrame,
        **params
    ) -> pd.Series:
        """
        Generate entry signals based on indicator values.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with indicator values
        **params
            Signal parameters (constants, factors, etc.)

        Returns
        -------
        pd.Series
            Boolean series where True = entry signal
        """
        pass

    @abstractmethod
    def generate_exit_signal(
        self,
        df: pd.DataFrame,
        **params
    ) -> pd.Series:
        """
        Generate exit signals based on indicator values.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with indicator values
        **params
            Signal parameters (constants, factors, etc.)

        Returns
        -------
        pd.Series
            Boolean series where True = exit signal
        """
        pass

    def generate_signals(
        self,
        df: pd.DataFrame,
        **params
    ) -> pd.DataFrame:
        """
        Generate both entry and exit signals.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame
        **params
            All parameters for indicator and signals

        Returns
        -------
        pd.DataFrame
            DataFrame with entry_signal and exit_signal columns
        """
        # Merge default params with provided params
        full_params = {**self._default_params, **params}

        # Calculate indicator
        df = self.calculate_indicator(df.copy(), **full_params)

        # Generate signals
        df["entry_signal"] = self.generate_entry_signal(df, **full_params)
        df["exit_signal"] = self.generate_exit_signal(df, **full_params)

        return df

    def _crossed_above(
        self,
        series1: pd.Series,
        series2: Union[pd.Series, float]
    ) -> pd.Series:
        """
        Check if series1 crossed above series2.

        Equivalent to qtpylib.crossed_above()
        """
        if isinstance(series2, (int, float)):
            series2 = pd.Series([series2] * len(series1), index=series1.index)

        return (series1 > series2) & (series1.shift(1) <= series2.shift(1))

    def _crossed_below(
        self,
        series1: pd.Series,
        series2: Union[pd.Series, float]
    ) -> pd.Series:
        """
        Check if series1 crossed below series2.

        Equivalent to qtpylib.crossed_below()
        """
        if isinstance(series2, (int, float)):
            series2 = pd.Series([series2] * len(series1), index=series1.index)

        return (series1 < series2) & (series1.shift(1) >= series2.shift(1))

    def simulate_trades(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Vectorized trade simulation using NumPy operations.
        ~100x faster than row-by-row iteration.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with entry_signal, exit_signal, and close columns
        initial_capital : float
            Starting capital for simulation

        Returns
        -------
        dict
            Trade statistics including total_profit, win_rate, etc.
        """
        # Extract arrays (avoid pandas overhead)
        entry_signals = df.get("entry_signal", pd.Series(False, index=df.index))
        exit_signals = df.get("exit_signal", pd.Series(False, index=df.index))

        entry_signals = entry_signals.fillna(False).values.astype(bool)
        exit_signals = exit_signals.fillna(False).values.astype(bool)
        close_prices = df["close"].values.astype(np.float64)

        # Get signal indices
        entry_idx = np.where(entry_signals)[0]
        exit_idx = np.where(exit_signals)[0]

        # Handle no signals case
        if len(entry_idx) == 0:
            return self._empty_trade_result()

        # Find valid trade pairs using searchsorted
        valid_entries, valid_exits = self._find_trade_pairs(entry_idx, exit_idx)

        # Handle edge cases
        if len(valid_entries) == 0:
            return self._empty_trade_result()

        # Get prices
        entry_prices = close_prices[valid_entries]
        num_completed = len(valid_exits)
        has_open = len(valid_entries) > num_completed

        # Build exit prices array
        if num_completed > 0:
            exit_prices = close_prices[valid_exits]
        else:
            exit_prices = np.array([])

        # Handle open position at end
        if has_open:
            exit_prices = np.append(exit_prices, close_prices[-1])

        # Calculate profits vectorized
        profit_pcts = (exit_prices - entry_prices) / entry_prices * 100

        # Calculate final capital: capital * prod(exit/entry)
        multipliers = exit_prices / entry_prices
        final_capital = initial_capital * np.prod(multipliers)

        # Build trades list
        trades = [
            {
                "entry_price": float(entry_prices[i]),
                "exit_price": float(exit_prices[i]),
                "profit_pct": float(profit_pcts[i]),
                "profit_abs": float(entry_prices[i] * (multipliers[i] - 1))
            }
            for i in range(len(profit_pcts))
        ]

        # Calculate statistics vectorized
        num_trades = len(profit_pcts)
        wins = profit_pcts[profit_pcts > 0]

        return {
            "total_profit": float(final_capital - initial_capital),
            "total_profit_pct": float((final_capital - initial_capital) / initial_capital * 100),
            "num_trades": num_trades,
            "win_rate": float(len(wins) / num_trades * 100) if num_trades > 0 else 0.0,
            "avg_profit_pct": float(np.mean(profit_pcts)) if num_trades > 0 else 0.0,
            "max_drawdown": float(np.min(profit_pcts)) if num_trades > 0 else 0.0,
            "sharpe_ratio": float(np.mean(profit_pcts) / np.std(profit_pcts)) if num_trades > 1 and np.std(profit_pcts) > 0 else 0.0,
            "trades": trades
        }

    def _find_trade_pairs(
        self,
        entry_idx: np.ndarray,
        exit_idx: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find valid entry-exit pairs using searchsorted.

        Returns alternating entry-exit pairs respecting position state.
        """
        valid_entries = []
        valid_exits = []

        entry_pos = 0
        exit_search_start = 0

        while entry_pos < len(entry_idx):
            entry = entry_idx[entry_pos]

            # Find first exit after this entry
            exit_search = np.searchsorted(exit_idx[exit_search_start:], entry, side='right')
            exit_search += exit_search_start

            if exit_search >= len(exit_idx):
                # No more exits - entry stays open
                valid_entries.append(entry)
                break

            exit_val = exit_idx[exit_search]
            valid_entries.append(entry)
            valid_exits.append(exit_val)

            # Next entry must be after this exit
            entry_pos = np.searchsorted(entry_idx, exit_val, side='right')
            exit_search_start = exit_search + 1

        return np.array(valid_entries), np.array(valid_exits)

    def _empty_trade_result(self) -> Dict[str, Any]:
        """Return empty result when no trades."""
        return {
            "total_profit": 0.0,
            "total_profit_pct": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_profit_pct": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "trades": []
        }
