"""
Dataset Builder Module

Facade class that orchestrates all components to build a final
DataFrame with OHLCV data, SignalPopulator signals, and optimized
indicator signals from both grid search and hyperopt.

Supports parallel processing using ThreadPoolExecutor.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ..signal_population import SignalPopulator
from .base import BaseIndicatorOptimizer, OptimizationResult
from .config_loader import ConfigLoader
from .grid_search import GridSearchOptimizer
from .hyperopt_optimizer import HyperoptOptimizer
from .signal_matcher import SignalMatcher
from .indicators import INDICATOR_REGISTRY, get_indicator, list_indicators


@dataclass
class IndicatorResult:
    """Result from processing a single indicator."""
    indicator_name: str
    gs_result: Optional[OptimizationResult] = None
    ho_result: Optional[OptimizationResult] = None
    # Binary mode columns (entry/exit signals)
    gs_entry: Optional[pd.Series] = None
    gs_exit: Optional[pd.Series] = None
    ho_entry: Optional[pd.Series] = None
    ho_exit: Optional[pd.Series] = None
    # Raw mode columns (continuous indicator values)
    gs_outputs: Optional[Dict[str, pd.Series]] = None  # {output_name: series}
    ho_outputs: Optional[Dict[str, pd.Series]] = None  # {output_name: series}
    error: Optional[str] = None


class DatasetBuilder:
    """
    Build complete dataset with all signal columns.

    Orchestrates:
    1. SignalPopulator for target signals
    2. Grid search optimization for each indicator
    3. Hyperopt optimization for each indicator

    Supports parallel processing with configurable thread pool size.

    Final DataFrame includes:
    - OHLCV columns
    - SignalPopulator signals
    - Grid search signals per indicator ({indicator}_gs_entry, {indicator}_gs_exit)
    - Hyperopt signals per indicator ({indicator}_ho_entry, {indicator}_ho_exit)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        period_hours: int = 4,
        n_workers: Optional[int] = None,
        signal_shift: int = 4,
        output_mode: str = 'binary'
    ):
        """
        Initialize dataset builder.

        Parameters
        ----------
        data_dir : str or Path
            Directory containing feather files with OHLCV data
        config_path : str or Path, optional
            Path to technical_indicators_config.json
        period_hours : int
            Period length for SignalPopulator (default: 4)
        n_workers : int, optional
            Number of parallel workers for indicator processing.
            If None, uses min(4, cpu_count).
        signal_shift : int, default=4
            Number of steps to shift indicator signals forward.
            This makes indicator signals at time t predict target at time t+signal_shift.
            Aligns with LSTM target_shift for consistent future prediction.
        output_mode : str, default='binary'
            Output mode for indicator columns:
            - 'binary': Binary entry/exit signals (0/1). Column names: {indicator}_gs_entry, {indicator}_gs_exit
            - 'raw': Raw indicator values (continuous floats). Column names: {indicator}_gs_{output_name}
        """
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path) if config_path else None
        self.period_hours = period_hours
        self.signal_shift = signal_shift
        self.output_mode = output_mode

        # Validate output_mode
        if output_mode not in ('binary', 'raw'):
            raise ValueError(f"output_mode must be 'binary' or 'raw', got '{output_mode}'")

        # Set number of workers
        if n_workers is None:
            n_workers = min(4, os.cpu_count() or 1)
        self.n_workers = max(1, n_workers)

        # Initialize components
        self.signal_populator = SignalPopulator(data_dir, period_hours)
        self.signal_matcher = SignalMatcher()
        self.grid_search = GridSearchOptimizer(self.signal_matcher)

        # Load config if provided
        self.config_loader = None
        if config_path:
            self.config_loader = ConfigLoader(config_path)

        # Store optimization results
        self.optimization_results: Dict[str, Dict[str, OptimizationResult]] = {}

        # Thread safety for results
        self._results_lock = Lock()

    def _process_indicator(
        self,
        ind_name: str,
        df: pd.DataFrame,
        target_df: pd.DataFrame,
        grid_search: bool,
        hyperopt: bool,
        hyperopt_trials: int,
        signal_shift: int,
        verbose: bool
    ) -> IndicatorResult:
        """
        Process a single indicator (grid search + hyperopt).

        This method runs in a separate thread.

        Parameters
        ----------
        ind_name : str
            Indicator name
        df : pd.DataFrame
            OHLCV DataFrame
        target_df : pd.DataFrame
            DataFrame with target signals
        grid_search : bool
            Whether to run grid search
        hyperopt : bool
            Whether to run hyperopt
        hyperopt_trials : int
            Number of hyperopt trials
        signal_shift : int
            Number of steps to shift signals forward
        verbose : bool
            Print progress
        """
        result = IndicatorResult(indicator_name=ind_name)

        try:
            # Get indicator class
            indicator_class = get_indicator(ind_name)
            indicator = indicator_class()

            # Create thread-local optimizer instances
            local_signal_matcher = SignalMatcher()
            local_grid_search = GridSearchOptimizer(
                local_signal_matcher,
                step_sizes=self.grid_search.step_sizes
            )

            # Run grid search
            if grid_search:
                if verbose:
                    print(f"  [{ind_name}] Grid search optimization...")

                gs_result = local_grid_search.optimize(
                    indicator, df.copy(), target_df,
                    verbose=False  # Disable verbose in threads to avoid messy output
                )
                result.gs_result = gs_result

                if self.output_mode == 'binary':
                    # Binary mode: Generate entry/exit signals
                    gs_df = indicator.generate_signals(df.copy(), **gs_result.best_params)
                    gs_entry = gs_df["entry_signal"].astype(int)
                    gs_exit = gs_df["exit_signal"].astype(int)

                    # Shift signals forward to predict future targets
                    # NaN values at start will be dropped later in build()
                    if signal_shift > 0:
                        gs_entry = gs_entry.shift(signal_shift)
                        gs_exit = gs_exit.shift(signal_shift)

                    result.gs_entry = gs_entry
                    result.gs_exit = gs_exit
                else:
                    # Raw mode: Extract raw indicator values
                    gs_df = indicator.calculate_indicator(df.copy(), **gs_result.best_params)
                    gs_outputs = {}
                    for output_name in indicator.outputs:
                        output_series = gs_df[output_name]
                        # Shift values forward to predict future targets
                        if signal_shift > 0:
                            output_series = output_series.shift(signal_shift)
                        gs_outputs[output_name] = output_series
                    result.gs_outputs = gs_outputs

                if verbose:
                    print(f"  [{ind_name}] Grid search done - score: {gs_result.score}")

            # Run hyperopt
            if hyperopt:
                if verbose:
                    print(f"  [{ind_name}] Hyperopt optimization...")

                try:
                    ho_optimizer = HyperoptOptimizer(loss_function="profit")
                    ho_result = ho_optimizer.optimize(
                        indicator, df.copy(),
                        n_trials=hyperopt_trials,
                        verbose=False
                    )
                    result.ho_result = ho_result

                    if self.output_mode == 'binary':
                        # Binary mode: Generate entry/exit signals
                        ho_df = indicator.generate_signals(df.copy(), **ho_result.best_params)
                        ho_entry = ho_df["entry_signal"].astype(int)
                        ho_exit = ho_df["exit_signal"].astype(int)

                        # Shift signals forward to predict future targets
                        # NaN values at start will be dropped later in build()
                        if signal_shift > 0:
                            ho_entry = ho_entry.shift(signal_shift)
                            ho_exit = ho_exit.shift(signal_shift)

                        result.ho_entry = ho_entry
                        result.ho_exit = ho_exit
                    else:
                        # Raw mode: Extract raw indicator values
                        ho_df = indicator.calculate_indicator(df.copy(), **ho_result.best_params)
                        ho_outputs = {}
                        for output_name in indicator.outputs:
                            output_series = ho_df[output_name]
                            # Shift values forward to predict future targets
                            if signal_shift > 0:
                                output_series = output_series.shift(signal_shift)
                            ho_outputs[output_name] = output_series
                        result.ho_outputs = ho_outputs

                    if verbose:
                        print(f"  [{ind_name}] Hyperopt done - score: {ho_result.score:.2f}")

                except ImportError:
                    if verbose:
                        print(f"  [{ind_name}] Skipping hyperopt (optuna not installed)")

        except Exception as e:
            result.error = str(e)
            if verbose:
                print(f"  [{ind_name}] Error: {e}")

        return result

    def build(
        self,
        symbol: str,
        threshold_pct: float,
        indicators: Optional[List[str]] = None,
        grid_search: bool = True,
        hyperopt: bool = True,
        hyperopt_trials: int = 100,
        signal_shift: Optional[int] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Build complete dataset with all signals using parallel processing.

        Parameters
        ----------
        symbol : str
            Cryptocurrency symbol (e.g., "BTC", "ETH")
        threshold_pct : float
            Percentage threshold for SignalPopulator
        indicators : list, optional
            List of indicator names to include. If None, uses all available.
        grid_search : bool
            Whether to run grid search optimization
        hyperopt : bool
            Whether to run hyperopt optimization
        hyperopt_trials : int
            Number of trials for hyperopt
        signal_shift : int, optional
            Number of steps to shift indicator signals forward.
            If None, uses the value set in __init__ (default: 4).
            Set to 0 to disable shifting.
        verbose : bool
            Print progress information

        Returns
        -------
        pd.DataFrame
            Complete dataset with all signal columns
        """
        # Get target signals from SignalPopulator
        if verbose:
            print(f"Generating target signals for {symbol}...")

        df = self.signal_populator.populate_signals(symbol, threshold_pct)

        # Determine which indicators to use
        if indicators is None:
            indicators = list_indicators()
        elif indicators == "all":
            indicators = list_indicators()

        # Use instance signal_shift if not provided
        if signal_shift is None:
            signal_shift = self.signal_shift

        if verbose:
            print(f"Processing {len(indicators)} indicators with {self.n_workers} workers...")
            if signal_shift > 0:
                print(f"Signal shift: {signal_shift} steps (indicators predict t+{signal_shift})")

        # Store results
        self.optimization_results[symbol] = {}

        # Process indicators in parallel
        results: List[IndicatorResult] = []

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all indicator tasks
            futures = {
                executor.submit(
                    self._process_indicator,
                    ind_name,
                    df.copy(),
                    df,
                    grid_search,
                    hyperopt,
                    hyperopt_trials,
                    signal_shift,
                    verbose
                ): ind_name
                for ind_name in indicators
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                ind_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1

                    if verbose:
                        print(f"Completed {completed}/{len(indicators)}: {ind_name}")

                except Exception as e:
                    if verbose:
                        print(f"Error processing {ind_name}: {e}")

        # Merge results into DataFrame
        for result in results:
            ind_name = result.indicator_name

            if result.error:
                continue

            # Add grid search results
            if result.gs_result:
                with self._results_lock:
                    self.optimization_results[symbol][f"{ind_name}_gs"] = result.gs_result

                if self.output_mode == 'binary':
                    if result.gs_entry is not None:
                        df[f"{ind_name}_gs_entry"] = result.gs_entry
                        df[f"{ind_name}_gs_exit"] = result.gs_exit
                else:
                    # Raw mode: add raw indicator output columns
                    if result.gs_outputs is not None:
                        for output_name, output_series in result.gs_outputs.items():
                            df[f"{ind_name}_gs_{output_name}"] = output_series

            # Add hyperopt results
            if result.ho_result:
                with self._results_lock:
                    self.optimization_results[symbol][f"{ind_name}_ho"] = result.ho_result

                if self.output_mode == 'binary':
                    if result.ho_entry is not None:
                        df[f"{ind_name}_ho_entry"] = result.ho_entry
                        df[f"{ind_name}_ho_exit"] = result.ho_exit
                else:
                    # Raw mode: add raw indicator output columns
                    if result.ho_outputs is not None:
                        for output_name, output_series in result.ho_outputs.items():
                            df[f"{ind_name}_ho_{output_name}"] = output_series

        # Drop first signal_shift rows (they have NaN indicator signals due to shift)
        if signal_shift > 0:
            original_len = len(df)
            df = df.iloc[signal_shift:].reset_index(drop=True)

            if self.output_mode == 'binary':
                # Convert binary indicator columns back to int (they were float due to NaN)
                indicator_cols = [
                    col for col in df.columns
                    if col.endswith(('_gs_entry', '_gs_exit', '_ho_entry', '_ho_exit'))
                ]
                for col in indicator_cols:
                    df[col] = df[col].astype(int)
            # Raw mode: keep columns as float (continuous values)

            if verbose:
                print(f"Dropped first {signal_shift} rows ({original_len} -> {len(df)})")

        if verbose:
            print(f"\nDataset built with {len(df.columns)} columns")

        return df

    def build_sequential(
        self,
        symbol: str,
        threshold_pct: float,
        indicators: Optional[List[str]] = None,
        grid_search: bool = True,
        hyperopt: bool = True,
        hyperopt_trials: int = 100,
        signal_shift: Optional[int] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Build dataset sequentially (no parallelization).

        Useful for debugging or when thread safety is a concern.
        Same parameters as build().
        """
        # Get target signals from SignalPopulator
        if verbose:
            print(f"Generating target signals for {symbol}...")

        df = self.signal_populator.populate_signals(symbol, threshold_pct)

        # Determine which indicators to use
        if indicators is None:
            indicators = list_indicators()
        elif indicators == "all":
            indicators = list_indicators()

        # Use instance signal_shift if not provided
        if signal_shift is None:
            signal_shift = self.signal_shift

        if verbose and signal_shift > 0:
            print(f"Signal shift: {signal_shift} steps (indicators predict t+{signal_shift})")

        # Store results
        self.optimization_results[symbol] = {}

        # Process each indicator sequentially
        for i, ind_name in enumerate(indicators):
            if verbose:
                print(f"\n[{i+1}/{len(indicators)}] Processing {ind_name}...")

            result = self._process_indicator(
                ind_name, df.copy(), df,
                grid_search, hyperopt, hyperopt_trials, signal_shift, verbose
            )

            if result.error:
                continue

            # Add grid search results
            if result.gs_result:
                self.optimization_results[symbol][f"{ind_name}_gs"] = result.gs_result
                if self.output_mode == 'binary':
                    if result.gs_entry is not None:
                        df[f"{ind_name}_gs_entry"] = result.gs_entry
                        df[f"{ind_name}_gs_exit"] = result.gs_exit
                else:
                    # Raw mode: add raw indicator output columns
                    if result.gs_outputs is not None:
                        for output_name, output_series in result.gs_outputs.items():
                            df[f"{ind_name}_gs_{output_name}"] = output_series

            # Add hyperopt results
            if result.ho_result:
                self.optimization_results[symbol][f"{ind_name}_ho"] = result.ho_result
                if self.output_mode == 'binary':
                    if result.ho_entry is not None:
                        df[f"{ind_name}_ho_entry"] = result.ho_entry
                        df[f"{ind_name}_ho_exit"] = result.ho_exit
                else:
                    # Raw mode: add raw indicator output columns
                    if result.ho_outputs is not None:
                        for output_name, output_series in result.ho_outputs.items():
                            df[f"{ind_name}_ho_{output_name}"] = output_series

        # Drop first signal_shift rows (they have NaN indicator signals due to shift)
        if signal_shift > 0:
            original_len = len(df)
            df = df.iloc[signal_shift:].reset_index(drop=True)

            if self.output_mode == 'binary':
                # Convert binary indicator columns back to int (they were float due to NaN)
                indicator_cols = [
                    col for col in df.columns
                    if col.endswith(('_gs_entry', '_gs_exit', '_ho_entry', '_ho_exit'))
                ]
                for col in indicator_cols:
                    df[col] = df[col].astype(int)
            # Raw mode: keep columns as float (continuous values)

            if verbose:
                print(f"Dropped first {signal_shift} rows ({original_len} -> {len(df)})")

        if verbose:
            print(f"\nDataset built with {len(df.columns)} columns")

        return df

    def build_for_multiple_symbols(
        self,
        symbols: List[str],
        threshold_pct: float,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Build datasets for multiple symbols.

        Parameters
        ----------
        symbols : list
            List of cryptocurrency symbols
        threshold_pct : float
            Percentage threshold for SignalPopulator
        **kwargs
            Additional arguments passed to build()

        Returns
        -------
        dict
            Symbol -> DataFrame mapping
        """
        results = {}
        for symbol in symbols:
            try:
                df = self.build(symbol, threshold_pct, **kwargs)
                results[symbol] = df
            except Exception as e:
                print(f"Error building dataset for {symbol}: {e}")

        return results

    def get_optimization_results(
        self,
        symbol: str
    ) -> Dict[str, OptimizationResult]:
        """
        Get optimization results for a symbol.

        Returns
        -------
        dict
            Indicator name -> OptimizationResult mapping
        """
        return self.optimization_results.get(symbol, {})

    def get_best_params_summary(
        self,
        symbol: str
    ) -> pd.DataFrame:
        """
        Get summary of best parameters for all indicators.

        Returns
        -------
        pd.DataFrame
            Summary table with indicator, optimization type, and parameters
        """
        results = self.optimization_results.get(symbol, {})

        rows = []
        for name, result in results.items():
            row = {
                "indicator": result.indicator_name,
                "optimization_type": result.optimization_type,
                "score": result.score,
                **result.best_params
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def generate_signals_with_params(
        self,
        df: pd.DataFrame,
        indicator_name: str,
        params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Generate signals for an indicator with specific parameters.

        Useful for testing different parameter combinations.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame
        indicator_name : str
            Name of indicator
        params : dict
            Parameters to use

        Returns
        -------
        pd.DataFrame
            DataFrame with entry_signal and exit_signal columns
        """
        indicator_class = get_indicator(indicator_name)
        indicator = indicator_class()
        return indicator.generate_signals(df.copy(), **params)

    def compare_signals(
        self,
        df: pd.DataFrame,
        indicator_name: str
    ) -> pd.DataFrame:
        """
        Compare SignalPopulator signals with indicator signals.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with all signal columns
        indicator_name : str
            Indicator to compare

        Returns
        -------
        pd.DataFrame
            Comparison table showing signal alignment
        """
        # Get relevant columns
        cols = ["date", "close", "signal"]

        if f"{indicator_name}_gs_entry" in df.columns:
            cols.extend([f"{indicator_name}_gs_entry", f"{indicator_name}_gs_exit"])

        if f"{indicator_name}_ho_entry" in df.columns:
            cols.extend([f"{indicator_name}_ho_entry", f"{indicator_name}_ho_exit"])

        # Filter to rows with any signal
        mask = df["signal"].notna()
        for col in cols[3:]:
            if col in df.columns:
                mask |= df[col] == 1

        return df.loc[mask, cols].copy()

    def export_results(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        format: str = "feather"
    ):
        """
        Export dataset to file.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to export
        output_path : str or Path
            Output file path
        format : str
            Export format: 'feather', 'csv', or 'parquet'
        """
        output_path = Path(output_path)

        if format == "feather":
            df.to_feather(output_path)
        elif format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "parquet":
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"Exported to {output_path}")
