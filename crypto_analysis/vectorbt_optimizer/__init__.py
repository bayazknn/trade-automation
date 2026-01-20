"""
Vectorbt-based Indicator Optimizer

High-performance indicator optimization using vectorbt for backtesting
and Optuna for parameter optimization.

Usage
-----
Single indicator optimization:
    >>> from crypto_analysis.vectorbt_optimizer import optimize_indicator
    >>> result = optimize_indicator("RSI", "BTC")

Batch optimization:
    >>> from crypto_analysis.vectorbt_optimizer import optimize_all
    >>> results = optimize_all(symbols="whitelist", indicators="all")

Specific symbols and indicators:
    >>> results = optimize_all(
    ...     symbols=["BTC", "ETH", "SOL"],
    ...     indicators=["RSI", "MACD", "BBANDS"],
    ...     n_processes=8,
    ...     export_csv=True
    ... )
"""
import logging
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from .config import (
    FitnessResult,
    OptimizationConfig,
    OptimizationResult,
    RunConfig,
)
from .data_loader import DataLoader, load_feather, list_available_symbols, load_whitelist
from .fitness import VectorbtFitness, calculate_fitness
from .optimizer import OptunaOptimizer, optimize_indicator_params
from .output_builder import OutputBuilder, export_optimization_results
from .parallel_runner import MultiCryptoRunner, run_batch_optimization

__all__ = [
    # Main API functions
    "optimize_indicator",
    "optimize_all",
    # Config classes
    "OptimizationConfig",
    "RunConfig",
    "FitnessResult",
    "OptimizationResult",
    # Core classes
    "DataLoader",
    "VectorbtFitness",
    "OptunaOptimizer",
    "OutputBuilder",
    "MultiCryptoRunner",
    # Utility functions
    "load_feather",
    "list_available_symbols",
    "load_whitelist",
    "calculate_fitness",
    "optimize_indicator_params",
    "export_optimization_results",
    "run_batch_optimization",
]

logger = logging.getLogger(__name__)


def _get_indicator_registry() -> Dict:
    """Get indicator registry from indicator_optimizer module."""
    try:
        from crypto_analysis.indicator_optimizer.indicators import INDICATOR_REGISTRY
        return INDICATOR_REGISTRY
    except ImportError as e:
        logger.warning(f"Could not import INDICATOR_REGISTRY: {e}")
        return {}


def _resolve_data_dir(data_dir: str) -> Path:
    """Resolve data directory path, handling relative paths from module location."""
    data_path = Path(data_dir)
    if data_path.is_absolute() and data_path.exists():
        return data_path

    # Try relative to current working directory
    if data_path.exists():
        return data_path

    # Try relative to user_data directory (parent of crypto_analysis)
    module_dir = Path(__file__).parent  # vectorbt_optimizer
    user_data_dir = module_dir.parent.parent  # user_data
    resolved = user_data_dir / data_dir
    if resolved.exists():
        return resolved

    # Return original path and let it fail with clear error
    return data_path


def optimize_indicator(
    indicator_name: str,
    symbol: str,
    data_dir: str = "data/binance",
    max_grid_combinations: int = 500,
    max_tpe_trials: int = 1000,
    n_jobs: int = 4,
    init_cash: float = 100.0,
    fees: float = 0.001,
) -> OptimizationResult:
    """
    Optimize single indicator for single cryptocurrency.

    Parameters
    ----------
    indicator_name : str
        Name of indicator (e.g., "RSI", "MACD", "BBANDS")
    symbol : str
        Cryptocurrency symbol (e.g., "BTC", "ETH")
    data_dir : str
        Directory containing feather files
    max_grid_combinations : int
        Threshold for Grid vs TPE sampler selection
    max_tpe_trials : int
        Maximum TPE epochs if using TPE sampler
    n_jobs : int
        Optuna parallel jobs
    init_cash : float
        Initial capital for backtest
    fees : float
        Trading fees (e.g., 0.001 for 0.1%)

    Returns
    -------
    OptimizationResult
        Optimization result with best_params, score, and fitness details

    Raises
    ------
    ValueError
        If indicator not found in registry
    FileNotFoundError
        If data file not found

    Examples
    --------
    >>> result = optimize_indicator("RSI", "BTC")
    >>> print(result.best_params)
    {'timeperiod': 14, 'entry_constant': 30, 'exit_constant': 70}
    >>> print(result.score)
    0.234
    """
    # Get indicator from registry
    registry = _get_indicator_registry()
    if indicator_name not in registry:
        raise ValueError(
            f"Indicator '{indicator_name}' not found. "
            f"Available: {list(registry.keys())}"
        )

    indicator_class = registry[indicator_name]
    indicator = indicator_class()

    # Load data - resolve path relative to module if needed
    resolved_dir = _resolve_data_dir(data_dir)
    loader = DataLoader(resolved_dir)
    df = loader.load_feather(symbol)

    # Configure optimizer
    config = OptimizationConfig(
        max_grid_combinations=max_grid_combinations,
        max_tpe_trials=max_tpe_trials,
        n_jobs=n_jobs,
        init_cash=init_cash,
        fees=fees,
    )

    # Run optimization
    optimizer = OptunaOptimizer(config)
    result = optimizer.optimize(indicator, df)

    logger.info(
        f"Optimized {indicator_name} for {symbol}: "
        f"score={result.score:.4f}, params={result.best_params}"
    )

    return result


def optimize_all(
    symbols: Union[str, List[str]] = "whitelist",
    indicators: Union[str, List[str]] = "all",
    data_dir: str = "data/binance",
    output_dir: str = "output",
    config_path: str = "config.json",
    n_processes: int = 4,
    n_jobs_optuna: int = 4,
    threshold_pct: float = 2.0,
    period_hours: int = 24,
    export_csv: bool = True,
    export_params_json: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Main batch optimization entry point.

    Optimizes multiple indicators across multiple cryptocurrencies
    using parallel processing.

    Parameters
    ----------
    symbols : str or List[str]
        - "whitelist": Use pair_whitelist from config.json
        - "all": All available symbols in data directory
        - List[str]: Explicit list of symbols (e.g., ["BTC", "ETH"])
    indicators : str or List[str]
        - "all": All registered indicators
        - List[str]: Explicit list (e.g., ["RSI", "MACD"])
    data_dir : str
        Directory containing feather files
    output_dir : str
        Directory for output files
    config_path : str
        Path to config.json (for whitelist)
    n_processes : int
        ProcessPoolExecutor workers (crypto parallelization)
    n_jobs_optuna : int
        Optuna threads per process
    threshold_pct : float
        Threshold for tradeable labeling
    period_hours : int
        Period hours for tradeable calculation
    export_csv : bool
        Export CSV files for each symbol
    export_params_json : bool
        Export parameter JSON files

    Returns
    -------
    Dict[str, pd.DataFrame]
        Symbol -> DataFrame with optimized signals and indicators

    Examples
    --------
    # Optimize all indicators for whitelist cryptos
    >>> results = optimize_all(symbols="whitelist", indicators="all")

    # Specific cryptos and indicators
    >>> results = optimize_all(
    ...     symbols=["BTC", "ETH", "SOL"],
    ...     indicators=["RSI", "MACD", "BBANDS"],
    ...     n_processes=8,
    ...     export_csv=True
    ... )

    # Check output
    >>> "BTC" in results
    True
    >>> Path("output/BTC_optimized.csv").exists()
    True
    """
    # Resolve paths relative to module location if needed
    resolved_data_dir = _resolve_data_dir(data_dir)
    resolved_output_dir = _resolve_data_dir(output_dir)
    resolved_config_path = _resolve_data_dir(config_path)

    config = RunConfig(
        data_dir=resolved_data_dir,
        output_dir=resolved_output_dir,
        config_path=resolved_config_path,
        symbols=symbols,
        indicators=indicators,
        n_processes=n_processes,
        n_jobs_optuna=n_jobs_optuna,
        threshold_pct=threshold_pct,
        period_hours=period_hours,
        export_csv=export_csv,
        export_params_json=export_params_json,
    )

    runner = MultiCryptoRunner(config)
    results = runner.run()

    logger.info(f"Optimization complete. Processed {len(results)} symbols.")

    return results
