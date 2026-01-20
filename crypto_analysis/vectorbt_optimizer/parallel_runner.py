"""
Multi-crypto parallel optimization runner using ProcessPoolExecutor.
"""
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from .config import OptimizationConfig, RunConfig
from .data_loader import DataLoader
from .optimizer import OptunaOptimizer
from .output_builder import OutputBuilder

logger = logging.getLogger(__name__)


def _get_indicator_registry() -> Dict:
    """Get indicator registry from indicator_optimizer module."""
    try:
        from crypto_analysis.indicator_optimizer.indicators import INDICATOR_REGISTRY
        return INDICATOR_REGISTRY
    except ImportError as e:
        logger.warning(f"Could not import INDICATOR_REGISTRY: {e}")
        return {}


def _process_single_crypto(
    symbol: str,
    indicator_names: List[str],
    data_dir: Path,
    opt_config: OptimizationConfig,
    threshold_pct: float,
    period_hours: int,
) -> tuple:
    """
    Process single cryptocurrency (worker function for ProcessPoolExecutor).

    Parameters
    ----------
    symbol : str
        Cryptocurrency symbol
    indicator_names : list
        List of indicator names to optimize
    data_dir : Path
        Data directory
    opt_config : OptimizationConfig
        Optimization configuration
    threshold_pct : float
        Threshold for tradeable labeling
    period_hours : int
        Period hours for tradeable

    Returns
    -------
    tuple
        (symbol, results_dict, df_with_signals)
    """
    logger.info(f"Processing {symbol}...")

    # Load data
    loader = DataLoader(data_dir)
    try:
        df = loader.load_feather(symbol)
    except FileNotFoundError as e:
        logger.error(f"Data not found for {symbol}: {e}")
        return symbol, {}, None

    # Get indicator registry
    registry = _get_indicator_registry()

    # Initialize optimizer
    optimizer = OptunaOptimizer(opt_config)

    # Results storage
    results = {}
    df_result = df.copy()

    for indicator_name in indicator_names:
        if indicator_name not in registry:
            logger.warning(f"Indicator {indicator_name} not found in registry")
            continue

        try:
            # Create indicator instance
            indicator_class = registry[indicator_name]
            indicator = indicator_class()

            # Optimize
            result = optimizer.optimize(indicator, df)
            results[indicator_name] = result

            logger.info(
                f"{symbol}/{indicator_name}: score={result.score:.4f}, "
                f"params={result.best_params}"
            )

            # Calculate indicator and add signals to DataFrame
            df_with_indicator = indicator.calculate_indicator(df.copy(), **result.best_params)
            entries = indicator.generate_entry_signal(df_with_indicator, **result.best_params)
            exits = indicator.generate_exit_signal(df_with_indicator, **result.best_params)

            df_result[f"{indicator_name}_entry"] = entries
            df_result[f"{indicator_name}_exit"] = exits

            # Add indicator values
            for col in df_with_indicator.columns:
                if col not in df.columns and col not in df_result.columns:
                    df_result[f"{indicator_name}_{col}"] = df_with_indicator[col]

        except Exception as e:
            logger.error(f"Error optimizing {indicator_name} for {symbol}: {e}")
            results[indicator_name] = None

    # Add tradeable column
    try:
        from crypto_analysis.signal_population import SignalPopulator
        populator = SignalPopulator(data_dir=data_dir, period_hours=period_hours)
        df_result = populator.generate_tradeable_by_range(df_result, threshold_pct)
    except ImportError:
        logger.warning("Could not import SignalPopulator for tradeable column")
    except Exception as e:
        logger.warning(f"Error generating tradeable column: {e}")

    return symbol, results, df_result


class MultiCryptoRunner:
    """
    Run optimization across multiple cryptocurrencies in parallel.

    Uses ProcessPoolExecutor for CPU-bound parallel processing.
    """

    def __init__(self, config: RunConfig):
        """
        Initialize MultiCryptoRunner.

        Parameters
        ----------
        config : RunConfig
            Run configuration
        """
        self.config = config
        self.loader = DataLoader(config.data_dir, config.config_path)
        self.output_builder = OutputBuilder(config.output_dir)

    def run(
        self,
        symbols: Optional[Union[str, List[str]]] = None,
        indicators: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run batch optimization.

        Parameters
        ----------
        symbols : str or list, optional
            Symbols to optimize (overrides config)
        indicators : str or list, optional
            Indicators to optimize (overrides config)

        Returns
        -------
        dict
            Symbol -> DataFrame with optimized signals
        """
        # Resolve symbols
        symbol_spec = symbols or self.config.symbols
        crypto_list = self.loader.resolve_symbols(symbol_spec)

        # Resolve indicators
        indicator_spec = indicators or self.config.indicators
        indicator_names = self._resolve_indicators(indicator_spec)

        logger.info(
            f"Running optimization for {len(crypto_list)} cryptos, "
            f"{len(indicator_names)} indicators"
        )

        # Create optimization config
        opt_config = OptimizationConfig(
            n_jobs=self.config.n_jobs_optuna,
            init_cash=100.0,
            fees=0.001,
        )

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        all_params = {}

        # Run in parallel
        with ProcessPoolExecutor(max_workers=self.config.n_processes) as executor:
            futures = {
                executor.submit(
                    _process_single_crypto,
                    symbol,
                    indicator_names,
                    self.config.data_dir,
                    opt_config,
                    self.config.threshold_pct,
                    self.config.period_hours,
                ): symbol
                for symbol in crypto_list
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    symbol_result, opt_results, df_result = future.result()

                    if df_result is not None:
                        results[symbol] = df_result

                        # Collect params
                        all_params[symbol] = {
                            name: res.to_dict() if res else None
                            for name, res in opt_results.items()
                        }

                        # Export if configured
                        if self.config.export_csv:
                            self.output_builder.export_csv(symbol, df_result)

                        if self.config.export_params_json:
                            self.output_builder.export_params(
                                symbol, all_params[symbol]
                            )

                        logger.info(f"Completed {symbol}")

                except Exception as e:
                    logger.error(f"Failed to process {symbol}: {e}")

        # Export combined params
        if self.config.export_params_json:
            self._export_combined_params(all_params)

        return results

    def _resolve_indicators(self, indicators: Union[str, List[str]]) -> List[str]:
        """Resolve indicator specification to list of names."""
        if isinstance(indicators, list):
            return indicators

        if indicators == "all":
            registry = _get_indicator_registry()
            return list(registry.keys())

        # Single indicator
        return [indicators]

    def _export_combined_params(self, all_params: Dict) -> None:
        """Export combined params JSON."""
        output_path = self.config.output_dir / "all_params.json"
        with open(output_path, "w") as f:
            json.dump(all_params, f, indent=2)
        logger.info(f"Exported combined params to {output_path}")


def run_batch_optimization(
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
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function for batch optimization.

    Parameters
    ----------
    symbols : str or list
        "whitelist", "all", or list of symbols
    indicators : str or list
        "all" or list of indicator names
    data_dir : str
        Data directory path
    output_dir : str
        Output directory path
    config_path : str
        Config file path
    n_processes : int
        ProcessPoolExecutor workers
    n_jobs_optuna : int
        Optuna parallel jobs per process
    threshold_pct : float
        Threshold for tradeable labeling
    period_hours : int
        Period hours for tradeable
    export_csv : bool
        Export CSV files

    Returns
    -------
    dict
        Symbol -> DataFrame results
    """
    config = RunConfig(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        config_path=Path(config_path),
        symbols=symbols,
        indicators=indicators,
        n_processes=n_processes,
        n_jobs_optuna=n_jobs_optuna,
        threshold_pct=threshold_pct,
        period_hours=period_hours,
        export_csv=export_csv,
    )

    runner = MultiCryptoRunner(config)
    return runner.run()
