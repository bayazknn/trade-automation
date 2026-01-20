#!/usr/bin/env python
"""
Batch Indicator Optimization Script

Runs vectorbt-based indicator optimization for all cryptocurrencies and indicators.
Designed for server execution with high parallelization.

Usage:
    python run_optimization.py
    python run_optimization.py --symbols whitelist
    python run_optimization.py --symbols all
"""
import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("optimization.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run batch indicator optimization")
    parser.add_argument(
        "--symbols",
        type=str,
        default="whitelist",
        help="Symbols to optimize: 'whitelist', 'all', or comma-separated list (default: whitelist)",
    )
    parser.add_argument(
        "--indicators",
        type=str,
        default="all",
        help="Indicators to optimize: 'all' or comma-separated list (default: all)",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=10,
        help="Number of parallel processes for crypto optimization (default: 10)",
    )
    parser.add_argument(
        "--n-jobs-optuna",
        type=int,
        default=15,
        help="Number of Optuna parallel jobs per process (default: 15)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="notebooks/csv",
        help="Output directory for CSV files (default: notebooks/csv)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/binance",
        help="Data directory containing feather files (default: data/binance)",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=2.0,
        help="Threshold percentage for tradeable labeling (default: 2.0)",
    )
    args = parser.parse_args()

    # Parse symbols if comma-separated
    if args.symbols not in ("whitelist", "all"):
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = args.symbols

    # Parse indicators if comma-separated
    if args.indicators != "all":
        indicators = [i.strip() for i in args.indicators.split(",")]
    else:
        indicators = args.indicators

    logger.info("=" * 60)
    logger.info("Starting Batch Indicator Optimization")
    logger.info("=" * 60)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Indicators: {indicators}")
    logger.info(f"Processes: {args.n_processes}")
    logger.info(f"Optuna jobs: {args.n_jobs_optuna}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info("=" * 60)

    # Import here to avoid import errors if dependencies missing
    try:
        from crypto_analysis.vectorbt_optimizer import optimize_all
    except ImportError as e:
        logger.error(f"Failed to import optimizer: {e}")
        logger.error("Make sure vectorbt and optuna are installed:")
        logger.error("  pip install vectorbt optuna")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run optimization
    try:
        results = optimize_all(
            symbols=symbols,
            indicators=indicators,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config_path="config.json",
            n_processes=args.n_processes,
            n_jobs_optuna=args.n_jobs_optuna,
            threshold_pct=args.threshold_pct,
            period_hours=24,
            export_csv=True,
            export_params_json=False,
        )

        logger.info("=" * 60)
        logger.info("Optimization Complete")
        logger.info("=" * 60)
        logger.info(f"Processed {len(results)} symbols")

        for symbol, df in results.items():
            logger.info(f"  {symbol}: {df.shape[0]} rows, {df.shape[1]} columns")

        # List output files
        csv_files = list(output_path.glob("*.csv"))
        logger.info(f"\nGenerated {len(csv_files)} CSV files in {output_path}")

    except Exception as e:
        logger.exception(f"Optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
