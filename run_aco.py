#!/usr/bin/env python3
"""
Run ACO Optimization for Trading Strategy Indicator Selection.

Usage:
    python run_aco.py                    # Run with default settings (20 ants, 50 iterations)
    python run_aco.py --ants 10 --iters 20   # Custom settings
    python run_aco.py --quick            # Quick test (5 ants, 3 iterations)
"""
import argparse
import ctypes
import logging
import sys
from datetime import datetime
from pathlib import Path

# Windows sleep prevention constants
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def prevent_sleep():
    """Prevent Windows from sleeping or turning off display during optimization."""
    if sys.platform == 'win32':
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        return True
    return False


def allow_sleep():
    """Restore normal Windows sleep behavior."""
    if sys.platform == 'win32':
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        return True
    return False

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from aco_optimizer import ACOAlgorithm, ACO_CONFIG
from aco_optimizer.config import STRATEGIES_DIR


def cleanup_previous_aco_files() -> int:
    """
    Delete all strategy files from previous ACO runs (files starting with 'aco_').

    Returns:
        Number of files deleted
    """
    deleted_count = 0
    if STRATEGIES_DIR.exists():
        for file_path in STRATEGIES_DIR.glob("aco_*.py"):
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")
    return deleted_count


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the optimization run."""
    log_dir = Path(__file__).parent / "aco_logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"aco_run_{timestamp}.log"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info(f"Log file: {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="ACO Optimization for Trading Strategy Indicator Selection"
    )
    parser.add_argument(
        "--ants", "-a",
        type=int,
        default=ACO_CONFIG["n_ants"],
        help=f"Number of ants (default: {ACO_CONFIG['n_ants']})"
    )
    parser.add_argument(
        "--iters", "-i",
        type=int,
        default=ACO_CONFIG["n_iterations"],
        help=f"Number of iterations (default: {ACO_CONFIG['n_iterations']})"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick test mode (5 ants, 3 iterations)"
    )
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ACO_CONFIG["alpha"],
        help=f"Pheromone importance (default: {ACO_CONFIG['alpha']})"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=ACO_CONFIG["beta"],
        help=f"Heuristic importance (default: {ACO_CONFIG['beta']})"
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=ACO_CONFIG["rho"],
        help=f"Evaporation rate (default: {ACO_CONFIG['rho']})"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Cleanup previous ACO strategy files
    deleted = cleanup_previous_aco_files()
    if deleted > 0:
        logger.info(f"Cleaned up {deleted} previous ACO strategy files from strategies/")

    # Quick mode overrides
    if args.quick:
        args.ants = 5
        args.iters = 3
        logger.info("Quick test mode enabled")

    # Update config
    config = ACO_CONFIG.copy()
    config["n_ants"] = args.ants
    config["n_iterations"] = args.iters
    config["alpha"] = args.alpha
    config["beta"] = args.beta
    config["rho"] = args.rho

    # Print configuration
    logger.info("=" * 60)
    logger.info("ACO Optimization Configuration")
    logger.info("=" * 60)
    logger.info(f"Ants: {config['n_ants']}")
    logger.info(f"Iterations: {config['n_iterations']}")
    logger.info(f"Alpha (pheromone): {config['alpha']}")
    logger.info(f"Beta (heuristic): {config['beta']}")
    logger.info(f"Rho (evaporation): {config['rho']}")
    logger.info(f"Min entry indicators: {config['min_entry_indicators']}")
    logger.info(f"Max entry indicators: {config['max_entry_indicators']}")
    logger.info(f"Min exit indicators: {config['min_exit_indicators']}")
    logger.info(f"Max exit indicators: {config['max_exit_indicators']}")

    estimated_time = config['n_ants'] * config['n_iterations'] * 45 / 3600
    logger.info(f"Estimated runtime: ~{estimated_time:.1f} hours")
    logger.info("=" * 60)

    # Confirm before long runs
    if estimated_time > 1 and not args.quick:
        logger.info("\nThis optimization will take a significant amount of time.")
        logger.info("Consider running in a screen/tmux session.")
        response = input("\nProceed? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Optimization cancelled.")
            return

    # Prevent Windows from sleeping during optimization
    if prevent_sleep():
        logger.info("Sleep prevention enabled (Windows will stay awake)")

    # Create and run ACO
    try:
        aco = ACOAlgorithm(config)
        best = aco.run()

        if best:
            logger.info("\n" + "=" * 60)
            logger.info("OPTIMIZATION COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Best Strategy: {best.strategy_name}")
            logger.info(f"Best Fitness: {best.fitness:.2f}")
            logger.info(f"\nEntry Indicators ({len(best.entry_indicators)}):")
            for ind in best.entry_indicators:
                logger.info(f"  - {ind}")
            logger.info(f"\nExit Indicators ({len(best.exit_indicators)}):")
            for ind in best.exit_indicators:
                logger.info(f"  - {ind}")

            if best.backtest_result:
                logger.info("\nBacktest Results:")
                for key, value in best.backtest_result.items():
                    logger.info(f"  {key}: {value}")
        else:
            logger.warning("No valid solution found.")

    except KeyboardInterrupt:
        logger.info("\nOptimization interrupted by user.")
        logger.info("Partial results may be saved in aco_checkpoints/")

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise

    finally:
        # Restore normal sleep behavior
        if allow_sleep():
            logger.info("Sleep prevention disabled (normal power settings restored)")


if __name__ == "__main__":
    main()
