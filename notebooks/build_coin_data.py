"""
Multi-Coin Data Building Script with Multiprocessing

This script builds binary indicator dataframes for all coins in parallel
using multiprocessing Pool to speed up the grid search optimization process.

Usage:
    python build_coin_data.py

Output:
    - coin_csvs/{symbol}_binary.csv for each coin
    - coin_csvs/optimization_params.json with all optimization results
    - coin_csvs/data_summary.json with summary statistics
"""

import sys
import json
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from functools import partial

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from crypto_analysis.indicator_optimizer.dataset_builder import DatasetBuilder


# === CONFIGURATION ===

DATA_DIR = Path(__file__).parent.parent / "data" / "binance"
OUTPUT_DIR = Path(__file__).parent / "coin_csvs"

# Available symbols (from config.json pair_whitelist)
AVAILABLE_SYMBOLS = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOT", "LINK",
    "AVAX", "MATIC", "ATOM", "NEAR", "LTC", "ARB", "OP", "LRC",
    "TRX", "XLM", "DOGE", "VET", "ALGO", "HBAR", "FIL", "EOS",
    "CHZ", "IOTA", "CRO", "ZIL", "ONE", "LDO", "ICP"
]

# Set to True to only process coins without existing CSV files (incremental mode)
INCREMENTAL_MODE = True

# DatasetBuilder parameters
DATASET_CONFIG = {
    "period_hours": 4,
    "signal_shift": 0,
    "threshold_pct": 1.2,
    "grid_search": True,
    "hyperopt": False,
}

# Number of parallel workers
# Cloud server: Xeon E5-2673 v4, 20/80 effective vCPUs (shared)
# Using ~50% of effective cores to leave headroom for OS and other tenants
NUM_WORKERS = 10


@dataclass
class CoinResult:
    """Result from processing a single coin."""
    symbol: str
    success: bool
    shape: Optional[Tuple[int, int]] = None
    trade_count: int = 0
    hold_count: int = 0
    trade_pct: float = 0.0
    n_indicators: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None
    optimization_results: Optional[Dict] = None


def get_available_symbols(data_dir: Path, requested_symbols: List[str]) -> List[str]:
    """Filter requested symbols to only those with available data files."""
    available = []
    for symbol in requested_symbols:
        pattern = f"{symbol}_USDT-*.feather"
        files = list(data_dir.glob(pattern))
        if files:
            available.append(symbol)
    return available


def filter_symbols_without_csv(symbols: List[str], output_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Filter symbols to only those without existing CSV files (for incremental updates).

    Returns
    -------
    Tuple[List[str], List[str]]
        (symbols_to_process, symbols_already_done)
    """
    to_process = []
    already_done = []

    for symbol in symbols:
        csv_path = output_dir / f"{symbol.lower()}_binary.csv"
        if csv_path.exists():
            already_done.append(symbol)
        else:
            to_process.append(symbol)

    return to_process, already_done


def build_single_coin(
    symbol: str,
    data_dir: Path,
    config: Dict
) -> CoinResult:
    """
    Build binary indicator dataframe for a single coin.

    This function is designed to be called by multiprocessing Pool.
    Each process creates its own DatasetBuilder instance.

    Parameters
    ----------
    symbol : str
        Coin symbol (e.g., "BTC")
    data_dir : Path
        Directory with data files
    config : Dict
        DatasetBuilder configuration

    Returns
    -------
    CoinResult
        Result with dataframe info or error
    """
    start_time = time.time()

    try:
        # Each process creates its own builder instance
        # n_workers=1 avoids nested parallelism (Pool already parallelizes across coins)
        # This prevents thread contention and CPU oversubscription on shared servers
        builder = DatasetBuilder(
            data_dir=data_dir,
            period_hours=config["period_hours"],
            signal_shift=config["signal_shift"],
            output_mode='binary',
            n_workers=1
        )

        # Build dataframe with grid search
        df = builder.build(
            symbol=symbol,
            threshold_pct=config["threshold_pct"],
            grid_search=config["grid_search"],
            hyperopt=config["hyperopt"],
            verbose=False
        )

        # Get optimization results
        opt_results = builder.get_optimization_results(symbol)

        # Convert optimization results to serializable dict
        opt_dict = {}
        for ind_name, result in opt_results.items():
            opt_dict[ind_name] = {
                "indicator_name": result.indicator_name,
                "optimization_type": result.optimization_type,
                "score": float(result.score),
                "best_params": result.best_params
            }

        # Calculate statistics
        trade_count = int((df['tradeable'] == 'trade').sum())
        hold_count = int((df['tradeable'] == 'hold').sum())
        trade_pct = trade_count / len(df) * 100

        # Count indicator columns
        metadata_cols = ['date', 'signal', 'signal_pct_change', 'period_id', 'tradeable']
        n_indicators = len([c for c in df.columns if c not in metadata_cols])

        processing_time = time.time() - start_time

        # Save dataframe to CSV
        output_path = OUTPUT_DIR / f"{symbol.lower()}_binary.csv"
        df.to_csv(output_path, index=False)

        return CoinResult(
            symbol=symbol,
            success=True,
            shape=df.shape,
            trade_count=trade_count,
            hold_count=hold_count,
            trade_pct=trade_pct,
            n_indicators=n_indicators,
            processing_time=processing_time,
            optimization_results=opt_dict
        )

    except Exception as e:
        processing_time = time.time() - start_time
        return CoinResult(
            symbol=symbol,
            success=False,
            processing_time=processing_time,
            error=str(e)
        )


def build_all_coins_parallel(
    symbols: List[str],
    data_dir: Path,
    config: Dict,
    num_workers: Optional[int] = None
) -> List[CoinResult]:
    """
    Build dataframes for all coins in parallel using multiprocessing Pool.

    Parameters
    ----------
    symbols : List[str]
        List of coin symbols to process
    data_dir : Path
        Directory with data files
    config : Dict
        DatasetBuilder configuration
    num_workers : int, optional
        Number of parallel workers. None = use all CPUs.

    Returns
    -------
    List[CoinResult]
        Results for all coins
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    # Limit workers to number of symbols
    num_workers = min(num_workers, len(symbols))

    print(f"Building data for {len(symbols)} coins using {num_workers} workers...")
    print("=" * 60)

    # Create partial function with fixed arguments
    build_func = partial(
        build_single_coin,
        data_dir=data_dir,
        config=config
    )

    # Use multiprocessing Pool
    results = []
    with mp.Pool(processes=num_workers) as pool:
        # Use imap_unordered for progress updates
        for result in pool.imap_unordered(build_func, symbols):
            if result.success:
                print(f"[OK] {result.symbol}: {result.shape} | "
                      f"Trade: {result.trade_count} ({result.trade_pct:.1f}%) | "
                      f"Time: {result.processing_time:.1f}s")
            else:
                print(f"[FAIL] {result.symbol}: {result.error}")
            results.append(result)

    return results


def save_optimization_params(results: List[CoinResult], output_dir: Path, merge: bool = True) -> Path:
    """Save optimization parameters for all coins to JSON (merges with existing if merge=True)."""
    output_path = output_dir / "optimization_params.json"

    # Load existing params if merging
    params_dict = {}
    if merge and output_path.exists():
        with open(output_path, 'r') as f:
            params_dict = json.load(f)

    # Add/update with new results
    for result in results:
        if result.success and result.optimization_results:
            params_dict[result.symbol] = result.optimization_results

    with open(output_path, 'w') as f:
        json.dump(params_dict, f, indent=2)

    return output_path


def save_data_summary(results: List[CoinResult], output_dir: Path, merge: bool = True) -> Path:
    """Save data summary for all coins to JSON (merges with existing if merge=True)."""
    output_path = output_dir / "data_summary.json"

    # Load existing summary if merging
    existing_coins = {}
    if merge and output_path.exists():
        with open(output_path, 'r') as f:
            existing = json.load(f)
            existing_coins = existing.get("coins", {})

    # Merge existing coins with new results
    all_coins = existing_coins.copy()
    for result in results:
        all_coins[result.symbol] = {
            "success": result.success,
            "shape": list(result.shape) if result.shape else None,
            "trade_count": result.trade_count,
            "hold_count": result.hold_count,
            "trade_pct": round(result.trade_pct, 2),
            "n_indicators": result.n_indicators,
            "processing_time": round(result.processing_time, 2),
            "error": result.error
        }

    # Recalculate totals based on all coins
    successful_coins = [c for c in all_coins.values() if c["success"]]
    summary = {
        "total_coins": len(all_coins),
        "successful": len(successful_coins),
        "failed": len(all_coins) - len(successful_coins),
        "total_processing_time": sum(c["processing_time"] for c in all_coins.values()),
        "coins": all_coins
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return output_path


def print_summary(results: List[CoinResult]):
    """Print summary of all results."""
    print("\n" + "=" * 70)
    print("DATA BUILDING SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\nTotal coins: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        total_time = sum(r.processing_time for r in successful)
        avg_time = total_time / len(successful)
        print(f"\nTotal processing time: {total_time:.1f}s")
        print(f"Average time per coin: {avg_time:.1f}s")

        total_rows = sum(r.shape[0] for r in successful)
        total_trade = sum(r.trade_count for r in successful)
        total_hold = sum(r.hold_count for r in successful)
        overall_trade_pct = total_trade / (total_trade + total_hold) * 100

        print(f"\nTotal rows: {total_rows:,}")
        print(f"Total trade: {total_trade:,} ({overall_trade_pct:.2f}%)")
        print(f"Total hold: {total_hold:,}")

    if failed:
        print("\nFailed coins:")
        for r in failed:
            print(f"  {r.symbol}: {r.error}")

    print("=" * 70)


def main():
    """Main function to run the data building process."""
    print("\n" + "=" * 70)
    print("MULTI-COIN DATA BUILDING (Multiprocessing)")
    print("=" * 70)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Config: period_hours={DATASET_CONFIG['period_hours']}, "
          f"threshold_pct={DATASET_CONFIG['threshold_pct']}, "
          f"signal_shift={DATASET_CONFIG['signal_shift']}")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)

    # Get available symbols (those with data files)
    symbols = get_available_symbols(DATA_DIR, AVAILABLE_SYMBOLS)
    print(f"\nAvailable symbols ({len(symbols)}): {symbols}")

    if not symbols:
        print("ERROR: No data files found!")
        return 1

    # Incremental mode: skip symbols that already have CSV files
    if INCREMENTAL_MODE:
        symbols_to_process, already_done = filter_symbols_without_csv(symbols, OUTPUT_DIR)
        if already_done:
            print(f"\nSkipping {len(already_done)} coins with existing CSVs: {already_done}")
        if not symbols_to_process:
            print("\nAll coins already processed! Set INCREMENTAL_MODE=False to rebuild.")
            return 0
        symbols = symbols_to_process
        print(f"Processing {len(symbols)} new coins: {symbols}")

    # Build all coins in parallel
    start_time = time.time()
    results = build_all_coins_parallel(
        symbols=symbols,
        data_dir=DATA_DIR,
        config=DATASET_CONFIG,
        num_workers=NUM_WORKERS
    )
    total_time = time.time() - start_time

    # Save optimization params (merge with existing in incremental mode)
    params_path = save_optimization_params(results, OUTPUT_DIR, merge=INCREMENTAL_MODE)
    print(f"\nSaved optimization params: {params_path}")

    # Save data summary (merge with existing in incremental mode)
    summary_path = save_data_summary(results, OUTPUT_DIR, merge=INCREMENTAL_MODE)
    print(f"Saved data summary: {summary_path}")

    # Print summary
    print_summary(results)

    print(f"\nTotal wall-clock time: {total_time:.1f}s")
    print(f"Speedup from parallelization: {sum(r.processing_time for r in results) / total_time:.1f}x")

    # List created files
    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)
    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        size_kb = path.stat().st_size / 1024
        print(f"  {path.name}: {size_kb:.1f} KB")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    sys.exit(main())
