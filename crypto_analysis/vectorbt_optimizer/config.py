"""
Configuration dataclasses for vectorbt-based indicator optimization.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class OptimizationConfig:
    """Configuration for the Optuna optimization process."""

    max_grid_combinations: int = 500  # Grid vs TPE threshold
    max_tpe_trials: int = 1000        # Max TPE epochs
    n_jobs: int = 4                   # Optuna parallel jobs
    init_cash: float = 100.0          # Initial capital for backtest
    fees: float = 0.001               # Trading fees (0.1%)

    # Optimization direction
    direction: str = "maximize"       # "maximize" for total return

    # Optuna study settings
    study_name: Optional[str] = None
    storage: Optional[str] = None     # Optional SQLite storage for persistence


@dataclass
class RunConfig:
    """Configuration for batch optimization runs."""

    data_dir: Path = field(default_factory=lambda: Path("data/binance"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    config_path: Path = field(default_factory=lambda: Path("config.json"))

    # What to optimize
    indicators: Union[str, List[str]] = "all"   # "all" or list of indicator names
    symbols: Union[str, List[str]] = "whitelist"  # "whitelist", "all", or list of symbols

    # Tradeable column settings
    threshold_pct: float = 2.0        # Threshold for tradeable labeling
    period_hours: int = 6            # Look-ahead period for tradeable

    # Parallelization
    n_processes: int = 4              # ProcessPoolExecutor workers
    n_jobs_optuna: int = 4            # Optuna threads per process

    # Output options
    export_csv: bool = True
    export_params_json: bool = True

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.config_path, str):
            self.config_path = Path(self.config_path)


@dataclass
class FitnessResult:
    """Result from vectorbt fitness calculation."""

    total_return: float      # Primary fitness metric
    num_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float = 0.0

    def __repr__(self) -> str:
        return (
            f"FitnessResult(return={self.total_return:.4f}, "
            f"trades={self.num_trades}, win_rate={self.win_rate:.2%}, "
            f"sharpe={self.sharpe_ratio:.2f})"
        )


@dataclass
class OptimizationResult:
    """Result from indicator optimization."""

    indicator_name: str
    best_params: dict
    score: float                      # Best fitness score
    sampler_type: str                 # "grid" or "tpe"
    n_trials: int                     # Number of trials run
    fitness_details: Optional[FitnessResult] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "best_params": self.best_params,
            "score": self.score,
            "type": self.sampler_type,
            "n_trials": self.n_trials,
            "fitness": {
                "total_return": self.fitness_details.total_return,
                "num_trades": self.fitness_details.num_trades,
                "win_rate": self.fitness_details.win_rate,
                "sharpe_ratio": self.fitness_details.sharpe_ratio,
                "max_drawdown": self.fitness_details.max_drawdown,
            } if self.fitness_details else None
        }
