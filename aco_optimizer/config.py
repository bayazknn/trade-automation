"""
ACO Configuration for Trading Strategy Optimization
"""
from pathlib import Path

# Base paths
USER_DATA_DIR = Path(__file__).parent.parent  # user_data/
FREQTRADE_DIR = USER_DATA_DIR.parent          # freqtrade/ (where CLI runs)
INDICATORS_JSON = USER_DATA_DIR / "predefined_indicators.json"
STRATEGIES_DIR = USER_DATA_DIR / "strategies"  # Standard freqtrade convention
PROMPTS_DIR = USER_DATA_DIR / "aco_prompts"
CONFIG_JSON = USER_DATA_DIR / "config.json"
VENV_ACTIVATE = FREQTRADE_DIR / ".venv" / "Scripts" / "Activate.ps1"

# ACO Algorithm Parameters
ACO_CONFIG = {
    # Population settings (LARGE - thorough search)
    "n_ants": 20,
    "n_iterations": 50,

    # Pheromone parameters
    "alpha": 1.0,              # Pheromone importance
    "beta": 1.0,               # Heuristic importance
    "rho": 0.1,                # Evaporation rate (0.1 = 10% evaporation)
    "Q": 10,                   # Pheromone deposit constant
    "tau_min": 0.1,            # Minimum pheromone level
    "tau_max": 15.0,           # Maximum pheromone level
    "tau_initial": 1.0,        # Initial pheromone level

    # Solution constraints
    "min_entry_indicators": 2,  # Minimum indicators for entry
    "min_exit_indicators": 2,   # Minimum indicators for exit
    "max_entry_indicators": 5,  # Maximum indicators for entry
    "max_exit_indicators": 5,   # Maximum indicators for exit

    # Fitness settings
    "penalty_score": -1000,     # Penalty for failed backtests
    "fitness_metric": "total_profit",  # Primary fitness metric

    # Claude Code settings
    "claude_model": "haiku",
    "claude_timeout": 240,      # Seconds per evaluation (increased for safety)
    "max_parallel_evaluations": 20,  # Limit concurrent Claude processes

    # Backtest settings
    "timerange": "20251001-",
    "timeframe": "1h",

    # Logging
    "log_interval": 1,          # Log every N iterations
    "save_best_every": 5,       # Save best strategy every N iterations
}

# Heuristic weights for indicator categories (higher = more likely to be selected)
INDICATOR_HEURISTICS = {
    "momentum": 1.1,    # Slightly prefer momentum indicators
    "overlap": 1.0,     # Standard weight for moving averages
    "volume": 0.9,      # Lower weight for volume (less reliable in crypto)
    "volatility": 0.95,  # Volatility filters
}
