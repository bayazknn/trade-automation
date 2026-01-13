"""
Indicator Optimizer Package

Provides tools for optimizing technical indicator parameters using:
1. Grid Search - Match signals with SignalPopulator
2. Hyperopt (Optuna) - Optimize for trading performance
"""

from .base import BaseIndicatorOptimizer, OptimizationResult
from .config_loader import ConfigLoader, IndicatorConfig
from .signal_matcher import SignalMatcher, MatchResult
from .grid_search import GridSearchOptimizer
from .hyperopt_optimizer import HyperoptOptimizer
from .dataset_builder import DatasetBuilder

__all__ = [
    "BaseIndicatorOptimizer",
    "OptimizationResult",
    "ConfigLoader",
    "IndicatorConfig",
    "SignalMatcher",
    "MatchResult",
    "GridSearchOptimizer",
    "HyperoptOptimizer",
    "DatasetBuilder",
]
