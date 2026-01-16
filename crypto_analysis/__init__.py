"""
Cryptocurrency Analysis Package

A package for cryptocurrency data analysis and signal generation.

Modules:
- signal_population: Generate entry/exit signals based on price changes
- indicator_optimizer: Optimize technical indicator parameters
- lstm: LSTM-based binary signal prediction models (hold/trade)
"""

from .signal_population import SignalPopulator

# Import indicator optimizer components
from .indicator_optimizer import (
    BaseIndicatorOptimizer,
    OptimizationResult,
    ConfigLoader,
    IndicatorConfig,
    SignalMatcher,
    MatchResult,
    GridSearchOptimizer,
    HyperoptOptimizer,
    DatasetBuilder,
)

# Import LSTM components
from .lstm import (
    DataPreprocessor,
    SignalDataset,
    create_sequences,
    LSTMSignalPredictor,
    ModelConfig,
    BinarySignalLoss,
    FocalBinaryLoss,
    WeightedSignalLoss,  # Alias for backwards compatibility
    FocalWeightedLoss,   # Alias for backwards compatibility
    Trainer,
    TrainingConfig,
    Predictor,
    # Dual-CNN LSTM
    DualModelConfig,
    DualCNNLSTMPredictor,
    DualDataPreprocessor,
    DualSignalDataset,
    create_dual_sequences,
    DualCNNPredictor,
    DualPredictionResult,
    # Deprecated
    SequenceValidator,
    SequenceType,
)

# Import LSTM optimizer components
from .lstm_optimizer import (
    LSTMMetaheuristicOptimizer,
    LSTMOptimizationResult,
    HyperparamConfig,
    OptimizationCheckpoint,
)

# Import Dual-CNN LSTM optimizer components
from .dual_cnn_optimizer import (
    DualCNNMetaheuristicOptimizer,
    DualCNNOptimizationResult,
)

# Import log analyzer components
from .log_analyzer import (
    LSTMLogAnalyzer,
    FeatureImportanceResult,
    ParameterAnalysisResult,
    EvolutionAnalysisResult,
    ConfigurationRecommendation,
    AnalysisReport,
)

# Import indicator association analyzer components
from .indicator_association_analyzer import (
    IndicatorAssociationAnalyzer,
    IndicatorAssociationResult,
    FeatureStatistics,
    AssociationRule,
)

__version__ = "0.6.0"  # Added Dual-CNN LSTM optimizer
__all__ = [
    # Signal Population
    "SignalPopulator",
    # Indicator Optimizer
    "BaseIndicatorOptimizer",
    "OptimizationResult",
    "ConfigLoader",
    "IndicatorConfig",
    "SignalMatcher",
    "MatchResult",
    "GridSearchOptimizer",
    "HyperoptOptimizer",
    "DatasetBuilder",
    # LSTM
    "DataPreprocessor",
    "SignalDataset",
    "create_sequences",
    "LSTMSignalPredictor",
    "ModelConfig",
    "BinarySignalLoss",
    "FocalBinaryLoss",
    "WeightedSignalLoss",  # Alias for backwards compatibility
    "FocalWeightedLoss",   # Alias for backwards compatibility
    "Trainer",
    "TrainingConfig",
    "Predictor",
    # Dual-CNN LSTM
    "DualModelConfig",
    "DualCNNLSTMPredictor",
    "DualDataPreprocessor",
    "DualSignalDataset",
    "create_dual_sequences",
    "DualCNNPredictor",
    "DualPredictionResult",
    # Deprecated LSTM components
    "SequenceValidator",
    "SequenceType",
    # LSTM Optimizer
    "LSTMMetaheuristicOptimizer",
    "LSTMOptimizationResult",
    "HyperparamConfig",
    "OptimizationCheckpoint",
    # Dual-CNN LSTM Optimizer
    "DualCNNMetaheuristicOptimizer",
    "DualCNNOptimizationResult",
    # Log Analyzer
    "LSTMLogAnalyzer",
    "FeatureImportanceResult",
    "ParameterAnalysisResult",
    "EvolutionAnalysisResult",
    "ConfigurationRecommendation",
    "AnalysisReport",
    # Indicator Association Analyzer
    "IndicatorAssociationAnalyzer",
    "IndicatorAssociationResult",
    "FeatureStatistics",
    "AssociationRule",
]
