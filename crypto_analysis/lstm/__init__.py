"""
LSTM Signal Prediction Package

PyTorch-based models for binary classification of trading signals (hold/trade).

Supported model architectures:
- LSTMSignalPredictor: Standard LSTM with input projection
- CNNLSTMSignalPredictor: CNN feature extractor + LSTM encoder (hybrid model)
- DualCNNLSTMPredictor: Dual-CNN LSTM for separate binary and technical inputs

Predicts whether the next period is tradeable:
- hold (0): No trade opportunity
- trade (1): Trade opportunity exists

Usage:
    from crypto_analysis.lstm import (
        DataPreprocessor,
        SignalDataset,
        LSTMSignalPredictor,
        CNNLSTMSignalPredictor,
        BinarySignalLoss,
        Trainer,
        TrainingConfig,
        Predictor,
        create_sequences,
        # Dual-CNN LSTM
        DualModelConfig,
        DualCNNLSTMPredictor,
        DualDataPreprocessor,
        DualSignalDataset,
        create_dual_sequences,
        DualCNNPredictor,
    )

Example:
    # Preprocess data
    preprocessor = DataPreprocessor()
    features, targets = preprocessor.fit_transform(df)

    # Create sequences
    X, y = create_sequences(features, targets)

    # Create dataset
    dataset = SignalDataset(X, y)

    # Train model
    config = TrainingConfig(epochs=50)
    model = LSTMSignalPredictor(ModelConfig(input_size=len(preprocessor.feature_columns)))
    trainer = Trainer(model, config, preprocessor=preprocessor)
    history = trainer.train(dataset)

    # Predict
    predictor = Predictor(model, preprocessor)
    result = predictor.predict(new_df)
    print(result.labels)  # ['hold', 'trade', 'hold', ...]
"""

from .data_preprocessor import DataPreprocessor
from .dataset import SignalDataset, create_sequences
from .model import LSTMSignalPredictor, CNNLSTMSignalPredictor, ModelConfig
from .loss import BinarySignalLoss, FocalBinaryLoss, WeightedSignalLoss, FocalWeightedLoss
from .trainer import Trainer, TrainingConfig, TrainingHistory
from .predictor import Predictor, PredictionResult, ThresholdResult

# Dual-CNN LSTM components
from .dual_model import DualModelConfig, DualCNNLSTMPredictor
from .dual_preprocessor import DualDataPreprocessor, DualSignalDataset, create_dual_sequences
from .dual_predictor import DualCNNPredictor, DualPredictionResult

# Deprecated: SequenceValidator is no longer needed for binary classification
# Import with deprecation warnings
from .sequence_validator import SequenceValidator, SequenceType, ValidationResult

__all__ = [
    # Data preprocessing
    "DataPreprocessor",
    # Dataset
    "SignalDataset",
    "create_sequences",
    # Models
    "LSTMSignalPredictor",
    "CNNLSTMSignalPredictor",
    "ModelConfig",
    # Loss (new names)
    "BinarySignalLoss",
    "FocalBinaryLoss",
    # Loss (backwards compatibility aliases)
    "WeightedSignalLoss",
    "FocalWeightedLoss",
    # Training
    "Trainer",
    "TrainingConfig",
    "TrainingHistory",
    # Prediction
    "Predictor",
    "PredictionResult",
    "ThresholdResult",
    # Dual-CNN LSTM
    "DualModelConfig",
    "DualCNNLSTMPredictor",
    "DualDataPreprocessor",
    "DualSignalDataset",
    "create_dual_sequences",
    "DualCNNPredictor",
    "DualPredictionResult",
    # Deprecated (kept for backwards compatibility)
    "SequenceValidator",
    "SequenceType",
    "ValidationResult",
]
