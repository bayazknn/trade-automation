"""
Dual-CNN LSTM Predictor Module

Utilities for inference with trained Dual-CNN LSTM models for binary classification.

Features:
- Load trained models from checkpoint + metadata
- Batch predictions on dual DataFrame inputs
- Single sequence prediction for real-time use
- Model evaluation with binary metrics
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from .dual_model import DualModelConfig, DualCNNLSTMPredictor
from .dual_preprocessor import DualDataPreprocessor, create_dual_sequences


@dataclass
class DualPredictionResult:
    """Container for prediction results."""
    predictions: np.ndarray      # Shape (n_sequences,) - predicted class indices (0=hold, 1=trade)
    probabilities: np.ndarray    # Shape (n_sequences, 2) - class probabilities [hold, trade]
    labels: List[str]            # Human-readable labels ('hold' or 'trade')
    confidence: np.ndarray       # Max probability for each prediction


class DualCNNPredictor:
    """
    Inference manager for trained Dual-CNN LSTM binary signal prediction models.

    Handles:
    - Loading trained models from checkpoints
    - Preprocessing dual input data
    - Making batch predictions
    - Making real-time single predictions
    - Evaluating model performance

    Attributes
    ----------
    model : DualCNNLSTMPredictor
        Trained Dual-CNN LSTM model
    preprocessor : DualDataPreprocessor
        Fitted data preprocessor
    metadata : dict
        Model metadata from metadata.json
    device : torch.device
        Device for inference

    Examples
    --------
    >>> predictor = DualCNNPredictor.from_checkpoint('output_dir')
    >>> result = predictor.predict(df_binary, df_technical)
    >>> print(result.labels[:5])
    ['hold', 'hold', 'trade', 'hold', 'trade']
    """

    SIGNAL_LABELS = {0: 'hold', 1: 'trade'}

    def __init__(
        self,
        model: DualCNNLSTMPredictor,
        preprocessor: DualDataPreprocessor,
        metadata: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize predictor.

        Parameters
        ----------
        model : DualCNNLSTMPredictor
            Trained model
        preprocessor : DualDataPreprocessor
            Fitted data preprocessor
        metadata : dict, optional
            Model metadata
        device : torch.device, optional
            Device for inference (default: CPU)
        """
        self.device = device or torch.device('cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocessor = preprocessor
        self.metadata = metadata or {}

        # Get sequence length from model config
        self.input_seq_length = model.config.input_seq_length

        # Get expected feature lists from metadata if available
        self.binary_features = metadata.get('feature_selection', {}).get('binary_features', []) if metadata else []
        self.technical_features = metadata.get('feature_selection', {}).get('technical_features', []) if metadata else []

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Union[str, Path],
        device: Optional[torch.device] = None
    ) -> 'DualCNNPredictor':
        """
        Load predictor from saved checkpoint directory.

        Expects the following files in checkpoint_dir:
        - best_model.pt: Model checkpoint
        - preprocessor.pkl: Fitted preprocessor
        - metadata.json: Model metadata

        Parameters
        ----------
        checkpoint_dir : str or Path
            Directory containing model artifacts
        device : torch.device, optional
            Device for inference

        Returns
        -------
        DualCNNPredictor
            Initialized predictor ready for inference
        """
        checkpoint_dir = Path(checkpoint_dir)
        device = device or torch.device('cpu')

        # Load metadata
        metadata_path = checkpoint_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load model checkpoint
        model_path = checkpoint_dir / 'best_model.pt'
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Reconstruct model from config
        model_config = checkpoint['model_config']
        model = DualCNNLSTMPredictor(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load preprocessor
        preprocessor_path = checkpoint_dir / 'preprocessor.pkl'
        preprocessor = DualDataPreprocessor.load(preprocessor_path)

        return cls(model, preprocessor, metadata, device)

    def predict(
        self,
        df_binary: pd.DataFrame,
        df_technical: pd.DataFrame,
        batch_size: int = 64
    ) -> DualPredictionResult:
        """
        Make predictions on dual DataFrames.

        Parameters
        ----------
        df_binary : pd.DataFrame
            Binary indicator signals DataFrame
        df_technical : pd.DataFrame
            Technical indicators + OHLCV DataFrame
        batch_size : int, default=64
            Batch size for inference

        Returns
        -------
        DualPredictionResult
            Container with predictions, probabilities, labels, and confidence
        """
        # Filter to expected features if we know them
        if self.binary_features:
            df_binary_filtered = df_binary[['tradeable'] + [
                c for c in self.binary_features if c in df_binary.columns
            ]].copy()
        else:
            df_binary_filtered = df_binary.copy()

        if self.technical_features:
            df_technical_filtered = df_technical[['tradeable'] + [
                c for c in self.technical_features if c in df_technical.columns
            ]].copy()
        else:
            df_technical_filtered = df_technical.copy()

        # Preprocess data
        binary_feat, technical_feat, _ = self.preprocessor.transform(
            df_binary_filtered, df_technical_filtered
        )

        # Create sequences
        dummy_targets = np.zeros(len(binary_feat), dtype=np.int64)
        binary_seqs, technical_seqs, _ = create_dual_sequences(
            binary_feat, technical_feat, dummy_targets,
            input_seq_length=self.input_seq_length,
            output_seq_length=1
        )

        # Convert to tensors
        binary_tensor = torch.tensor(binary_seqs, dtype=torch.float32)
        technical_tensor = torch.tensor(technical_seqs, dtype=torch.float32)

        # Predict in batches
        all_predictions = []
        all_probabilities = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(binary_tensor), batch_size):
                binary_batch = binary_tensor[i:i + batch_size].to(self.device)
                technical_batch = technical_tensor[i:i + batch_size].to(self.device)

                logits = self.model(binary_batch, technical_batch)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                all_predictions.append(preds.cpu().numpy())
                all_probabilities.append(probs.cpu().numpy())

        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)
        probabilities = np.concatenate(all_probabilities, axis=0)

        # Compute confidence
        confidence = np.max(probabilities, axis=-1)

        # Convert to labels
        labels = [self.SIGNAL_LABELS[p] for p in predictions]

        return DualPredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            labels=labels,
            confidence=confidence
        )

    def predict_with_threshold(
        self,
        df_binary: pd.DataFrame,
        df_technical: pd.DataFrame,
        trade_threshold: float = 0.5,
        batch_size: int = 64
    ) -> DualPredictionResult:
        """
        Make predictions with confidence threshold for trade signals.

        Only predicts trade if probability exceeds threshold, otherwise hold.

        Parameters
        ----------
        df_binary : pd.DataFrame
            Binary indicator signals DataFrame
        df_technical : pd.DataFrame
            Technical indicators + OHLCV DataFrame
        trade_threshold : float, default=0.5
            Minimum probability required to predict 'trade'
        batch_size : int, default=64
            Batch size for inference

        Returns
        -------
        DualPredictionResult
            Container with thresholded predictions
        """
        result = self.predict(df_binary, df_technical, batch_size)

        predictions = result.predictions.copy()
        probabilities = result.probabilities

        # Apply threshold
        for i in range(len(predictions)):
            if predictions[i] == 1 and probabilities[i, 1] < trade_threshold:
                predictions[i] = 0

        confidence = np.max(probabilities, axis=-1)
        labels = [self.SIGNAL_LABELS[p] for p in predictions]

        return DualPredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            labels=labels,
            confidence=confidence
        )

    def predict_next(
        self,
        recent_binary: pd.DataFrame,
        recent_technical: pd.DataFrame
    ) -> Dict:
        """
        Predict whether the next period is tradeable given recent data.

        Parameters
        ----------
        recent_binary : pd.DataFrame
            Most recent rows of binary data
        recent_technical : pd.DataFrame
            Most recent rows of technical data

        Returns
        -------
        dict
            {
                'signal': 'hold' or 'trade',
                'probabilities': [hold_prob, trade_prob],
                'confidence': float,
                'predicted_class': 0 or 1
            }
        """
        min_rows = self.input_seq_length + self.preprocessor.target_shift

        if len(recent_binary) < min_rows:
            raise ValueError(f"Need at least {min_rows} rows, got {len(recent_binary)}")
        if len(recent_technical) < min_rows:
            raise ValueError(f"Need at least {min_rows} rows, got {len(recent_technical)}")

        # Take required rows
        recent_binary = recent_binary.tail(min_rows)
        recent_technical = recent_technical.tail(min_rows)

        # Filter to expected features
        if self.binary_features:
            cols_to_use = ['tradeable'] + [c for c in self.binary_features if c in recent_binary.columns]
            recent_binary = recent_binary[cols_to_use].copy()
        if self.technical_features:
            cols_to_use = ['tradeable'] + [c for c in self.technical_features if c in recent_technical.columns]
            recent_technical = recent_technical[cols_to_use].copy()

        # Preprocess
        binary_feat, technical_feat, _ = self.preprocessor.transform(
            recent_binary, recent_technical
        )

        if len(binary_feat) < self.input_seq_length:
            raise ValueError(
                f"After preprocessing, need {self.input_seq_length} rows, got {len(binary_feat)}"
            )

        # Take last input_seq_length rows
        binary_feat = binary_feat[-self.input_seq_length:]
        technical_feat = technical_feat[-self.input_seq_length:]

        # Create tensors
        binary_tensor = torch.tensor(
            binary_feat.reshape(1, self.input_seq_length, -1),
            dtype=torch.float32,
            device=self.device
        )
        technical_tensor = torch.tensor(
            technical_feat.reshape(1, self.input_seq_length, -1),
            dtype=torch.float32,
            device=self.device
        )

        # Predict
        self.model.eval()
        with torch.no_grad():
            logits = self.model(binary_tensor, technical_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred = torch.argmax(logits, dim=-1).cpu().numpy()[0]

        return {
            'signal': self.SIGNAL_LABELS[pred],
            'probabilities': probs.tolist(),
            'confidence': float(np.max(probs)),
            'predicted_class': int(pred)
        }

    def evaluate(
        self,
        df_binary: pd.DataFrame,
        df_technical: pd.DataFrame,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Evaluate model on datasets with ground truth targets.

        Parameters
        ----------
        df_binary : pd.DataFrame
            Binary indicator signals DataFrame with tradeable column
        df_technical : pd.DataFrame
            Technical indicators + OHLCV DataFrame with tradeable column
        batch_size : int, default=64
            Batch size for inference

        Returns
        -------
        dict
            Evaluation metrics including accuracy, per-class metrics, confusion matrix
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix
        )

        # Get predictions
        result = self.predict(df_binary, df_technical, batch_size)

        # Filter DataFrames to expected features
        if self.binary_features:
            df_binary_filtered = df_binary[['tradeable'] + [
                c for c in self.binary_features if c in df_binary.columns
            ]].copy()
        else:
            df_binary_filtered = df_binary.copy()

        if self.technical_features:
            df_technical_filtered = df_technical[['tradeable'] + [
                c for c in self.technical_features if c in df_technical.columns
            ]].copy()
        else:
            df_technical_filtered = df_technical.copy()

        # Get true targets
        binary_feat, technical_feat, targets = self.preprocessor.transform(
            df_binary_filtered, df_technical_filtered
        )
        _, _, target_seqs = create_dual_sequences(
            binary_feat, technical_feat, targets,
            input_seq_length=self.input_seq_length,
            output_seq_length=1
        )

        y_true = target_seqs.flatten()
        y_pred = result.predictions.flatten()

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(
                y_true, y_pred, average=None, labels=[0, 1], zero_division=0
            )

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'hold_precision': float(precision_per_class[0]),
            'hold_recall': float(recall_per_class[0]),
            'hold_f1': float(f1_per_class[0]),
            'hold_support': int(support_per_class[0]) if support_per_class is not None else 0,
            'trade_precision': float(precision_per_class[1]),
            'trade_recall': float(recall_per_class[1]),
            'trade_f1': float(f1_per_class[1]),
            'trade_support': int(support_per_class[1]) if support_per_class is not None else 0,
            'confusion_matrix': cm.tolist()
        }

    def print_evaluation_report(self, metrics: Dict[str, float]):
        """Print a formatted evaluation report."""
        print("=" * 60)
        print("Dual-CNN LSTM Evaluation Report (Binary: hold=0, trade=1)")
        print("=" * 60)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")

        print(f"\nPer-Class Metrics:")
        print("-" * 60)
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-" * 60)

        for cls in ['hold', 'trade']:
            print(
                f"{cls:<10} "
                f"{metrics[f'{cls}_precision']:<12.4f} "
                f"{metrics[f'{cls}_recall']:<12.4f} "
                f"{metrics[f'{cls}_f1']:<12.4f} "
                f"{metrics[f'{cls}_support']:<10}"
            )

        print("-" * 60)

        if 'confusion_matrix' in metrics:
            print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
            print("              Predicted")
            print("              hold   trade")
            cm = metrics['confusion_matrix']
            labels = ['hold', 'trade']
            for i, row in enumerate(cm):
                print(f"Actual {labels[i]:<5} {row[0]:<6} {row[1]:<6}")

        print("=" * 60)

    def __repr__(self) -> str:
        return (
            f"DualCNNPredictor("
            f"input_seq={self.input_seq_length}, "
            f"n_binary={len(self.binary_features)}, "
            f"n_technical={len(self.technical_features)}, "
            f"device={self.device})"
        )
