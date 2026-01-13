"""
Predictor Module

Utilities for inference with trained LSTM models for binary classification.

For binary classification (hold=0, trade=1):
- Input: 12 timesteps of features
- Output: Single prediction per sequence (0=hold, 1=trade)

Features:
- Load trained models from checkpoints
- Batch predictions on DatasetBuilder output
- Single sequence prediction for real-time use
- Model evaluation with binary metrics
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from .model import LSTMSignalPredictor, ModelConfig
from .data_preprocessor import DataPreprocessor
from .dataset import create_sequences


@dataclass
class PredictionResult:
    """Container for prediction results."""
    predictions: np.ndarray      # Shape (n_sequences,) - predicted class indices (0=hold, 1=trade)
    probabilities: np.ndarray    # Shape (n_sequences, 2) - class probabilities [hold, trade]
    labels: List[str]            # Human-readable labels for each sequence ('hold' or 'trade')
    confidence: np.ndarray       # Max probability for each prediction


@dataclass
class ThresholdResult:
    """Container for threshold evaluation results."""
    threshold: float
    accuracy: float
    hold_precision: float
    hold_recall: float
    hold_f1: float
    trade_precision: float
    trade_recall: float
    trade_f1: float
    confusion_matrix: List[List[int]]


class Predictor:
    """
    Inference manager for trained LSTM binary signal prediction models.

    Handles:
    - Loading trained models from checkpoints
    - Preprocessing input data
    - Making batch predictions
    - Making real-time single predictions
    - Evaluating model performance

    Predicts whether the next period is tradeable:
    - 0 = hold (no trade opportunity)
    - 1 = trade (trade opportunity exists)

    Attributes
    ----------
    model : LSTMSignalPredictor
        Trained LSTM model
    preprocessor : DataPreprocessor
        Fitted data preprocessor
    device : torch.device
        Device for inference

    Examples
    --------
    >>> predictor = Predictor.from_checkpoint(
    ...     'checkpoints/best_model.pt',
    ...     'checkpoints/preprocessor.pkl'
    ... )
    >>> result = predictor.predict(df)
    >>> print(result.labels[:5])
    ['hold', 'hold', 'trade', 'hold', 'trade']
    """

    SIGNAL_LABELS = {0: 'hold', 1: 'trade'}

    def __init__(
        self,
        model: LSTMSignalPredictor,
        preprocessor: DataPreprocessor,
        device: Optional[torch.device] = None
    ):
        """
        Initialize predictor.

        Parameters
        ----------
        model : LSTMSignalPredictor
            Trained model
        preprocessor : DataPreprocessor
            Fitted data preprocessor
        device : torch.device, optional
            Device for inference (default: CPU)
        """
        self.device = device or torch.device('cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocessor = preprocessor

        # Get sequence length from model config
        self.input_seq_length = model.config.input_seq_length

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        preprocessor_path: Union[str, Path],
        device: Optional[torch.device] = None
    ) -> 'Predictor':
        """
        Load predictor from saved checkpoint and preprocessor.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to model checkpoint (.pt file)
        preprocessor_path : str or Path
            Path to saved preprocessor (.pkl file)
        device : torch.device, optional
            Device for inference

        Returns
        -------
        Predictor
            Initialized predictor ready for inference
        """
        device = device or torch.device('cpu')

        # Load checkpoint (weights_only=False needed for custom classes like ModelConfig)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Reconstruct model from config
        model_config = checkpoint['model_config']
        model = LSTMSignalPredictor(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load preprocessor
        preprocessor = DataPreprocessor.load(preprocessor_path)

        return cls(model, preprocessor, device)

    def predict(
        self,
        df: pd.DataFrame,
        batch_size: int = 64
    ) -> PredictionResult:
        """
        Make predictions on DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DatasetBuilder output DataFrame
        batch_size : int, default=64
            Batch size for inference

        Returns
        -------
        PredictionResult
            Container with predictions (0=hold, 1=trade), probabilities, labels, and confidence
        """
        # Preprocess data
        features, _ = self.preprocessor.transform(df)

        # Create sequences (dummy targets since we're predicting)
        dummy_targets = np.zeros(len(features), dtype=np.int64)
        feature_sequences, _ = create_sequences(
            features,
            dummy_targets,
            input_seq_length=self.input_seq_length,
            output_seq_length=1
        )

        # Convert to tensor
        x = torch.tensor(feature_sequences, dtype=torch.float32)

        # Predict in batches
        all_predictions = []
        all_probabilities = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = x[i:i + batch_size].to(self.device)
                logits = self.model(batch)  # (batch, 2)
                probs = torch.softmax(logits, dim=-1)  # (batch, 2)
                preds = torch.argmax(logits, dim=-1)  # (batch,)

                all_predictions.append(preds.cpu().numpy())
                all_probabilities.append(probs.cpu().numpy())

        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)  # (n_sequences,)
        probabilities = np.concatenate(all_probabilities, axis=0)  # (n_sequences, 2)

        # Compute confidence (max probability for each prediction)
        confidence = np.max(probabilities, axis=-1)  # (n_sequences,)

        # Convert to human-readable labels
        labels = [self.SIGNAL_LABELS[p] for p in predictions]

        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            labels=labels,
            confidence=confidence
        )

    def predict_with_threshold(
        self,
        df: pd.DataFrame,
        trade_threshold: float = 0.5,
        batch_size: int = 64
    ) -> PredictionResult:
        """
        Make predictions with confidence threshold for trade signals.

        Only predicts trade if the model's confidence exceeds the threshold,
        otherwise defaults to 'hold'. This reduces false positives for the
        minority class (trade).

        Parameters
        ----------
        df : pd.DataFrame
            DatasetBuilder output DataFrame
        trade_threshold : float, default=0.5
            Minimum probability required to predict 'trade' (class 1)
        batch_size : int, default=64
            Batch size for inference

        Returns
        -------
        PredictionResult
            Container with thresholded predictions
        """
        # Get raw predictions
        result = self.predict(df, batch_size)

        # Apply threshold
        predictions = result.predictions.copy()
        probabilities = result.probabilities

        # For each prediction
        for i in range(len(predictions)):
            # If predicted trade but confidence < threshold, switch to hold
            if predictions[i] == 1 and probabilities[i, 1] < trade_threshold:
                predictions[i] = 0  # hold

        # Recompute confidence and labels
        confidence = np.max(probabilities, axis=-1)
        labels = [self.SIGNAL_LABELS[p] for p in predictions]

        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            labels=labels,
            confidence=confidence
        )

    def find_optimal_threshold(
        self,
        df: pd.DataFrame,
        thresholds: Optional[List[float]] = None,
        metric: str = 'f1',
        batch_size: int = 64,
        verbose: bool = True
    ) -> Dict:
        """
        Find optimal confidence threshold by evaluating different values.

        Tests multiple thresholds and returns the one that maximizes the
        specified metric for trade predictions.

        Parameters
        ----------
        df : pd.DataFrame
            DatasetBuilder output DataFrame with tradeable column (for evaluation)
        thresholds : list, optional
            List of thresholds to try. Default: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        metric : str, default='f1'
            Metric to optimize: 'f1', 'precision', or 'recall'
        batch_size : int, default=64
            Batch size for inference
        verbose : bool, default=True
            Print results for each threshold

        Returns
        -------
        dict
            {
                'best_threshold': float,
                'results': List[ThresholdResult],
                'best_metrics': dict
            }
        """
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Get raw predictions first
        raw_result = self.predict(df, batch_size)

        # Get true targets
        features, targets = self.preprocessor.transform(df)
        _, target_sequences = create_sequences(
            features, targets,
            input_seq_length=self.input_seq_length,
            output_seq_length=1
        )
        y_true = target_sequences.flatten()

        results = []
        best_score = -1
        best_threshold = 0.5

        if verbose:
            print("\nThreshold Evaluation Results (Binary: hold=0, trade=1):")
            print("=" * 80)
            print(f"{'Threshold':<10} {'Accuracy':<10} {'Hold P':<10} {'Hold R':<10} "
                  f"{'Hold F1':<10} {'Trade P':<10} {'Trade R':<10} {'Trade F1':<10}")
            print("-" * 80)

        for threshold in thresholds:
            # Apply threshold
            predictions = raw_result.predictions.copy()
            probabilities = raw_result.probabilities

            for i in range(len(predictions)):
                if predictions[i] == 1 and probabilities[i, 1] < threshold:
                    predictions[i] = 0

            y_pred = predictions

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)

            precision_per_class, recall_per_class, f1_per_class, _ = \
                precision_recall_fscore_support(
                    y_true, y_pred, average=None, labels=[0, 1], zero_division=0
                )

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            result = ThresholdResult(
                threshold=threshold,
                accuracy=accuracy,
                hold_precision=precision_per_class[0],
                hold_recall=recall_per_class[0],
                hold_f1=f1_per_class[0],
                trade_precision=precision_per_class[1],
                trade_recall=recall_per_class[1],
                trade_f1=f1_per_class[1],
                confusion_matrix=cm.tolist()
            )
            results.append(result)

            # Calculate score for this threshold (optimize for trade class)
            if metric == 'f1':
                score = f1_per_class[1]  # Trade F1
            elif metric == 'precision':
                score = precision_per_class[1]  # Trade precision
            elif metric == 'recall':
                score = recall_per_class[1]  # Trade recall
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

            if verbose:
                print(f"{threshold:<10.2f} {accuracy:<10.4f} "
                      f"{precision_per_class[0]:<10.4f} {recall_per_class[0]:<10.4f} "
                      f"{f1_per_class[0]:<10.4f} {precision_per_class[1]:<10.4f} "
                      f"{recall_per_class[1]:<10.4f} {f1_per_class[1]:<10.4f}")

        if verbose:
            print("-" * 80)
            print(f"\nBest threshold: {best_threshold:.2f} (optimizing for trade {metric})")
            print(f"Best trade {metric}: {best_score:.4f}")

        # Get best result
        best_result = next(r for r in results if r.threshold == best_threshold)

        return {
            'best_threshold': best_threshold,
            'results': results,
            'best_metrics': {
                'accuracy': best_result.accuracy,
                'hold_precision': best_result.hold_precision,
                'hold_recall': best_result.hold_recall,
                'hold_f1': best_result.hold_f1,
                'trade_precision': best_result.trade_precision,
                'trade_recall': best_result.trade_recall,
                'trade_f1': best_result.trade_f1,
                'confusion_matrix': best_result.confusion_matrix
            }
        }

    def evaluate_with_threshold(
        self,
        df: pd.DataFrame,
        trade_threshold: float = 0.5,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Evaluate model with confidence threshold applied.

        Parameters
        ----------
        df : pd.DataFrame
            DatasetBuilder output DataFrame with tradeable column
        trade_threshold : float, default=0.5
            Minimum probability required to predict 'trade'
        batch_size : int, default=64
            Batch size for inference

        Returns
        -------
        dict
            Same format as evaluate() but with thresholded predictions
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix
        )

        # Get thresholded predictions
        result = self.predict_with_threshold(df, trade_threshold, batch_size)

        # Get true targets
        features, targets = self.preprocessor.transform(df)
        _, target_sequences = create_sequences(
            features, targets,
            input_seq_length=self.input_seq_length,
            output_seq_length=1
        )

        # Flatten for metrics
        y_true = target_sequences.flatten()
        y_pred = result.predictions.flatten()

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(
                y_true, y_pred, average=None, labels=[0, 1], zero_division=0
            )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'threshold_trade': trade_threshold,
            # Per-class metrics
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

    def predict_next(
        self,
        recent_data: pd.DataFrame
    ) -> Dict:
        """
        Predict whether the next period is tradeable given recent data.

        This is useful for real-time prediction where you have the
        most recent 12+ rows of data and want to predict ahead.

        Parameters
        ----------
        recent_data : pd.DataFrame
            Most recent rows of data (at least input_seq_length rows)

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
        # Ensure we have enough rows
        if len(recent_data) < self.input_seq_length:
            raise ValueError(
                f"Need at least {self.input_seq_length} rows, "
                f"got {len(recent_data)}"
            )

        # Take the required number of rows
        recent_data = recent_data.tail(self.input_seq_length + self.preprocessor.target_shift)

        # Preprocess
        features, _ = self.preprocessor.transform(recent_data)

        # We need exactly input_seq_length rows after preprocessing
        if len(features) < self.input_seq_length:
            raise ValueError(
                f"After preprocessing, need {self.input_seq_length} rows, "
                f"got {len(features)}"
            )

        # Take the last input_seq_length rows
        features = features[-self.input_seq_length:]

        # Create single sequence tensor
        x = torch.tensor(
            features.reshape(1, self.input_seq_length, -1),
            dtype=torch.float32,
            device=self.device
        )

        # Predict
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)  # (1, 2)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # (2,)
            pred = torch.argmax(logits, dim=-1).cpu().numpy()[0]  # scalar

        return {
            'signal': self.SIGNAL_LABELS[pred],
            'probabilities': probs.tolist(),
            'confidence': float(np.max(probs)),
            'predicted_class': int(pred)
        }

    def evaluate(
        self,
        df: pd.DataFrame,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset with ground truth targets.

        Parameters
        ----------
        df : pd.DataFrame
            DatasetBuilder output DataFrame with tradeable column
        batch_size : int, default=64
            Batch size for inference

        Returns
        -------
        dict
            Evaluation metrics including:
            - accuracy: Overall accuracy
            - precision, recall, f1: Weighted averages
            - {class}_precision/recall/f1: Per-class metrics (hold, trade)
            - confusion_matrix: 2x2 confusion matrix
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix
        )

        # Get predictions
        result = self.predict(df, batch_size)

        # Get true targets
        features, targets = self.preprocessor.transform(df)
        _, target_sequences = create_sequences(
            features, targets,
            input_seq_length=self.input_seq_length,
            output_seq_length=1
        )

        # Flatten for metrics
        y_true = target_sequences.flatten()
        y_pred = result.predictions.flatten()

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # Per-class metrics (hold=0, trade=1)
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(
                y_true, y_pred, average=None, labels=[0, 1], zero_division=0
            )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            # Per-class metrics
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

    def save(
        self,
        model_path: Union[str, Path],
        preprocessor_path: Union[str, Path]
    ):
        """
        Save model and preprocessor for later use.

        Parameters
        ----------
        model_path : str or Path
            Path to save model checkpoint
        preprocessor_path : str or Path
            Path to save preprocessor
        """
        # Save model
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.config
        }
        torch.save(checkpoint, model_path)

        # Save preprocessor
        self.preprocessor.save(preprocessor_path)

    def print_evaluation_report(self, metrics: Dict[str, float]):
        """
        Print a formatted evaluation report.

        Parameters
        ----------
        metrics : dict
            Metrics from evaluate()
        """
        print("=" * 60)
        print("Model Evaluation Report (Binary: hold=0, trade=1)")
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
            f"Predictor("
            f"input_seq={self.input_seq_length}, "
            f"num_classes=2 (hold/trade), "
            f"device={self.device})"
        )
