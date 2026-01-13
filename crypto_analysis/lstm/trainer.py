"""
Trainer Module

Handles training loop, validation, checkpointing, and early stopping
for binary classification (hold=0, trade=1).

Features:
- AdamW optimizer with learning rate scheduling
- Early stopping with configurable patience
- Model checkpointing (best model and periodic saves)
- Training history tracking
- Gradient clipping for stability
- Class weight computation for imbalanced data
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, _LRScheduler
from torch.utils.data import DataLoader, Subset

from .dataset import SignalDataset
from .model import LSTMSignalPredictor, ModelConfig
from .loss import WeightedSignalLoss, FocalWeightedLoss
from .data_preprocessor import DataPreprocessor


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model architecture (used when creating model from config)
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3  # Increased from 0.2 to reduce overfitting
    bidirectional: bool = False

    # Training parameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3  # Increased from 1e-4 for stronger regularization
    optimizer: str = 'adamw'  # 'adam' or 'adamw'
    grad_clip_norm: float = 1.0  # Max gradient norm for clipping

    # Learning rate scheduler
    scheduler: str = 'plateau'  # 'plateau', 'cosine', or 'none'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5

    # Class imbalance handling
    auto_class_weights: bool = True  # Automatically compute class weights from data
    class_weight_power: float = 0.5  # Power for class weights (0.5=sqrt, 1.0=full inverse)
    focal_loss: bool = False  # Use focal loss for hard example mining
    focal_gamma: float = 2.0  # Focal loss gamma parameter
    label_smoothing: float = 0.1  # Label smoothing (0.0=off, helps generalization)

    # Data split (train/val/test)
    val_split: float = 0.2
    test_split: float = 0.2  # Train=0.6, Val=0.2, Test=0.2

    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: Optional[str] = None
    save_best_only: bool = True
    save_interval: int = 10  # Save every N epochs (if not save_best_only)

    # Device
    device: str = 'auto'  # 'cuda', 'cpu', or 'auto'

    # Logging
    log_interval: int = 10  # Log every N batches
    verbose: bool = True


@dataclass
class TrainingHistory:
    """Stores training history."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')

    def to_dict(self) -> Dict:
        """Convert history to dictionary."""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss
        }


class EarlyStopping:
    """Early stopping handler to prevent overfitting."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        """
        Initialize early stopping.

        Parameters
        ----------
        patience : int
            Number of epochs to wait before stopping
        min_delta : float
            Minimum improvement to reset patience counter
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Parameters
        ----------
        val_loss : float
            Current validation loss

        Returns
        -------
        bool
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False


class Trainer:
    """
    Training manager for LSTM signal prediction model.

    Handles:
    - Training loop with batched gradient updates
    - Validation after each epoch
    - Learning rate scheduling (ReduceLROnPlateau or Cosine)
    - Early stopping
    - Model checkpointing
    - Training history tracking

    Examples
    --------
    >>> config = TrainingConfig(epochs=50, batch_size=32)
    >>> model_config = ModelConfig(input_size=50)
    >>> model = LSTMSignalPredictor(model_config)
    >>> trainer = Trainer(model, config)
    >>> history = trainer.train(train_dataset)
    """

    def __init__(
        self,
        model: LSTMSignalPredictor,
        config: TrainingConfig,
        loss_fn: Optional[nn.Module] = None,
        preprocessor: Optional[DataPreprocessor] = None
    ):
        """
        Initialize trainer.

        Parameters
        ----------
        model : LSTMSignalPredictor
            Model to train
        config : TrainingConfig
            Training configuration
        loss_fn : nn.Module, optional
            Loss function. If None, uses WeightedSignalLoss with config weights.
            Class weights will be computed from training data if auto_class_weights=True.
        preprocessor : DataPreprocessor, optional
            Data preprocessor to save alongside model checkpoints.
            If provided, will be saved as 'preprocessor.pkl' in checkpoint directory.
        """
        self.config = config
        self.device = self._get_device(config.device)

        # Store preprocessor for checkpointing
        self.preprocessor = preprocessor

        # Move model to device
        self.model = model.to(self.device)

        # Loss function (class weights computed later from training data)
        self._user_loss_fn = loss_fn  # Store user-provided loss function
        self.loss_fn = None  # Will be created in train() with class weights

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler (created in train() to know total epochs)
        self.scheduler: Optional[_LRScheduler] = None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        ) if config.early_stopping else None

        # History
        self.history = TrainingHistory()

        # Stored datasets (populated after train() is called)
        self.train_dataset: Optional[SignalDataset] = None
        self.val_dataset: Optional[SignalDataset] = None
        self.test_dataset: Optional[SignalDataset] = None

        # Class weights (computed from training data)
        self.class_weights: Optional[torch.Tensor] = None

        # Checkpoint directory
        if config.checkpoint_dir:
            self.checkpoint_dir = Path(config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

    def _get_device(self, device_str: str) -> torch.device:
        """Get the appropriate device."""
        if device_str == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_str)

    def _create_optimizer(self) -> Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer.lower() == 'adam':
            return Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adamw':
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler based on config."""
        if self.config.scheduler == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience
            )
        elif self.config.scheduler == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        return None

    def _compute_class_weights(self, dataset: SignalDataset) -> torch.Tensor:
        """
        Compute class weights from dataset to handle imbalance.

        Uses inverse frequency weighting with configurable power:
        - power=1.0: Full inverse frequency (strong rebalancing)
        - power=0.5: Square root of inverse frequency (balanced)
        - power=0.0: No rebalancing (all weights = 1)

        For binary classification (hold=0, trade=1), returns weights of shape (2,).

        Parameters
        ----------
        dataset : SignalDataset
            Dataset to compute weights from

        Returns
        -------
        torch.Tensor
            Class weights of shape (2,) for [hold_weight, trade_weight]
        """
        # Flatten all targets
        targets_flat = dataset.targets.flatten()
        class_counts = torch.bincount(targets_flat, minlength=2).float()
        total = class_counts.sum()

        # Avoid division by zero
        class_counts = torch.clamp(class_counts, min=1.0)

        # Inverse frequency weighting: weight = total / (num_classes * count)
        inverse_freq = total / (2.0 * class_counts)

        # Apply power to control strength of rebalancing
        weights = inverse_freq ** self.config.class_weight_power

        # Normalize so mean weight = 1 (doesn't change relative weights, just scale)
        weights = weights / weights.mean()

        return weights.to(self.device)

    def _create_loss_fn(self, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
        """
        Create loss function with class weights for binary classification.

        Parameters
        ----------
        class_weights : torch.Tensor, optional
            Per-class weights of shape (2,) for [hold, trade]

        Returns
        -------
        nn.Module
            Loss function (BinarySignalLoss or FocalBinaryLoss)
        """
        if self._user_loss_fn is not None:
            return self._user_loss_fn.to(self.device)

        if self.config.focal_loss:
            return FocalWeightedLoss(
                gamma=self.config.focal_gamma,
                class_weights=class_weights,
                label_smoothing=self.config.label_smoothing
            ).to(self.device)
        else:
            return WeightedSignalLoss(
                class_weights=class_weights,
                label_smoothing=self.config.label_smoothing
            ).to(self.device)

    def train(
        self,
        train_dataset: SignalDataset,
        val_dataset: Optional[SignalDataset] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> TrainingHistory:
        """
        Train the model.

        Parameters
        ----------
        train_dataset : SignalDataset
            Training dataset
        val_dataset : SignalDataset, optional
            Validation dataset. If None, splits from train_dataset.
        callbacks : list, optional
            Callback functions called after each epoch with (epoch, history)

        Returns
        -------
        TrainingHistory
            Training history with losses and metrics
        """
        # Reset history and early stopping
        self.history = TrainingHistory()
        if self.early_stopping:
            self.early_stopping.reset()

        # Create scheduler
        self.scheduler = self._create_scheduler()

        # Split dataset into train/val/test if not provided
        # (need to split first to compute class weights from training data only)
        if val_dataset is None:
            train_subset, val_subset, test_subset = self._split_dataset(
                train_dataset,
                self.config.val_split,
                self.config.test_split
            )
            # Store as instance attributes for later evaluation
            self.train_dataset = train_subset
            self.val_dataset = val_subset
            self.test_dataset = test_subset
        else:
            # Use provided datasets
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = None

        # Compute class weights from training data
        if self.config.auto_class_weights:
            # Get the underlying dataset from Subset if needed
            if hasattr(self.train_dataset, 'dataset'):
                # It's a Subset - get class distribution from underlying data
                self.class_weights = self._compute_class_weights(self.train_dataset.dataset)
            else:
                self.class_weights = self._compute_class_weights(self.train_dataset)

            if self.config.verbose:
                print(f"Class weights: hold={self.class_weights[0]:.3f}, "
                      f"trade={self.class_weights[1]:.3f}")
        else:
            self.class_weights = None

        # Create loss function with class weights
        self.loss_fn = self._create_loss_fn(self.class_weights)

        # Create data loaders (training uses only train_dataset)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=SignalDataset.collate_fn,
            drop_last=False
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=SignalDataset.collate_fn
        )

        if self.config.verbose:
            print(f"Training on {self.device}")
            print(f"Train samples: {len(self.train_dataset)}, "
                  f"Val samples: {len(self.val_dataset)}", end="")
            if self.test_dataset:
                print(f", Test samples: {len(self.test_dataset)}")
            else:
                print()
            print(f"Batches per epoch: {len(train_loader)}")
            print("-" * 50)

        # Training loop
        for epoch in range(self.config.epochs):
            # Train one epoch
            train_loss, train_acc = self._train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc = self._validate(val_loader)

            # Update history
            self.history.train_losses.append(train_loss)
            self.history.val_losses.append(val_loss)
            self.history.train_accuracies.append(train_acc)
            self.history.val_accuracies.append(val_acc)
            self.history.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            # Log epoch results
            if self.config.verbose:
                print(f"Epoch {epoch + 1}/{self.config.epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Checkpoint best model
            if val_loss < self.history.best_val_loss:
                self.history.best_val_loss = val_loss
                self.history.best_epoch = epoch
                if self.checkpoint_dir and self.config.save_best_only:
                    self._save_checkpoint('best_model.pt', epoch)
                    if self.config.verbose:
                        print(f"  Saved best model (val_loss: {val_loss:.4f})")

            # Periodic checkpoint
            if (self.checkpoint_dir and
                not self.config.save_best_only and
                (epoch + 1) % self.config.save_interval == 0):
                self._save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt', epoch)

            # Early stopping check
            if self.early_stopping and self.early_stopping(val_loss):
                if self.config.verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    print(f"Best epoch was {self.history.best_epoch + 1} "
                          f"with val_loss={self.history.best_val_loss:.4f}")
                break

            # Run callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, self.history)

        return self.history

    def _split_dataset(
        self,
        dataset: SignalDataset,
        val_ratio: float,
        test_ratio: float
    ) -> Tuple[Subset, Subset, Optional[Subset]]:
        """
        Split dataset into train/val/test sets sequentially (no shuffle).

        For time series data, we use sequential split to preserve temporal order.

        Parameters
        ----------
        dataset : SignalDataset
            Full dataset to split
        val_ratio : float
            Proportion for validation set (e.g., 0.2)
        test_ratio : float
            Proportion for test set (e.g., 0.2)

        Returns
        -------
        tuple
            (train_subset, val_subset, test_subset)
            test_subset is None if test_ratio <= 0
        """
        n = len(dataset)
        test_size = int(n * test_ratio) if test_ratio > 0 else 0
        val_size = int(n * val_ratio)
        train_size = n - val_size - test_size

        # Sequential split (no shuffling for time series)
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, train_size + val_size))

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        if test_size > 0:
            test_indices = list(range(train_size + val_size, n))
            test_subset = Subset(dataset, test_indices)
        else:
            test_subset = None

        return train_subset, val_subset, test_subset

    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.loss_fn(logits, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.grad_clip_norm
                )

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * features.size(0)
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == targets).sum().item()
            total += targets.numel()

            # Log progress
            if self.config.verbose and batch_idx % self.config.log_interval == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = correct / total

        return avg_loss, accuracy

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)

                logits = self.model(features)
                loss = self.loss_fn(logits, targets)

                total_loss += loss.item() * features.size(0)
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == targets).sum().item()
                total += targets.numel()

        avg_loss = total_loss / len(val_loader.dataset)
        accuracy = correct / total

        return avg_loss, accuracy

    def _save_checkpoint(self, filename: str, epoch: int):
        """Save model checkpoint and preprocessor."""
        import pickle

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.config,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_config': self.config,
            'history': self.history.to_dict()
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

        # Save preprocessor alongside model checkpoint
        if self.preprocessor is not None:
            preprocessor_path = self.checkpoint_dir / 'preprocessor.pkl'
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(self.preprocessor, f)

    def load_checkpoint(self, path: Union[str, Path], load_preprocessor: bool = True):
        """
        Load model from checkpoint.

        Parameters
        ----------
        path : str or Path
            Path to checkpoint file
        load_preprocessor : bool, default=True
            If True, also loads preprocessor.pkl from same directory if it exists
        """
        import pickle

        path = Path(path)
        # weights_only=False needed for custom classes like ModelConfig and TrainingConfig
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'history' in checkpoint:
            hist = checkpoint['history']
            self.history = TrainingHistory(
                train_losses=hist.get('train_losses', []),
                val_losses=hist.get('val_losses', []),
                train_accuracies=hist.get('train_accuracies', []),
                val_accuracies=hist.get('val_accuracies', []),
                learning_rates=hist.get('learning_rates', []),
                best_epoch=hist.get('best_epoch', 0),
                best_val_loss=hist.get('best_val_loss', float('inf'))
            )

        # Load preprocessor if it exists
        if load_preprocessor:
            preprocessor_path = path.parent / 'preprocessor.pkl'
            if preprocessor_path.exists():
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)

    def evaluate(
        self,
        dataset: SignalDataset,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate the trained model on a dataset with comprehensive metrics.

        Parameters
        ----------
        dataset : SignalDataset
            Dataset to evaluate on (can be train, val, or test)
        verbose : bool, default=True
            If True, prints a formatted evaluation report

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

        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=SignalDataset.collate_fn
        )

        # Collect all predictions and targets
        all_predictions = []
        all_targets = []

        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)

                logits = self.model(features)
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # Flatten arrays
        y_pred = np.concatenate(all_predictions).flatten()
        y_true = np.concatenate(all_targets).flatten()

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

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            # Per-class metrics (hold=0, trade=1)
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

        if verbose:
            self._print_single_evaluation(metrics)

        return metrics

    def evaluate_all(self, verbose: bool = True) -> Dict[str, Dict]:
        """
        Evaluate the trained model on all datasets (train, val, test).

        Parameters
        ----------
        verbose : bool, default=True
            If True, prints a comprehensive evaluation report

        Returns
        -------
        dict
            Dictionary with keys 'train', 'val', 'test' containing metrics for each dataset.
            'test' is None if no test dataset was created.
        """
        if self.train_dataset is None:
            raise RuntimeError("No datasets available. Call train() first.")

        results = {
            'train': self.evaluate(self.train_dataset, verbose=False),
            'val': self.evaluate(self.val_dataset, verbose=False),
            'test': self.evaluate(self.test_dataset, verbose=False) if self.test_dataset else None
        }

        if verbose:
            self.print_evaluation_report(results)

        return results

    def _print_single_evaluation(self, metrics: Dict) -> None:
        """Print evaluation report for a single dataset."""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION REPORT (Binary: hold=0, trade=1)")
        print("=" * 60)

        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f} (weighted)")
        print(f"  Recall:    {metrics['recall']:.4f} (weighted)")
        print(f"  F1 Score:  {metrics['f1']:.4f} (weighted)")

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

        print("=" * 60 + "\n")

    def print_evaluation_report(self, results: Dict[str, Dict]) -> None:
        """
        Print a comprehensive evaluation report for all datasets.

        Parameters
        ----------
        results : dict
            Results from evaluate_all() with keys 'train', 'val', 'test'
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE MODEL EVALUATION REPORT (Binary: hold=0, trade=1)")
        print("=" * 80)

        # Determine which datasets are available
        datasets = ['train', 'val']
        if results.get('test') is not None:
            datasets.append('test')

        # Overall metrics comparison table
        print("\n" + "-" * 80)
        print("OVERALL METRICS COMPARISON")
        print("-" * 80)
        print(f"{'Metric':<15} ", end="")
        for ds in datasets:
            print(f"{ds.upper():<20} ", end="")
        print()
        print("-" * 80)

        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            print(f"{metric.capitalize():<15} ", end="")
            for ds in datasets:
                val = results[ds][metric]
                print(f"{val:<20.4f} ", end="")
            print()

        # Per-class metrics for each dataset
        for ds in datasets:
            metrics = results[ds]
            print("\n" + "-" * 80)
            print(f"{ds.upper()} DATASET - Per-Class Metrics")
            print("-" * 80)
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

        # Confusion matrices
        print("\n" + "-" * 80)
        print("CONFUSION MATRICES (rows=actual, cols=predicted)")
        print("-" * 80)

        for ds in datasets:
            metrics = results[ds]
            if 'confusion_matrix' in metrics:
                print(f"\n{ds.upper()}:")
                print("              Predicted")
                print("              hold   trade")
                cm = metrics['confusion_matrix']
                labels = ['hold', 'trade']
                for i, row in enumerate(cm):
                    print(f"Actual {labels[i]:<5} {row[0]:<6} {row[1]:<6}")

        print("\n" + "=" * 80 + "\n")

    def __repr__(self) -> str:
        return (
            f"Trainer(device={self.device}, "
            f"epochs={self.config.epochs}, "
            f"batch_size={self.config.batch_size})"
        )
