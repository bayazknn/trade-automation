"""
PyTorch Dataset Module

Provides SignalDataset for loading and batching LSTM training data,
and create_sequences function for creating sliding window sequences.

For binary classification (hold=0, trade=1):
- Input: 12 timesteps of features
- Output: 1 target value (predicting if next period is tradeable)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    input_seq_length: int = 12,
    output_seq_length: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences from flat arrays.

    For each valid start position i, creates:
    - Input sequence: features[i : i + input_seq_length]
    - Output: single target value at targets[i + input_seq_length]

    Parameters
    ----------
    features : np.ndarray
        Feature array, shape (n_timesteps, n_features)
    targets : np.ndarray
        Target array, shape (n_timesteps,)
    input_seq_length : int, default=12
        Length of input sequences
    output_seq_length : int, default=1
        Length of output/target (1 for binary classification)

    Returns
    -------
    tuple
        - feature_sequences: shape (n_sequences, input_seq_length, n_features)
        - target_sequences: shape (n_sequences,) for single output

    Raises
    ------
    ValueError
        If not enough timesteps for the requested sequence lengths

    Examples
    --------
    >>> features = np.random.randn(100, 10)  # 100 timesteps, 10 features
    >>> targets = np.random.randint(0, 2, 100)  # Binary: 0=hold, 1=trade
    >>> feat_seqs, tgt_seqs = create_sequences(features, targets)
    >>> print(feat_seqs.shape, tgt_seqs.shape)
    (88, 12, 10) (88,)
    """
    n_timesteps = len(features)
    n_features = features.shape[1] if features.ndim > 1 else 1

    # Calculate valid sequence start positions
    # Input: features[i : i + input_seq_length]
    # Target: targets[i + input_seq_length] (single value)
    # So we need i + input_seq_length + output_seq_length <= n_timesteps
    max_start = n_timesteps - input_seq_length - output_seq_length + 1

    if max_start <= 0:
        raise ValueError(
            f"Not enough timesteps ({n_timesteps}) for "
            f"input_seq={input_seq_length} and output_seq={output_seq_length}. "
            f"Need at least {input_seq_length + output_seq_length} timesteps."
        )

    # Pre-allocate arrays for efficiency
    feature_sequences = np.zeros(
        (max_start, input_seq_length, n_features),
        dtype=np.float32
    )
    # Single target per sequence
    target_sequences = np.zeros(max_start, dtype=np.int64)

    # Create sequences using sliding window
    for i in range(max_start):
        feature_sequences[i] = features[i:i + input_seq_length]
        # Single target value at the next timestep after input sequence
        target_sequences[i] = targets[i + input_seq_length]

    return feature_sequences, target_sequences


class SignalDataset(Dataset):
    """
    PyTorch Dataset for LSTM binary classification.

    Stores feature sequences and single target values for predicting
    whether the next period is tradeable (hold=0, trade=1).

    Input shape: (batch, input_seq_length, n_features) = (batch, 12, n_feat)
    Output shape: (batch,) - single target class index per sample

    Attributes
    ----------
    features : torch.Tensor
        Feature sequences, shape (n_samples, input_seq_length, n_features)
    targets : torch.Tensor
        Target values, shape (n_samples,) - 0=hold, 1=trade

    Examples
    --------
    >>> features = np.random.randn(100, 12, 10).astype(np.float32)
    >>> targets = np.random.randint(0, 2, 100)  # Binary: 0=hold, 1=trade
    >>> dataset = SignalDataset(features, targets)
    >>> print(len(dataset))
    100
    >>> sample = dataset[0]
    >>> print(sample['features'].shape, sample['targets'].shape)
    torch.Size([12, 10]) torch.Size([])
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        device: Optional[torch.device] = None
    ):
        """
        Initialize dataset.

        Parameters
        ----------
        features : np.ndarray
            Feature sequences, shape (n_samples, input_seq_length, n_features)
        targets : np.ndarray
            Target values, shape (n_samples,) - 0=hold, 1=trade
        device : torch.device, optional
            Device to store tensors on (default: CPU)
        """
        self.device = device or torch.device('cpu')

        # Convert to tensors
        self.features = torch.tensor(
            features,
            dtype=torch.float32,
            device=self.device
        )
        self.targets = torch.tensor(
            targets,
            dtype=torch.long,
            device=self.device
        )

        # Store metadata
        self.n_samples = len(features)
        self.input_seq_length = features.shape[1]
        self.n_features = features.shape[2]
        self.num_classes = 2  # Binary: hold=0, trade=1

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        dict
            'features': tensor of shape (input_seq_length, n_features)
            'targets': scalar tensor (0=hold, 1=trade)
        """
        return {
            'features': self.features[idx],
            'targets': self.targets[idx]
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for DataLoader.

        Stacks individual samples into batched tensors.

        Parameters
        ----------
        batch : list
            List of dictionaries from __getitem__

        Returns
        -------
        dict
            'features': tensor of shape (batch_size, input_seq_length, n_features)
            'targets': tensor of shape (batch_size,)
        """
        return {
            'features': torch.stack([b['features'] for b in batch]),
            'targets': torch.stack([b['targets'] for b in batch])
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced data.

        Returns inverse frequency weights for each class (hold, trade).

        Returns
        -------
        torch.Tensor
            Class weights of shape (2,) for [hold_weight, trade_weight]
        """
        class_counts = torch.bincount(self.targets, minlength=2)
        total = class_counts.sum().float()

        # Inverse frequency weighting
        weights = total / (class_counts.float() * 2 + 1e-6)
        return weights

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get distribution of target classes.

        Returns
        -------
        dict
            {'hold': count, 'trade': count}
        """
        class_counts = torch.bincount(self.targets, minlength=2)
        return {
            'hold': int(class_counts[0]),
            'trade': int(class_counts[1])
        }

    def to(self, device: torch.device) -> 'SignalDataset':
        """
        Move dataset tensors to specified device.

        Parameters
        ----------
        device : torch.device
            Target device

        Returns
        -------
        SignalDataset
            Self with tensors on new device
        """
        self.device = device
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        return self

    def __repr__(self) -> str:
        dist = self.get_class_distribution()
        return (
            f"SignalDataset("
            f"n_samples={self.n_samples}, "
            f"input_seq_length={self.input_seq_length}, "
            f"n_features={self.n_features}, "
            f"hold={dist['hold']}, trade={dist['trade']})"
        )
