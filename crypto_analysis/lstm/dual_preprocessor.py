"""
Dual Data Preprocessor Module

Handles preprocessing for Dual-CNN LSTM model with separate binary and technical inputs.

Preprocessing steps:
1. Binary features (df_binary): 114 binary columns (57 indicators x entry/exit), values 0/1
   - Pass through unchanged (already in well-defined 0/1 range)

2. Technical features (df_technical): 76 continuous columns + OHLCV
   - Scale OHLCV columns using StandardScaler
   - Scale technical indicator values (many already normalized, but some need scaling)

3. Targets: Use 'tradeable' column for binary classification (hold=0, trade=1)
4. Shift targets by target_shift steps (features at t predict tradeable at t+target_shift)
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DualDataPreprocessor:
    """
    Preprocesses dual DataFrames for Dual-CNN LSTM training.

    Handles two separate input DataFrames:
    - df_binary: Binary indicator signals (0/1 entry/exit columns)
    - df_technical: Technical indicators + OHLCV (continuous values)

    Binary features are passed through unchanged.
    Technical and OHLCV features are scaled using StandardScaler.

    Attributes
    ----------
    binary_columns : list
        List of binary feature column names
    technical_columns : list
        List of technical + OHLCV feature column names
    target_column : str
        Column to use as target (default: 'tradeable')
    target_shift : int
        Number of steps to shift target (features at t predict t+shift)
    """

    # Drop signal-related columns (tradeable is the target)
    DEFAULT_COLUMNS_TO_DROP = ['date', 'signal_pct_change', 'period_id', 'signal']
    # Binary encoding for tradeable column
    DEFAULT_TARGET_ENCODING = {'hold': 0, 'trade': 1}
    # OHLCV columns that should be in technical DataFrame
    OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

    def __init__(
        self,
        columns_to_drop: Optional[List[str]] = None,
        target_encoding: Optional[Dict[str, int]] = None,
        target_column: str = 'tradeable',
        target_shift: int = 4,
    ):
        """
        Initialize preprocessor.

        Parameters
        ----------
        columns_to_drop : list, optional
            Columns to remove from features.
            Default: ['date', 'signal_pct_change', 'period_id', 'signal']
        target_encoding : dict, optional
            Mapping from target string to integer.
            Default: {'hold': 0, 'trade': 1} for binary classification
        target_column : str, default='tradeable'
            Column to use as prediction target
        target_shift : int, default=4
            Number of steps to shift target (features at t predict t+shift)
        """
        self.columns_to_drop = columns_to_drop or self.DEFAULT_COLUMNS_TO_DROP.copy()
        self.target_encoding = target_encoding or self.DEFAULT_TARGET_ENCODING.copy()
        self.target_column = target_column
        self.target_shift = target_shift

        # Scalers for technical features
        self.technical_scaler: Optional[StandardScaler] = None

        # Feature column lists (populated during fit)
        self.binary_columns: List[str] = []
        self.technical_columns: List[str] = []  # Includes OHLCV

        self._is_fitted = False

    def fit(
        self,
        df_binary: pd.DataFrame,
        df_technical: pd.DataFrame
    ) -> 'DualDataPreprocessor':
        """
        Fit the preprocessor on training data.

        Parameters
        ----------
        df_binary : pd.DataFrame
            DataFrame with binary indicator signals (0/1 columns)
        df_technical : pd.DataFrame
            DataFrame with technical indicators + OHLCV

        Returns
        -------
        self
            Fitted preprocessor
        """
        # Identify binary feature columns (exclude metadata columns)
        all_drop_cols = set(self.columns_to_drop + [self.target_column])
        self.binary_columns = [
            col for col in df_binary.columns
            if col not in all_drop_cols
        ]

        # Identify technical feature columns (includes OHLCV)
        self.technical_columns = [
            col for col in df_technical.columns
            if col not in all_drop_cols
        ]

        # Fit scaler on all technical columns (including OHLCV)
        self.technical_scaler = StandardScaler()
        self.technical_scaler.fit(df_technical[self.technical_columns].values)

        self._is_fitted = True
        return self

    def transform(
        self,
        df_binary: pd.DataFrame,
        df_technical: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform DataFrames to features and targets.

        Parameters
        ----------
        df_binary : pd.DataFrame
            DataFrame with binary indicator signals
        df_technical : pd.DataFrame
            DataFrame with technical indicators + OHLCV

        Returns
        -------
        tuple
            (binary_features, technical_features, targets)
            - binary_features shape: (n_samples - target_shift, n_binary)
            - technical_features shape: (n_samples - target_shift, n_technical)
            - targets shape: (n_samples - target_shift,)
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call fit() first.")

        # Validate DataFrames have same length
        if len(df_binary) != len(df_technical):
            raise ValueError(
                f"DataFrames must have same length: "
                f"binary={len(df_binary)}, technical={len(df_technical)}"
            )

        # 1. Extract and encode targets from either DataFrame (they should have same tradeable column)
        if self.target_column in df_binary.columns:
            target_df = df_binary
        elif self.target_column in df_technical.columns:
            target_df = df_technical
        else:
            raise ValueError(f"Target column '{self.target_column}' not found in either DataFrame")

        target_series = target_df[self.target_column].fillna('hold')

        # Use numpy vectorized mapping to avoid FutureWarning from pandas replace
        target_values = target_series.values
        targets = np.zeros(len(target_values), dtype=np.int64)
        valid_mask = np.zeros(len(target_values), dtype=bool)

        for key, val in self.target_encoding.items():
            mask = target_values == key
            targets[mask] = val
            valid_mask |= mask

        # Validate encoding
        if not valid_mask.all():
            unmapped = np.unique(target_values[~valid_mask]).tolist()
            raise ValueError(f"Unknown target values: {unmapped}")

        # 2. Extract binary features (pass through unchanged)
        binary_features = df_binary[self.binary_columns].values.astype(np.float32)

        # 3. Extract and scale technical features
        if self.technical_scaler is None:
            raise RuntimeError("Technical scaler not fitted")
        technical_values = self.technical_scaler.transform(
            df_technical[self.technical_columns].values
        )
        technical_features = np.asarray(technical_values, dtype=np.float32)

        # 4. Apply target shift (features at t predict t+shift)
        binary_features = binary_features[:-self.target_shift]
        technical_features = technical_features[:-self.target_shift]
        targets = targets[self.target_shift:]

        return binary_features, technical_features, targets

    def fit_transform(
        self,
        df_binary: pd.DataFrame,
        df_technical: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit and transform in one step.

        Parameters
        ----------
        df_binary : pd.DataFrame
            DataFrame with binary indicator signals
        df_technical : pd.DataFrame
            DataFrame with technical indicators + OHLCV

        Returns
        -------
        tuple
            (binary_features, technical_features, targets)
        """
        return self.fit(df_binary, df_technical).transform(df_binary, df_technical)

    def inverse_transform_targets(self, encoded: np.ndarray) -> List[str]:
        """
        Convert encoded targets back to string labels.

        Parameters
        ----------
        encoded : np.ndarray
            Array of encoded target integers (0=hold, 1=trade)

        Returns
        -------
        list
            String target labels ('hold' or 'trade')
        """
        inverse_encoding = {v: k for k, v in self.target_encoding.items()}
        return [inverse_encoding[int(e)] for e in encoded.flatten()]

    def get_num_binary_features(self) -> int:
        """Return number of binary features."""
        return len(self.binary_columns)

    def get_num_technical_features(self) -> int:
        """Return number of technical features (includes OHLCV)."""
        return len(self.technical_columns)

    def get_binary_feature_names(self) -> List[str]:
        """Return list of binary feature column names."""
        return self.binary_columns.copy()

    def get_technical_feature_names(self) -> List[str]:
        """Return list of technical feature column names."""
        return self.technical_columns.copy()

    def save(self, path: Union[str, Path]) -> None:
        """
        Save fitted preprocessor state to file.

        Parameters
        ----------
        path : str or Path
            Path to save preprocessor
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")

        state = {
            'columns_to_drop': self.columns_to_drop,
            'target_encoding': self.target_encoding,
            'target_column': self.target_column,
            'target_shift': self.target_shift,
            'technical_scaler': self.technical_scaler,
            'binary_columns': self.binary_columns,
            'technical_columns': self.technical_columns,
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'DualDataPreprocessor':
        """
        Load a fitted preprocessor from file.

        Parameters
        ----------
        path : str or Path
            Path to saved preprocessor

        Returns
        -------
        DualDataPreprocessor
            Loaded and fitted preprocessor
        """
        with open(path, 'rb') as f:
            loaded = pickle.load(f)

        # Handle both direct object pickle and state dict
        if isinstance(loaded, cls):
            return loaded

        state = loaded

        preprocessor = cls(
            columns_to_drop=state['columns_to_drop'],
            target_encoding=state['target_encoding'],
            target_column=state['target_column'],
            target_shift=state['target_shift'],
        )
        preprocessor.technical_scaler = state['technical_scaler']
        preprocessor.binary_columns = state['binary_columns']
        preprocessor.technical_columns = state['technical_columns']
        preprocessor._is_fitted = True

        return preprocessor

    def __repr__(self) -> str:
        if self._is_fitted:
            return (
                f"DualDataPreprocessor("
                f"target_column='{self.target_column}', "
                f"target_shift={self.target_shift}, "
                f"n_binary={len(self.binary_columns)}, "
                f"n_technical={len(self.technical_columns)}, "
                f"num_classes=2)"
            )
        else:
            return (
                f"DualDataPreprocessor("
                f"target_column='{self.target_column}', "
                f"target_shift={self.target_shift}, "
                f"num_classes=2, "
                f"status=unfitted)"
            )


def create_dual_sequences(
    binary_features: np.ndarray,
    technical_features: np.ndarray,
    targets: np.ndarray,
    input_seq_length: int = 16,
    output_seq_length: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding window sequences from dual feature arrays.

    For each valid start position i, creates:
    - Binary input sequence: binary_features[i : i + input_seq_length]
    - Technical input sequence: technical_features[i : i + input_seq_length]
    - Output: single target value at targets[i + input_seq_length]

    Parameters
    ----------
    binary_features : np.ndarray
        Binary feature array, shape (n_timesteps, n_binary)
    technical_features : np.ndarray
        Technical feature array, shape (n_timesteps, n_technical)
    targets : np.ndarray
        Target array, shape (n_timesteps,)
    input_seq_length : int, default=16
        Length of input sequences
    output_seq_length : int, default=1
        Length of output/target (1 for binary classification)

    Returns
    -------
    tuple
        - binary_sequences: shape (n_sequences, input_seq_length, n_binary)
        - technical_sequences: shape (n_sequences, input_seq_length, n_technical)
        - target_sequences: shape (n_sequences,) for single output

    Raises
    ------
    ValueError
        If not enough timesteps or feature arrays have different lengths
    """
    n_timesteps = len(binary_features)

    if len(technical_features) != n_timesteps or len(targets) != n_timesteps:
        raise ValueError(
            f"All arrays must have same length: "
            f"binary={len(binary_features)}, technical={len(technical_features)}, "
            f"targets={len(targets)}"
        )

    n_binary = binary_features.shape[1] if binary_features.ndim > 1 else 1
    n_technical = technical_features.shape[1] if technical_features.ndim > 1 else 1

    # Calculate valid sequence start positions
    max_start = n_timesteps - input_seq_length - output_seq_length + 1

    if max_start <= 0:
        raise ValueError(
            f"Not enough timesteps ({n_timesteps}) for "
            f"input_seq={input_seq_length} and output_seq={output_seq_length}. "
            f"Need at least {input_seq_length + output_seq_length} timesteps."
        )

    # Pre-allocate arrays
    binary_sequences = np.zeros(
        (max_start, input_seq_length, n_binary),
        dtype=np.float32
    )
    technical_sequences = np.zeros(
        (max_start, input_seq_length, n_technical),
        dtype=np.float32
    )
    target_sequences = np.zeros(max_start, dtype=np.int64)

    # Create sequences using sliding window
    for i in range(max_start):
        binary_sequences[i] = binary_features[i:i + input_seq_length]
        technical_sequences[i] = technical_features[i:i + input_seq_length]
        target_sequences[i] = targets[i + input_seq_length]

    return binary_sequences, technical_sequences, target_sequences


class DualSignalDataset:
    """
    Dataset for Dual-CNN LSTM binary classification.

    Stores separate binary and technical feature sequences along with
    target values for predicting whether the next period is tradeable.

    Attributes
    ----------
    binary_features : torch.Tensor
        Binary feature sequences, shape (n_samples, seq_len, n_binary)
    technical_features : torch.Tensor
        Technical feature sequences, shape (n_samples, seq_len, n_technical)
    targets : torch.Tensor
        Target values, shape (n_samples,) - 0=hold, 1=trade
    """

    def __init__(
        self,
        binary_features: np.ndarray,
        technical_features: np.ndarray,
        targets: np.ndarray,
        device: Optional[str] = None
    ):
        """
        Initialize dataset.

        Parameters
        ----------
        binary_features : np.ndarray
            Binary feature sequences, shape (n_samples, seq_len, n_binary)
        technical_features : np.ndarray
            Technical feature sequences, shape (n_samples, seq_len, n_technical)
        targets : np.ndarray
            Target values, shape (n_samples,) - 0=hold, 1=trade
        device : str, optional
            Device to store tensors on (default: CPU)
        """
        import torch

        self.device = torch.device(device) if device else torch.device('cpu')

        # Convert to tensors
        self.binary_features = torch.tensor(
            binary_features,
            dtype=torch.float32,
            device=self.device
        )
        self.technical_features = torch.tensor(
            technical_features,
            dtype=torch.float32,
            device=self.device
        )
        self.targets = torch.tensor(
            targets,
            dtype=torch.long,
            device=self.device
        )

        # Store metadata
        self.n_samples = len(binary_features)
        self.input_seq_length = binary_features.shape[1]
        self.n_binary = binary_features.shape[2]
        self.n_technical = technical_features.shape[2]
        self.num_classes = 2

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Returns
        -------
        dict
            'binary_features': tensor of shape (seq_len, n_binary)
            'technical_features': tensor of shape (seq_len, n_technical)
            'targets': scalar tensor (0=hold, 1=trade)
        """
        return {
            'binary_features': self.binary_features[idx],
            'technical_features': self.technical_features[idx],
            'targets': self.targets[idx]
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Custom collate function for DataLoader.

        Returns
        -------
        dict
            'binary_features': tensor of shape (batch_size, seq_len, n_binary)
            'technical_features': tensor of shape (batch_size, seq_len, n_technical)
            'targets': tensor of shape (batch_size,)
        """
        import torch

        return {
            'binary_features': torch.stack([b['binary_features'] for b in batch]),
            'technical_features': torch.stack([b['technical_features'] for b in batch]),
            'targets': torch.stack([b['targets'] for b in batch])
        }

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get distribution of target classes.

        Returns
        -------
        dict
            {'hold': count, 'trade': count}
        """
        import torch

        class_counts = torch.bincount(self.targets, minlength=2)
        return {
            'hold': int(class_counts[0]),
            'trade': int(class_counts[1])
        }

    def __repr__(self) -> str:
        dist = self.get_class_distribution()
        return (
            f"DualSignalDataset("
            f"n_samples={self.n_samples}, "
            f"seq_len={self.input_seq_length}, "
            f"n_binary={self.n_binary}, "
            f"n_technical={self.n_technical}, "
            f"hold={dist['hold']}, trade={dist['trade']})"
        )
