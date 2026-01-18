"""
Data Preprocessor Module

Handles loading, cleaning, encoding, and preparing DatasetBuilder output for LSTM training.

Preprocessing steps:
1. Drop columns: date, signal_pct_change, period_id, signal (keep tradeable)
2. Use 'tradeable' column for targets (hold/trade)
3. Encode: hold=0, trade=1 (binary classification)
4. Scale ONLY continuous columns (OHLCV) - binary indicator columns are kept as-is
5. Shift targets by target_shift steps (features at t predict tradeable at t+target_shift)
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessor:
    """
    Preprocesses DatasetBuilder output for LSTM training.

    Only OHLCV columns (open, high, low, close, volume) are scaled.
    Binary indicator columns (0/1 values) are kept unchanged since
    they are already in a well-defined range.

    Uses 'tradeable' column for binary classification:
    - hold = 0
    - trade = 1

    Attributes
    ----------
    columns_to_drop : list
        Columns to remove from features
    target_encoding : dict
        Mapping from tradeable string to integer (hold=0, trade=1)
    target_column : str
        Column to use as target (default: 'tradeable')
    target_shift : int
        Number of steps to shift target (features at t predict t+shift)
    scaler_type : str
        'standard' for StandardScaler, 'minmax' for MinMaxScaler
    """

    # Drop signal-related columns (tradeable is the target, signal is for indicator optimization)
    DEFAULT_COLUMNS_TO_DROP = ['date', 'signal_pct_change', 'period_id', 'signal']
    # Binary encoding for tradeable column
    DEFAULT_TARGET_ENCODING = {'hold': 0, 'trade': 1}
    # Continuous columns that need scaling
    CONTINUOUS_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    # Metadata columns to exclude from NaN checking (not feature columns)
    METADATA_COLUMNS = ['date', 'signal', 'signal_pct_change', 'period_id', 'tradeable']

    @staticmethod
    def align_dataframe(
        df: pd.DataFrame,
        period_size: int = 4,
        target_column: str = 'tradeable',
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Align DataFrame for period-consistent sequences.

        Drops rows with NaN values (feature columns only) and trims the DataFrame
        so that row count is divisible by period_size and periods have consistent
        target values (all hold or all trade within each period).

        This ensures that when using stride=period_size in create_sequences,
        each sequence's target will be from a consistent period.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to align
        period_size : int, default=4
            Ensure output row count is divisible by this value.
            Set to 1 to disable period alignment.
        target_column : str, default='tradeable'
            Column containing target values for period consistency check
        verbose : bool, default=False
            Print information about dropped/trimmed rows

        Returns
        -------
        pd.DataFrame
            Aligned DataFrame with row count divisible by period_size

        Examples
        --------
        >>> df = pd.read_csv('data.csv')
        >>> df_aligned = DataPreprocessor.align_dataframe(df, period_size=4)
        >>> # Now use with stride=4 for period-aligned sequences
        >>> feat_seqs, tgt_seqs = create_sequences(features, targets, stride=4)
        """
        original_len = len(df)

        # Find rows with NaN in feature columns (exclude metadata)
        metadata_cols = set(DataPreprocessor.METADATA_COLUMNS)
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        nan_mask = df[feature_cols].isna().any(axis=1)
        n_nan_rows = int(nan_mask.sum())

        if n_nan_rows > 0:
            df_aligned = df[~nan_mask].copy().reset_index(drop=True)
        else:
            df_aligned = df.copy()

        # Find optimal offset for period consistency
        if period_size > 1 and target_column in df_aligned.columns:
            best_offset = 0
            best_consistency = 0

            for offset in range(period_size):
                consistent_count = 0
                total_count = 0

                for i in range(offset, len(df_aligned) - period_size + 1, period_size):
                    chunk = df_aligned.iloc[i:i + period_size][target_column].tolist()
                    total_count += 1
                    if len(set(chunk)) == 1:
                        consistent_count += 1

                if total_count > 0 and consistent_count > best_consistency:
                    best_consistency = consistent_count
                    best_offset = offset

            # Trim from start using best offset
            if best_offset > 0:
                df_aligned = df_aligned.iloc[best_offset:].reset_index(drop=True)

            # Trim from end to ensure divisibility
            remaining_len = len(df_aligned)
            extra_trim = remaining_len % period_size
            if extra_trim > 0:
                df_aligned = df_aligned.iloc[:-extra_trim].reset_index(drop=True)

            total_trim = best_offset + extra_trim
            if verbose and (total_trim > 0 or n_nan_rows > 0):
                final_groups = len(df_aligned) // period_size
                print(f"Dropped {n_nan_rows} NaN rows, trimmed {best_offset} from start, "
                      f"{extra_trim} from end for {final_groups} aligned periods")
        elif period_size > 1:
            # No target column, just ensure divisibility
            current_len = len(df_aligned)
            if current_len % period_size != 0:
                trim_count = current_len % period_size
                df_aligned = df_aligned.iloc[trim_count:].reset_index(drop=True)

                if verbose:
                    print(f"Trimmed {trim_count} rows from start to ensure divisibility by {period_size}")

        if verbose:
            final_len = len(df_aligned)
            print(f"Original: {original_len}, Final: {final_len} (divisible by {period_size})")

        return df_aligned

    def __init__(
        self,
        columns_to_drop: Optional[List[str]] = None,
        target_encoding: Optional[Dict[str, int]] = None,
        target_column: str = 'tradeable',
        target_shift: int = 4,
        scaler_type: str = 'standard',
        continuous_columns: Optional[List[str]] = None
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
        scaler_type : str, default='standard'
            'standard' for StandardScaler, 'minmax' for MinMaxScaler
        continuous_columns : list, optional
            Columns that should be scaled. Default: ['open', 'high', 'low', 'close', 'volume']
            Binary indicator columns (not in this list) are kept as-is.
        """
        self.columns_to_drop = columns_to_drop or self.DEFAULT_COLUMNS_TO_DROP.copy()
        self.target_encoding = target_encoding or self.DEFAULT_TARGET_ENCODING.copy()
        self.target_column = target_column
        self.target_shift = target_shift
        self.scaler_type = scaler_type
        self.continuous_columns = continuous_columns or self.CONTINUOUS_COLUMNS.copy()

        # For backwards compatibility
        self.signal_encoding = self.target_encoding

        self.scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.feature_columns: List[str] = []        # All feature columns in order
        self._continuous_indices: List[int] = []    # Indices of continuous columns
        self._binary_indices: List[int] = []        # Indices of binary columns
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.

        Only fits the scaler on continuous columns (OHLCV).
        Binary indicator columns are identified and stored for pass-through.

        Parameters
        ----------
        df : pd.DataFrame
            DatasetBuilder output DataFrame

        Returns
        -------
        self
            Fitted preprocessor
        """
        df = df.copy()

        # Fill target column nulls with 'hold'
        if self.target_column in df.columns:
            df[self.target_column] = df[self.target_column].fillna('hold')

        # Determine feature columns (exclude target and columns to drop)
        all_drop_cols = self.columns_to_drop + [self.target_column]
        self.feature_columns = [
            col for col in df.columns
            if col not in all_drop_cols and col in df.columns
        ]

        # Identify continuous vs binary column indices
        self._continuous_indices = []
        self._binary_indices = []
        for i, col in enumerate(self.feature_columns):
            if col in self.continuous_columns:
                self._continuous_indices.append(i)
            else:
                self._binary_indices.append(i)

        # Initialize and fit scaler ONLY on continuous columns
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {self.scaler_type}")

        # Fit scaler only on continuous columns
        if self._continuous_indices:
            continuous_cols = [self.feature_columns[i] for i in self._continuous_indices]
            self.scaler.fit(df[continuous_cols].values)

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform DataFrame to features and targets.

        Only continuous columns (OHLCV) are scaled.
        Binary indicator columns are passed through unchanged.

        Parameters
        ----------
        df : pd.DataFrame
            DatasetBuilder output DataFrame

        Returns
        -------
        tuple
            (features: np.ndarray, targets: np.ndarray)
            - features shape: (n_samples - target_shift, n_features)
            - targets shape: (n_samples - target_shift,) - encoded target integers (0=hold, 1=trade)
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call fit() first.")

        df = df.copy()

        # 1. Fill target column nulls with 'hold'
        if self.target_column in df.columns:
            df[self.target_column] = df[self.target_column].fillna('hold')
        else:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")

        # 2. Encode targets
        df['target_encoded'] = df[self.target_column].map(self.target_encoding)

        # Validate encoding (check for unmapped values)
        if df['target_encoded'].isna().any():
            unmapped = df[df['target_encoded'].isna()][self.target_column].unique()
            raise ValueError(f"Unknown target values: {unmapped}")

        # 3. Extract targets
        targets = df['target_encoded'].values.astype(np.int64)

        # 4. Extract features - scale only continuous columns
        feature_df = df[self.feature_columns]
        features = feature_df.values.astype(np.float32)

        # Scale only continuous columns, leave binary columns unchanged
        if self._continuous_indices and self.scaler is not None:
            continuous_cols = [self.feature_columns[i] for i in self._continuous_indices]
            scaled_continuous = self.scaler.transform(df[continuous_cols].values)
            # Replace continuous columns with scaled values
            for new_idx, orig_idx in enumerate(self._continuous_indices):
                features[:, orig_idx] = scaled_continuous[:, new_idx]

        # 5. Shift targets by target_shift steps
        # Features at index i predict target at index i + target_shift
        features = features[:-self.target_shift]
        targets = targets[self.target_shift:]

        return features.astype(np.float32), targets

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform in one step.

        Parameters
        ----------
        df : pd.DataFrame
            DatasetBuilder output DataFrame

        Returns
        -------
        tuple
            (features, targets)
        """
        return self.fit(df).transform(df)

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

    def get_feature_names(self) -> List[str]:
        """
        Return list of feature column names.

        Returns
        -------
        list
            Feature column names in order
        """
        return self.feature_columns.copy()

    def get_num_features(self) -> int:
        """
        Return number of features.

        Returns
        -------
        int
            Number of feature columns
        """
        return len(self.feature_columns)

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
            'scaler_type': self.scaler_type,
            'continuous_columns': self.continuous_columns,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            '_continuous_indices': self._continuous_indices,
            '_binary_indices': self._binary_indices,
            # For backwards compatibility
            'signal_encoding': self.target_encoding,
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'DataPreprocessor':
        """
        Load a fitted preprocessor from file.

        Parameters
        ----------
        path : str or Path
            Path to saved preprocessor

        Returns
        -------
        DataPreprocessor
            Loaded and fitted preprocessor
        """
        with open(path, 'rb') as f:
            loaded = pickle.load(f)

        # Handle both direct object pickle (from Trainer) and state dict (from save())
        if isinstance(loaded, cls):
            # Directly pickled DataPreprocessor object
            return loaded

        # State dictionary format (from DataPreprocessor.save())
        state = loaded

        # Handle both old format (signal_encoding) and new format (target_encoding)
        target_encoding = state.get('target_encoding', state.get('signal_encoding', cls.DEFAULT_TARGET_ENCODING))
        target_column = state.get('target_column', 'tradeable')

        preprocessor = cls(
            columns_to_drop=state['columns_to_drop'],
            target_encoding=target_encoding,
            target_column=target_column,
            target_shift=state['target_shift'],
            scaler_type=state['scaler_type'],
            continuous_columns=state.get('continuous_columns', cls.CONTINUOUS_COLUMNS)
        )
        preprocessor.scaler = state['scaler']
        preprocessor.feature_columns = state['feature_columns']
        preprocessor._continuous_indices = state.get('_continuous_indices', [])
        preprocessor._binary_indices = state.get('_binary_indices', [])
        preprocessor._is_fitted = True

        return preprocessor

    def __repr__(self) -> str:
        if self._is_fitted:
            return (
                f"DataPreprocessor("
                f"target_column='{self.target_column}', "
                f"target_shift={self.target_shift}, "
                f"scaler_type='{self.scaler_type}', "
                f"n_features={len(self.feature_columns)}, "
                f"n_scaled={len(self._continuous_indices)}, "
                f"n_binary={len(self._binary_indices)}, "
                f"num_classes=2)"
            )
        else:
            return (
                f"DataPreprocessor("
                f"target_column='{self.target_column}', "
                f"target_shift={self.target_shift}, "
                f"scaler_type='{self.scaler_type}', "
                f"num_classes=2, "
                f"status=unfitted)"
            )
