"""
VectorBT Data Preprocessor Module

Preprocesses CSV output from vectorbt_optimizer for LSTM training.

Input CSV columns:
- OHLCV: date, open, high, low, close, volume
- Entry/Exit signals: {INDICATOR}_entry, {INDICATOR}_exit (binary 0/1)
- Raw indicator values: {INDICATOR}_{output} (e.g., RSI_rsi, MACD_macd)
- Target: tradeable ("hold"/"trade")

Output:
- X: (n_sequences, sequence_length, n_features) float32
- y: (n_sequences,) int64 (0=hold, 1=trade)
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from crypto_analysis.lstm.dataset import create_sequences


class VectorBTDataPreprocessor:
    """
    Preprocesses vectorbt_optimizer CSV output for LSTM training.

    Handles loading, filtering, scaling, and sequence creation for
    multiple CSV files while ensuring sequences don't span across files.

    Key features:
    - Column filtering: Keep OHLCV + entry/exit signals, optionally raw indicators
    - Scaling: Only OHLCV columns are scaled (entry/exit are already 0/1)
    - Target shifting: Features at t predict target at t+shift
    - Sequence creation: Sliding windows for LSTM input
    - Multi-file support: Sequences don't span across different files

    Attributes
    ----------
    remove_raw_indicators : bool
        If True, drop raw indicator columns (keep only OHLCV + *_entry/*_exit)
    target_shift : int
        Number of steps to shift target (features at t predict t+shift)
    sequence_length : int
        Length of LSTM input sequences
    stride : int
        Step size between consecutive sequences
    scaler_type : str
        'standard' or 'minmax'
    target_column : str
        Column to use as target (default: 'tradeable')
    target_encoding : dict
        Mapping from target string to integer (hold=0, trade=1)
    ohlcv_columns : list
        Continuous columns to scale
    columns_to_drop : list
        Columns to always remove from features

    Examples
    --------
    >>> prep = VectorBTDataPreprocessor(
    ...     remove_raw_indicators=True,
    ...     sequence_length=24,
    ...     target_shift=1
    ... )
    >>> dfs = VectorBTDataPreprocessor.load_directory("notebooks/csv/")
    >>> X, y = prep.fit(dfs).create_sequences_from_multiple(dfs)
    >>> print(X.shape, y.shape)
    (1000, 24, 15) (1000,)
    """

    DEFAULT_OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    DEFAULT_COLUMNS_TO_DROP = ['date']
    DEFAULT_TARGET_ENCODING = {'hold': 0, 'trade': 1}

    def __init__(
        self,
        remove_raw_indicators: bool = True,
        target_shift: int = 1,
        sequence_length: int = 24,
        stride: int = 1,
        scaler_type: str = 'standard',
        target_column: str = 'tradeable',
        target_encoding: Optional[Dict[str, int]] = None,
        ohlcv_columns: Optional[List[str]] = None,
        columns_to_drop: Optional[List[str]] = None,
    ):
        """
        Initialize preprocessor.

        Parameters
        ----------
        remove_raw_indicators : bool, default=True
            If True, drop raw indicator value columns (e.g., RSI_rsi, MACD_macd).
            Keep only OHLCV + *_entry/*_exit signal columns.
        target_shift : int, default=1
            Number of steps to shift target. Features at t predict target at t+shift.
        sequence_length : int, default=24
            Length of LSTM input sequences.
        stride : int, default=1
            Step size between consecutive sequences.
        scaler_type : str, default='standard'
            'standard' for StandardScaler, 'minmax' for MinMaxScaler.
        target_column : str, default='tradeable'
            Column to use as prediction target.
        target_encoding : dict, optional
            Mapping from target string to integer.
            Default: {'hold': 0, 'trade': 1}
        ohlcv_columns : list, optional
            Continuous columns to scale.
            Default: ['open', 'high', 'low', 'close', 'volume']
        columns_to_drop : list, optional
            Columns to always remove from features.
            Default: ['date']
        """
        self.remove_raw_indicators = remove_raw_indicators
        self.target_shift = target_shift
        self.sequence_length = sequence_length
        self.stride = stride
        self.scaler_type = scaler_type
        self.target_column = target_column
        self.target_encoding = target_encoding or self.DEFAULT_TARGET_ENCODING.copy()
        self.ohlcv_columns = ohlcv_columns or self.DEFAULT_OHLCV_COLUMNS.copy()
        self.columns_to_drop = columns_to_drop or self.DEFAULT_COLUMNS_TO_DROP.copy()

        # Fitted state
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.feature_columns: List[str] = []
        self._ohlcv_indices: List[int] = []
        self._signal_indices: List[int] = []
        self._is_fitted = False

    @staticmethod
    def load_csv(path: Union[str, Path]) -> pd.DataFrame:
        """
        Load a single CSV file.

        Parameters
        ----------
        path : str or Path
            Path to CSV file

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame
        """
        return pd.read_csv(path)

    @staticmethod
    def load_directory(
        directory: Union[str, Path],
        pattern: str = "*.csv"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all matching CSV files from a directory.

        Parameters
        ----------
        directory : str or Path
            Directory containing CSV files
        pattern : str, default="*.csv"
            Glob pattern to match files

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping filename (without extension) to DataFrame
        """
        directory = Path(directory)
        dfs = {}

        for csv_path in sorted(directory.glob(pattern)):
            name = csv_path.stem
            dfs[name] = pd.read_csv(csv_path)

        return dfs

    def _filter_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Determine which columns to keep as features.

        Always keeps:
        - OHLCV columns
        - Columns ending with '_entry'
        - Columns ending with '_exit'

        Always drops:
        - Columns in columns_to_drop
        - Target column (it's the target, not a feature)

        Conditionally drops (if remove_raw_indicators=True):
        - Raw indicator value columns (not ending in _entry/_exit)

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame

        Returns
        -------
        List[str]
            List of column names to keep as features
        """
        keep_columns = []

        for col in df.columns:
            # Skip columns to drop
            if col in self.columns_to_drop:
                continue

            # Skip target column
            if col == self.target_column:
                continue

            # Always keep OHLCV
            if col in self.ohlcv_columns:
                keep_columns.append(col)
                continue

            # Always keep entry/exit signals
            if col.endswith('_entry') or col.endswith('_exit'):
                keep_columns.append(col)
                continue

            # Keep raw indicators only if remove_raw_indicators=False
            if not self.remove_raw_indicators:
                keep_columns.append(col)

        return keep_columns

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean DataFrame by removing NaN rows and all-zero columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame
        """
        df = df.copy()

        # Remove rows with NaN in any column (except target which we handle separately)
        feature_cols = [c for c in df.columns if c != self.target_column]
        df = df.dropna(subset=feature_cols).reset_index(drop=True)

        # Remove all-zero columns (excluding target)
        for col in feature_cols:
            if (df[col] == 0).all():
                df = df.drop(columns=[col])

        return df

    def fit(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame], List[pd.DataFrame]]
    ) -> 'VectorBTDataPreprocessor':
        """
        Fit the preprocessor on data.

        Fits the scaler on OHLCV columns from all provided data.

        Parameters
        ----------
        data : DataFrame, Dict[str, DataFrame], or List[DataFrame]
            Training data. Can be:
            - Single DataFrame
            - Dict of {name: DataFrame} (from load_directory)
            - List of DataFrames

        Returns
        -------
        self
            Fitted preprocessor
        """
        # Normalize input to list of DataFrames
        if isinstance(data, pd.DataFrame):
            dfs = [data]
        elif isinstance(data, dict):
            dfs = list(data.values())
        else:
            dfs = list(data)

        if not dfs:
            raise ValueError("No data provided for fitting")

        # Determine feature columns from first DataFrame
        first_df = self._clean_data(dfs[0])
        self.feature_columns = self._filter_columns(first_df)

        # Identify OHLCV vs signal column indices
        self._ohlcv_indices = []
        self._signal_indices = []
        for i, col in enumerate(self.feature_columns):
            if col in self.ohlcv_columns:
                self._ohlcv_indices.append(i)
            else:
                self._signal_indices.append(i)

        # Initialize scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {self.scaler_type}")

        # Collect OHLCV data from all DataFrames for fitting
        ohlcv_cols = [self.feature_columns[i] for i in self._ohlcv_indices]
        if ohlcv_cols:
            all_ohlcv = []
            for df in dfs:
                df_clean = self._clean_data(df)
                all_ohlcv.append(df_clean[ohlcv_cols].values)

            combined_ohlcv = np.vstack(all_ohlcv)
            self.scaler.fit(combined_ohlcv)

        self._is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform a DataFrame to features and targets.

        Applies column filtering, scaling (OHLCV only), and target shifting.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame

        Returns
        -------
        tuple
            (features, targets)
            - features: shape (n_samples - target_shift, n_features), float32
            - targets: shape (n_samples - target_shift,), int64

        Raises
        ------
        RuntimeError
            If preprocessor is not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call fit() first.")

        df = self._clean_data(df)

        # Validate target column exists
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")

        # Validate feature columns exist
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        # Extract features
        features = df[self.feature_columns].values.astype(np.float32)

        # Scale OHLCV columns only
        if self._ohlcv_indices and self.scaler is not None:
            ohlcv_cols = [self.feature_columns[i] for i in self._ohlcv_indices]
            scaled_ohlcv = self.scaler.transform(df[ohlcv_cols].values)
            for new_idx, orig_idx in enumerate(self._ohlcv_indices):
                features[:, orig_idx] = scaled_ohlcv[:, new_idx]

        # Encode targets
        df_target = df[self.target_column].fillna('hold')
        targets = df_target.map(self.target_encoding).values.astype(np.int64)

        # Check for unmapped target values
        if np.any(np.isnan(targets.astype(float))):
            unmapped = df_target[~df_target.isin(self.target_encoding)].unique()
            raise ValueError(f"Unknown target values: {unmapped}")

        # Shift targets: features at t predict target at t+shift
        if self.target_shift > 0:
            features = features[:-self.target_shift]
            targets = targets[self.target_shift:]

        return features, targets

    def fit_transform(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame], List[pd.DataFrame]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform in one step.

        Note: For multiple DataFrames, this returns the concatenated
        features and targets, NOT sequences. Use create_sequences_from_multiple
        if you need proper sequence handling across multiple files.

        Parameters
        ----------
        data : DataFrame, Dict[str, DataFrame], or List[DataFrame]
            Input data

        Returns
        -------
        tuple
            (features, targets) from all data concatenated
        """
        self.fit(data)

        # Normalize input
        if isinstance(data, pd.DataFrame):
            return self.transform(data)

        if isinstance(data, dict):
            dfs = list(data.values())
        else:
            dfs = list(data)

        # Transform and concatenate
        all_features = []
        all_targets = []
        for df in dfs:
            features, targets = self.transform(df)
            all_features.append(features)
            all_targets.append(targets)

        return np.vstack(all_features), np.concatenate(all_targets)

    def create_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create LSTM sequences from flat arrays.

        Uses sliding window approach to create sequences. Each sequence
        uses features from t to t+sequence_length-1 to predict the target
        at t+sequence_length.

        Parameters
        ----------
        features : np.ndarray
            Feature array, shape (n_timesteps, n_features)
        targets : np.ndarray
            Target array, shape (n_timesteps,)

        Returns
        -------
        tuple
            (X, y)
            - X: shape (n_sequences, sequence_length, n_features), float32
            - y: shape (n_sequences,), int64
        """
        return create_sequences(
            features=features,
            targets=targets,
            input_seq_length=self.sequence_length,
            output_seq_length=1,
            stride=self.stride
        )

    def create_sequences_from_multiple(
        self,
        data: Union[Dict[str, pd.DataFrame], List[pd.DataFrame]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create LSTM sequences from multiple DataFrames.

        Processes each DataFrame separately and concatenates the resulting
        sequences. This ensures sequences never span across different files.

        Parameters
        ----------
        data : Dict[str, DataFrame] or List[DataFrame]
            Multiple DataFrames to process

        Returns
        -------
        tuple
            (X, y)
            - X: shape (total_sequences, sequence_length, n_features), float32
            - y: shape (total_sequences,), int64

        Raises
        ------
        RuntimeError
            If preprocessor is not fitted
        ValueError
            If no valid sequences could be created
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted. Call fit() first.")

        if isinstance(data, dict):
            dfs = list(data.values())
        else:
            dfs = list(data)

        all_X = []
        all_y = []

        for df in dfs:
            try:
                features, targets = self.transform(df)

                # Skip if not enough data for sequences
                min_length = self.sequence_length + 1
                if len(features) < min_length:
                    continue

                X, y = self.create_sequences(features, targets)
                all_X.append(X)
                all_y.append(y)
            except ValueError as e:
                # Skip files that can't be processed (missing columns, etc.)
                continue

        if not all_X:
            raise ValueError("No valid sequences could be created from the data")

        return np.vstack(all_X), np.concatenate(all_y)

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

        Raises
        ------
        RuntimeError
            If preprocessor is not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")

        state = {
            'remove_raw_indicators': self.remove_raw_indicators,
            'target_shift': self.target_shift,
            'sequence_length': self.sequence_length,
            'stride': self.stride,
            'scaler_type': self.scaler_type,
            'target_column': self.target_column,
            'target_encoding': self.target_encoding,
            'ohlcv_columns': self.ohlcv_columns,
            'columns_to_drop': self.columns_to_drop,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            '_ohlcv_indices': self._ohlcv_indices,
            '_signal_indices': self._signal_indices,
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'VectorBTDataPreprocessor':
        """
        Load a fitted preprocessor from file.

        Parameters
        ----------
        path : str or Path
            Path to saved preprocessor

        Returns
        -------
        VectorBTDataPreprocessor
            Loaded and fitted preprocessor
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        # Handle direct object pickle
        if isinstance(state, cls):
            return state

        preprocessor = cls(
            remove_raw_indicators=state['remove_raw_indicators'],
            target_shift=state['target_shift'],
            sequence_length=state['sequence_length'],
            stride=state['stride'],
            scaler_type=state['scaler_type'],
            target_column=state['target_column'],
            target_encoding=state['target_encoding'],
            ohlcv_columns=state['ohlcv_columns'],
            columns_to_drop=state['columns_to_drop'],
        )
        preprocessor.scaler = state['scaler']
        preprocessor.feature_columns = state['feature_columns']
        preprocessor._ohlcv_indices = state['_ohlcv_indices']
        preprocessor._signal_indices = state['_signal_indices']
        preprocessor._is_fitted = True

        return preprocessor

    def __repr__(self) -> str:
        if self._is_fitted:
            return (
                f"VectorBTDataPreprocessor("
                f"sequence_length={self.sequence_length}, "
                f"target_shift={self.target_shift}, "
                f"stride={self.stride}, "
                f"remove_raw_indicators={self.remove_raw_indicators}, "
                f"n_features={len(self.feature_columns)}, "
                f"n_ohlcv={len(self._ohlcv_indices)}, "
                f"n_signals={len(self._signal_indices)}, "
                f"scaler_type='{self.scaler_type}')"
            )
        else:
            return (
                f"VectorBTDataPreprocessor("
                f"sequence_length={self.sequence_length}, "
                f"target_shift={self.target_shift}, "
                f"stride={self.stride}, "
                f"remove_raw_indicators={self.remove_raw_indicators}, "
                f"scaler_type='{self.scaler_type}', "
                f"status=unfitted)"
            )
