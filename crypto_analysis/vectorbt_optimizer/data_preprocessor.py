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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from crypto_analysis.lstm.dataset import create_sequences, SignalDataset


class VectorBTDataPreprocessor:
    """
    Preprocesses vectorbt_optimizer CSV output for LSTM training.

    Handles loading, filtering, scaling, and sequence creation for
    multiple CSV files while ensuring sequences don't span across files.

    Key features:
    - Column filtering: Keep OHLCV + entry/exit signals, optionally raw indicators
    - Scaling: All non-binary columns are scaled (OHLCV + raw indicators).
      Entry/exit signals are already 0/1 and are not scaled.
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

    Note
    ----
    This class is not thread-safe. Each thread should use its own instance.

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

    # Small epsilon value to avoid division by zero
    EPSILON = 1e-10

    # Default number of partitions for resampling (24 hours * 7 days)
    DEFAULT_RESAMPLE_PARTITIONS = 168

    # Bounded oscillators that are already percentage-based and should not be
    # normalized by close price. These indicators are already in a fixed range
    # (e.g., 0-100, -100 to 100) and are not price-dependent.
    BOUNDED_OSCILLATORS = (
        'RSI', 'STOCH', 'STOCHRSI', 'STOCHF',  # 0-100
        'WILLR',  # -100 to 0
        'ULTOSC',  # 0-100
        'MFI',  # 0-100
        'ADX', 'ADXR', 'PLUS_DI', 'MINUS_DI', 'DX',  # 0-100
        'AROON', 'AROONOSC',  # 0-100 or -100 to 100
        'CCI',  # unbounded but percentile-based
        'CMO',  # -100 to 100
        'ROC', 'ROCP', 'ROCR',  # percentage-based
        'MOM',  # momentum (not price-based)
        'PPO',  # percentage price oscillator
        'APO',  # absolute price oscillator (but percentage-like)
        'BOP',  # balance of power (-1 to 1)
        'TRIX',  # rate of change
    )

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
        normalize_by_close: bool = False,
        # Time feature extraction
        extract_time_features: bool = False,
        # DataFrame clustering
        enable_dataframe_clustering: bool = False,
        df_cluster_columns: str = 'indicators',
        cluster_k_range: Tuple[int, int] = (2, 10),
        cluster_k: Optional[int] = None,
        # Signal clustering
        enable_signal_clustering: bool = False,
        signal_cluster_k_range: Tuple[int, int] = (2, 8),
        signal_cluster_k: Optional[int] = None,
        keep_original_signals: bool = False,
        # Resampling
        enable_resampling: bool = False,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        resample_random_state: int = 42,
        # Derived feature engineering
        enable_derived_features: bool = False,
        enable_derived_clustering: bool = False,
        derived_cluster_k_range: Tuple[int, int] = (2, 10),
        derived_cluster_k: Optional[int] = None,
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
        normalize_by_close : bool, default=False
            If True, divide all raw technical indicator columns by the close price
            after cleaning data but before fit/transform. This normalizes indicators
            relative to price level, making them more comparable across different
            price ranges. Only affects non-binary columns (not _entry/_exit signals).
        extract_time_features : bool, default=False
            If True, extract cyclical time features (day_sin, day_cos, hour_sin, hour_cos)
            from the date column.
        enable_dataframe_clustering : bool, default=False
            If True, apply KMeans clustering to the entire DataFrame features and add
            one-hot encoded cluster columns.
        df_cluster_columns : str, default='indicators'
            Controls which columns are used for DataFrame clustering:
            - 'all': All DataFrame columns (except target and date)
            - 'indicators': Only raw technical indicators + OHLCV columns
            - 'signals': Entry/exit signal columns + OHLCV columns
            - 'indicators_signals': Raw indicators + entry/exit signals + OHLCV columns
        cluster_k_range : tuple, default=(2, 10)
            Range of k values to search for optimal number of clusters using elbow method.
        cluster_k : int, optional
            If provided, override elbow detection and use this k for DataFrame clustering.
        enable_signal_clustering : bool, default=False
            If True, cluster entry/exit signal columns and create aggregated cluster columns.
        signal_cluster_k_range : tuple, default=(2, 8)
            Range of k values to search for optimal number of signal clusters.
        signal_cluster_k : int, optional
            If provided, override elbow detection and use this k for signal clustering.
        keep_original_signals : bool, default=False
            If True, keep original entry/exit columns alongside cluster aggregates.
            If False, replace original signals with cluster aggregates.
        enable_resampling : bool, default=False
            If True, enable train/val/test splitting and resampling functionality.
        train_ratio : float, default=0.6
            Ratio of data to use for training (chronological split).
        val_ratio : float, default=0.2
            Ratio of data to use for validation. Test ratio = 1 - train_ratio - val_ratio.
        resample_random_state : int, default=42
            Random state for reproducible resampling.
        enable_derived_features : bool, default=False
            If True, create derived features from raw technical indicators.
            Features include momentum, volatility, trend, and composite indicators.
            All derived features use 'derived_' prefix.
        enable_derived_clustering : bool, default=False
            If True, cluster derived features using KMeans and add one-hot encoded
            cluster columns (derived_cluster_0, derived_cluster_1, etc.).
        derived_cluster_k_range : tuple, default=(2, 10)
            Range of k values to search for optimal number of derived clusters.
        derived_cluster_k : int, optional
            If provided, override elbow detection and use this k for derived clustering.
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
        self.normalize_by_close = normalize_by_close

        # Feature engineering parameters
        self.extract_time_features = extract_time_features
        self.enable_dataframe_clustering = enable_dataframe_clustering
        self.df_cluster_columns = df_cluster_columns
        self.cluster_k_range = cluster_k_range
        self.cluster_k = cluster_k

        # Validate df_cluster_columns
        valid_df_cluster_options = {'all', 'indicators', 'signals', 'indicators_signals'}
        if df_cluster_columns not in valid_df_cluster_options:
            raise ValueError(
                f"df_cluster_columns must be one of {valid_df_cluster_options}, "
                f"got '{df_cluster_columns}'"
            )
        self.enable_signal_clustering = enable_signal_clustering
        self.signal_cluster_k_range = signal_cluster_k_range
        self.signal_cluster_k = signal_cluster_k
        self.keep_original_signals = keep_original_signals

        # Resampling parameters
        self.enable_resampling = enable_resampling
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.resample_random_state = resample_random_state

        # Derived feature engineering parameters
        self.enable_derived_features = enable_derived_features
        self.enable_derived_clustering = enable_derived_clustering
        self.derived_cluster_k_range = derived_cluster_k_range
        self.derived_cluster_k = derived_cluster_k

        # Validate ratios
        if train_ratio + val_ratio > 1.0:
            raise ValueError(f"train_ratio + val_ratio must be <= 1.0, got {train_ratio + val_ratio}")

        # Fitted state
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.feature_columns: List[str] = []
        self._ohlcv_indices: List[int] = []
        self._signal_indices: List[int] = []
        self._is_fitted = False

        # DataFrame clustering state
        self._dataframe_cluster_model: Optional[KMeans] = None
        self._dataframe_cluster_k: int = 0
        self._dataframe_cluster_columns: List[str] = []
        self._df_cluster_fit_columns: List[str] = []  # Columns used during fit

        # Signal clustering state
        self._entry_cluster_model: Optional[KMeans] = None
        self._exit_cluster_model: Optional[KMeans] = None
        self._entry_column_clusters: Dict[str, int] = {}
        self._exit_column_clusters: Dict[str, int] = {}
        self._entry_cluster_k: int = 0
        self._exit_cluster_k: int = 0

        # Derived clustering state
        self._derived_cluster_model: Optional[KMeans] = None
        self._derived_cluster_scaler: Optional[StandardScaler] = None
        self._derived_cluster_k: int = 0
        self._derived_cluster_columns: List[str] = []
        self._derived_cluster_fit_columns: List[str] = []

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

            # Always keep time feature columns if extract_time_features is enabled
            if self.extract_time_features and col in ('day_sin', 'day_cos', 'hour_sin', 'hour_cos'):
                keep_columns.append(col)
                continue

            # Always keep signal cluster aggregate columns
            if col.startswith('entry_cluster_') or col.startswith('exit_cluster_'):
                keep_columns.append(col)
                continue

            # Always keep DataFrame cluster one-hot columns
            if col.startswith('df_cluster_'):
                keep_columns.append(col)
                continue

            # Always keep derived feature columns
            if col.startswith('derived_'):
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
        # During transform, preserve columns that are part of fitted feature_columns
        cols_to_preserve = set(self.feature_columns) if self._is_fitted else set()
        for col in feature_cols:
            if col not in cols_to_preserve and (df[col] == 0).all():
                df = df.drop(columns=[col])

        return df

    def _apply_normalize_by_close(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Divide raw technical indicator columns by close price.

        This normalizes indicators relative to price level, making them more
        comparable across different price ranges. Binary columns (_entry/_exit)
        and volume are not normalized.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with 'close' column

        Returns
        -------
        pd.DataFrame
            DataFrame with normalized indicator columns
        """
        if 'close' not in df.columns:
            return df

        close = np.asarray(df['close'].values, dtype=np.float64)

        # Avoid division by zero
        close = np.where(close == 0, self.EPSILON, close)

        # Columns to skip normalization
        skip_cols = set(self.columns_to_drop) | {self.target_column, 'volume'}

        for col in df.columns:
            # Skip metadata and target columns
            if col in skip_cols:
                continue

            # Skip binary signal columns (_entry/_exit)
            if col.endswith('_entry') or col.endswith('_exit'):
                continue

            # Skip OHLCV columns (they will be scaled separately)
            if col in self.ohlcv_columns:
                continue

            # Skip cluster columns (they are counts or binary, not price-related)
            if col.startswith('entry_cluster_') or col.startswith('exit_cluster_') or col.startswith('df_cluster_'):
                continue

            # Skip time feature columns (they are cyclical features, not price-related)
            if col in ('day_sin', 'day_cos', 'hour_sin', 'hour_cos'):
                continue

            # Skip bounded oscillators (already percentage-based, not price-related)
            col_upper = col.upper()
            if any(osc in col_upper for osc in self.BOUNDED_OSCILLATORS):
                continue

            # Normalize this column by dividing by close price
            df[col] = df[col].values / close

        return df

    def _has_columns(self, df: pd.DataFrame, *cols) -> bool:
        """
        Check if DataFrame has all specified columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to check
        *cols : str
            Column names to check for

        Returns
        -------
        bool
            True if all columns exist, False otherwise
        """
        return all(col in df.columns for col in cols)

    def _apply_scaling_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted scaler to continuous columns in a DataFrame.

        Uses the fitted scaler (MinMax or Standard) to scale continuous columns
        while preserving binary/categorical columns unchanged.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with columns to scale

        Returns
        -------
        pd.DataFrame
            DataFrame with scaled continuous columns
        """
        if self.scaler is None:
            return df

        df = df.copy()

        # Get columns to scale (continuous columns from feature_columns)
        cols_to_scale = [self.feature_columns[i] for i in self._ohlcv_indices
                         if self.feature_columns[i] in df.columns]

        if not cols_to_scale:
            return df

        # Extract values for columns to scale
        values_to_scale = df[cols_to_scale].values.astype(np.float64)

        # Apply scaler transform
        scaled_values = self.scaler.transform(values_to_scale)

        # Put scaled values back into DataFrame
        df[cols_to_scale] = scaled_values

        return df

    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract cyclical time features from date column.

        Parses the date column and creates sin/cos encoded features for
        day of week and hour of day, preserving cyclical nature.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with 'date' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added time feature columns:
            - day_sin, day_cos: cyclical encoding of day of week (0-6)
            - hour_sin, hour_cos: cyclical encoding of hour (0-23)
        """
        if 'date' not in df.columns:
            return df

        df = df.copy()

        # Parse date column
        dt = pd.to_datetime(df['date'])

        # Extract day of week (0=Monday, 6=Sunday) and hour
        day_of_week = dt.dt.dayofweek.values
        hour = dt.dt.hour.values

        # Apply cyclical sin/cos encoding
        df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        return df

    def _find_optimal_k(
        self,
        data: np.ndarray,
        k_range: Tuple[int, int]
    ) -> int:
        """
        Find optimal number of clusters using elbow method.

        Uses the rate of change of inertia to detect the elbow point.

        Parameters
        ----------
        data : np.ndarray
            Data to cluster, shape (n_samples, n_features)
        k_range : tuple
            (min_k, max_k) range of k values to test

        Returns
        -------
        int
            Optimal number of clusters
        """
        min_k, max_k = k_range

        # Ensure we have enough samples
        n_samples = data.shape[0]
        max_k = min(max_k, n_samples - 1)

        if max_k < min_k:
            return min_k

        k_values = list(range(min_k, max_k + 1))
        inertias = []

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)

        # Find elbow point using rate of change
        if len(inertias) < 3:
            return k_values[0]

        # Calculate second derivative (rate of change of slope)
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)

        # Elbow is where second derivative is maximum (least negative)
        elbow_idx = np.argmax(diffs2) + 1  # +1 to account for diff offset
        optimal_k = k_values[elbow_idx]

        return optimal_k

    def _get_df_cluster_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get columns to use for DataFrame clustering based on df_cluster_columns setting.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame

        Returns
        -------
        List[str]
            List of column names to use for clustering
        """
        # Always exclude target and date
        cols_to_exclude = {self.target_column, 'date', 'split'}

        # Also exclude cluster columns that might already exist
        cols_to_exclude.update(c for c in df.columns if c.startswith('df_cluster_'))
        cols_to_exclude.update(c for c in df.columns if c.startswith('entry_cluster_'))
        cols_to_exclude.update(c for c in df.columns if c.startswith('exit_cluster_'))

        if self.df_cluster_columns == 'all':
            # All columns except excluded
            return [c for c in df.columns if c not in cols_to_exclude]

        elif self.df_cluster_columns == 'indicators':
            # Only raw technical indicators + OHLCV
            # Raw indicators are columns that don't end with _entry or _exit
            cluster_cols = []
            for c in df.columns:
                if c in cols_to_exclude:
                    continue
                # Include OHLCV columns
                if c in self.ohlcv_columns:
                    cluster_cols.append(c)
                # Include raw indicators (not entry/exit signals)
                elif not c.endswith('_entry') and not c.endswith('_exit'):
                    cluster_cols.append(c)
            return cluster_cols

        elif self.df_cluster_columns == 'signals':
            # Entry/exit signal columns + OHLCV
            cluster_cols = []
            for c in df.columns:
                if c in cols_to_exclude:
                    continue
                # Include OHLCV columns
                if c in self.ohlcv_columns:
                    cluster_cols.append(c)
                # Include entry/exit signals
                elif c.endswith('_entry') or c.endswith('_exit'):
                    cluster_cols.append(c)
            return cluster_cols

        elif self.df_cluster_columns == 'indicators_signals':
            # Raw indicators + entry/exit signals + OHLCV (everything except excluded)
            return [c for c in df.columns if c not in cols_to_exclude]

        else:
            # Fallback to indicators (shouldn't reach here due to validation)
            return [c for c in df.columns
                    if c not in cols_to_exclude
                    and (c in self.ohlcv_columns or
                         (not c.endswith('_entry') and not c.endswith('_exit')))]

    def _fit_dataframe_cluster(
        self,
        dfs: List[pd.DataFrame]
    ) -> None:
        """
        Fit KMeans clustering on concatenated DataFrame features.

        Parameters
        ----------
        dfs : List[pd.DataFrame]
            List of cleaned DataFrames (already processed with _clean_data)
        """
        # Concatenate all DataFrames
        combined = pd.concat(dfs, ignore_index=True)

        # Get columns to use based on df_cluster_columns setting
        cluster_cols = self._get_df_cluster_columns(combined)

        if not cluster_cols:
            return

        # Store the columns used for clustering (needed for transform)
        self._df_cluster_fit_columns = cluster_cols

        cluster_data = combined[cluster_cols].values

        # Handle any remaining NaN
        cluster_data = np.nan_to_num(cluster_data, nan=0.0)

        # Determine optimal k
        if self.cluster_k is not None:
            optimal_k = self.cluster_k
        else:
            optimal_k = self._find_optimal_k(cluster_data, self.cluster_k_range)

        self._dataframe_cluster_k = optimal_k

        # Fit KMeans
        self._dataframe_cluster_model = KMeans(
            n_clusters=optimal_k,
            random_state=42,
            n_init=10
        )
        self._dataframe_cluster_model.fit(cluster_data)

        # Store cluster column names for one-hot encoding
        self._dataframe_cluster_columns = [
            f'df_cluster_{i}' for i in range(optimal_k)
        ]

    def _transform_dataframe_cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame by adding one-hot encoded cluster columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with added cluster columns (df_cluster_0, df_cluster_1, ...)
        """
        if self._dataframe_cluster_model is None:
            return df

        df = df.copy()

        # Use the same columns that were used during fit
        cluster_cols = [c for c in self._df_cluster_fit_columns if c in df.columns]

        if not cluster_cols:
            return df

        cluster_data = df[cluster_cols].values
        cluster_data = np.nan_to_num(cluster_data, nan=0.0)

        # Predict clusters
        cluster_labels = self._dataframe_cluster_model.predict(cluster_data)

        # One-hot encode clusters as binary columns
        for i in range(self._dataframe_cluster_k):
            col_name = f'df_cluster_{i}'
            df[col_name] = (cluster_labels == i).astype(np.float32)

        return df

    def _fit_signal_clusters(
        self,
        dfs: List[pd.DataFrame]
    ) -> None:
        """
        Fit KMeans clustering on entry/exit signal columns.

        Clusters the signal columns (not rows) based on their activation patterns
        across all samples. Each column becomes a data point where features are
        its values across all samples.

        Parameters
        ----------
        dfs : List[pd.DataFrame]
            List of cleaned DataFrames
        """
        # Concatenate all DataFrames
        combined = pd.concat(dfs, ignore_index=True)

        # Identify entry and exit columns
        entry_cols = [c for c in combined.columns if c.endswith('_entry')]
        exit_cols = [c for c in combined.columns if c.endswith('_exit')]

        if not entry_cols or not exit_cols:
            return

        # Transpose: columns become rows (each column is a sample)
        # Shape: (n_columns, n_samples)
        entry_data = combined[entry_cols].values.T
        exit_data = combined[exit_cols].values.T

        # Determine optimal k for entry signals
        if self.signal_cluster_k is not None:
            entry_k = self.signal_cluster_k
        else:
            entry_k = self._find_optimal_k(entry_data, self.signal_cluster_k_range)

        # Determine optimal k for exit signals
        if self.signal_cluster_k is not None:
            exit_k = self.signal_cluster_k
        else:
            exit_k = self._find_optimal_k(exit_data, self.signal_cluster_k_range)

        self._entry_cluster_k = entry_k
        self._exit_cluster_k = exit_k

        # Fit entry cluster model
        self._entry_cluster_model = KMeans(
            n_clusters=entry_k,
            random_state=42,
            n_init=10
        )
        entry_labels = self._entry_cluster_model.fit_predict(entry_data)
        self._entry_column_clusters = {
            col: int(label) for col, label in zip(entry_cols, entry_labels)
        }

        # Fit exit cluster model
        self._exit_cluster_model = KMeans(
            n_clusters=exit_k,
            random_state=42,
            n_init=10
        )
        exit_labels = self._exit_cluster_model.fit_predict(exit_data)
        self._exit_column_clusters = {
            col: int(label) for col, label in zip(exit_cols, exit_labels)
        }

    def _transform_signal_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame by aggregating entry/exit signals into cluster columns.

        For each cluster, sums all signal columns belonging to that cluster.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with cluster aggregate columns (entry_cluster_0, exit_cluster_0, ...)
        """
        if not self._entry_column_clusters or not self._exit_column_clusters:
            return df

        df = df.copy()

        # Create entry cluster aggregates
        for k in range(self._entry_cluster_k):
            cols_in_cluster = [
                col for col, cluster_id in self._entry_column_clusters.items()
                if cluster_id == k and col in df.columns
            ]
            if cols_in_cluster:
                df[f'entry_cluster_{k}'] = df[cols_in_cluster].sum(axis=1)
            else:
                df[f'entry_cluster_{k}'] = 0

        # Create exit cluster aggregates
        for k in range(self._exit_cluster_k):
            cols_in_cluster = [
                col for col, cluster_id in self._exit_column_clusters.items()
                if cluster_id == k and col in df.columns
            ]
            if cols_in_cluster:
                df[f'exit_cluster_{k}'] = df[cols_in_cluster].sum(axis=1)
            else:
                df[f'exit_cluster_{k}'] = 0

        # Optionally remove original signal columns
        if not self.keep_original_signals:
            entry_cols = [c for c in df.columns if c.endswith('_entry')]
            exit_cols = [c for c in df.columns if c.endswith('_exit')]
            df = df.drop(columns=entry_cols + exit_cols)

        return df

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from existing indicator columns.

        All derived features are prefixed with 'derived_' for easy identification.
        Only creates features for indicators that exist in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with indicator columns

        Returns
        -------
        pd.DataFrame
            DataFrame with derived feature columns added
        """
        if not self.enable_derived_features:
            return df

        df = df.copy()

        # Require close price for many calculations
        if 'close' not in df.columns:
            return df

        close = df['close'].values.astype(np.float64)
        # Avoid division by zero
        close_safe = np.where(close == 0, self.EPSILON, close)

        # Auto-detect and create features for each indicator category
        df = self._create_bb_derived(df, close_safe)
        df = self._create_macd_derived(df, close_safe)
        df = self._create_ma_derived(df, close_safe)
        df = self._create_atr_derived(df, close_safe)
        df = self._create_rsi_derived(df, close_safe)
        df = self._create_stoch_derived(df)
        df = self._create_adx_derived(df)
        df = self._create_volume_derived(df, close_safe)
        df = self._create_aroon_derived(df)
        df = self._create_cycle_derived(df)
        df = self._create_composite_derived(df, close_safe)

        return df

    def _create_bb_derived(self, df: pd.DataFrame, close: np.ndarray) -> pd.DataFrame:
        """Create Bollinger Bands derived features."""
        # Look for BBANDS columns (case-insensitive prefix matching)
        upper_col = None
        middle_col = None
        lower_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'upperband' in col_lower:
                upper_col = col
            elif 'middleband' in col_lower:
                middle_col = col
            elif 'lowerband' in col_lower:
                lower_col = col

        if not all([upper_col, middle_col, lower_col]):
            return df

        upper = df[upper_col].values.astype(np.float64)
        middle = df[middle_col].values.astype(np.float64)
        lower = df[lower_col].values.astype(np.float64)

        # Band width normalized by close (volatility measure)
        bb_width = upper - lower
        df['derived_bb_width_norm'] = bb_width / close

        # Position within bands (-1 to +1, where 0 is middle)
        band_half_width = np.where(bb_width == 0, self.EPSILON, bb_width / 2)
        df['derived_bb_position'] = (close - middle) / band_half_width

        # Distance from bands as percentage
        df['derived_bb_upper_dist'] = (upper - close) / close
        df['derived_bb_lower_dist'] = (close - lower) / close

        # Squeeze indicator (low volatility)
        middle_safe = np.where(middle == 0, self.EPSILON, middle)
        df['derived_bb_squeeze'] = bb_width / middle_safe

        # Band expansion rate (volatility change)
        bb_width_norm = bb_width / close
        df['derived_bb_expansion'] = pd.Series(bb_width_norm).diff().values

        return df

    def _create_macd_derived(self, df: pd.DataFrame, close: np.ndarray) -> pd.DataFrame:
        """Create MACD derived features."""
        # Look for MACD columns
        macd_col = None
        signal_col = None
        hist_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'macdhist' in col_lower:
                hist_col = col
            elif 'macdsignal' in col_lower:
                signal_col = col
            elif 'macd' in col_lower and 'hist' not in col_lower and 'signal' not in col_lower:
                macd_col = col

        if not all([macd_col, signal_col, hist_col]):
            return df

        macd = df[macd_col].values.astype(np.float64)
        signal = df[signal_col].values.astype(np.float64)
        hist = df[hist_col].values.astype(np.float64)

        # Histogram momentum (change in histogram)
        df['derived_macd_hist_momentum'] = pd.Series(hist).diff().values

        # Histogram acceleration
        hist_momentum = pd.Series(hist).diff()
        df['derived_macd_hist_accel'] = hist_momentum.diff().values

        # Signal line distance normalized
        df['derived_macd_signal_dist'] = (macd - signal) / close

        # Zero line distance normalized
        df['derived_macd_zero_dist'] = macd / close

        # Convergence/divergence strength
        df['derived_macd_strength'] = np.abs(hist) / close

        return df

    def _create_ma_derived(self, df: pd.DataFrame, close: np.ndarray) -> pd.DataFrame:
        """Create moving average derived features."""
        # Find all MA-related columns
        ma_cols = {}
        for col in df.columns:
            col_lower = col.lower()
            # Match SMA, EMA, DEMA, TEMA, KAMA, WMA, TRIMA, T3, MA patterns
            for ma_type in ['sma', 'ema', 'dema', 'tema', 'kama', 'wma', 'trima', 't3']:
                if ma_type in col_lower:
                    ma_cols[col] = ma_type
                    break

        # Create features for each MA column found
        for col, ma_type in ma_cols.items():
            ma_values = df[col].values.astype(np.float64)
            prefix = f'derived_{col.lower()}'

            # Price distance from MA (normalized)
            df[f'{prefix}_dist_norm'] = (close - ma_values) / close

            # MA slope (trend direction and strength)
            ma_series = pd.Series(ma_values)
            ma_shifted = ma_series.shift(1)
            ma_shifted_safe = np.where(ma_shifted == 0, self.EPSILON, ma_shifted)
            df[f'{prefix}_slope'] = (ma_values - ma_shifted.values) / ma_shifted_safe

            # MA curvature (trend acceleration)
            slope = pd.Series(df[f'{prefix}_slope'].values)
            df[f'{prefix}_curvature'] = slope.diff().values

        # Check for fast/slow MA pairs (e.g., sma_fast, sma_slow or ema_fast, ema_slow)
        for ma_type in ['sma', 'ema']:
            fast_col = None
            slow_col = None
            for col in df.columns:
                col_lower = col.lower()
                if f'{ma_type}_fast' in col_lower:
                    fast_col = col
                elif f'{ma_type}_slow' in col_lower:
                    slow_col = col

            if fast_col and slow_col:
                fast = df[fast_col].values.astype(np.float64)
                slow = df[slow_col].values.astype(np.float64)

                # MA spread (fast - slow) normalized
                df[f'derived_{ma_type}_spread'] = (fast - slow) / close

                # MA spread momentum
                spread = pd.Series((fast - slow) / close)
                df[f'derived_{ma_type}_spread_momentum'] = spread.diff().values

        return df

    def _create_atr_derived(self, df: pd.DataFrame, close: np.ndarray) -> pd.DataFrame:
        """Create ATR/volatility derived features."""
        # Look for ATR column
        atr_col = None
        natr_col = None
        trange_col = None

        for col in df.columns:
            col_lower = col.lower()
            if col_lower.endswith('_natr') or col_lower == 'natr':
                natr_col = col
            elif col_lower.endswith('_atr') or col_lower == 'atr':
                atr_col = col
            elif 'trange' in col_lower and 'avg' not in col_lower:
                trange_col = col

        # ATR-based features
        if atr_col:
            atr = df[atr_col].values.astype(np.float64)
            atr_safe = np.where(atr == 0, self.EPSILON, atr)

            # Volatility ratio (current vs average)
            atr_series = pd.Series(atr)
            atr_mean = atr_series.rolling(20, min_periods=1).mean().values
            atr_mean_safe = np.where(atr_mean == 0, self.EPSILON, atr_mean)
            df['derived_atr_ratio'] = atr / atr_mean_safe

            # Volatility expansion/contraction
            atr_shifted = atr_series.shift(1).values
            atr_shifted_safe = np.where(atr_shifted == 0, self.EPSILON, atr_shifted)
            df['derived_atr_change'] = (atr - atr_shifted) / atr_shifted_safe

            # Price range as multiple of ATR (if high/low available)
            if self._has_columns(df, 'high', 'low'):
                high = df['high'].values.astype(np.float64)
                low = df['low'].values.astype(np.float64)
                df['derived_price_range_atr'] = (high - low) / atr_safe

            # Normalized volatility (ATR / close)
            df['derived_volatility_norm'] = atr / close

            # Volatility percentile (rolling rank)
            atr_percentile = atr_series.rolling(50, min_periods=1).rank(pct=True).values
            df['derived_atr_percentile'] = atr_percentile

        # NATR-based features (already normalized)
        if natr_col:
            natr = df[natr_col].values.astype(np.float64)
            natr_series = pd.Series(natr)

            # NATR momentum
            df['derived_natr_momentum'] = natr_series.diff().values

            # NATR percentile
            df['derived_natr_percentile'] = natr_series.rolling(50, min_periods=1).rank(pct=True).values

        return df

    def _create_rsi_derived(self, df: pd.DataFrame, close: np.ndarray) -> pd.DataFrame:
        """Create RSI derived features."""
        # Look for RSI column
        rsi_col = None
        for col in df.columns:
            col_lower = col.lower()
            if col_lower.endswith('_rsi') or col_lower == 'rsi':
                rsi_col = col
                break

        if not rsi_col:
            return df

        rsi = df[rsi_col].values.astype(np.float64)
        rsi_series = pd.Series(rsi)

        # RSI momentum (rate of change)
        df['derived_rsi_momentum'] = rsi_series.diff().values

        # RSI acceleration
        rsi_momentum = rsi_series.diff()
        df['derived_rsi_accel'] = rsi_momentum.diff().values

        # Distance from oversold (30) / overbought (70)
        df['derived_rsi_oversold_dist'] = rsi - 30  # negative when oversold
        df['derived_rsi_overbought_dist'] = 70 - rsi  # negative when overbought

        # RSI divergence from price
        rsi_pct_change = rsi_series.pct_change().values
        close_pct_change = pd.Series(close).pct_change().values
        df['derived_rsi_price_divergence'] = rsi_pct_change - close_pct_change

        return df

    def _create_stoch_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Stochastic derived features."""
        # Look for slow stochastic columns first, then fast
        slowk_col = None
        slowd_col = None
        fastk_col = None
        fastd_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'slowk' in col_lower:
                slowk_col = col
            elif 'slowd' in col_lower:
                slowd_col = col
            elif 'fastk' in col_lower:
                fastk_col = col
            elif 'fastd' in col_lower:
                fastd_col = col

        # Prefer slow stochastic, fall back to fast
        k_col = slowk_col or fastk_col
        d_col = slowd_col or fastd_col

        if not k_col or not d_col:
            return df

        k_values = df[k_col].values.astype(np.float64)
        d_values = df[d_col].values.astype(np.float64)

        # K-D crossover distance
        df['derived_stoch_kd_diff'] = k_values - d_values

        # K-D crossover momentum
        kd_diff = pd.Series(k_values - d_values)
        df['derived_stoch_kd_momentum'] = kd_diff.diff().values

        # Combined stochastic strength
        df['derived_stoch_strength'] = (k_values + d_values) / 2

        # Stochastic momentum
        k_series = pd.Series(k_values)
        df['derived_stoch_k_momentum'] = k_series.diff().values

        return df

    def _create_adx_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ADX/directional derived features."""
        # Look for ADX and DI columns
        adx_col = None
        plus_di_col = None
        minus_di_col = None

        for col in df.columns:
            col_lower = col.lower()
            if col_lower.endswith('_adx') or col_lower == 'adx':
                adx_col = col
            elif 'plus_di' in col_lower:
                plus_di_col = col
            elif 'minus_di' in col_lower:
                minus_di_col = col

        if not adx_col:
            return df

        adx = df[adx_col].values.astype(np.float64)
        adx_series = pd.Series(adx)

        # Trend strength change
        df['derived_adx_momentum'] = adx_series.diff().values

        # If DI columns available, create directional features
        if plus_di_col and minus_di_col:
            plus_di = df[plus_di_col].values.astype(np.float64)
            minus_di = df[minus_di_col].values.astype(np.float64)

            # Trend strength with direction
            df['derived_adx_directional'] = adx * np.sign(plus_di - minus_di)

            # DI spread (trend direction clarity)
            di_spread = plus_di - minus_di
            df['derived_di_spread'] = di_spread

            # DI spread momentum
            df['derived_di_spread_momentum'] = pd.Series(di_spread).diff().values

            # Combined directional strength
            df['derived_di_strength'] = (plus_di + minus_di) / 2

        return df

    def _create_volume_derived(self, df: pd.DataFrame, close: np.ndarray) -> pd.DataFrame:
        """Create volume-based derived features."""
        # Look for OBV and AD columns
        obv_col = None
        obv_sma_col = None
        ad_col = None
        ad_sma_col = None

        for col in df.columns:
            col_lower = col.lower()
            if col_lower.endswith('_obv_sma') or 'obv_sma' in col_lower:
                obv_sma_col = col
            elif col_lower.endswith('_obv') or col_lower == 'obv':
                obv_col = col
            elif col_lower.endswith('_ad_sma') or 'ad_sma' in col_lower:
                ad_sma_col = col
            elif col_lower.endswith('_ad') or col_lower == 'ad':
                ad_col = col

        close_series = pd.Series(close)
        close_pct_change = close_series.pct_change().values

        # OBV-based features
        if obv_col:
            obv = df[obv_col].values.astype(np.float64)
            obv_series = pd.Series(obv)

            # OBV momentum
            df['derived_obv_momentum'] = obv_series.diff().values

            # OBV divergence from price
            obv_pct_change = obv_series.pct_change().values
            df['derived_obv_divergence'] = obv_pct_change - close_pct_change

            # Volume-price trend alignment
            obv_momentum = obv_series.diff().values
            df['derived_volume_price_align'] = np.sign(obv_momentum) * np.sign(close_pct_change)

            # OBV trend deviation (if obv_sma available)
            if obv_sma_col:
                obv_sma = df[obv_sma_col].values.astype(np.float64)
                df['derived_obv_trend_deviation'] = obv - obv_sma

        # A/D line features
        if ad_col:
            ad = df[ad_col].values.astype(np.float64)
            ad_series = pd.Series(ad)

            # A/D line divergence
            ad_pct_change = ad_series.pct_change().values
            df['derived_ad_divergence'] = ad_pct_change - close_pct_change

        return df

    def _create_aroon_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Aroon derived features."""
        # Look for Aroon columns
        aroonup_col = None
        aroondown_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'aroonup' in col_lower:
                aroonup_col = col
            elif 'aroondown' in col_lower:
                aroondown_col = col

        if not aroonup_col or not aroondown_col:
            return df

        aroonup = df[aroonup_col].values.astype(np.float64)
        aroondown = df[aroondown_col].values.astype(np.float64)

        # Aroon spread
        aroon_spread = aroonup - aroondown
        df['derived_aroon_spread'] = aroon_spread

        # Aroon trend strength
        df['derived_aroon_strength'] = (aroonup + aroondown) / 2

        # Aroon momentum
        df['derived_aroon_momentum'] = pd.Series(aroon_spread).diff().values

        return df

    def _create_cycle_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Hilbert Transform cycle derived features."""
        # Look for HT columns
        ht_dcphase_col = None
        inphase_col = None
        quadrature_col = None
        sine_col = None
        leadsine_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'ht_dcphase' in col_lower:
                ht_dcphase_col = col
            elif 'inphase' in col_lower:
                inphase_col = col
            elif 'quadrature' in col_lower:
                quadrature_col = col
            elif 'leadsine' in col_lower:
                leadsine_col = col
            elif 'sine' in col_lower and 'lead' not in col_lower:
                sine_col = col

        # HT phase momentum
        if ht_dcphase_col:
            ht_dcphase = df[ht_dcphase_col].values.astype(np.float64)
            df['derived_ht_phase_momentum'] = pd.Series(ht_dcphase).diff().values

        # Phasor features
        if inphase_col and quadrature_col:
            inphase = df[inphase_col].values.astype(np.float64)
            quadrature = df[quadrature_col].values.astype(np.float64)

            # Phasor magnitude
            df['derived_phasor_magnitude'] = np.sqrt(inphase**2 + quadrature**2)

            # Phasor angle
            df['derived_phasor_angle'] = np.arctan2(quadrature, inphase)

        # Sine-leadsine spread
        if sine_col and leadsine_col:
            sine = df[sine_col].values.astype(np.float64)
            leadsine = df[leadsine_col].values.astype(np.float64)
            df['derived_sine_spread'] = sine - leadsine

        return df

    def _create_composite_derived(self, df: pd.DataFrame, close: np.ndarray) -> pd.DataFrame:
        """Create cross-indicator composite features."""
        close_series = pd.Series(close)

        # Momentum consensus score (multiple oscillators)
        # Check which oscillators are available
        momentum_signals = []

        # RSI > 50
        for col in df.columns:
            if col.lower().endswith('_rsi') or col.lower() == 'rsi':
                rsi = df[col].values
                momentum_signals.append((rsi > 50).astype(np.float64))
                break

        # MFI > 50
        for col in df.columns:
            if col.lower().endswith('_mfi') or col.lower() == 'mfi':
                mfi = df[col].values
                momentum_signals.append((mfi > 50).astype(np.float64))
                break

        # Stoch K > 50
        for col in df.columns:
            col_lower = col.lower()
            if 'slowk' in col_lower or 'fastk' in col_lower:
                stoch_k = df[col].values
                momentum_signals.append((stoch_k > 50).astype(np.float64))
                break

        # Williams %R > -50
        for col in df.columns:
            if col.lower().endswith('_willr') or col.lower() == 'willr':
                willr = df[col].values
                momentum_signals.append((willr > -50).astype(np.float64))
                break

        if len(momentum_signals) >= 2:
            consensus = np.mean(momentum_signals, axis=0)
            df['derived_momentum_consensus'] = consensus

        # Trend-momentum alignment
        # Find MA distance and RSI
        ma_dist_col = None
        rsi_col = None

        for col in df.columns:
            if col.startswith('derived_') and '_dist_norm' in col:
                ma_dist_col = col
                break

        for col in df.columns:
            if col.lower().endswith('_rsi') or col.lower() == 'rsi':
                rsi_col = col
                break

        if ma_dist_col and rsi_col:
            ma_dist = df[ma_dist_col].values
            rsi = df[rsi_col].values
            df['derived_trend_momentum_align'] = np.sign(ma_dist) * np.sign(rsi - 50)

        # Volatility-adjusted momentum
        rsi_momentum_col = 'derived_rsi_momentum' if 'derived_rsi_momentum' in df.columns else None
        natr_col = None
        for col in df.columns:
            if col.lower().endswith('_natr') or col.lower() == 'natr':
                natr_col = col
                break

        if rsi_momentum_col and natr_col:
            rsi_momentum = df[rsi_momentum_col].values
            natr = df[natr_col].values
            natr_safe = np.where(natr == 0, self.EPSILON, natr)
            df['derived_vol_adj_momentum'] = rsi_momentum / natr_safe

        # Volume-confirmed trend
        ma_slope_col = None
        obv_momentum_col = 'derived_obv_momentum' if 'derived_obv_momentum' in df.columns else None

        for col in df.columns:
            if col.startswith('derived_') and '_slope' in col:
                ma_slope_col = col
                break

        if ma_slope_col and obv_momentum_col:
            ma_slope = df[ma_slope_col].values
            obv_momentum = df[obv_momentum_col].values
            df['derived_volume_trend_confirm'] = np.sign(ma_slope) * np.sign(obv_momentum)

        return df

    def _fit_derived_cluster(self, dfs: List[pd.DataFrame]) -> None:
        """
        Fit KMeans clustering on derived feature columns.

        Parameters
        ----------
        dfs : List[pd.DataFrame]
            List of DataFrames with derived_ columns
        """
        # Concatenate all DataFrames
        combined = pd.concat(dfs, ignore_index=True)

        # Get all derived feature columns (excluding cluster columns)
        derived_cols = [c for c in combined.columns
                        if c.startswith('derived_') and not c.startswith('derived_cluster_')]

        if not derived_cols:
            return

        # Store columns used for clustering
        self._derived_cluster_fit_columns = derived_cols

        # Extract data and handle NaN
        cluster_data = combined[derived_cols].values
        cluster_data = np.nan_to_num(cluster_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize for clustering
        self._derived_cluster_scaler = StandardScaler()
        cluster_data_scaled = self._derived_cluster_scaler.fit_transform(cluster_data)

        # Determine optimal k using elbow method
        if self.derived_cluster_k is not None:
            optimal_k = self.derived_cluster_k
        else:
            optimal_k = self._find_optimal_k(cluster_data_scaled, self.derived_cluster_k_range)

        self._derived_cluster_k = optimal_k

        # Fit KMeans
        self._derived_cluster_model = KMeans(
            n_clusters=optimal_k,
            random_state=42,
            n_init=10
        )
        self._derived_cluster_model.fit(cluster_data_scaled)

        # Store one-hot column names
        self._derived_cluster_columns = [
            f'derived_cluster_{i}' for i in range(optimal_k)
        ]

    def _transform_derived_cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame by adding one-hot encoded derived cluster columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with derived_ columns

        Returns
        -------
        pd.DataFrame
            DataFrame with derived_cluster_0, derived_cluster_1, ... columns
        """
        if self._derived_cluster_model is None:
            return df

        df = df.copy()

        # Get derived columns used during fit
        derived_cols = [c for c in self._derived_cluster_fit_columns if c in df.columns]

        if not derived_cols:
            return df

        # Extract and preprocess data
        cluster_data = df[derived_cols].values
        cluster_data = np.nan_to_num(cluster_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale using fitted scaler
        cluster_data_scaled = self._derived_cluster_scaler.transform(cluster_data)

        # Predict clusters
        cluster_labels = self._derived_cluster_model.predict(cluster_data_scaled)

        # One-hot encode clusters
        for i in range(self._derived_cluster_k):
            col_name = f'derived_cluster_{i}'
            df[col_name] = (cluster_labels == i).astype(np.float32)

        return df

    def _split_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame chronologically into train, validation, test sets.

        Time series data - no shuffling, preserve temporal order.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to split

        Returns
        -------
        tuple
            (train_df, val_df, test_df) with reset indices
        """
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train_df = df.iloc[:train_end].reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].reset_index(drop=True)
        test_df = df.iloc[val_end:].reset_index(drop=True)

        return train_df, val_df, test_df

    def _get_all_cluster_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get all cluster columns from DataFrame.

        Returns columns matching: entry_cluster_*, exit_cluster_*, df_cluster_*, derived_cluster_*

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame

        Returns
        -------
        List[str]
            Sorted list of cluster column names
        """
        cluster_cols = []
        for col in df.columns:
            if (col.startswith('entry_cluster_') or
                col.startswith('exit_cluster_') or
                col.startswith('derived_cluster_') or
                col.startswith('df_cluster_')):
                cluster_cols.append(col)
        return sorted(cluster_cols)

    def _calculate_cluster_distribution(
        self,
        df: pd.DataFrame,
        cluster_cols: Optional[List[str]] = None
    ) -> Optional[np.ndarray]:
        """
        Calculate probability distribution over all cluster columns.

        For each row, normalizes cluster values and averages across rows
        to get the overall distribution.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with cluster columns
        cluster_cols : List[str], optional
            Specific cluster columns to use. If None, auto-detects all cluster columns.

        Returns
        -------
        np.ndarray or None
            1D array with probability distribution over cluster columns,
            or None if no cluster columns found
        """
        if cluster_cols is None:
            cluster_cols = self._get_all_cluster_columns(df)

        if not cluster_cols:
            return None

        # Extract cluster values
        cluster_data = df[cluster_cols].values.astype(np.float64)

        # Normalize per row (avoid division by zero)
        row_sums = cluster_data.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        normalized = cluster_data / row_sums

        # Average across rows to get distribution
        distribution = normalized.mean(axis=0)

        # Normalize to sum to 1
        dist_sum = distribution.sum()
        if dist_sum > 0:
            distribution = distribution / dist_sum

        return distribution

    def _resample_by_distribution(
        self,
        df: pd.DataFrame,
        target_distribution: np.ndarray,
        cluster_cols: List[str],
        target_size: int,
        n_partitions: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Resample DataFrame to match target cluster distribution using partitioned sampling.

        Divides the DataFrame into n_partitions chronological partitions, samples from
        each partition proportionally to its size, then concatenates the results.
        This ensures temporal diversity in the resampled data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to resample
        target_distribution : np.ndarray
            Target probability distribution over cluster columns (1D array)
        cluster_cols : List[str]
            List of cluster column names
        target_size : int
            Total number of rows to sample across all partitions
        n_partitions : int, optional
            Number of partitions to divide the data into.
            Defaults to DEFAULT_RESAMPLE_PARTITIONS (168 = 24 hours * 7 days).

        Returns
        -------
        pd.DataFrame
            Resampled DataFrame with rows sampled from each partition
        """
        if len(df) == 0 or target_size == 0:
            return df.iloc[:0]  # Return empty DataFrame with same columns

        # Use default if not specified
        if n_partitions is None:
            n_partitions = self.DEFAULT_RESAMPLE_PARTITIONS

        # Adjust n_partitions if larger than DataFrame length
        n_partitions = min(n_partitions, len(df))

        # Calculate partition boundaries
        partition_indices = np.array_split(np.arange(len(df)), n_partitions)

        # Calculate samples per partition (proportional to partition size)
        partition_sizes = np.array([len(p) for p in partition_indices])
        samples_per_partition = np.round(
            partition_sizes / partition_sizes.sum() * target_size
        ).astype(int)

        # Adjust to ensure total equals target_size
        diff = target_size - samples_per_partition.sum()
        if diff != 0:
            # Add/remove from largest partitions
            sorted_idx = np.argsort(partition_sizes)[::-1]
            for i in range(abs(diff)):
                idx = sorted_idx[i % len(sorted_idx)]
                samples_per_partition[idx] += 1 if diff > 0 else -1

        rng = np.random.RandomState(self.resample_random_state)
        resampled_partitions = []

        for partition_idx, (indices, n_samples) in enumerate(zip(partition_indices, samples_per_partition)):
            if len(indices) == 0 or n_samples <= 0:
                continue

            partition_df = df.iloc[indices]

            # Extract and normalize cluster values for this partition
            cluster_data = partition_df[cluster_cols].values.astype(np.float64)
            row_sums = cluster_data.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            normalized = cluster_data / row_sums

            # Calculate current distribution in this partition
            current_distribution = normalized.mean(axis=0)
            current_sum = current_distribution.sum()
            if current_sum > 0:
                current_distribution = current_distribution / current_sum

            # Compute weight for each row in partition
            # Weight = sum over clusters of (row's contribution * target/current ratio)
            weights = np.zeros(len(partition_df))
            for row_idx in range(len(partition_df)):
                row_weight = 0.0
                for col_idx in range(len(cluster_cols)):
                    contrib = normalized[row_idx, col_idx]
                    if contrib > 0:
                        current_ratio = current_distribution[col_idx] if current_distribution[col_idx] > 0 else 1e-10
                        row_weight += contrib * (target_distribution[col_idx] / current_ratio)
                weights[row_idx] = row_weight

            # Normalize weights to probabilities
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights /= weight_sum
            else:
                # Fallback to uniform sampling
                weights = np.ones(len(partition_df)) / len(partition_df)

            # Sample from this partition with replacement based on weights
            # Use different seed per partition for variety while maintaining reproducibility
            partition_rng = np.random.RandomState(self.resample_random_state + partition_idx)
            local_indices = partition_rng.choice(len(partition_df), size=n_samples, replace=True, p=weights)

            # Get original DataFrame indices
            sampled_df = partition_df.iloc[local_indices]
            resampled_partitions.append(sampled_df)

        if not resampled_partitions:
            return df.iloc[:0]

        return pd.concat(resampled_partitions, ignore_index=True)

    def _resample_train_data(
        self,
        train_df: pd.DataFrame,
        val_test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Resample training data to match val+test class ratio AND cluster distribution per class.

        For each class (hold, trade):
        1. Calculate class ratio from val_test_df
        2. Determine target count for each class in train
        3. Calculate cluster distribution from val_test_df for this class
        4. Resample train rows of that class to match target count and cluster distribution

        Parameters
        ----------
        train_df : pd.DataFrame
            Training DataFrame to resample
        val_test_df : pd.DataFrame
            Combined validation + test DataFrame to calculate target distributions from

        Returns
        -------
        pd.DataFrame
            Resampled training DataFrame with class ratio matching val+test
        """
        cluster_cols = self._get_all_cluster_columns(train_df)
        if not cluster_cols:
            return train_df

        # Calculate class distribution from val+test
        val_test_class_counts = val_test_df[self.target_column].value_counts()
        val_test_total = len(val_test_df)

        if val_test_total == 0:
            return train_df

        # Determine target counts for train based on val+test class ratio
        train_total = len(train_df)
        target_counts = {}
        for class_value in ['hold', 'trade']:
            if class_value in val_test_class_counts.index:
                ratio = val_test_class_counts[class_value] / val_test_total
                target_counts[class_value] = int(train_total * ratio)
            else:
                target_counts[class_value] = 0

        resampled_dfs = []

        for class_value in ['hold', 'trade']:
            # Filter by class
            train_class = train_df[train_df[self.target_column] == class_value]
            val_test_class = val_test_df[val_test_df[self.target_column] == class_value]
            target_count = target_counts.get(class_value, 0)

            if len(train_class) == 0 or len(val_test_class) == 0 or target_count == 0:
                continue

            # Calculate target cluster distribution for this class
            target_dist = self._calculate_cluster_distribution(val_test_class, cluster_cols)

            if target_dist is None:
                # No cluster columns, just sample uniformly
                rng = np.random.RandomState(self.resample_random_state)
                indices = rng.choice(len(train_class), size=target_count, replace=True)
                resampled_dfs.append(train_class.iloc[indices].reset_index(drop=True))
            else:
                # Resample train data for this class to match cluster distribution
                resampled = self._resample_by_distribution(
                    train_class, target_dist, cluster_cols, target_count
                )
                resampled_dfs.append(resampled)

        if not resampled_dfs:
            return train_df

        return pd.concat(resampled_dfs, ignore_index=True)

    def resample(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split and resample data based on cluster distributions.

        Applies all transforms, splits data chronologically, calculates cluster
        distribution from validation+test data, then resamples training data
        to match that distribution.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (should be the full dataset)

        Returns
        -------
        tuple
            (resampled_train_df, val_df, test_df)

        Raises
        ------
        RuntimeError
            If preprocessor is not fitted or signal clustering is not enabled
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before resample. Call fit() first.")

        # Apply all transforms to get cluster columns
        transformed_df = self.transform_dataframe(df)

        # Check if any cluster columns exist
        cluster_cols = self._get_all_cluster_columns(transformed_df)
        if not cluster_cols:
            raise RuntimeError("Resampling requires cluster columns (enable_signal_clustering or enable_dataframe_clustering)")

        # Split chronologically
        train_df, val_df, test_df = self._split_data(transformed_df)

        # Combine val+test for distribution calculation
        val_test_combined = pd.concat([val_df, test_df], ignore_index=True)

        # Resample train data to match val+test class ratio and cluster distribution
        resampled_train = self._resample_train_data(train_df, val_test_combined)

        # Sort resampled train by date to restore chronological order
        if 'date' in resampled_train.columns:
            resampled_train = resampled_train.sort_values('date').reset_index(drop=True)

        return resampled_train, val_df, test_df

    def fit(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame], List[pd.DataFrame]]
    ) -> 'VectorBTDataPreprocessor':
        """
        Fit the preprocessor on data.

        Fits the scaler on OHLCV columns from all provided data.
        Also fits clustering models if enabled.

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

        # Step 1: Clean all DataFrames
        cleaned_dfs = [self._clean_data(df) for df in dfs]

        # Step 2: Extract time features if enabled
        if self.extract_time_features:
            cleaned_dfs = [self._extract_time_features(df) for df in cleaned_dfs]

        # Step 3: Fit signal clustering BEFORE any transforms (on original signal columns)
        if self.enable_signal_clustering:
            self._fit_signal_clusters(cleaned_dfs)

        # Step 4: Apply signal clustering transform to get the new column structure
        if self.enable_signal_clustering:
            cleaned_dfs = [self._transform_signal_clusters(df) for df in cleaned_dfs]

        # Step 5: Create derived features if enabled
        if self.enable_derived_features:
            cleaned_dfs = [self._create_derived_features(df) for df in cleaned_dfs]

        # Step 6: Fit derived clustering if enabled
        if self.enable_derived_clustering:
            self._fit_derived_cluster(cleaned_dfs)

        # Step 7: Apply derived clustering transform
        if self.enable_derived_clustering:
            cleaned_dfs = [self._transform_derived_cluster(df) for df in cleaned_dfs]

        # Step 8: Fit DataFrame clustering (on all features)
        if self.enable_dataframe_clustering:
            self._fit_dataframe_cluster(cleaned_dfs)

        # Step 9: Apply DataFrame clustering transform
        if self.enable_dataframe_clustering:
            cleaned_dfs = [self._transform_dataframe_cluster(df) for df in cleaned_dfs]

        # Step 10: Apply normalize by close if enabled (after clustering)
        if self.normalize_by_close:
            cleaned_dfs = [self._apply_normalize_by_close(df) for df in cleaned_dfs]

        # Determine feature columns from first processed DataFrame
        first_df = cleaned_dfs[0]
        self.feature_columns = self._filter_columns(first_df)

        # Add time feature columns if extracted
        if self.extract_time_features:
            time_cols = ['day_sin', 'day_cos', 'hour_sin', 'hour_cos']
            for col in time_cols:
                if col in first_df.columns and col not in self.feature_columns:
                    self.feature_columns.append(col)

        # Add clustering columns if enabled
        if self.enable_signal_clustering:
            for col in first_df.columns:
                if col.startswith('entry_cluster_') or col.startswith('exit_cluster_'):
                    if col not in self.feature_columns:
                        self.feature_columns.append(col)

        if self.enable_dataframe_clustering:
            for col in self._dataframe_cluster_columns:
                if col not in self.feature_columns:
                    self.feature_columns.append(col)

        # Add derived cluster columns if enabled
        if self.enable_derived_clustering:
            for col in self._derived_cluster_columns:
                if col not in self.feature_columns:
                    self.feature_columns.append(col)

        # Identify columns to scale vs binary signal columns
        # Scale all non-binary columns (OHLCV + raw indicator values + time features)
        # Binary signals and cluster columns are not scaled
        self._ohlcv_indices = []  # Indices of columns to scale
        self._signal_indices = []  # Binary signal columns (not scaled)
        for i, col in enumerate(self.feature_columns):
            # Binary signal columns (not scaled)
            if col.endswith('_entry') or col.endswith('_exit'):
                self._signal_indices.append(i)
            # DataFrame cluster one-hot columns (not scaled - they are binary)
            elif col.startswith('df_cluster_'):
                self._signal_indices.append(i)
            # Derived cluster one-hot columns (not scaled - they are binary)
            elif col.startswith('derived_cluster_'):
                self._signal_indices.append(i)
            else:
                # Scale continuous columns (OHLCV, raw indicators, time features, cluster aggregates, derived features)
                self._ohlcv_indices.append(i)

        # Initialize scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {self.scaler_type}")

        # Collect data from all scalable columns for fitting
        scale_cols = [self.feature_columns[i] for i in self._ohlcv_indices]
        if scale_cols:
            all_scale_data = []
            for df in cleaned_dfs:
                all_scale_data.append(df[scale_cols].values)

            combined_scale_data = np.vstack(all_scale_data)
            self.scaler.fit(combined_scale_data)

        self._is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform a DataFrame to features and targets.

        Applies column filtering, scaling (all non-binary columns), and target shifting.

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

        # Step 1: Clean data
        df = self._clean_data(df)

        # Step 2: Extract time features if enabled
        if self.extract_time_features:
            df = self._extract_time_features(df)

        # Step 3: Apply signal clustering transform
        if self.enable_signal_clustering:
            df = self._transform_signal_clusters(df)

        # Step 4: Create derived features if enabled
        if self.enable_derived_features:
            df = self._create_derived_features(df)

        # Step 5: Apply derived clustering transform
        if self.enable_derived_clustering:
            df = self._transform_derived_cluster(df)

        # Step 6: Apply DataFrame clustering transform
        if self.enable_dataframe_clustering:
            df = self._transform_dataframe_cluster(df)

        # Step 7: Apply normalize by close if enabled (after clustering)
        if self.normalize_by_close:
            df = self._apply_normalize_by_close(df)

        # Validate target column exists
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")

        # Validate feature columns exist
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        # Extract features
        features = df[self.feature_columns].values.astype(np.float32)

        # Scale all non-binary columns (OHLCV + raw indicator values + time features)
        if self._ohlcv_indices and self.scaler is not None:
            scale_cols = [self.feature_columns[i] for i in self._ohlcv_indices]
            scaled_data = self.scaler.transform(df[scale_cols].values)
            for new_idx, orig_idx in enumerate(self._ohlcv_indices):
                features[:, orig_idx] = scaled_data[:, new_idx]

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

    def fit_transform_dataframe(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame], List[pd.DataFrame]],
        apply_resampling: bool = False,
        apply_scaling: bool = False
    ) -> pd.DataFrame:
        """
        Fit and transform to DataFrame in one efficient step.

        Combines fit() and transform_dataframe() without redundant processing.
        All feature engineering (cleaning, time features, clustering, normalization)
        is done once during fitting, and the transformed data is returned directly.

        Parameters
        ----------
        data : DataFrame, Dict[str, DataFrame], or List[DataFrame]
            Input data. If multiple DataFrames, they are concatenated.
        apply_resampling : bool, default=False
            If True and enable_resampling is enabled, split data into train/val/test,
            resample train data to match cluster distribution in val+test, and return
            concatenated DataFrame with a 'split' column.
        apply_scaling : bool, default=False
            If True, apply the fitted scaler to continuous columns.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with feature engineering columns added:
            - Time features (day_sin, day_cos, hour_sin, hour_cos) if enabled
            - Signal cluster aggregates (entry_cluster_*, exit_cluster_*) if enabled
            - DataFrame cluster one-hot columns (df_cluster_*) if enabled
            If apply_resampling=True, includes a 'split' column ('train', 'val', 'test').
            If apply_scaling=True, continuous columns are scaled.

        Examples
        --------
        >>> prep = VectorBTDataPreprocessor(
        ...     enable_signal_clustering=True,
        ...     enable_resampling=True
        ... )
        >>> transformed = prep.fit_transform_dataframe(df, apply_resampling=True, apply_scaling=True)
        >>> (X_train, y_train), (X_val, y_val), (X_test, y_test) = prep.create_sequences_by_split(transformed)
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

        # Step 1: Clean all DataFrames
        cleaned_dfs = [self._clean_data(df) for df in dfs]

        # Step 2: Extract time features if enabled
        if self.extract_time_features:
            cleaned_dfs = [self._extract_time_features(df) for df in cleaned_dfs]

        # Step 3: Fit signal clustering BEFORE any transforms (on original signal columns)
        if self.enable_signal_clustering:
            self._fit_signal_clusters(cleaned_dfs)

        # Step 4: Apply signal clustering transform to get the new column structure
        if self.enable_signal_clustering:
            cleaned_dfs = [self._transform_signal_clusters(df) for df in cleaned_dfs]

        # Step 5: Create derived features if enabled
        if self.enable_derived_features:
            cleaned_dfs = [self._create_derived_features(df) for df in cleaned_dfs]

        # Step 6: Fit derived clustering if enabled
        if self.enable_derived_clustering:
            self._fit_derived_cluster(cleaned_dfs)

        # Step 7: Apply derived clustering transform
        if self.enable_derived_clustering:
            cleaned_dfs = [self._transform_derived_cluster(df) for df in cleaned_dfs]

        # Step 8: Fit DataFrame clustering (on all features)
        if self.enable_dataframe_clustering:
            self._fit_dataframe_cluster(cleaned_dfs)

        # Step 9: Apply DataFrame clustering transform
        if self.enable_dataframe_clustering:
            cleaned_dfs = [self._transform_dataframe_cluster(df) for df in cleaned_dfs]

        # Step 10: Apply normalize by close if enabled (after clustering)
        if self.normalize_by_close:
            cleaned_dfs = [self._apply_normalize_by_close(df) for df in cleaned_dfs]

        # Determine feature columns from first processed DataFrame
        first_df = cleaned_dfs[0]
        self.feature_columns = self._filter_columns(first_df)

        # Add time feature columns if extracted
        if self.extract_time_features:
            time_cols = ['day_sin', 'day_cos', 'hour_sin', 'hour_cos']
            for col in time_cols:
                if col in first_df.columns and col not in self.feature_columns:
                    self.feature_columns.append(col)

        # Add clustering columns if enabled
        if self.enable_signal_clustering:
            for col in first_df.columns:
                if col.startswith('entry_cluster_') or col.startswith('exit_cluster_'):
                    if col not in self.feature_columns:
                        self.feature_columns.append(col)

        if self.enable_dataframe_clustering:
            for col in self._dataframe_cluster_columns:
                if col not in self.feature_columns:
                    self.feature_columns.append(col)

        # Add derived cluster columns if enabled
        if self.enable_derived_clustering:
            for col in self._derived_cluster_columns:
                if col not in self.feature_columns:
                    self.feature_columns.append(col)

        # Identify columns to scale vs binary signal columns
        self._ohlcv_indices = []
        self._signal_indices = []
        for i, col in enumerate(self.feature_columns):
            if col.endswith('_entry') or col.endswith('_exit'):
                self._signal_indices.append(i)
            elif col.startswith('df_cluster_'):
                self._signal_indices.append(i)
            elif col.startswith('derived_cluster_'):
                self._signal_indices.append(i)
            else:
                self._ohlcv_indices.append(i)

        # Initialize scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {self.scaler_type}")

        # Collect data from all scalable columns for fitting
        scale_cols = [self.feature_columns[i] for i in self._ohlcv_indices]
        if scale_cols:
            all_scale_data = []
            for df in cleaned_dfs:
                all_scale_data.append(df[scale_cols].values)

            combined_scale_data = np.vstack(all_scale_data)
            self.scaler.fit(combined_scale_data)

        self._is_fitted = True

        # Concatenate all transformed DataFrames
        result_df = pd.concat(cleaned_dfs, ignore_index=True)

        # Apply resampling if enabled
        if apply_resampling and self.enable_resampling:
            cluster_cols = self._get_all_cluster_columns(result_df)
            if cluster_cols:
                train_df, val_df, test_df = self._split_data(result_df)

                val_test_combined = pd.concat([val_df, test_df], ignore_index=True)

                # Resample train to match val+test class ratio and cluster distribution
                resampled_train = self._resample_train_data(train_df, val_test_combined)

                if 'date' in resampled_train.columns:
                    resampled_train = resampled_train.sort_values('date').reset_index(drop=True)

                # Add split column
                resampled_train['split'] = 'train'
                val_df = val_df.copy()
                val_df['split'] = 'val'
                test_df = test_df.copy()
                test_df['split'] = 'test'

                result_df = pd.concat([resampled_train, val_df, test_df], ignore_index=True)

        # Apply scaling if enabled
        if apply_scaling and self.scaler is not None:
            result_df = self._apply_scaling_to_dataframe(result_df)

        return result_df

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

    @staticmethod
    def create_sequences_by_split(
        df: pd.DataFrame,
        sequence_length: int,
        target_column: str = 'tradeable',
        target_encoding: Optional[Dict[str, int]] = None,
        target_shift: int = 1,
        stride: int = 1,
        exclude_columns: Optional[List[str]] = None,
        device: Optional[str] = None
    ) -> Tuple[SignalDataset, SignalDataset, SignalDataset]:
        """
        Create LSTM sequences from a DataFrame with a 'split' column.

        Static method that splits the DataFrame by the 'split' column into
        train/val/test sets, then creates sequences for each set separately.
        This ensures sequences never span across different splits.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a 'split' column containing 'train', 'val', 'test' values.
            Should be the output of fit_transform_dataframe(apply_resampling=True, apply_scaling=True).
        sequence_length : int
            Length of each input sequence for LSTM.
        target_column : str, default='tradeable'
            Name of the target column.
        target_encoding : dict, optional
            Mapping from target string to integer. Default: {'hold': 0, 'trade': 1}
        target_shift : int, default=1
            Number of steps to shift target. Features at t predict target at t+shift.
        stride : int, default=1
            Step size between consecutive sequences.
        exclude_columns : list, optional
            Additional columns to exclude from features (besides 'split', 'date', target).
        device : str, optional
            Device for SignalDataset tensors ('cuda', 'cpu'). If None, auto-detects.

        Returns
        -------
        tuple
            (train_dataset, val_dataset, test_dataset)
            Each is a SignalDataset instance ready for PyTorch DataLoader.

        Raises
        ------
        ValueError
            If 'split' column is missing or no valid sequences could be created

        Examples
        --------
        >>> transformed = prep.fit_transform_dataframe(df, apply_resampling=True, apply_scaling=True)
        >>> train_ds, val_ds, test_ds = VectorBTDataPreprocessor.create_sequences_by_split(
        ...     transformed, sequence_length=24
        ... )
        >>> train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        """
        import torch

        if target_encoding is None:
            target_encoding = {'hold': 0, 'trade': 1}

        if 'split' not in df.columns:
            raise ValueError("DataFrame must have a 'split' column.")

        # Determine device
        if device is not None:
            torch_device = torch.device(device)
        else:
            torch_device = None  # Let SignalDataset auto-detect

        # Determine columns to exclude from features
        cols_to_exclude = {'split', target_column}
        if 'date' in df.columns:
            cols_to_exclude.add('date')
        if exclude_columns:
            cols_to_exclude.update(exclude_columns)

        # Determine feature columns (all columns except excluded ones)
        feature_columns = [c for c in df.columns if c not in cols_to_exclude]

        results = {}

        for split_name in ['train', 'val', 'test']:
            # Filter rows for this split
            split_df = df[df['split'] == split_name].copy()

            if len(split_df) == 0:
                # Empty split - create empty SignalDataset
                n_features = len(feature_columns)
                X = np.empty((0, sequence_length, n_features), dtype=np.float32)
                y = np.empty((0,), dtype=np.int64)
                results[split_name] = SignalDataset(X, y, device=torch_device)
                continue

            # Sort by date to ensure chronological order
            if 'date' in split_df.columns:
                split_df = split_df.sort_values('date').reset_index(drop=True)

            # Extract features (exclude non-feature columns)
            features = split_df[feature_columns].values.astype(np.float32)

            # Encode targets
            df_target = split_df[target_column].fillna('hold')
            mapped_targets = df_target.map(target_encoding)

            # Check for unmapped target values
            if mapped_targets.isna().any():
                unmapped = df_target[mapped_targets.isna()].unique()
                raise ValueError(f"Unknown target values in {split_name} split: {unmapped}")

            targets = np.asarray(mapped_targets, dtype=np.int64)

            # Shift targets: features at t predict target at t+shift
            if target_shift > 0:
                features = features[:-target_shift]
                targets = targets[target_shift:]

            # Check if enough data for sequences
            min_length = sequence_length + 1
            if len(features) < min_length:
                n_features = len(feature_columns)
                X = np.empty((0, sequence_length, n_features), dtype=np.float32)
                y = np.empty((0,), dtype=np.int64)
                results[split_name] = SignalDataset(X, y, device=torch_device)
                continue

            # Create sequences using the imported function
            X, y = create_sequences(
                features=features,
                targets=targets,
                input_seq_length=sequence_length,
                output_seq_length=1,
                stride=stride
            )
            results[split_name] = SignalDataset(X, y, device=torch_device)

        return results['train'], results['val'], results['test']

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

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        apply_resampling: bool = False,
        apply_scaling: bool = False
    ) -> pd.DataFrame:
        """
        Transform a DataFrame and return the transformed DataFrame.

        Applies all feature engineering (time features, clustering) and returns
        the transformed DataFrame with all new columns, without converting to
        numpy arrays or creating sequences.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        apply_resampling : bool, default=False
            If True and enable_resampling is enabled, split data into train/val/test,
            resample train data to match cluster distribution in val+test, and return
            concatenated DataFrame (resampled_train + val + test).
        apply_scaling : bool, default=False
            If True, apply the fitted scaler (MinMax or Standard) to continuous columns.
            This normalizes the data using the scaler fitted during fit().

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with feature engineering columns added:
            - Time features (day_sin, day_cos, hour_sin, hour_cos) if enabled
            - Signal cluster aggregates (entry_cluster_*, exit_cluster_*) if enabled
            - DataFrame cluster one-hot columns (df_cluster_*) if enabled
            If apply_resampling=True, returns concatenated resampled train + val + test
            with a 'split' column indicating 'train', 'val', or 'test' for each row.
            If apply_scaling=True, continuous columns are scaled.

        Raises
        ------
        RuntimeError
            If preprocessor is not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call fit() first.")

        # Step 1: Clean data
        df = self._clean_data(df)

        # Step 2: Extract time features if enabled
        if self.extract_time_features:
            df = self._extract_time_features(df)

        # Step 3: Apply signal clustering transform
        if self.enable_signal_clustering:
            df = self._transform_signal_clusters(df)

        # Step 4: Create derived features if enabled
        if self.enable_derived_features:
            df = self._create_derived_features(df)

        # Step 5: Apply derived clustering transform
        if self.enable_derived_clustering:
            df = self._transform_derived_cluster(df)

        # Step 6: Apply DataFrame clustering transform
        if self.enable_dataframe_clustering:
            df = self._transform_dataframe_cluster(df)

        # Step 7: Apply normalize by close if enabled (after clustering)
        if self.normalize_by_close:
            df = self._apply_normalize_by_close(df)

        # Step 8: Apply resampling if enabled
        if apply_resampling and self.enable_resampling:
            cluster_cols = self._get_all_cluster_columns(df)
            if cluster_cols:
                # Split the transformed data
                train_df, val_df, test_df = self._split_data(df)

                # Combine val+test for distribution calculation
                val_test_combined = pd.concat([val_df, test_df], ignore_index=True)

                # Resample train to match val+test class ratio and cluster distribution
                resampled_train = self._resample_train_data(train_df, val_test_combined)

                # Sort resampled train by date to restore chronological order
                if 'date' in resampled_train.columns:
                    resampled_train = resampled_train.sort_values('date').reset_index(drop=True)

                # Add split column to identify train/val/test rows
                resampled_train['split'] = 'train'
                val_df = val_df.copy()
                val_df['split'] = 'val'
                test_df = test_df.copy()
                test_df['split'] = 'test'

                # Concatenate
                df = pd.concat([resampled_train, val_df, test_df], ignore_index=True)

        # Step 7: Apply scaling if enabled
        if apply_scaling and self.scaler is not None:
            df = self._apply_scaling_to_dataframe(df)

        return df

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

        Warning
        -------
        This method uses pickle for serialization. Only load preprocessors
        from trusted sources, as pickle can execute arbitrary code.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")

        state = {
            # Basic parameters
            'remove_raw_indicators': self.remove_raw_indicators,
            'target_shift': self.target_shift,
            'sequence_length': self.sequence_length,
            'stride': self.stride,
            'scaler_type': self.scaler_type,
            'target_column': self.target_column,
            'target_encoding': self.target_encoding,
            'ohlcv_columns': self.ohlcv_columns,
            'columns_to_drop': self.columns_to_drop,
            'normalize_by_close': self.normalize_by_close,
            # Feature engineering parameters
            'extract_time_features': self.extract_time_features,
            'enable_dataframe_clustering': self.enable_dataframe_clustering,
            'df_cluster_columns': self.df_cluster_columns,
            'cluster_k_range': self.cluster_k_range,
            'cluster_k': self.cluster_k,
            'enable_signal_clustering': self.enable_signal_clustering,
            'signal_cluster_k_range': self.signal_cluster_k_range,
            'signal_cluster_k': self.signal_cluster_k,
            'keep_original_signals': self.keep_original_signals,
            # Resampling parameters
            'enable_resampling': self.enable_resampling,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'resample_random_state': self.resample_random_state,
            # Derived feature engineering parameters
            'enable_derived_features': self.enable_derived_features,
            'enable_derived_clustering': self.enable_derived_clustering,
            'derived_cluster_k_range': self.derived_cluster_k_range,
            'derived_cluster_k': self.derived_cluster_k,
            # Fitted state
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            '_ohlcv_indices': self._ohlcv_indices,
            '_signal_indices': self._signal_indices,
            # DataFrame clustering state
            '_dataframe_cluster_model': self._dataframe_cluster_model,
            '_dataframe_cluster_k': self._dataframe_cluster_k,
            '_dataframe_cluster_columns': self._dataframe_cluster_columns,
            '_df_cluster_fit_columns': self._df_cluster_fit_columns,
            # Signal clustering state
            '_entry_cluster_model': self._entry_cluster_model,
            '_exit_cluster_model': self._exit_cluster_model,
            '_entry_column_clusters': self._entry_column_clusters,
            '_exit_column_clusters': self._exit_column_clusters,
            '_entry_cluster_k': self._entry_cluster_k,
            '_exit_cluster_k': self._exit_cluster_k,
            # Derived clustering state
            '_derived_cluster_model': self._derived_cluster_model,
            '_derived_cluster_scaler': self._derived_cluster_scaler,
            '_derived_cluster_k': self._derived_cluster_k,
            '_derived_cluster_columns': self._derived_cluster_columns,
            '_derived_cluster_fit_columns': self._derived_cluster_fit_columns,
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

        Warning
        -------
        This method uses pickle for deserialization. Only load preprocessors
        from trusted sources, as pickle can execute arbitrary code.
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
            normalize_by_close=state.get('normalize_by_close', False),
            # Feature engineering parameters
            extract_time_features=state.get('extract_time_features', False),
            enable_dataframe_clustering=state.get('enable_dataframe_clustering', False),
            df_cluster_columns=state.get('df_cluster_columns', 'indicators'),
            cluster_k_range=state.get('cluster_k_range', (2, 10)),
            cluster_k=state.get('cluster_k', None),
            enable_signal_clustering=state.get('enable_signal_clustering', False),
            signal_cluster_k_range=state.get('signal_cluster_k_range', (2, 8)),
            signal_cluster_k=state.get('signal_cluster_k', None),
            keep_original_signals=state.get('keep_original_signals', False),
            # Resampling parameters
            enable_resampling=state.get('enable_resampling', False),
            train_ratio=state.get('train_ratio', 0.6),
            val_ratio=state.get('val_ratio', 0.2),
            resample_random_state=state.get('resample_random_state', 42),
            # Derived feature engineering parameters
            enable_derived_features=state.get('enable_derived_features', False),
            enable_derived_clustering=state.get('enable_derived_clustering', False),
            derived_cluster_k_range=state.get('derived_cluster_k_range', (2, 10)),
            derived_cluster_k=state.get('derived_cluster_k', None),
        )
        # Fitted state
        preprocessor.scaler = state['scaler']
        preprocessor.feature_columns = state['feature_columns']
        preprocessor._ohlcv_indices = state['_ohlcv_indices']
        preprocessor._signal_indices = state['_signal_indices']
        preprocessor._is_fitted = True

        # DataFrame clustering state
        preprocessor._dataframe_cluster_model = state.get('_dataframe_cluster_model', None)
        preprocessor._dataframe_cluster_k = state.get('_dataframe_cluster_k', 0)
        preprocessor._dataframe_cluster_columns = state.get('_dataframe_cluster_columns', [])
        preprocessor._df_cluster_fit_columns = state.get('_df_cluster_fit_columns', [])

        # Signal clustering state
        preprocessor._entry_cluster_model = state.get('_entry_cluster_model', None)
        preprocessor._exit_cluster_model = state.get('_exit_cluster_model', None)
        preprocessor._entry_column_clusters = state.get('_entry_column_clusters', {})
        preprocessor._exit_column_clusters = state.get('_exit_column_clusters', {})
        preprocessor._entry_cluster_k = state.get('_entry_cluster_k', 0)
        preprocessor._exit_cluster_k = state.get('_exit_cluster_k', 0)

        # Derived clustering state
        preprocessor._derived_cluster_model = state.get('_derived_cluster_model', None)
        preprocessor._derived_cluster_scaler = state.get('_derived_cluster_scaler', None)
        preprocessor._derived_cluster_k = state.get('_derived_cluster_k', 0)
        preprocessor._derived_cluster_columns = state.get('_derived_cluster_columns', [])
        preprocessor._derived_cluster_fit_columns = state.get('_derived_cluster_fit_columns', [])

        return preprocessor

    def __repr__(self) -> str:
        base_info = (
            f"sequence_length={self.sequence_length}, "
            f"target_shift={self.target_shift}, "
            f"stride={self.stride}, "
            f"remove_raw_indicators={self.remove_raw_indicators}, "
            f"normalize_by_close={self.normalize_by_close}, "
            f"scaler_type='{self.scaler_type}'"
        )

        if self._is_fitted:
            fitted_info = (
                f"n_features={len(self.feature_columns)}, "
                f"n_scaled={len(self._ohlcv_indices)}, "
                f"n_binary={len(self._signal_indices)}, "
            )
            extras = []
            if self.extract_time_features:
                extras.append("time_features=True")
            if self.enable_dataframe_clustering:
                extras.append(f"df_clusters={self._dataframe_cluster_k}")
            if self.enable_signal_clustering:
                extras.append(f"entry_clusters={self._entry_cluster_k}")
                extras.append(f"exit_clusters={self._exit_cluster_k}")
            if self.enable_derived_features:
                # Count derived feature columns (excluding cluster columns)
                n_derived = len([c for c in self.feature_columns
                                 if c.startswith('derived_') and not c.startswith('derived_cluster_')])
                extras.append(f"derived_features={n_derived}")
            if self.enable_derived_clustering:
                extras.append(f"derived_clusters={self._derived_cluster_k}")
            if self.enable_resampling:
                test_ratio = 1 - self.train_ratio - self.val_ratio
                extras.append(
                    f"resampling=({self.train_ratio:.0%}/{self.val_ratio:.0%}/{test_ratio:.0%})"
                )
            extras_str = ", ".join(extras)
            if extras_str:
                extras_str = ", " + extras_str
            return f"VectorBTDataPreprocessor({fitted_info}{base_info}{extras_str})"
        else:
            extras = []
            if self.extract_time_features:
                extras.append("extract_time_features=True")
            if self.enable_dataframe_clustering:
                extras.append("enable_dataframe_clustering=True")
            if self.enable_signal_clustering:
                extras.append("enable_signal_clustering=True")
            if self.enable_derived_features:
                extras.append("enable_derived_features=True")
            if self.enable_derived_clustering:
                extras.append("enable_derived_clustering=True")
            if self.enable_resampling:
                extras.append("enable_resampling=True")
            extras_str = ", ".join(extras)
            if extras_str:
                extras_str = ", " + extras_str
            return f"VectorBTDataPreprocessor({base_info}{extras_str}, status=unfitted)"
