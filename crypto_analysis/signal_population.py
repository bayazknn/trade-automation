"""
Signal Population Module

Scans OHLCV data in configurable periods and generates entry/exit signals
based on price percentage changes.

Entry is always at the first hour of the period.
Exit is at the hour with the highest positive percentage change >= threshold
(can be any hour from 1 to period_hours-1, not just the last hour).

Also generates a 'tradeable' column for LSTM prediction:
- "trade" for periods with entry/exit signals
- "hold" for periods with no signals
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd


class SignalPopulator:
    """
    Generates entry/exit signals from cryptocurrency OHLCV data.

    Scans data in configurable periods and tags:
    - "entry" at start price (hour 0 of period)
    - "exit" at the hour with highest positive % change >= threshold

    Also generates 'tradeable' column:
    - "trade" for periods with entry/exit
    - "hold" for periods without signals

    Parameters
    ----------
    data_dir : str or Path
        Directory containing feather files with OHLCV data
    period_hours : int, optional
        Period length in hours for signal scanning (default: 4)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        period_hours: int = 4
    ):
        self.data_dir = Path(data_dir)
        self.period_hours = period_hours

    def load_data(self, symbol: str) -> pd.DataFrame:
        """
        Load OHLCV data for a cryptocurrency symbol.

        Parameters
        ----------
        symbol : str
            Cryptocurrency symbol (e.g., "BTC", "ETH")
            Will search for files matching {symbol}_USDT-*.feather

        Returns
        -------
        pd.DataFrame
            DataFrame with OHLCV columns
        """
        # Search for matching feather file
        pattern = f"{symbol}_USDT-*.feather"
        files = list(self.data_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No feather file found for symbol '{symbol}' in {self.data_dir}"
            )

        # Use the first matching file
        file_path = files[0]
        df = pd.read_feather(file_path)

        # Ensure date column is datetime and sorted
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

        return df

    def _find_best_exit_in_period(
        self,
        df: pd.DataFrame,
        start_idx: int,
        threshold_pct: float
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Find the best exit hour in a period (hour with highest positive % change).

        Searches hours 1 to period_hours-1 for the hour with the maximum
        positive percentage change that meets the threshold.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame
        start_idx : int
            Index of the entry hour (hour 0 of the period)
        threshold_pct : float
            Minimum percentage gain to consider as valid exit

        Returns
        -------
        Tuple[Optional[int], Optional[float]]
            (best_exit_idx, best_pct_change) or (None, None) if no valid exit
        """
        start_close = df.loc[start_idx, "close"]
        best_exit_idx = None
        best_pct_change = None

        # Search hours 1 to period_hours-1 for exit
        for offset in range(1, self.period_hours):
            check_idx = start_idx + offset
            if check_idx >= len(df):
                break

            check_close = df.loc[check_idx, "close"]
            pct_change = ((check_close - start_close) / start_close) * 100

            # Only consider positive changes that meet threshold
            if pct_change >= threshold_pct:
                if best_pct_change is None or pct_change > best_pct_change:
                    best_exit_idx = check_idx
                    best_pct_change = pct_change

        return best_exit_idx, best_pct_change

    def _count_signals_from_index(
        self,
        df: pd.DataFrame,
        start_index: int,
        threshold_pct: float
    ) -> Tuple[int, int]:
        """
        Count entry and exit signals starting from a given index.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame
        start_index : int
            Starting index for signal generation
        threshold_pct : float
            Minimum percentage gain to trigger signals

        Returns
        -------
        Tuple[int, int]
            (entry_count, exit_count)
        """
        rows_per_period = self.period_hours
        entry_count = 0
        exit_count = 0
        i = start_index

        while i + rows_per_period <= len(df):
            # Find best exit in this period
            best_exit_idx, _ = self._find_best_exit_in_period(
                df, i, threshold_pct
            )

            if best_exit_idx is not None:
                entry_count += 1
                exit_count += 1
                # Move past this period (after the exit)
                i = i + rows_per_period
            else:
                # No valid exit found, slide window by 1
                i += 1

        return entry_count, exit_count

    def generate_signals(
        self,
        df: pd.DataFrame,
        threshold_pct: float
    ) -> pd.DataFrame:
        """
        Generate entry/exit signals based on price change threshold.

        Scans OHLCV data in periods of `period_hours` and marks:
        - "entry" signal at start (hour 0 of period)
        - "exit" signal at the hour with highest positive % change >= threshold
          (can be any hour from 1 to period_hours-1)

        Also generates 'tradeable' column:
        - "trade" for all rows in periods with entry/exit signals
        - "hold" for all rows in periods without signals

        Finds optimal starting index (0 to period_hours) that maximizes
        signal count, drops rows before that index.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with columns: date, open, high, low, close, volume
        threshold_pct : float
            Minimum percentage gain (k%) to trigger entry/exit signals
            e.g., 2.0 means end price must be 2% above start price

        Returns
        -------
        pd.DataFrame
            DataFrame (starting from optimal index) with additional columns:
            - signal: "entry", "exit", or None (for indicator optimization)
            - signal_pct_change: percentage change for signal periods
            - period_id: unique identifier for entry/exit pairs
            - tradeable: "trade" or "hold" (for LSTM prediction)
        """
        df = df.copy()

        # Find optimal starting index by testing 0 to period_hours
        initial_entry, initial_exit = self._count_signals_from_index(
            df, 0, threshold_pct
        )
        print(f"Initial index (0): entry={initial_entry}, exit={initial_exit}")

        best_index = 0
        best_total = initial_entry + initial_exit
        best_entry = initial_entry
        best_exit = initial_exit

        for idx in range(1, self.period_hours + 1):
            if idx >= len(df):
                break
            entry_count, exit_count = self._count_signals_from_index(
                df, idx, threshold_pct
            )
            total = entry_count + exit_count
            if total > best_total:
                best_total = total
                best_index = idx
                best_entry = entry_count
                best_exit = exit_count

        print(
            f"Found optimal index ({best_index}): "
            f"entry={best_entry}, exit={best_exit}"
        )

        # Drop rows before optimal starting index
        if best_index > 0:
            df = df.iloc[best_index:].reset_index(drop=True)

        # Initialize signal columns
        df["signal"] = None
        df["signal_pct_change"] = None
        df["period_id"] = None
        df["tradeable"] = None  # Will be filled per period

        # Calculate number of rows per period
        rows_per_period = self.period_hours

        period_id = 0
        i = 0

        while i + rows_per_period <= len(df):
            start_idx = i
            end_idx = i + rows_per_period - 1

            # Find best exit in this period (hour with highest % change >= threshold)
            best_exit_idx, best_pct_change = self._find_best_exit_in_period(
                df, start_idx, threshold_pct
            )

            if best_exit_idx is not None:
                # Valid trade period found
                # Tag entry at start
                df.loc[start_idx, "signal"] = "entry"
                df.loc[start_idx, "signal_pct_change"] = best_pct_change
                df.loc[start_idx, "period_id"] = period_id

                # Tag exit at best exit hour
                df.loc[best_exit_idx, "signal"] = "exit"
                df.loc[best_exit_idx, "signal_pct_change"] = best_pct_change
                df.loc[best_exit_idx, "period_id"] = period_id

                # Mark entire period as "trade" for LSTM
                for j in range(start_idx, end_idx + 1):
                    df.loc[j, "tradeable"] = "trade"

                period_id += 1

                # Move past this period
                i = end_idx + 1
            else:
                # No valid exit found - mark this period as "hold"
                for j in range(start_idx, end_idx + 1):
                    df.loc[j, "tradeable"] = "hold"

                # Move past this period (non-overlapping hold periods)
                i = end_idx + 1

        # Handle any remaining rows at the end (incomplete period)
        remaining_start = i
        if remaining_start < len(df):
            for j in range(remaining_start, len(df)):
                if df.loc[j, "tradeable"] is None:
                    df.loc[j, "tradeable"] = "hold"

        return df

    def populate_signals(
        self,
        symbol: str,
        threshold_pct: float
    ) -> pd.DataFrame:
        """
        Load data and generate signals for a cryptocurrency symbol.

        Parameters
        ----------
        symbol : str
            Cryptocurrency symbol (e.g., "BTC", "ETH")
        threshold_pct : float
            Minimum percentage gain (k%) to trigger entry/exit signals

        Returns
        -------
        pd.DataFrame
            DataFrame with OHLCV data and signal columns
        """
        df = self.load_data(symbol)
        return self.generate_signals(df, threshold_pct)

    def get_signals_only(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter DataFrame to only rows with signals.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with signal column

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame containing only entry/exit signals
        """
        return df[df["signal"].notna()].copy()

    def get_signal_summary(
        self,
        df: pd.DataFrame
    ) -> dict:
        """
        Get summary statistics for generated signals.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with signal columns

        Returns
        -------
        dict
            Summary containing:
            - total_rows: Total number of rows analyzed
            - signal_count: Number of entry/exit pairs
            - avg_pct_change: Average percentage change for signals
            - min_pct_change: Minimum percentage change
            - max_pct_change: Maximum percentage change
            - trade_rows: Number of rows with tradeable="trade"
            - hold_rows: Number of rows with tradeable="hold"
            - trade_ratio: Ratio of trade rows to total rows
        """
        signals_df = self.get_signals_only(df)
        entries = signals_df[signals_df["signal"] == "entry"]

        # Count tradeable distribution
        trade_rows = (df["tradeable"] == "trade").sum()
        hold_rows = (df["tradeable"] == "hold").sum()
        total_rows = len(df)

        if len(entries) == 0:
            return {
                "total_rows": total_rows,
                "signal_count": 0,
                "avg_pct_change": None,
                "min_pct_change": None,
                "max_pct_change": None,
                "trade_rows": int(trade_rows),
                "hold_rows": int(hold_rows),
                "trade_ratio": 0.0
            }

        return {
            "total_rows": total_rows,
            "signal_count": len(entries),
            "avg_pct_change": entries["signal_pct_change"].mean(),
            "min_pct_change": entries["signal_pct_change"].min(),
            "max_pct_change": entries["signal_pct_change"].max(),
            "trade_rows": int(trade_rows),
            "hold_rows": int(hold_rows),
            "trade_ratio": trade_rows / total_rows if total_rows > 0 else 0.0
        }

    def process_all_symbols(
        self,
        threshold_pct: float,
        symbols: Optional[list] = None
    ) -> dict[str, pd.DataFrame]:
        """
        Process multiple cryptocurrency symbols.

        Parameters
        ----------
        threshold_pct : float
            Minimum percentage gain to trigger signals
        symbols : list, optional
            List of symbols to process. If None, processes all
            available feather files in data_dir.

        Returns
        -------
        dict
            Dictionary mapping symbol names to DataFrames with signals
        """
        if symbols is None:
            # Discover all available symbols
            feather_files = list(self.data_dir.glob("*_USDT-*.feather"))
            symbols = [
                f.stem.split("_USDT")[0]
                for f in feather_files
            ]

        results = {}
        for symbol in symbols:
            try:
                df = self.populate_signals(symbol, threshold_pct)
                results[symbol] = df
            except FileNotFoundError:
                print(f"Warning: No data found for {symbol}")

        return results
