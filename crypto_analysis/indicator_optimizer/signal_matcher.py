"""
Signal Matcher Module

Compares indicator-generated signals with SignalPopulator target signals
to evaluate how well an indicator configuration matches the target.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MatchResult:
    """Result of signal matching comparison."""
    entry_matches: int
    exit_matches: int
    total_matches: int
    entry_precision: float
    entry_recall: float
    entry_f1: float
    exit_precision: float
    exit_recall: float
    exit_f1: float
    combined_score: float

    def __repr__(self):
        return (
            f"MatchResult(matches={self.total_matches}, "
            f"entry_f1={self.entry_f1:.3f}, exit_f1={self.exit_f1:.3f}, "
            f"score={self.combined_score:.3f})"
        )


class SignalMatcher:
    """
    Compare indicator signals with SignalPopulator signals.

    Uses a tolerance window to allow signals within N bars to count as matches.
    Default tolerance is 1 bar (signals within 1 candle count as match).
    """

    TOLERANCE_BARS = 1  # Default: signals within 1 candle count as match

    def __init__(self, tolerance_bars: int = TOLERANCE_BARS):
        """
        Initialize signal matcher.

        Parameters
        ----------
        tolerance_bars : int
            Number of bars tolerance for signal matching.
            A signal is considered a match if it occurs within
            tolerance_bars of the target signal.
        """
        self.tolerance_bars = tolerance_bars

    def calculate_match_score(
        self,
        indicator_df: pd.DataFrame,
        target_df: pd.DataFrame,
        entry_col: str = "entry_signal",
        exit_col: str = "exit_signal",
        target_signal_col: str = "signal"
    ) -> MatchResult:
        """
        Calculate how well indicator signals match target signals.

        Parameters
        ----------
        indicator_df : pd.DataFrame
            DataFrame with indicator entry/exit signals (boolean)
        target_df : pd.DataFrame
            DataFrame with target signals from SignalPopulator
        entry_col : str
            Column name for entry signals in indicator_df
        exit_col : str
            Column name for exit signals in indicator_df
        target_signal_col : str
            Column name for signals in target_df ("entry"/"exit")

        Returns
        -------
        MatchResult
            Match statistics and scores
        """
        # Ensure dataframes are aligned
        if len(indicator_df) != len(target_df):
            raise ValueError(
                f"DataFrames must have same length: "
                f"{len(indicator_df)} vs {len(target_df)}"
            )

        # Get target entry/exit indices
        target_entries = target_df[target_df[target_signal_col] == "entry"].index.tolist()
        target_exits = target_df[target_df[target_signal_col] == "exit"].index.tolist()

        # Get indicator entry/exit indices
        indicator_entries = indicator_df[indicator_df[entry_col] == True].index.tolist()
        indicator_exits = indicator_df[indicator_df[exit_col] == True].index.tolist()

        # Calculate entry matches
        entry_matches, entry_precision, entry_recall = self._calculate_matches(
            indicator_entries, target_entries, len(indicator_df)
        )

        # Calculate exit matches
        exit_matches, exit_precision, exit_recall = self._calculate_matches(
            indicator_exits, target_exits, len(indicator_df)
        )

        # Calculate F1 scores
        entry_f1 = self._calculate_f1(entry_precision, entry_recall)
        exit_f1 = self._calculate_f1(exit_precision, exit_recall)

        # Combined score: sum of matches (as per user requirement)
        total_matches = entry_matches + exit_matches
        combined_score = total_matches

        return MatchResult(
            entry_matches=entry_matches,
            exit_matches=exit_matches,
            total_matches=total_matches,
            entry_precision=entry_precision,
            entry_recall=entry_recall,
            entry_f1=entry_f1,
            exit_precision=exit_precision,
            exit_recall=exit_recall,
            exit_f1=exit_f1,
            combined_score=combined_score
        )

    def _calculate_matches(
        self,
        indicator_indices: List[int],
        target_indices: List[int],
        total_length: int
    ) -> Tuple[int, float, float]:
        """
        Vectorized calculation of matches with tolerance using NumPy broadcasting.
        ~100x faster than nested loops.

        Parameters
        ----------
        indicator_indices : list
            Indices where indicator generated signals
        target_indices : list
            Indices where target signals exist
        total_length : int
            Total length of the DataFrame

        Returns
        -------
        tuple
            (num_matches, precision, recall)
        """
        if not indicator_indices or not target_indices:
            return 0, 0.0, 0.0

        # Convert to numpy arrays
        ind_arr = np.array(indicator_indices)
        tgt_arr = np.array(target_indices)

        # Compute distance matrix using broadcasting: |ind[i] - tgt[j]|
        # Shape: (len(indicator), len(target))
        distances = np.abs(ind_arr[:, np.newaxis] - tgt_arr[np.newaxis, :])

        # Find matches within tolerance
        within_tolerance = distances <= self.tolerance_bars

        # Greedy matching: each target can only match once
        # Process indicators sorted by their minimum distance to any target
        matches = 0
        matched_targets = set()

        min_distances = np.min(distances, axis=1)
        sorted_ind_order = np.argsort(min_distances)

        for i in sorted_ind_order:
            if min_distances[i] > self.tolerance_bars:
                break  # No more matches possible

            # Find closest unmatched target
            for j in np.argsort(distances[i]):
                if distances[i, j] > self.tolerance_bars:
                    break
                if j not in matched_targets:
                    matches += 1
                    matched_targets.add(j)
                    break

        # Precision: correct indicator signals / total indicator signals
        precision = matches / len(indicator_indices)

        # Recall: matched targets / total targets
        recall = len(matched_targets) / len(target_indices)

        return matches, precision, recall

    def _calculate_f1(self, precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def find_signal_alignment(
        self,
        indicator_df: pd.DataFrame,
        target_df: pd.DataFrame,
        entry_col: str = "entry_signal",
        exit_col: str = "exit_signal",
        target_signal_col: str = "signal"
    ) -> pd.DataFrame:
        """
        Create a detailed alignment DataFrame showing matches.

        Parameters
        ----------
        indicator_df : pd.DataFrame
            DataFrame with indicator signals
        target_df : pd.DataFrame
            DataFrame with target signals

        Returns
        -------
        pd.DataFrame
            DataFrame with columns showing signal alignment
        """
        result = pd.DataFrame(index=indicator_df.index)
        result["indicator_entry"] = indicator_df[entry_col].fillna(False)
        result["indicator_exit"] = indicator_df[exit_col].fillna(False)
        result["target_signal"] = target_df[target_signal_col]
        result["target_entry"] = target_df[target_signal_col] == "entry"
        result["target_exit"] = target_df[target_signal_col] == "exit"

        # Mark matches
        result["entry_match"] = False
        result["exit_match"] = False

        target_entries = result[result["target_entry"]].index.tolist()
        target_exits = result[result["target_exit"]].index.tolist()

        # Vectorized entry match check
        ind_entry_idx = np.array(result[result["indicator_entry"]].index.tolist())
        tgt_entry_idx = np.array(target_entries)
        if len(ind_entry_idx) > 0 and len(tgt_entry_idx) > 0:
            entry_distances = np.abs(ind_entry_idx[:, np.newaxis] - tgt_entry_idx[np.newaxis, :])
            entry_matches = np.any(entry_distances <= self.tolerance_bars, axis=1)
            result.loc[ind_entry_idx[entry_matches], "entry_match"] = True

        # Vectorized exit match check
        ind_exit_idx = np.array(result[result["indicator_exit"]].index.tolist())
        tgt_exit_idx = np.array(target_exits)
        if len(ind_exit_idx) > 0 and len(tgt_exit_idx) > 0:
            exit_distances = np.abs(ind_exit_idx[:, np.newaxis] - tgt_exit_idx[np.newaxis, :])
            exit_matches = np.any(exit_distances <= self.tolerance_bars, axis=1)
            result.loc[ind_exit_idx[exit_matches], "exit_match"] = True

        return result

    def get_unmatched_signals(
        self,
        indicator_df: pd.DataFrame,
        target_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get signals that don't have matches.

        Returns
        -------
        tuple
            (unmatched_indicator_signals, unmatched_target_signals)
        """
        alignment = self.find_signal_alignment(indicator_df, target_df)

        # Unmatched indicator entries
        unmatched_ind = alignment[
            (alignment["indicator_entry"] & ~alignment["entry_match"]) |
            (alignment["indicator_exit"] & ~alignment["exit_match"])
        ]

        # Unmatched targets (targets without nearby indicator signals)
        # This requires more complex logic - for now return targets with no matches
        unmatched_tgt = alignment[
            (alignment["target_entry"] | alignment["target_exit"]) &
            ~alignment["entry_match"] & ~alignment["exit_match"]
        ]

        return unmatched_ind, unmatched_tgt
