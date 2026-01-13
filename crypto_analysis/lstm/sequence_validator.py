"""
Sequence Validator Module (DEPRECATED)

This module is deprecated as of the 2-class binary classification update.

The LSTM model now predicts a single binary value (hold=0, trade=1) instead of
a sequence of entry/hold/exit signals. Sequence validation is no longer needed.

This file is kept for backwards compatibility but should not be used in new code.
"""

import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple

import numpy as np


# Emit deprecation warning when module is imported
warnings.warn(
    "sequence_validator module is deprecated. "
    "The LSTM model now uses binary classification (hold/trade) instead of "
    "sequence validation (entry/hold/exit patterns).",
    DeprecationWarning,
    stacklevel=2
)


class SequenceType(Enum):
    """
    DEPRECATED: Classification of sequence type.

    This enum is kept for backwards compatibility but is no longer used
    in the binary classification model.
    """
    ALL_HOLD = auto()      # Previously: all hold signals (1,1,1,1)
    ENTRY_EXIT = auto()    # Previously: starts entry, ends exit (0,*,*,2)
    INVALID = auto()       # Previously: does not match valid patterns


@dataclass
class ValidationResult:
    """DEPRECATED: Result of sequence validation."""
    valid_indices: np.ndarray
    invalid_indices: np.ndarray
    sequence_types: np.ndarray
    stats: dict


class SequenceValidator:
    """
    DEPRECATED: Validates target sequences for LSTM training.

    This class is deprecated as the LSTM model now uses binary classification
    (hold=0, trade=1) with single output per sequence instead of 4-step
    entry/hold/exit patterns.

    Kept for backwards compatibility only.
    """

    # Signal encoding constants (deprecated)
    ENTRY = 0
    HOLD = 1
    EXIT = 2

    def __init__(self, sequence_length: int = 4):
        """
        Initialize validator.

        Parameters
        ----------
        sequence_length : int, default=4
            DEPRECATED: Length of target sequences to validate
        """
        warnings.warn(
            "SequenceValidator is deprecated. "
            "Use binary classification (hold=0, trade=1) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.sequence_length = sequence_length

    def classify_sequence(self, sequence: np.ndarray) -> SequenceType:
        """DEPRECATED: Classify a single sequence into its type."""
        warnings.warn(
            "classify_sequence is deprecated",
            DeprecationWarning,
            stacklevel=2
        )
        if len(sequence) != self.sequence_length:
            return SequenceType.INVALID
        if np.all(sequence == self.HOLD):
            return SequenceType.ALL_HOLD
        if (sequence[0] == self.ENTRY and
            sequence[-1] == self.EXIT and
            np.all(sequence[1:-1] == self.HOLD)):
            return SequenceType.ENTRY_EXIT
        return SequenceType.INVALID

    def validate_sequences(self, sequences: np.ndarray) -> ValidationResult:
        """DEPRECATED: Validate batch of sequences."""
        warnings.warn(
            "validate_sequences is deprecated",
            DeprecationWarning,
            stacklevel=2
        )
        n_sequences = len(sequences)
        all_hold_mask = np.all(sequences == self.HOLD, axis=1)
        starts_entry = sequences[:, 0] == self.ENTRY
        ends_exit = sequences[:, -1] == self.EXIT
        if self.sequence_length > 2:
            middle_all_hold = np.all(sequences[:, 1:-1] == self.HOLD, axis=1)
        else:
            middle_all_hold = np.ones(n_sequences, dtype=bool)
        entry_exit_mask = starts_entry & ends_exit & middle_all_hold
        valid_mask = all_hold_mask | entry_exit_mask
        valid_indices = np.where(valid_mask)[0]
        invalid_indices = np.where(~valid_mask)[0]

        sequence_types = np.empty(len(valid_indices), dtype=object)
        for i, idx in enumerate(valid_indices):
            if all_hold_mask[idx]:
                sequence_types[i] = SequenceType.ALL_HOLD
            else:
                sequence_types[i] = SequenceType.ENTRY_EXIT

        stats = {
            'total_sequences': n_sequences,
            'valid_sequences': len(valid_indices),
            'invalid_sequences': len(invalid_indices),
            'all_hold_count': int(np.sum(all_hold_mask)),
            'entry_exit_count': int(np.sum(entry_exit_mask)),
            'validity_rate': len(valid_indices) / n_sequences if n_sequences > 0 else 0.0
        }

        return ValidationResult(
            valid_indices=valid_indices,
            invalid_indices=invalid_indices,
            sequence_types=sequence_types,
            stats=stats
        )

    def filter_valid_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """DEPRECATED: Filter to keep only valid sequences."""
        warnings.warn(
            "filter_valid_sequences is deprecated",
            DeprecationWarning,
            stacklevel=2
        )
        result = self.validate_sequences(targets)
        return features[result.valid_indices], targets[result.valid_indices], result.sequence_types

    def __repr__(self) -> str:
        return f"SequenceValidator(DEPRECATED, sequence_length={self.sequence_length})"
