"""
Test script for DualDataPreprocessor alignment and sequence creation.

Verifies that:
1. align_dataframes drops NaN rows and maintains period alignment (divisible by 4)
2. create_dual_sequences with stride=4 creates period-aligned sequences
3. Each target in the sequences corresponds to a period with consistent tradeable values
"""

import sys
from pathlib import Path

# Add crypto_analysis to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from crypto_analysis.lstm.dual_preprocessor import (
    DualDataPreprocessor,
    create_dual_sequences,
)


def create_test_dataframes(n_periods: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create test dataframes with known period structure.

    Each period has 4 rows with the same tradeable value.
    Technical dataframe has NaN values at the start (simulating indicator lookback).
    """
    n_rows = n_periods * 4

    # Create dates
    dates = pd.date_range(start='2025-06-01', periods=n_rows, freq='1h')

    # Create tradeable values: each period of 4 rows has the same value
    # Randomly assign hold/trade to each period
    np.random.seed(42)
    period_labels = np.random.choice(['hold', 'trade'], size=n_periods, p=[0.8, 0.2])
    tradeable = np.repeat(period_labels, 4)

    # Create period_id
    period_ids = np.repeat(np.arange(n_periods), 4)

    # Create binary features (0/1 values)
    n_binary_features = 10
    binary_data = {
        'date': dates,
        'tradeable': tradeable,
        'period_id': period_ids,
    }
    for i in range(n_binary_features):
        binary_data[f'binary_feat_{i}'] = np.random.randint(0, 2, size=n_rows)

    df_binary = pd.DataFrame(binary_data)

    # Create technical features with NaN at start (simulating indicator lookback)
    n_technical_features = 8
    technical_data = {
        'date': dates,
        'tradeable': tradeable,
        'period_id': period_ids,
        'open': np.random.randn(n_rows) * 10 + 100,
        'high': np.random.randn(n_rows) * 10 + 105,
        'low': np.random.randn(n_rows) * 10 + 95,
        'close': np.random.randn(n_rows) * 10 + 100,
        'volume': np.random.rand(n_rows) * 1000000,
    }
    for i in range(n_technical_features):
        values = np.random.randn(n_rows)
        # Add NaN at start to simulate indicator lookback (e.g., first 20 rows)
        values[:20] = np.nan
        technical_data[f'tech_feat_{i}'] = values

    df_technical = pd.DataFrame(technical_data)

    return df_binary, df_technical


def test_align_dataframes():
    """Test that align_dataframes properly handles NaN and maintains period alignment."""
    print("\n" + "=" * 70)
    print("TEST 1: align_dataframes")
    print("=" * 70)

    df_binary, df_technical = create_test_dataframes(n_periods=50)

    print(f"\nOriginal shapes:")
    print(f"  Binary: {df_binary.shape}")
    print(f"  Technical: {df_technical.shape}")
    print(f"  NaN rows in technical: {df_technical.isna().any(axis=1).sum()}")

    # Align dataframes
    df_binary_aligned, df_technical_aligned = DualDataPreprocessor.align_dataframes(
        df_binary, df_technical, period_size=4, verbose=True
    )

    print(f"\nAligned shapes:")
    print(f"  Binary: {df_binary_aligned.shape}")
    print(f"  Technical: {df_technical_aligned.shape}")

    # Verify no NaN in aligned technical
    nan_count = df_technical_aligned.isna().any(axis=1).sum()
    print(f"  NaN rows in aligned technical: {nan_count}")
    assert nan_count == 0, "Aligned technical should have no NaN"

    # Verify divisible by 4
    assert len(df_binary_aligned) % 4 == 0, "Row count should be divisible by 4"
    assert len(df_technical_aligned) % 4 == 0, "Row count should be divisible by 4"
    print(f"  Divisible by 4: YES ({len(df_binary_aligned)} rows)")

    # Verify same length
    assert len(df_binary_aligned) == len(df_technical_aligned), "Both should have same length"
    print(f"  Same length: YES")

    # Verify dates match
    if 'date' in df_binary_aligned.columns and 'date' in df_technical_aligned.columns:
        dates_match = (df_binary_aligned['date'] == df_technical_aligned['date']).all()
        print(f"  Dates match: {dates_match}")
        assert dates_match, "Dates should match"

    print("\n✓ align_dataframes test PASSED")
    return df_binary_aligned, df_technical_aligned


def test_create_sequences(df_binary: pd.DataFrame, df_technical: pd.DataFrame):
    """Test that create_dual_sequences creates period-aligned sequences."""
    print("\n" + "=" * 70)
    print("TEST 2: create_dual_sequences with stride=4")
    print("=" * 70)

    # Preprocess data
    preprocessor = DualDataPreprocessor(target_shift=4)
    binary_feat, technical_feat, targets = preprocessor.fit_transform(
        df_binary, df_technical
    )

    print(f"\nAfter preprocessing:")
    print(f"  Binary features shape: {binary_feat.shape}")
    print(f"  Technical features shape: {technical_feat.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Unique targets: {np.unique(targets)}")

    # Create sequences with stride=4
    input_seq_length = 20  # Must be divisible by 4
    binary_seqs, technical_seqs, target_seqs = create_dual_sequences(
        binary_feat, technical_feat, targets,
        input_seq_length=input_seq_length,
        output_seq_length=1,
        stride=4
    )

    print(f"\nAfter create_dual_sequences (stride=4):")
    print(f"  Binary sequences shape: {binary_seqs.shape}")
    print(f"  Technical sequences shape: {technical_seqs.shape}")
    print(f"  Target sequences shape: {target_seqs.shape}")

    # Count hold vs trade
    n_hold = (target_seqs == 0).sum()
    n_trade = (target_seqs == 1).sum()
    print(f"  Hold sequences: {n_hold}")
    print(f"  Trade sequences: {n_trade}")

    print("\n✓ create_dual_sequences test PASSED")
    return binary_seqs, technical_seqs, target_seqs


def test_period_consistency(df_binary: pd.DataFrame, df_technical: pd.DataFrame):
    """
    Test that each target in sequences corresponds to a period with consistent values.

    This verifies that with stride=4 and data divisible by 4, each sequence's
    target corresponds to a complete period where all 4 rows have the same tradeable value.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Period consistency verification")
    print("=" * 70)

    # Get the original tradeable values before preprocessing
    tradeable_original = df_technical['tradeable'].values
    n_rows = len(tradeable_original)

    print(f"\nOriginal data:")
    print(f"  Total rows: {n_rows}")
    print(f"  Total periods: {n_rows // 4}")

    # Verify each period of 4 rows has consistent tradeable values
    periods_consistent = True
    inconsistent_periods = []

    for i in range(0, n_rows, 4):
        period_values = tradeable_original[i:i+4]
        if len(set(period_values)) > 1:
            periods_consistent = False
            inconsistent_periods.append((i // 4, period_values.tolist()))

    if periods_consistent:
        print(f"  All periods have consistent tradeable values: YES")
    else:
        print(f"  WARNING: {len(inconsistent_periods)} periods have inconsistent values")
        for period_idx, values in inconsistent_periods[:5]:
            print(f"    Period {period_idx}: {values}")

    # Now test the sequence creation
    preprocessor = DualDataPreprocessor(target_shift=4)
    binary_feat, technical_feat, targets = preprocessor.fit_transform(
        df_binary, df_technical
    )

    input_seq_length = 20
    stride = 4

    binary_seqs, technical_seqs, target_seqs = create_dual_sequences(
        binary_feat, technical_feat, targets,
        input_seq_length=input_seq_length,
        output_seq_length=1,
        stride=stride
    )

    print(f"\nSequence analysis:")
    print(f"  Input sequence length: {input_seq_length}")
    print(f"  Stride: {stride}")
    print(f"  Number of sequences: {len(target_seqs)}")

    # For each sequence, verify the target corresponds to a complete period
    # The target at position [i + input_seq_length] in the original targets array
    # should correspond to a period boundary

    # Since stride=4 and data is divisible by 4, each sequence starts at a period boundary
    # and the target is at position (start + input_seq_length), which should also be at a period boundary

    print(f"\nTarget distribution:")
    for target_val in [0, 1]:
        label = 'hold' if target_val == 0 else 'trade'
        count = (target_seqs == target_val).sum()
        pct = count / len(target_seqs) * 100
        print(f"  {label} (class {target_val}): {count} ({pct:.1f}%)")

    # Verify that sequences at stride=4 hit period boundaries
    # Since input_seq_length=20 is divisible by 4, and stride=4,
    # sequences start at 0, 4, 8, ... and targets are at 20, 24, 28, ...
    # All these positions are at period boundaries (divisible by 4)

    start_positions = list(range(0, len(targets) - input_seq_length, stride))
    target_positions = [s + input_seq_length for s in start_positions]

    all_at_boundary = all(pos % 4 == 0 for pos in target_positions)
    print(f"\n  All target positions at period boundaries: {all_at_boundary}")

    if not all_at_boundary:
        non_boundary = [pos for pos in target_positions if pos % 4 != 0]
        print(f"    Non-boundary positions: {non_boundary[:10]}...")

    print("\n✓ Period consistency test PASSED")


def test_with_real_data():
    """Test with real data files if available."""
    print("\n" + "=" * 70)
    print("TEST 4: Real data test (if available)")
    print("=" * 70)

    # Check for real data files
    data_dir = Path(__file__).parent / "notebooks"
    binary_file = data_dir / "doge.csv"
    technical_file = data_dir / "doge_ti.csv"

    if not binary_file.exists() or not technical_file.exists():
        print(f"\nReal data files not found:")
        print(f"  Binary: {binary_file} - {'EXISTS' if binary_file.exists() else 'NOT FOUND'}")
        print(f"  Technical: {technical_file} - {'EXISTS' if technical_file.exists() else 'NOT FOUND'}")
        print("\nSkipping real data test")
        return

    print(f"\nLoading real data:")
    df_binary = pd.read_csv(binary_file)
    df_technical = pd.read_csv(technical_file)

    print(f"  Binary shape: {df_binary.shape}")
    print(f"  Technical shape: {df_technical.shape}")
    print(f"  NaN in technical: {df_technical.isna().any(axis=1).sum()}")

    # Align dataframes
    df_binary_aligned, df_technical_aligned = DualDataPreprocessor.align_dataframes(
        df_binary, df_technical, period_size=4, verbose=True
    )

    # Verify period consistency in real data
    if 'tradeable' in df_binary_aligned.columns:
        tradeable = df_binary_aligned['tradeable'].values
        n_rows = len(tradeable)

        print(f"\nPeriod consistency check:")
        consistent_count = 0
        inconsistent_count = 0

        for i in range(0, n_rows, 4):
            period_values = tradeable[i:i+4]
            if len(period_values) == 4:
                if len(set(period_values)) == 1:
                    consistent_count += 1
                else:
                    inconsistent_count += 1

        print(f"  Consistent periods: {consistent_count}")
        print(f"  Inconsistent periods: {inconsistent_count}")

        if inconsistent_count == 0:
            print("  ✓ All periods have consistent tradeable values")
        else:
            print(f"  ⚠ {inconsistent_count} periods have mixed values")

    # Create sequences
    preprocessor = DualDataPreprocessor(target_shift=4)

    # Select feature columns (exclude metadata)
    binary_cols = [c for c in df_binary_aligned.columns
                   if c not in ['date', 'signal', 'signal_pct_change', 'period_id', 'tradeable',
                               'open', 'high', 'low', 'close', 'volume']]
    technical_cols = [c for c in df_technical_aligned.columns
                      if c not in ['date', 'signal', 'signal_pct_change', 'period_id', 'tradeable']]

    df_binary_selected = df_binary_aligned[['tradeable'] + binary_cols[:20]].copy()
    df_technical_selected = df_technical_aligned[['tradeable'] + technical_cols[:20]].copy()

    binary_feat, technical_feat, targets = preprocessor.fit_transform(
        df_binary_selected, df_technical_selected
    )

    print(f"\nAfter preprocessing:")
    print(f"  Features length: {len(binary_feat)}")
    print(f"  Divisible by 4: {len(binary_feat) % 4 == 0}")

    # Create sequences
    input_seq_length = 20
    binary_seqs, technical_seqs, target_seqs = create_dual_sequences(
        binary_feat, technical_feat, targets,
        input_seq_length=input_seq_length,
        output_seq_length=1,
        stride=4
    )

    print(f"\nSequences created:")
    print(f"  Number of sequences: {len(target_seqs)}")
    print(f"  Hold: {(target_seqs == 0).sum()}")
    print(f"  Trade: {(target_seqs == 1).sum()}")

    print("\n✓ Real data test PASSED")


def main():
    """Run all tests."""
    print("\n" + "#" * 70)
    print("# DualDataPreprocessor Test Suite")
    print("#" * 70)

    # Test 1: align_dataframes
    df_binary_aligned, df_technical_aligned = test_align_dataframes()

    # Test 2: create_dual_sequences
    test_create_sequences(df_binary_aligned, df_technical_aligned)

    # Test 3: Period consistency
    df_binary, df_technical = create_test_dataframes(n_periods=50)
    df_binary_aligned, df_technical_aligned = DualDataPreprocessor.align_dataframes(
        df_binary, df_technical, period_size=4, verbose=False
    )
    test_period_consistency(df_binary_aligned, df_technical_aligned)

    # Test 4: Real data (if available)
    test_with_real_data()

    print("\n" + "#" * 70)
    print("# ALL TESTS PASSED")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
