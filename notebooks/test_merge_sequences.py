"""
Test script for merge_dataframes and create_sequences functions.

Verifies:
1. merge_dataframes correctly merges binary and technical dataframes
2. create_sequences produces sequences with no mixed labels in targets
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from crypto_analysis.lstm_optimizer import LSTMMetaheuristicOptimizer
from crypto_analysis.lstm import DataPreprocessor, create_sequences


def test_merge_dataframes():
    """Test merge_dataframes function."""
    print("=" * 70)
    print("TEST: merge_dataframes()")
    print("=" * 70)

    # Load dataframes
    df_binary = pd.read_csv("doge.csv")
    df_technical = pd.read_csv("doge_ti.csv")

    print(f"\nInput DataFrames:")
    print(f"  Binary:    {df_binary.shape[0]} rows, {df_binary.shape[1]} columns")
    print(f"  Technical: {df_technical.shape[0]} rows, {df_technical.shape[1]} columns")

    # Test merge
    df_merged = LSTMMetaheuristicOptimizer.merge_dataframes(
        df_binary, df_technical, verbose=True
    )

    # Verify merged dataframe
    print(f"\nMerged DataFrame:")
    print(f"  Shape: {df_merged.shape}")
    print(f"  Columns: {len(df_merged.columns)}")

    # Check no duplicate columns
    dup_cols = [c for c in df_merged.columns if c.endswith('_dup')]
    assert len(dup_cols) == 0, f"Found duplicate columns: {dup_cols}"
    print(f"  No duplicate columns: PASS")

    # Check required columns exist
    required = ['date', 'tradeable', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df_merged.columns]
    assert len(missing) == 0, f"Missing required columns: {missing}"
    print(f"  Required columns present: PASS")

    # Check row count divisible by period_size (4)
    assert len(df_merged) % 4 == 0, f"Row count {len(df_merged)} not divisible by 4"
    print(f"  Row count divisible by 4: PASS ({len(df_merged)} rows)")

    # Check period consistency (tradeable values within each period)
    consistent_periods = 0
    total_periods = len(df_merged) // 4
    for i in range(0, len(df_merged), 4):
        period = df_merged.iloc[i:i+4]['tradeable'].tolist()
        if len(set(period)) == 1:
            consistent_periods += 1

    consistency_pct = (consistent_periods / total_periods) * 100
    print(f"  Period consistency: {consistency_pct:.1f}% ({consistent_periods}/{total_periods})")

    # Check binary columns from doge.csv exist
    binary_sample = ['TRIX_gs_entry', 'TRIX_gs_exit', 'RSI_gs_entry', 'RSI_gs_exit']
    binary_present = [c for c in binary_sample if c in df_merged.columns]
    print(f"  Binary columns present: {len(binary_present)}/{len(binary_sample)}")

    # Check technical columns from doge_ti.csv exist
    tech_sample = ['TRIX_gs_trix', 'RSI_gs_rsi', 'MACD_gs_macd', 'BOP_gs_bop']
    tech_present = [c for c in tech_sample if c in df_merged.columns]
    print(f"  Technical columns present: {len(tech_present)}/{len(tech_sample)}")

    print("\nmerge_dataframes: ALL TESTS PASSED")
    return df_merged


def test_create_sequences(df_merged: pd.DataFrame):
    """Test create_sequences function for no mixed labels."""
    print("\n" + "=" * 70)
    print("TEST: create_sequences() - No Mixed Labels")
    print("=" * 70)

    # Prepare data using DataPreprocessor
    preprocessor = DataPreprocessor(target_shift=4)

    # Align dataframe first
    df_aligned = DataPreprocessor.align_dataframe(df_merged, period_size=4, verbose=True)

    # Fit and transform
    features, targets = preprocessor.fit_transform(df_aligned)

    print(f"\nPreprocessed Data:")
    print(f"  Features shape: {features.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Unique targets: {np.unique(targets)}")

    # Test different sequence configurations
    configs = [
        {'input_seq_length': 12, 'output_seq_length': 1, 'stride': 4},
        {'input_seq_length': 16, 'output_seq_length': 1, 'stride': 4},
        {'input_seq_length': 24, 'output_seq_length': 1, 'stride': 4},
    ]

    all_passed = True

    for cfg in configs:
        print(f"\n--- Config: input={cfg['input_seq_length']}, output={cfg['output_seq_length']}, stride={cfg['stride']} ---")

        feat_seqs, tgt_seqs = create_sequences(
            features, targets,
            input_seq_length=cfg['input_seq_length'],
            output_seq_length=cfg['output_seq_length'],
            stride=cfg['stride']
        )

        print(f"  Feature sequences: {feat_seqs.shape}")
        print(f"  Target sequences: {tgt_seqs.shape}")

        # Check each target sequence for mixed labels
        mixed_count = 0
        for i, tgt in enumerate(tgt_seqs):
            unique_labels = np.unique(tgt)
            if len(unique_labels) > 1:
                mixed_count += 1
                if mixed_count <= 3:  # Show first 3 examples
                    print(f"    Mixed labels at index {i}: {tgt.flatten()} -> unique: {unique_labels}")

        if mixed_count == 0:
            print(f"  No mixed labels: PASS ({len(tgt_seqs)} sequences checked)")
        else:
            print(f"  No mixed labels: FAIL ({mixed_count}/{len(tgt_seqs)} sequences have mixed labels)")
            all_passed = False

        # Show label distribution
        if cfg['output_seq_length'] == 1:
            hold_count = (tgt_seqs.flatten() == 0).sum()
            trade_count = (tgt_seqs.flatten() == 1).sum()
            print(f"  Label distribution: hold={hold_count}, trade={trade_count}")

    if all_passed:
        print("\ncreate_sequences: ALL TESTS PASSED")
    else:
        print("\ncreate_sequences: SOME TESTS FAILED")

    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("MERGE DATAFRAMES & CREATE SEQUENCES TEST SUITE")
    print("=" * 70)

    # Change to notebook directory
    import os
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Test 1: merge_dataframes
    df_merged = test_merge_dataframes()

    # Test 2: create_sequences
    sequences_passed = test_create_sequences(df_merged)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  merge_dataframes: PASS")
    print(f"  create_sequences (no mixed labels): {'PASS' if sequences_passed else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
