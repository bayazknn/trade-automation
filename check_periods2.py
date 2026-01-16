"""Deeper analysis of period structure."""
import pandas as pd
import numpy as np

df_binary = pd.read_csv('notebooks/doge.csv')
df_technical = pd.read_csv('notebooks/doge_ti.csv')

print('=== Full Data Overview ===')
print(f'Total rows: {len(df_binary)}')
print(f'Rows with period_id: {df_binary["period_id"].notna().sum()}')
print(f'Rows without period_id (NaN): {df_binary["period_id"].isna().sum()}')

print(f'\n=== Tradeable Distribution ===')
print(df_binary['tradeable'].value_counts())

# Check if rows with period_id correspond to 'trade'
print(f'\n=== Period_id vs Tradeable ===')
with_pid = df_binary[df_binary['period_id'].notna()]
without_pid = df_binary[df_binary['period_id'].isna()]
print(f'Rows WITH period_id - tradeable distribution:')
print(with_pid['tradeable'].value_counts())
print(f'\nRows WITHOUT period_id - tradeable distribution:')
print(without_pid['tradeable'].value_counts())

# Check technical indicators NaN
print(f'\n=== Technical NaN Analysis ===')
metadata_cols = ['date', 'signal', 'signal_pct_change', 'period_id', 'tradeable']
feature_cols = [c for c in df_technical.columns if c not in metadata_cols]
nan_per_row = df_technical[feature_cols].isna().sum(axis=1)
print(f'Rows with 0 NaN in features: {(nan_per_row == 0).sum()}')
print(f'Rows with >0 NaN in features: {(nan_per_row > 0).sum()}')

# Check overlap: rows with period_id AND no NaN in features
valid_features = nan_per_row == 0
has_pid = df_binary['period_id'].notna()
print(f'\n=== Overlap Analysis ===')
print(f'Rows with period_id AND valid features: {(has_pid & valid_features).sum()}')
print(f'Rows with period_id BUT invalid features: {(has_pid & ~valid_features).sum()}')
print(f'Rows without period_id AND valid features: {(~has_pid & valid_features).sum()}')
print(f'Rows without period_id AND invalid features: {(~has_pid & ~valid_features).sum()}')

# Where are the valid feature rows?
valid_idx = np.where(valid_features)[0]
print(f'\n=== Valid Feature Row Indices ===')
print(f'First 20 valid indices: {valid_idx[:20].tolist()}')
print(f'Last 20 valid indices: {valid_idx[-20:].tolist()}')
print(f'Min: {valid_idx.min()}, Max: {valid_idx.max()}')

# Check if valid rows are contiguous
if len(valid_idx) > 1:
    gaps = np.diff(valid_idx)
    if np.all(gaps == 1):
        print(f'Valid rows are CONTIGUOUS')
    else:
        print(f'Valid rows have GAPS')
        gap_counts = pd.Series(gaps).value_counts().sort_index()
        print(f'Gap distribution: {gap_counts.head(10).to_dict()}')
