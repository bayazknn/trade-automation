"""Check period_id structure and tradeable consistency."""
import pandas as pd
import numpy as np

df_binary = pd.read_csv('notebooks/doge.csv')

print('=== Period ID Analysis ===')
print(f'period_id unique values: {df_binary["period_id"].nunique()}')
print(f'period_id NaN count: {df_binary["period_id"].isna().sum()}')

# Check period sizes
period_counts = df_binary.groupby('period_id').size()
print(f'\nPeriod size distribution:')
print(period_counts.value_counts().sort_index())

# Check tradeable consistency per period (excluding NaN period_id)
print(f'\n=== Tradeable Consistency per Period ===')
df_valid = df_binary[df_binary['period_id'].notna()]
period_tradeable = df_valid.groupby('period_id')['tradeable'].nunique()
consistent = (period_tradeable == 1).sum()
inconsistent = (period_tradeable > 1).sum()
print(f'Consistent periods (1 unique value): {consistent}')
print(f'Inconsistent periods (>1 unique value): {inconsistent}')

# Show some inconsistent periods if any
if inconsistent > 0:
    incon_pids = period_tradeable[period_tradeable > 1].index[:5]
    print(f'\nFirst 5 inconsistent periods:')
    for pid in incon_pids:
        rows = df_binary[df_binary['period_id'] == pid]
        print(f'  Period {int(pid)}:')
        print(f'    tradeable values: {rows["tradeable"].tolist()}')

# Check first few periods
print(f'\nFirst 10 periods:')
for pid in sorted(df_binary['period_id'].dropna().unique())[:10]:
    rows = df_binary[df_binary['period_id'] == pid]
    tradeable_vals = rows['tradeable'].unique()
    print(f'  Period {int(pid)}: {len(rows)} rows, tradeable={tradeable_vals}')
