"""Show actual tradeable pattern in real data."""
import pandas as pd

df = pd.read_csv('notebooks/doge.csv')

# Show tradeable values for first 100 rows after NaN warmup
print('=== Tradeable values (rows 63-162) ===')
for i in range(63, 163, 4):
    chunk = df.iloc[i:i+4]['tradeable'].tolist()
    consistent = len(set(chunk)) == 1
    marker = '✓' if consistent else '✗'
    print(f'Rows {i:4d}-{i+3:4d}: {chunk} {marker}')

# Count how many 4-row groups are consistent
print('\n=== 4-row group consistency (from row 63) ===')
consistent_count = 0
inconsistent_count = 0
for i in range(63, len(df) - 3, 4):
    chunk = df.iloc[i:i+4]['tradeable'].tolist()
    if len(set(chunk)) == 1:
        consistent_count += 1
    else:
        inconsistent_count += 1

print(f'Consistent 4-row groups: {consistent_count}')
print(f'Inconsistent 4-row groups: {inconsistent_count}')
print(f'Total 4-row groups: {consistent_count + inconsistent_count}')
