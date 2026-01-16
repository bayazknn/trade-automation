"""Find optimal starting position for 4-row period alignment."""
import pandas as pd

df = pd.read_csv('notebooks/doge.csv')

# First valid row after NaN warmup
first_valid = 63

# Try different starting offsets (0, 1, 2, 3)
print('=== Testing different start offsets ===')
for offset in range(4):
    start = first_valid + offset
    consistent_count = 0
    inconsistent_count = 0

    for i in range(start, len(df) - 3, 4):
        chunk = df.iloc[i:i+4]['tradeable'].tolist()
        if len(set(chunk)) == 1:
            consistent_count += 1
        else:
            inconsistent_count += 1

    total = consistent_count + inconsistent_count
    pct = consistent_count / total * 100 if total > 0 else 0
    print(f'Offset {offset} (start row {start}): {consistent_count}/{total} consistent ({pct:.1f}%)')

    # Show first few groups for this offset
    print(f'  First 5 groups:')
    for i, idx in enumerate(range(start, min(start + 20, len(df) - 3), 4)):
        chunk = df.iloc[idx:idx+4]['tradeable'].tolist()
        marker = '✓' if len(set(chunk)) == 1 else '✗'
        print(f'    {idx}-{idx+3}: {chunk} {marker}')
    print()
