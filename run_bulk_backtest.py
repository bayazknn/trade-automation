import os
import subprocess
import glob
import sys

def main():
    strategies_dir = os.path.join('user_data', 'strategies')
    if not os.path.exists(strategies_dir):
        print(f"Directory not found: {strategies_dir}")
        return

    # Get all .py files excluding __init__.py
    files = glob.glob(os.path.join(strategies_dir, "*.py"))
    strategy_names = []
    for f in files:
        basename = os.path.basename(f)
        if basename == "__init__.py":
            continue
        strategy_names.append(basename[:-3]) # Remove .py extension

    if not strategy_names:
        print("No strategies found.")
        return

    print(f"Found {len(strategy_names)} strategies.")
    
    # Construct command
    # Using relative path for strategy-path
    cmd = [
        sys.executable, "-m", "freqtrade", "backtesting",
        "--strategy-path", "user_data/strategies",
        "--timerange", "20240101-20240105",
        "--timeframe", "1h",
        "--strategy-list"
    ] + strategy_names

    print("Running command:", " ".join(cmd))
    
    # Run the command with output capture
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8' # Force utf-8
        )
        print(result.stdout)
        
        # Parse results
        print("\n" + "="*50)
        print("PARSED RESULTS")
        print("="*50)
        
        parsed_data = parse_results(result.stdout)
        for row in parsed_data:
            print(row)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running backtest: {e}")
        if e.stdout:
            print(e.stdout)
    except FileNotFoundError:
        print("freqtrade command not found. Ensure you are in the virtual environment.")

import re

def parse_results(output):
    lines = output.splitlines()
    data = []
    in_summary = False
    
    for line in lines:
        # Detect start of summary table
        if "STRATEGY SUMMARY" in line:
            in_summary = True
            continue
            
        if in_summary:
            clean_line = line.strip()
            # Stop at the bottom border (Unicode └ or ASCII + at end of table)
            # ASCII table bottom is +--------+
            # But the header separator is also +--------+
            # We can distinguish by checking if we have gathered data?
            # Or just check if line contains table chars
            
            # If we see 'Backtesting result caching' or other headers, we passed it
            if "Backtest result caching" in line:
                break

            # Parse data rows (starting with │ or |)
            if clean_line.startswith('│') or clean_line.startswith('|'):
                parts = re.split(r'[│|]', clean_line)
                # Structure: | Strategy | Trades | ...
                if len(parts) >= 6:
                    st_name = parts[1].strip()
                    # Skip headers and separators
                    if st_name == 'Strategy' or '---' in st_name:
                        continue
                        
                    row = {
                        'Strategy': st_name,
                        'Trades': parts[2].strip(),
                        'Avg Profit %': parts[3].strip(),
                        'Tot Profit USDT': parts[4].strip(),
                        'Tot Profit %': parts[5].strip(),
                    }
                    data.append(row)
    return data

if __name__ == "__main__":
    main()
