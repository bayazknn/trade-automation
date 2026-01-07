import subprocess
import sys
import re
import os

def parse_backtest_results(output):
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
            
            # If we see 'Backtesting result caching' or other headers, we passed it
            if "Backtest result caching" in line:
                break
                
            # Stop at the bottom border if visible (Unicode └ or ASCII + at end of table)
            if clean_line.startswith('└'):
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
                        'strategy': st_name,
                        'trade_count': parts[2].strip(),
                        'avg_profit': parts[3].strip(),
                        'tot_profit_usdt': parts[4].strip(),
                        'tot_profit_percent': parts[5].strip(),
                    }
                    data.append(row)
    return data

def run_strategies_backtest(strategy_names, timerange="20240101-20240105", timeframe="1h"):
    """
    Run backtest for a list of strategies and return parsed results.
    
    :param strategy_names: List of string strategy names (class names).
    :param timerange: Timerange string (default 20240101-20240105).
    :param timeframe: Timeframe string (default 1h).
    :return: List of dicts containing backtest results.
    """
    if not strategy_names:
        print("No strategies provided.")
        return []

    # Strategy path - assumes running from project root
    strategy_path = os.path.join("user_data", "strategies")
    
    cmd = [
        sys.executable, "-m", "freqtrade", "backtesting",
        "--strategy-path", strategy_path,
        "--timerange", timerange,
        "--timeframe", timeframe,
        "--strategy-list"
    ] + strategy_names

    print(f"Running backtest for {len(strategy_names)} strategies...")
    # print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8' # Force utf-8
        )
        # Parse output
        parsed_data = parse_backtest_results(result.stdout)
        return parsed_data
            
    except subprocess.CalledProcessError as e:
        print(f"Error running backtest: {e}")
        if e.stdout:
            print("--- Output Capture ---")
            print(e.stdout[-500:]) # Print last 500 chars of output for debug
            print("----------------------")
        return []
    except FileNotFoundError:
        print("freqtrade command not found. Ensure you are in the virtual environment.")
        return []

if __name__ == "__main__":
    # Simple test if run directly
    test_strategies = ["Test_Dynamic_Gen_Multi"]  # Use a known existing strategy
    results = run_strategies_backtest(test_strategies)
    print("Results:", results)
