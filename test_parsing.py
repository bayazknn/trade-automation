
import re

def parse_output(output, strategy_name):
    lines = output.splitlines()
    stats = {
        "strategy_name": strategy_name,
        "total_profit": 0.0,
        "total_trades": 0,
        "win_rate": 0.0,
        "max_drawdown": 0.0
    }
    
    header_map = {}
    normalized_lines = [line.replace('│', '|') for line in lines]
    
    print(f"Parsing for {strategy_name}...")
    
    for line in normalized_lines:
        # Flexible header detection
        if "| Strategy" in line and ("Backtesting from" not in line) and not header_map: 
            # This is likely the header line
            # Normalize headers by reducing multiple spaces to single space
            temp_headers = [h.strip() for h in line.split('|') if h.strip()]
            headers = [" ".join(h.split()) for h in temp_headers]
            
            for i, h in enumerate(headers):
                header_map[h] = i
            print(f"Detected headers (normalized): {header_map}")
        
        # Robust row detection using split
        if "|" in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if not parts: continue
            
            # Check if first valid part is strategy name
            if parts[0] == strategy_name and header_map:
                print(f"Row parts: {parts}")
                
                def parse_val(val_str):
                    return float(re.sub(r'[^\d.-]', '', val_str))

                try:
                    # Trades
                    if "Trades" in header_map:
                            stats["total_trades"] = int(parts[header_map["Trades"]])
                            print(f"Found Trades: {stats['total_trades']}")
                    elif "Buys" in header_map:
                        stats["total_trades"] = int(parts[header_map["Buys"]])
                    
                    # Total Profit
                    if "Tot Profit USDT" in header_map:
                        stats["total_profit"] = parse_val(parts[header_map["Tot Profit USDT"]])
                        print(f"Found Profit: {stats['total_profit']}")
                    
                    # Win Rate
                    if "Win Draw Loss Win%" in header_map:
                            # Matches "Win Draw Loss Win%" even if original had multiple spaces
                            wdl = parts[header_map["Win Draw Loss Win%"]]
                            stats["win_rate"] = parse_val(wdl.split()[-1]) / 100.0
                            print(f"Found Win Rate: {stats['win_rate']}")
                    
                    # Drawdown
                    if "Drawdown" in header_map:
                        dd_str = parts[header_map["Drawdown"]]
                        stats["max_drawdown"] = parse_val(dd_str.split(' ')[0])
                        print(f"Found Drawdown: {stats['max_drawdown']}")
                    
                except Exception as e:
                    print(f"Error parsing: {e}")

    return stats

snippet = """
+--------------------------------------------------------------------------------------------------------------------------------+
| Strategy | Trades | Avg Profit % | Tot Profit USDT | Tot Profit % | Avg Duration |  Win  Draw  Loss  Win% |          Drawdown |
|----------+--------+--------------+-----------------+--------------+--------------+------------------------+-------------------|
|  ACO_9_9 |      8 |         0.26 |           6.558 |         0.66 |      2:15:00 |    3     4     1  37.5 | 17.425 USDT  1.71% |
+--------------------------------------------------------------------------------------------------------------------------------+
"""

stats = parse_output(snippet, "ACO_9_9")
print(stats)
