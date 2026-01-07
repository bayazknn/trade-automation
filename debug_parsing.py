
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
        if "Strategy" in line:
             print(f"DEBUG: Found 'Strategy' in line: {repr(line)}")

        if "| Strategy" in line and ("Backtesting from" not in line) and not header_map: 
            print(f"Header line candidate: {line}")
            temp_headers = [h.strip() for h in line.split('|') if h.strip()]
            headers = [" ".join(h.split()) for h in temp_headers]
            for i, h in enumerate(headers):
                header_map[h] = i
            print(f"Detected headers: {header_map}")
        
        if "|" in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if not parts: continue

            if parts[0] == strategy_name and header_map:
                print(f"Row parts: {parts}")
                
                def parse_val(val_str):
                    return float(re.sub(r'[^\d.-]', '', val_str))

                try:
                    if "Trades" in header_map:
                         stats["total_trades"] = int(parts[header_map["Trades"]])
                         print(f"Trades: {stats['total_trades']}")
                    elif "Buys" in header_map:
                        stats["total_trades"] = int(parts[header_map["Buys"]])
                    
                    if "Tot Profit USDT" in header_map:
                        stats["total_profit"] = parse_val(parts[header_map["Tot Profit USDT"]])
                        print(f"Profit: {stats['total_profit']}")
                    elif "Tot Profit" in header_map:
                        stats["total_profit"] = parse_val(parts[header_map["Tot Profit"]])
                        print(f"Profit (alt): {stats['total_profit']}")
                    
                    if "Win Draw Loss Win%" in header_map:
                         wdl = parts[header_map["Win Draw Loss Win%"]]
                         stats["win_rate"] = parse_val(wdl.split()[-1]) / 100.0
                         print(f"WinRate: {stats['win_rate']}")
                    elif "Win Draw Loss" in header_map:
                        wdl = parts[header_map["Win Draw Loss"]]
                        wins = float(wdl.split('/')[0].strip())
                        stats["win_rate"] = wins / stats["total_trades"] if stats["total_trades"] > 0 else 0
                    
                    found_stats = True
                    break
                except Exception as e:
                    print(f"Error parsing row: {e}")

    return stats

# EXACT output provided by user for ACO_45_22
snippet = """
2026-01-07 01:44:08,747 | DEBUG    | Result for strategy ACO_45_22
                                              BACKTESTING REPORT
+------------------------------------------------------------------------------------------------------------+
|      Pair | Trades | Avg Profit % | Tot Profit USDT | Tot Profit % | Avg Duration |  Win  Draw  Loss  Win% |
|-----------+--------+--------------+-----------------+--------------+--------------+------------------------|
| DOGE/USDT |      8 |         0.78 |          20.756 |         2.08 |      2:00:00 |    4     4     0   100 |
|     TOTAL |      8 |         0.78 |          20.756 |         2.08 |      2:00:00 |    4     4     0   100 |
+------------------------------------------------------------------------------------------------------------+
                                         LEFT OPEN TRADES REPORT
+--------------------------------------------------------------------------------------------------------+
|  Pair | Trades | Avg Profit % | Tot Profit USDT | Tot Profit % | Avg Duration |  Win  Draw  Loss  Win% |
|-------+--------+--------------+-----------------+--------------+--------------+------------------------|
| TOTAL |      0 |          0.0 |           0.000 |          0.0 |         0:00 |    0     0     0     0 |
+--------------------------------------------------------------------------------------------------------+
                                                ENTER TAG STATS
+-------------------------------------------------------------------------------------------------------------+
| Enter Tag | Entries | Avg Profit % | Tot Profit USDT | Tot Profit % | Avg Duration |  Win  Draw  Loss  Win% |
|-----------+---------+--------------+-----------------+--------------+--------------+------------------------|
|     OTHER |       8 |         0.78 |          20.756 |         2.08 |      2:00:00 |    4     4     0   100 |
|     TOTAL |       8 |         0.78 |          20.756 |         2.08 |      2:00:00 |    4     4     0   100 |
+-------------------------------------------------------------------------------------------------------------+
                                               EXIT REASON STATS
+-------------------------------------------------------------------------------------------------------------+
| Exit Reason | Exits | Avg Profit % | Tot Profit USDT | Tot Profit % | Avg Duration |  Win  Draw  Loss  Win% |
|-------------+-------+--------------+-----------------+--------------+--------------+------------------------|
|         roi |     8 |         0.78 |          20.756 |         2.08 |      2:00:00 |    4     4     0   100 |
|       TOTAL |     8 |         0.78 |          20.756 |         2.08 |      2:00:00 |    4     4     0   100 |
+-------------------------------------------------------------------------------------------------------------+
                                                      MIXED TAG STATS
+--------------------------------------------------------------------------------------------------------------------------+
| Enter Tag | Exit Reason | Trades | Avg Profit % | Tot Profit USDT | Tot Profit % | Avg Duration |  Win  Draw  Loss  Win% |
|-----------+-------------+--------+--------------+-----------------+--------------+--------------+------------------------|
|           |         roi |      8 |         0.78 |          20.756 |         2.08 |      2:00:00 |    4     4     0   100 |
|     TOTAL |             |      8 |         0.78 |          20.756 |         2.08 |      2:00:00 |    4     4     0   100 |
+--------------------------------------------------------------------------------------------------------------------------+
                         SUMMARY METRICS
+----------------------------------------------------------------+
| Metric                        | Value                          |
|-------------------------------+--------------------------------|
| Backtesting from              | 2025-10-01 00:00:00            |
| Backtesting to                | 2026-01-06 20:00:00            |
| Trading Mode                  | Spot                           |
| Max open trades               | 1                              |
|                               |                                |
| Total/Daily Avg Trades        | 8 / 0.08                       |
| Starting balance              | 1000 USDT                      |
| Final balance                 | 1020.756 USDT                  |
| Absolute profit               | 20.756 USDT                    |
| Total profit %                | 2.08%                          |
| CAGR %                        | 8.04%                          |
| Sortino                       | -100.00                        |
| Sharpe                        | 1.18                           |
| Calmar                        | -100.00                        |
| SQN                           | 1.99                           |
| Profit factor                 | 0.00                           |
| Expectancy (Ratio)            | 2.59 (100.00)                  |
| Avg. daily profit             | 0.214 USDT                     |
| Avg. stake amount             | 332.412 USDT                   |
| Total trade volume            | 5350.033 USDT                  |
|                               |                                |
| Best Pair                     | DOGE/USDT 2.08%                |
| Worst Pair                    | DOGE/USDT 2.08%                |
| Best trade                    | DOGE/USDT 2.97%                |
| Worst trade                   | DOGE/USDT 0.00%                |
| Best day                      | 9.859 USDT                     |
| Worst day                     | 0 USDT                         |
| Days win/draw/lose            | 4 / 79 / 0                     |
| Min/Max/Avg. Duration Winners | 0d 02:00 / 0d 02:00 / 0d 02:00 |
| Min/Max/Avg. Duration Losers  | 0d 00:00 / 0d 00:00 / 0d 00:00 |
| Max Consecutive Wins / Loss   | 2 / 1                          |
| Rejected Entry signals        | 0                              |
| Entry/Exit Timeouts           | 0 / 0                          |
|                               |                                |
| Min balance                   | 1000 USDT                      |
| Max balance                   | 1020.756 USDT                  |
| Max % of account underwater   | 0.00%                          |
| Absolute drawdown             | 0 USDT (0.00%)                 |
| Drawdown duration             | 0 days 00:00:00                |
| Profit at drawdown start      | 0 USDT                         |
| Profit at drawdown end        | 0 USDT                         |
| Drawdown start                | 2025-10-10 00:00:00            |
| Drawdown end                  | 2025-10-10 00:00:00            |
| Market change                 | -36.63%                        |
+----------------------------------------------------------------+

Backtested 2025-10-01 00:00:00 -> 2026-01-06 20:00:00 | Max open trades : 1
                                                       STRATEGY SUMMARY
+----------------------------------------------------------------------------------------------------------------------------+
|  Strategy | Trades | Avg Profit % | Tot Profit USDT | Tot Profit % | Avg Duration |  Win  Draw  Loss  Win% |      Drawdown |
|-----------+--------+--------------+-----------------+--------------+--------------+------------------------+---------------|
| ACO_45_22 |      8 |         0.78 |          20.756 |         2.08 |      2:00:00 |    4     4     0   100 | 0 USDT  0.00% |
+----------------------------------------------------------------------------------------------------------------------------+
"""

stats = parse_output(snippet, "ACO_45_22")
print(stats)
