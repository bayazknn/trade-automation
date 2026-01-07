import subprocess
import json
import logging
import time
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from .config import ACO_CONFIG, STRATEGIES_DIR, FREQTRADE_DIR, USER_DATA_DIR
from .individual import Individual
from .strategy_generator import StrategyGenerator

# Import dynamic generator
try:
    from dynamic_strategy_generator import generate_strategy_file
except ImportError:
    # Fallback if user_data is not in path directly but we are in it
    sys.path.append(str(Path(__file__).parent.parent))
    from dynamic_strategy_generator import generate_strategy_file

logger = logging.getLogger(__name__)

# Separate logger for detailed local logs (reusing the name for consistency)
claude_logger = logging.getLogger("claude_sessions")


def setup_claude_logger():
    """Setup separate log file and console output for session monitoring."""
    if claude_logger.handlers:
        return  # Already configured

    log_dir = USER_DATA_DIR / "aco_logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"evaluation_logs_{timestamp}.log"

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler - all levels
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    claude_logger.addHandler(file_handler)

    # Console handler - INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    claude_logger.addHandler(console_handler)

    claude_logger.setLevel(logging.DEBUG)

    logger.info(f"Evaluation log: {log_file}")


class Evaluator:
    """
    Evaluates individuals by generating strategies locally and running freqtrade backtests.
    """

    def __init__(self, strategy_generator: StrategyGenerator,
                 config: Dict[str, Any] = None):
        self.generator = strategy_generator
        self.config = config or ACO_CONFIG
        self.evaluation_count = 0
        self.successful_evaluations = 0
        self.failed_evaluations = 0

        # Setup session logging
        setup_claude_logger()

    def evaluate(self, individual: Individual) -> float:
        """
        Evaluate an individual by generating strategy and running backtest.

        Args:
            individual: The individual to evaluate

        Returns:
            Fitness value (total_profit_abs or penalty)
        """
        self.evaluation_count += 1

        # Decode individual if not already done
        if not individual.entry_indicators or not individual.exit_indicators:
            individual.decode(self.generator.indicator_names)

        # Validate individual
        if not individual.is_valid(
            min_entry=self.config["min_entry_indicators"],
            max_entry=self.config["max_entry_indicators"],
            min_exit=self.config["min_exit_indicators"],
            max_exit=self.config["max_exit_indicators"]
        ):
            logger.warning(f"{individual.strategy_name}: Invalid solution (constraints violated)")
            individual.fitness = self.config["penalty_score"]
            return individual.fitness

        try:
            # 1. Generate Strategy Locally
            self._generate_strategy_locally(individual)

            # 2. Run Backtest Locally
            result = self._run_backtest_locally(individual.strategy_name)

        except Exception as e:
            logger.error(f"Evaluation failed for {individual.strategy_name}: {e}", exc_info=True)
            result = {
                "status": "error",
                "error_message": str(e)
            }

        # Parse result and extract fitness
        individual.fitness = self._extract_fitness(result, individual)

        if individual.fitness > self.config["penalty_score"]:
            self.successful_evaluations += 1
        else:
            self.failed_evaluations += 1

        return individual.fitness

    def _generate_strategy_locally(self, individual: Individual) -> str:
        """
        Generate the strategy file using dynamic_strategy_generator.
        """
        # Get indicator definitions from JSON
        entry_defs = self.generator.get_indicator_definitions(
            individual.entry_indicators, condition_type="entry"
        )
        exit_defs = self.generator.get_indicator_definitions(
            individual.exit_indicators, condition_type="exit"
        )

        # Convert dicts to lists as expected by generator
        entries_list = list(entry_defs.values())
        exits_list = list(exit_defs.values())

        claude_logger.info(f"Generating strategy file for {individual.strategy_name}...")
        
        # Generate the file
        file_path = generate_strategy_file(
            individual.strategy_name,
            entries_list,
            exits_list
        )
        
        return str(file_path)

    def _run_backtest_locally(self, strategy_name: str) -> Dict[str, Any]:
        """
        Run freqtrade backtest for the strategy and parse results.
        """
        claude_logger.info(f"Running backtest for {strategy_name}...")
        start_time = time.time()

        # Build command
        # Using a fixed timerange for optimization speed, or from config
        # Assuming freqtrade is installed in the current env
        cmd = [
            sys.executable, "-m", "freqtrade", "backtesting",
            "--strategy", strategy_name,
            "--config", "user_data/config.json",
            "--timerange", self.config["timerange"],
            "--timeframe", self.config["timeframe"]
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(FREQTRADE_DIR),
                encoding='utf-8'
            )
            
            elapsed = time.time() - start_time
            claude_logger.info(f"Backtest finished in {elapsed:.1f}s")
            
            if result.returncode != 0:
                claude_logger.warning(f"Backtest returned code {result.returncode}. Stderr: {result.stderr[-1000:]}")
                # Try to parse anyway if we have output
                if "STRATEGY SUMMARY" not in result.stdout:
                    return {"status": "error", "error_message": result.stderr}
            
            # Parse output
            return self._parse_backtest_output(result.stdout, strategy_name)

        except Exception as e:
            claude_logger.error(f"Backtest execution error: {e}")
            return {"status": "error", "error_message": str(e)}

    def _parse_backtest_output(self, output: str, strategy_name: str) -> Dict[str, Any]:
        """
        Parse stdout from freqtrade backtesting.
        Looking for the Summary Line.
        """
        # Regex to find the row in the table
        # Format example:
        # | Strategy | ... | Tot Profit USDT | ... |
        # | MyStrat  | ... | 120.5           | ... |
        
        # We look for the line starting with | strategy_name
        lines = output.splitlines()
        
        found_stats = False
        stats = {
            "strategy_name": strategy_name,
            "status": "success",
            "total_profit": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "max_drawdown": 0.0
        }

        # Detect headers to find column indices? 
        # But simpler regex might work if table format is standard.
        # Columns often: Strategy, Entries, Avg Profit %, Tot Profit USDT, Tot Profit %, ... Win Rate, ... 
        
        for line in lines:
            if f"| {strategy_name}" in line:
                # Found the line
                parts = [p.strip() for p in line.split('|') if p.strip()]
                # Typically: [Name, Entries, Avg Profit %, Tot Profit USDT, Tot Profit %, ..., Win Rate, ...]
                # Be careful with indices.
                # Let's try to match patterns or use fixed indices if standard.
                # Freqtrade standard columns:
                # Strategy, Buys, Avg Profit %, Cum Profit, Tot Profit USDT, Tot Profit %, ..., Win Rate, Max Drawdown...
                
                # Check headers first?
                # Let's assume standard freqtrade output.
                # parts[0] = Strategy
                # parts[1] = Buys (Trades)
                # parts[2] = Avg Profit %
                # parts[3] = Cum Profit (abs) ? No, depends on version.
                # Usually: Strategy | Entries | Avg Profit % | Cum Profit | Tot Profit USDT | Tot Profit % | ...
                
                # Let's try to parse based on values.
                try:
                    stats["total_trades"] = int(parts[1])
                    
                    # Tot Profit USDT is usually index 4 or 5 depending on config
                    # Let's regex the line for numbers.
                    
                    # Alternative: Extract columns based on header mapping if possible.
                    # But since we don't have header handy here easily (it's earlier in lines), let's guess based on parts.
                    
                    # parts: ['MyStrat', '10', '1.5', '150.0', '15.0', ...]
                    # We want Tot Profit USDT (Absolute profit).
                    # Usually it's the one with USDT or currency.
                    # If we can't be sure, we can rely on indices.
                    # Index 4 is usually Tot Profit USDT.
                    
                    # Let's try to be safer:
                    # Look for the header line earlier
                    pass
                except:
                    pass

        if not stats["total_trades"] and found_stats:
             # Double check if we parsed correctly
             pass

        # Robust parsing with header detection
        header_map = {}
        # Normalize separators to pipe
        normalized_lines = [line.replace('│', '|') for line in lines]
        

        for line in normalized_lines:
            # Flexible header detection
            # Check for Strategy AND Trades to confirm it's the header row
            if "| Strategy" in line or "|  Strategy" in line: # Try valid combos or use key words
               if "Trades" in line and "Profit" in line and not header_map:
                    # This is likely the header line
                    # Normalize headers by reducing multiple spaces to single space
                    temp_headers = [h.strip() for h in line.split('|') if h.strip()]
                    headers = [" ".join(h.split()) for h in temp_headers]
                    
                    for i, h in enumerate(headers):
                        header_map[h] = i
                    claude_logger.debug(f"Detected headers: {header_map}")
            
            # Robust row detection using split
            if "|" in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if not parts: continue

                if parts[0] == strategy_name and header_map:
                    claude_logger.debug(f"Row parts: {parts}")
                    
                    # Helper to clean currency symbols
                    def parse_val(val_str):
                        return float(re.sub(r'[^\d.-]', '', val_str))

                    try:
                        # Trades
                        if "Trades" in header_map:
                             stats["total_trades"] = int(parts[header_map["Trades"]])
                        elif "Buys" in header_map:
                            stats["total_trades"] = int(parts[header_map["Buys"]])
                        elif "Entries" in header_map:
                            stats["total_trades"] = int(parts[header_map["Entries"]])
                            
                        # Total Profit
                        if "Tot Profit USDT" in header_map:
                            stats["total_profit"] = parse_val(parts[header_map["Tot Profit USDT"]])
                        elif "Tot Profit" in header_map:
                            stats["total_profit"] = parse_val(parts[header_map["Tot Profit"]])
                        elif "Profit" in header_map: # Fallback
                             stats["total_profit"] = parse_val(parts[header_map["Profit"]])
                            
                        # Win Rate
                        if "Win Draw Loss Win%" in header_map:
                             # Format: 3 4 1 37.5
                             wdl = parts[header_map["Win Draw Loss Win%"]]
                             # Last number is win rate
                             stats["win_rate"] = parse_val(wdl.split()[-1]) / 100.0
                        elif "Win Draw Loss" in header_map:
                            # Format: 10 / 2 / 0
                            wdl = parts[header_map["Win Draw Loss"]]
                            wins = float(wdl.split('/')[0].strip())
                            stats["win_rate"] = wins / stats["total_trades"] if stats["total_trades"] > 0 else 0
                        
                        # Drawdown
                        if "Drawdown" in header_map:
                            # 17.425 USDT 1.71%
                            dd_str = parts[header_map["Drawdown"]]
                            # Extract first number
                            stats["max_drawdown"] = parse_val(dd_str.split(' ')[0])
                        
                        found_stats = True
                        break
                    except Exception as e:
                        claude_logger.warning(f"Error parsing row: {e}")
                        claude_logger.debug(f"Row parts: {parts}")

        if not found_stats:
             # Fallback: Maybe 0 trades?
             if "No trades found" in output or "No trades made" in output:
                 stats["total_trades"] = 0
             else:
                 claude_logger.warning(f"Could not parse stats from output for {strategy_name}")
                 # Log output for debug
                 claude_logger.debug(output)

        return stats

    def _extract_fitness(self, result: Dict[str, Any], individual: Individual) -> float:
        """
        Extract fitness value from evaluation result.
        """
        # Store full result
        individual.backtest_result = result

        # Check for errors
        if result.get("status") == "error":
            return self.config["penalty_score"]

        # Extract primary fitness metric
        fitness_metric = self.config.get("fitness_metric", "total_profit")
        fitness = result.get(fitness_metric, 0.0)

        # Log success
        trades = result.get("total_trades", 0)
        win_rate = result.get("win_rate", 0)
        claude_logger.info(f"{individual.strategy_name}: fitness={fitness:.2f}, trades={trades}, win_rate={win_rate:.2%}")

        return float(fitness)

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        return {
            "total_evaluations": self.evaluation_count,
            "successful": self.successful_evaluations,
            "failed": self.failed_evaluations,
            "success_rate": self.successful_evaluations / max(self.evaluation_count, 1)
        }
