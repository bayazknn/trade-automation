"""
Strategy Generator - Parses indicator JSON and generates prompts for Claude Code.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from .config import INDICATORS_JSON, PROMPTS_DIR


class StrategyGenerator:
    """
    Generates prompts for Claude Code to create freqtrade strategies.
    """

    def __init__(self, indicators_path: Path = INDICATORS_JSON):
        self.indicators_path = indicators_path
        self.indicators_data = self._load_indicators()
        self.indicator_names = list(self.indicators_data.get("indicators", {}).keys())
        self.operators = self.indicators_data.get("operators", {})

    def _load_indicators(self) -> Dict[str, Any]:
        """Load indicators from JSON file."""
        with open(self.indicators_path, 'r') as f:
            return json.load(f)

    @property
    def n_indicators(self) -> int:
        """Total number of available indicators."""
        return len(self.indicator_names)

    def get_indicator_category(self, indicator_name: str) -> str:
        """Get the category of an indicator."""
        ind_data = self.indicators_data.get("indicators", {}).get(indicator_name, {})
        return ind_data.get("category", "unknown")

    def get_indicator_definitions(self, indicator_names: List[str],
                                     condition_type: str = None) -> Dict[str, Any]:
        """
        Get indicator definitions for given indicator names.

        Args:
            indicator_names: List of indicator names to retrieve
            condition_type: If "entry", only include entry conditions.
                           If "exit", only include exit conditions.
                           If None, include full definitions.

        Returns:
            Dictionary of indicator definitions
        """
        all_indicators = self.indicators_data.get("indicators", {})
        result = {}

        for name in indicator_names:
            if name not in all_indicators:
                continue

            indicator = all_indicators[name].copy()

            if condition_type == "entry":
                # Remove exit key to prevent confusion
                indicator.pop("exit", None)
            elif condition_type == "exit":
                # Remove entry key to prevent confusion
                indicator.pop("entry", None)

            result[name] = indicator

        return result

    def generate_prompt(self, entry_indicators: List[str],
                        exit_indicators: List[str],
                        strategy_name: str,
                        iteration: int,
                        index: int) -> str:
        """
        Generate a complete prompt for Claude Code to create a strategy.

        Args:
            entry_indicators: List of indicator names for entry conditions
            exit_indicators: List of indicator names for exit conditions
            strategy_name: Name for the strategy class
            iteration: Current ACO iteration
            index: Individual index within iteration

        Returns:
            Complete prompt string
        """
        # Get definitions with only relevant condition type to prevent hallucination
        entry_defs = self.get_indicator_definitions(entry_indicators, condition_type="entry")
        exit_defs = self.get_indicator_definitions(exit_indicators, condition_type="exit")

        prompt = f"""You are generating a freqtrade trading strategy. Follow these instructions EXACTLY.

## Task
Create a strategy file and run backtesting. Return results as JSON.

## Strategy Details
- Class Name: `{strategy_name}`
- File Path: `user_data/strategies/{strategy_name.lower()}.py`

## Entry Indicators (ALL must be TRUE - AND logic)
```json
{json.dumps(entry_defs, indent=2)}
```

## Exit Indicators (ALL must be TRUE - AND logic)
```json
{json.dumps(exit_defs, indent=2)}
```

## Operators Reference
```json
{json.dumps(self.operators, indent=2)}
```

## Instructions

1. **Create the strategy file** with this structure:
```python
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import qtpylib.indicators as qtpylib

class {strategy_name}(IStrategy):
    timeframe = '1h'
    stoploss = -0.05
    minimal_roi = {{"0": 0.03, "60": 0.02, "180": 0.01}}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate ALL required indicators here
        ...
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ALL entry conditions combined with &
        dataframe.loc[
            (condition1) & (condition2) & ... & (dataframe['volume'] > 0),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ALL exit conditions combined with &
        dataframe.loc[
            (condition1) & (condition2) & ... & (dataframe['volume'] > 0),
            'exit_long'] = 1
        return dataframe
```

2. **Implement indicators** based on the JSON definitions:
   - Use `ta.INDICATOR(dataframe, ...)` for TA-Lib functions
   - For dual moving averages (SMA, EMA with fast/slow), create both columns
   - For OBV/AD with sma_period, also calculate the SMA of the indicator

3. **Implement entry conditions** using the "entry" field from each indicator:
   - "threshold" type: compare indicator to constant (e.g., `dataframe['rsi'] < 30`)
   - "crossover" type: use qtpylib.crossed_above/crossed_below
   - For factor-based conditions (like BBANDS): `dataframe['close'] < dataframe['lowerband'] * factor`

4. **Implement exit conditions** similarly using the "exit" field

5. **Run backtest** with this exact command (activate venv first):
```bash
powershell -Command "& '.\.venv\Scripts\Activate.ps1'; freqtrade backtesting --strategy {strategy_name} --config user_data/config.json --timerange 20251001- --export none"
```

6. **Return JSON output** with exactly this format:
```json
{{
    "strategy_name": "{strategy_name}",
    "total_profit": <total_profit_abs from backtest>,
    "total_trades": <number of trades>,
    "win_rate": <wins / total_trades>,
    "max_drawdown": <max_drawdown_abs>,
    "sharpe_ratio": <sharpe if available, else 0>,
    "status": "success" or "error",
    "error_message": "<if error>"
}}
```

## Important Notes
- Entry/exit conditions use AND logic (all must be true)
- Always include `dataframe['volume'] > 0` in conditions
- Handle NaN values that occur at the start of indicator calculations
- If backtest produces 0 trades, still return the result with total_profit = 0
- If any error occurs, return status="error" with the error message

RESPOND ONLY WITH THE JSON OUTPUT after completing all steps.
"""
        return prompt

    def generate_prompt_file(self, entry_indicators: List[str],
                             exit_indicators: List[str],
                             strategy_name: str,
                             iteration: int,
                             index: int,
                             output_path: Optional[Path] = None) -> Path:
        """
        Generate prompt and save to file.

        Args:
            entry_indicators: List of indicator names for entry
            exit_indicators: List of indicator names for exit
            strategy_name: Name for the strategy
            iteration: Current iteration
            index: Individual index
            output_path: Optional custom output path

        Returns:
            Path to the saved prompt file
        """
        prompt = self.generate_prompt(
            entry_indicators, exit_indicators,
            strategy_name, iteration, index
        )

        if output_path is None:
            output_path = PROMPTS_DIR / f"prompt_{strategy_name.lower()}.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(prompt)

        return output_path

    def get_heuristic_value(self, indicator_name: str,
                            category_weights: Dict[str, float] = None) -> float:
        """
        Get heuristic value for an indicator (used in ACO probability calculation).

        Args:
            indicator_name: Name of the indicator
            category_weights: Optional weights per category

        Returns:
            Heuristic value (higher = more desirable)
        """
        if category_weights is None:
            category_weights = {
                "momentum": 1.2,
                "overlap": 1.0,
                "volume": 0.8,
                "volatility": 0.9,
            }

        category = self.get_indicator_category(indicator_name)
        return category_weights.get(category, 1.0)
