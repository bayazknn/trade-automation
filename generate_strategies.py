import json
import os

def load_operators(data):
    # Mapping logical operators to Python equivalents
    mapping = {
        'lt': '<',
        'gt': '>',
        'lte': '<=',
        'gte': '>=',
        'crosses_above': 'qtpylib.crossed_above',
        'crosses_below': 'qtpylib.crossed_below',
    }
    return mapping

def generate_condition(container, mapping):
    cond = container['condition']
    left = f"dataframe['{cond['left']}']"
    
    # Handle right side being a constant or another column
    if cond.get('right') == 'constant':
        # Constant is a sibling of 'condition', so it's in 'container'
        right = str(container['constant'])
    elif ' * factor' in cond.get('right', ''):
        # Handle "lowerband * factor"
        col_name = cond['right'].replace(' * factor', '')
        factor = container.get('factor', 1.0)
        right = f"dataframe['{col_name}'] * {factor}"
    else:
        right = f"dataframe['{cond['right']}']"
        
    op = cond['operator']
    
    # Check if it is a crossover function or a standard comparison
    if 'cross' in op:
        # qtpylib.crossed_above(dataframe['rsi'], 30)
        fn_name = mapping.get(op, op).replace('()', '')
        return f"{fn_name}({left}, {right})"
    else:
        # Standard comparison: dataframe['rsi'] < 30
        sym = mapping.get(op, op)
        return f"({left} {sym} {right})"

def main():
    json_path = 'user_data/predefined_indicators.json'
    output_dir = 'user_data/strategies/base'
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    # Create empty __init__.py
    with open(os.path.join(output_dir, '__init__.py'), 'w') as f:
        f.write("")

    ops_mapping = load_operators(data)

    print(f"Found {len(data['indicators'])} indicators. Generating strategies...")

    for key, indicator in data['indicators'].items():
        class_name = key
        
        # Base strategy template
        strategy_content = f"""# Source: generated from predefined_indicators.json
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class {class_name}(IStrategy):
    timeframe = '{data['metadata']['timeframe']}'
    
    # Standard ROI and Stoploss
    minimal_roi = {{"0": 0.1, "60": 0.05, "120": 0.0}}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
"""

        # --- Populate Indicators ---
        if 'talib_function' in indicator:
            func = indicator['talib_function']
            params = indicator.get('params', {})
            # params to string: key=value, key2=value2
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            
            # Handle multi-output indicators
            if func == "ta.MACD":
                # MACD returns macd, macdsignal, macdhist
                call_str = f"{func}(dataframe, {param_str})" if param_str else f"{func}(dataframe)"
                strategy_content += f"        macd = {call_str}\n"
                outputs = indicator.get('outputs', ['macd', 'macdsignal', 'macdhist'])
                # Map assumed ta-lib outputs to configured output names
                strategy_content += f"        dataframe['{outputs[0]}'] = macd['macd']\n"
                strategy_content += f"        dataframe['{outputs[1]}'] = macd['macdsignal']\n"
                strategy_content += f"        dataframe['{outputs[2]}'] = macd['macdhist']\n"
            
            elif func == "ta.STOCH":
                # STOCH returns slowk, slowd
                call_str = f"{func}(dataframe, {param_str})" if param_str else f"{func}(dataframe)"
                strategy_content += f"        stoch = {call_str}\n"
                outputs = indicator.get('outputs', ['slowk', 'slowd'])
                strategy_content += f"        dataframe['{outputs[0]}'] = stoch['slowk']\n"
                strategy_content += f"        dataframe['{outputs[1]}'] = stoch['slowd']\n"

            elif func == "ta.STOCHRSI":
                # STOCHRSI returns fastk, fastd
                call_str = f"{func}(dataframe, {param_str})" if param_str else f"{func}(dataframe)"
                strategy_content += f"        stochrsi = {call_str}\n"
                outputs = indicator.get('outputs', ['fastk', 'fastd'])
                strategy_content += f"        dataframe['{outputs[0]}'] = stochrsi['fastk']\n"
                strategy_content += f"        dataframe['{outputs[1]}'] = stochrsi['fastd']\n"
            
            elif func == "ta.BBANDS":
                 # BBANDS returns upperband, middleband, lowerband
                call_str = f"{func}(dataframe, {param_str})" if param_str else f"{func}(dataframe)"
                strategy_content += f"        bbands = {call_str}\n"
                outputs = indicator.get('outputs', ['upperband', 'middleband', 'lowerband'])
                # Assuming simple mapping if names match, otherwise 0->upper, 1->middle, 2->lower
                if len(outputs) == 3:
                    strategy_content += f"        dataframe['{outputs[0]}'] = bbands['upperband']\n"
                    strategy_content += f"        dataframe['{outputs[1]}'] = bbands['middleband']\n"
                    strategy_content += f"        dataframe['{outputs[2]}'] = bbands['lowerband']\n"
                else:
                    # Fallback or specific case
                    pass

            else:
                # Handle other multi-output or single-output functions
                call_str = f"{func}(dataframe, {param_str})" if param_str else f"{func}(dataframe)"
                outputs = indicator.get('outputs', ['dummy'])

                if 'fast_period' in params and 'slow_period' in params:
                    # Handle crossover strategies (SMA, EMA crosses)
                    for out_col in outputs:
                        period = 0
                        if 'fast' in out_col:
                            period = params['fast_period']
                        elif 'slow' in out_col:
                            period = params['slow_period']
                        
                        strategy_content += f"        dataframe['{out_col}'] = {func}(dataframe, timeperiod={period})\n"

                elif 'sma_period' in indicator:
                    # Special case: Indicator + SMA derived from it (e.g. AD, OBV)
                    # First output is the base indicator, second is the SMA
                    base_col = outputs[0]
                    # Generate call without special params (assume empty or default params for base)
                    # We need to be careful not to include keys that ta function doesn't understand? 
                    # Existing logic passed all params. For AD/OBV params is empty usually.
                    
                    real_call = f"{func}(dataframe, {param_str})" if param_str else f"{func}(dataframe)"
                    strategy_content += f"        dataframe['{base_col}'] = {real_call}\n"

                    sma_period = indicator['sma_period']
                    sma_col = f"{base_col}_sma"
                    if len(outputs) > 1:
                        sma_col = outputs[1]
                    
                    strategy_content += f"        dataframe['{sma_col}'] = ta.SMA(dataframe, timeperiod={sma_period}, price='{base_col}')\n"

                elif len(outputs) > 1:
                    strategy_content += f"        res = {call_str}\n"
                    for i, out_col in enumerate(outputs):
                        strategy_content += f"        dataframe['{out_col}'] = res.iloc[:, {i}]\n"
                else:
                    strategy_content += f"        dataframe['{outputs[0]}'] = {call_str}\n"

        elif 'talib_functions' in indicator:
            # Multiple separate functions (e.g., PLUS_DI, MINUS_DI)
            funcs = indicator['talib_functions']
            outputs = indicator['outputs']
            params = indicator.get('params', {})
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            
            for i, func in enumerate(funcs):
                output_col = outputs[i]
                call_str = f"{func}(dataframe, {param_str})" if param_str else f"{func}(dataframe)"
                strategy_content += f"        dataframe['{output_col}'] = {call_str}\n"
        
        strategy_content += "        return dataframe\n\n"
        
        # --- Populate Entry ---
        strategy_content += "    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:\n"
        if 'entry' in indicator and 'condition' in indicator['entry']:
            entry_cond = generate_condition(indicator['entry'], ops_mapping)
            strategy_content += f"        dataframe.loc[\n            {entry_cond},\n            'enter_long'] = 1\n"
        strategy_content += "        return dataframe\n\n"

        # --- Populate Exit ---
        strategy_content += "    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:\n"
        if 'exit' in indicator and 'condition' in indicator['exit']:
            exit_cond = generate_condition(indicator['exit'], ops_mapping)
            strategy_content += f"        dataframe.loc[\n            {exit_cond},\n            'exit_long'] = 1\n"
        strategy_content += "        return dataframe\n"

        # Write to file
        file_path = os.path.join(output_dir, f"{key}.py")
        with open(file_path, 'w') as f:
            f.write(strategy_content)
        
        print(f"Generated {file_path}")

    print("Done.")

if __name__ == '__main__':
    main()
