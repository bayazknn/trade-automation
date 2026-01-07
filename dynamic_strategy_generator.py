import os
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

def load_operators():
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

def generate_condition_string(container, mapping):
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

def generate_indicator_code(indicator):
    code = ""
    if 'talib_function' in indicator:
        func = indicator['talib_function']
        params = indicator.get('params', {})
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        
        # Handle multi-output indicators
        if func == "ta.MACD":
            call_str = f"{func}(dataframe, {param_str})" if param_str else f"{func}(dataframe)"
            code += f"        macd = {call_str}\n"
            outputs = indicator.get('outputs', ['macd', 'macdsignal', 'macdhist'])
            code += f"        dataframe['{outputs[0]}'] = macd['macd']\n"
            code += f"        dataframe['{outputs[1]}'] = macd['macdsignal']\n"
            code += f"        dataframe['{outputs[2]}'] = macd['macdhist']\n"
        
        elif func == "ta.STOCH":
            call_str = f"{func}(dataframe, {param_str})" if param_str else f"{func}(dataframe)"
            code += f"        stoch = {call_str}\n"
            outputs = indicator.get('outputs', ['slowk', 'slowd'])
            code += f"        dataframe['{outputs[0]}'] = stoch['slowk']\n"
            code += f"        dataframe['{outputs[1]}'] = stoch['slowd']\n"

        elif func == "ta.STOCHRSI":
            call_str = f"{func}(dataframe, {param_str})" if param_str else f"{func}(dataframe)"
            code += f"        stochrsi = {call_str}\n"
            outputs = indicator.get('outputs', ['fastk', 'fastd'])
            code += f"        dataframe['{outputs[0]}'] = stochrsi['fastk']\n"
            code += f"        dataframe['{outputs[1]}'] = stochrsi['fastd']\n"
        
        elif func == "ta.BBANDS":
            call_str = f"{func}(dataframe, {param_str})" if param_str else f"{func}(dataframe)"
            code += f"        bbands = {call_str}\n"
            outputs = indicator.get('outputs', ['upperband', 'middleband', 'lowerband'])
            if len(outputs) == 3:
                code += f"        dataframe['{outputs[0]}'] = bbands['upperband']\n"
                code += f"        dataframe['{outputs[1]}'] = bbands['middleband']\n"
                code += f"        dataframe['{outputs[2]}'] = bbands['lowerband']\n"

        else:
            outputs = indicator.get('outputs', ['dummy'])
            
            if 'fast_period' in params and 'slow_period' in params:
                 # Handle crossover strategies
                for out_col in outputs:
                    period = 0
                    if 'fast' in out_col:
                        period = params['fast_period']
                    elif 'slow' in out_col:
                        period = params['slow_period']
                    
                    code += f"        dataframe['{out_col}'] = {func}(dataframe, timeperiod={period})\n"

            elif 'sma_period' in indicator:
                 # Indicator + SMA
                base_col = outputs[0]
                real_call = f"{func}(dataframe, {param_str})" if param_str else f"{func}(dataframe)"
                code += f"        dataframe['{base_col}'] = {real_call}\n"

                sma_period = indicator['sma_period']
                sma_col = f"{base_col}_sma"
                if len(outputs) > 1:
                    sma_col = outputs[1]
                
                code += f"        dataframe['{sma_col}'] = ta.SMA(dataframe, timeperiod={sma_period}, price='{base_col}')\n"

            elif len(outputs) > 1:
                call_str = f"{func}(dataframe, {param_str})" if param_str else f"{func}(dataframe)"
                code += f"        res = {call_str}\n"
                for i, out_col in enumerate(outputs):
                    code += f"        dataframe['{out_col}'] = res.iloc[:, {i}]\n"
            else:
                call_str = f"{func}(dataframe, {param_str})" if param_str else f"{func}(dataframe)"
                code += f"        dataframe['{outputs[0]}'] = {call_str}\n"

    elif 'talib_functions' in indicator:
        funcs = indicator['talib_functions']
        outputs = indicator['outputs']
        params = indicator.get('params', {})
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        
        for i, func in enumerate(funcs):
            output_col = outputs[i]
            call_str = f"{func}(dataframe, {param_str})" if param_str else f"{func}(dataframe)"
            code += f"        dataframe['{output_col}'] = {call_str}\n"
            
    return code

def generate_strategy_file(name, entries, exits):
    mapping = load_operators()
    
    # Header
    strategy_content = f"""# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class {name}(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {{"0": 0.1, "60": 0.05, "120": 0.0}}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
"""

    # Populate Indicators
    # De-duplicate indicators? 
    # Or just generate all. Optimally we should avoid re-calculating same indicators.
    # But for now, user asked to "all of indicators in entries and exits will be populated"
    # To avoid redefining same column, we can check outputs.
    generated_outputs = set()
    
    all_indicators = entries + exits
    for ind in all_indicators:
        # Simple check to avoid duplicated blocks if passing same dict object or identical config
        # We can key by 'outputs'
        outputs = tuple(ind.get('outputs', []))
        if outputs and outputs in generated_outputs:
            continue
            
        code_block = generate_indicator_code(ind)
        if code_block:
            strategy_content += code_block
            if outputs:
                generated_outputs.add(outputs)

    strategy_content += "        return dataframe\n\n"

    # Populate Entry
    strategy_content += "    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:\n"
    
    entry_conditions = []
    for ind in entries:
        if 'entry' in ind and 'condition' in ind['entry']:
            cond_str = generate_condition_string(ind['entry'], mapping)
            entry_conditions.append(cond_str)
            
    if entry_conditions:
        # Join with brackets and &
        joined_cond = " & ".join([f"(\n            {c}\n        )" for c in entry_conditions])
        strategy_content += f"        dataframe.loc[\n        {joined_cond},\n        'enter_long'] = 1\n"
        
    strategy_content += "        return dataframe\n\n"

    # Populate Exit
    strategy_content += "    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:\n"
    
    exit_conditions = []
    for ind in exits:
        if 'exit' in ind and 'condition' in ind['exit']:
            cond_str = generate_condition_string(ind['exit'], mapping)
            exit_conditions.append(cond_str)
            
    if exit_conditions:
        # Join with brackets and &
        joined_cond = " & ".join([f"(\n            {c}\n        )" for c in exit_conditions])
        strategy_content += f"        dataframe.loc[\n        {joined_cond},\n        'exit_long'] = 1\n"
        
    strategy_content += "        return dataframe\n"

    # Save file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'strategies')
    file_path = os.path.join(output_dir, f"{name}.py")
    with open(file_path, 'w') as f:
        f.write(strategy_content)
    
    print(f"Generated strategy: {file_path}")
    return file_path
