import json
import os
import sys

# Ensure we can import the generator
sys.path.append(os.getcwd())

from user_data.dynamic_strategy_generator import generate_strategy_file

def main():
    json_path = 'user_data/predefined_indicators.json'
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    indicators = data['indicators']
    
    # Pick sample entries and exits (Multiple)
    entry_names = ['RSI_standard', 'CCI_standard']
    exit_names = ['MACD_standard', 'MFI_standard']
    
    entries = [indicators[name] for name in entry_names]
    exits = [indicators[name] for name in exit_names]
    
    strategy_name = "Test_Dynamic_Gen_Multi"
    
    print(f"Generating strategy {strategy_name}...")
    print(f"Entries: {entry_names}")
    print(f"Exits: {exit_names}")
    
    file_path = generate_strategy_file(strategy_name, entries, exits)
    
    if os.path.exists(file_path):
        print(f"Success! File created at: {file_path}")
        
        # Optional: Print content to verify
        with open(file_path, 'r') as f:
            content = f.read()
            print("\n--- Strategy Content Preview ---")
            print(content[:500] + "\n... (truncated) ...")
            
        print("\nVerifying imports...")
        # Basic check if it's importable (syntax check)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(strategy_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print("Module imported successfully - Syntax is valid.")
        except Exception as e:
            print(f"Syntax Error or Import Error: {e}")
            
    else:
        print("Error: File was not created.")

if __name__ == "__main__":
    main()
