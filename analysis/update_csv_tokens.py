#!/usr/bin/env python3
"""
Update the existing CSV file with correct token values from source files
"""

import json
import glob
import pandas as pd

# Pricing data
PRICING_DATA = [
    {"MODEL": "gemini-1.5-flash", "COST_PER_INPUT_TOKEN": 0.000075, "COST_PER_OUTPUT_TOKEN": 0.0003},
    {"MODEL": "gemini-1.5-flash-002", "COST_PER_INPUT_TOKEN": 0.000075, "COST_PER_OUTPUT_TOKEN": 0.0003},
    {"MODEL": "gemini-1.5-flash-8b", "COST_PER_INPUT_TOKEN": 0.000075, "COST_PER_OUTPUT_TOKEN": 0.0003},
    {"MODEL": "gemini-1.5-flash-latest", "COST_PER_INPUT_TOKEN": 0.000075, "COST_PER_OUTPUT_TOKEN": 0.0003},
    {"MODEL": "gemini-1.5-pro", "COST_PER_INPUT_TOKEN": 0.00125, "COST_PER_OUTPUT_TOKEN": 0.005},
    {"MODEL": "gemini-1.5-pro-002", "COST_PER_INPUT_TOKEN": 0.00125, "COST_PER_OUTPUT_TOKEN": 0.005},
    {"MODEL": "gemini-1.5-pro-latest", "COST_PER_INPUT_TOKEN": 0.00125, "COST_PER_OUTPUT_TOKEN": 0.005},
    {"MODEL": "gemini-2.0-flash", "COST_PER_INPUT_TOKEN": 0.0001, "COST_PER_OUTPUT_TOKEN": 0.0004},
    {"MODEL": "gemini-2.0-flash-001", "COST_PER_INPUT_TOKEN": 0.0001, "COST_PER_OUTPUT_TOKEN": 0.0004},
    {"MODEL": "gemini-2.0-flash-lite", "COST_PER_INPUT_TOKEN": 0.000075, "COST_PER_OUTPUT_TOKEN": 0.0003},
    {"MODEL": "gemini-2.0-flash-lite-001", "COST_PER_INPUT_TOKEN": 0.000075, "COST_PER_OUTPUT_TOKEN": 0.0003},
    {"MODEL": "gemini-2.5-flash-lite", "COST_PER_INPUT_TOKEN": 0.0001, "COST_PER_OUTPUT_TOKEN": 0.0004},
    {"MODEL": "gemini-2.5-pro", "COST_PER_INPUT_TOKEN": 0.00125, "COST_PER_OUTPUT_TOKEN": 0.0025},
    {"MODEL": "gemma-3-12b-it", "COST_PER_INPUT_TOKEN": 0, "COST_PER_OUTPUT_TOKEN": 0},
    {"MODEL": "gemma-3-1b-it", "COST_PER_INPUT_TOKEN": 0, "COST_PER_OUTPUT_TOKEN": 0},
    {"MODEL": "gemma-3-27b-it", "COST_PER_INPUT_TOKEN": 0, "COST_PER_OUTPUT_TOKEN": 0},
    {"MODEL": "gemma-3-4b-it", "COST_PER_INPUT_TOKEN": 0, "COST_PER_OUTPUT_TOKEN": 0},
    {"MODEL": "gemma-3n-e2b-it", "COST_PER_INPUT_TOKEN": 0, "COST_PER_OUTPUT_TOKEN": 0},
    {"MODEL": "gemma-3n-e4b-it", "COST_PER_INPUT_TOKEN": 0, "COST_PER_OUTPUT_TOKEN": 0},
    {"MODEL": "gpt-4o", "COST_PER_INPUT_TOKEN": 0.0025, "COST_PER_OUTPUT_TOKEN": 0.01},
    {"MODEL": "gpt-5", "COST_PER_INPUT_TOKEN": 0.00125, "COST_PER_OUTPUT_TOKEN": 0.01},
    {"MODEL": "grok-2-1212", "COST_PER_INPUT_TOKEN": 0.002, "COST_PER_OUTPUT_TOKEN": 0.01},
    {"MODEL": "grok-2-vision-1212", "COST_PER_INPUT_TOKEN": 0.002, "COST_PER_OUTPUT_TOKEN": 0.01},
    {"MODEL": "grok-3", "COST_PER_INPUT_TOKEN": 0.002, "COST_PER_OUTPUT_TOKEN": 0.01},
    {"MODEL": "gpt-3.5-turbo", "COST_PER_INPUT_TOKEN": 0, "COST_PER_OUTPUT_TOKEN": 0},
    {"MODEL": "gpt-4", "COST_PER_INPUT_TOKEN": 0, "COST_PER_OUTPUT_TOKEN": 0}
]

PRICING_MAP = {item['MODEL']: item for item in PRICING_DATA}

def get_model_pricing(model_name):
    """Get pricing for a model by extracting model name from full identifier"""
    if '_' in model_name:
        model_short = model_name.split('_', 1)[1]
    else:
        model_short = model_name
    
    if model_short in PRICING_MAP:
        return PRICING_MAP[model_short]
    
    for pricing_model, pricing in PRICING_MAP.items():
        if pricing_model in model_short:
            return pricing
    
    return {"COST_PER_INPUT_TOKEN": 0, "COST_PER_OUTPUT_TOKEN": 0}

def get_correct_token_data():
    """Get correct token data from source files"""
    token_data = {}
    
    model_files = sorted([f for f in glob.glob("output/phase4_balanced_refinement/*_balanced.json") 
                         if not f.endswith('comprehensive_results.json')])
    
    for model_file in model_files:
        model_name = model_file.split('/')[-1].replace('_balanced.json', '')
        
        try:
            with open(model_file, 'r') as f:
                data = json.load(f)
            
            usage = data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            
            # Use calculated total (prompt + completion)
            total_tokens = prompt_tokens + completion_tokens
            
            # Get pricing
            pricing = get_model_pricing(model_name)
            cost_per_input = pricing['COST_PER_INPUT_TOKEN']
            cost_per_output = pricing['COST_PER_OUTPUT_TOKEN']
            
            # Calculate token cost
            token_cost = (prompt_tokens * cost_per_input) + (completion_tokens * cost_per_output)
            
            token_data[model_name] = {
                'INPUT_TOKENS': prompt_tokens,
                'OUTPUT_TOKENS': completion_tokens,
                'TOTAL_TOKENS': total_tokens,
                'COST_PER_INPUT_TOKEN': cost_per_input,
                'COST_PER_OUTPUT_TOKEN': cost_per_output,
                'TOKEN_COST': token_cost
            }
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    return token_data

def main():
    """Update the CSV with correct token data"""
    
    print("=== UPDATING CSV WITH CORRECT TOKEN DATA ===\n")
    
    # Read current CSV
    csv_file = "output/step2_client_analysis/step2_all_models_FINAL_CORRECT.csv"
    df = pd.read_csv(csv_file)
    
    # Get correct token data
    token_data = get_correct_token_data()
    
    print("Updating token data for models:")
    for i, row in df.iterrows():
        model_name = row['Model Name']
        if model_name in token_data:
            data = token_data[model_name]
            
            print(f"  {model_name}:")
            print(f"    INPUT_TOKENS: {row['INPUT_TOKENS']} -> {data['INPUT_TOKENS']}")
            print(f"    OUTPUT_TOKENS: {row['OUTPUT_TOKENS']} -> {data['OUTPUT_TOKENS']}")
            print(f"    TOTAL_TOKENS: {row['TOTAL_TOKENS']} -> {data['TOTAL_TOKENS']}")
            print(f"    TOKEN_COST: {row['TOKEN_COST']} -> {data['TOKEN_COST']:.10f}")
            
            # Update the row
            df.at[i, 'INPUT_TOKENS'] = data['INPUT_TOKENS']
            df.at[i, 'OUTPUT_TOKENS'] = data['OUTPUT_TOKENS']
            df.at[i, 'TOTAL_TOKENS'] = data['TOTAL_TOKENS']
            df.at[i, 'COST_PER_INPUT_TOKEN'] = data['COST_PER_INPUT_TOKEN']
            df.at[i, 'COST_PER_OUTPUT_TOKEN'] = data['COST_PER_OUTPUT_TOKEN']
            df.at[i, 'TOKEN_COST'] = f"{data['TOKEN_COST']:.10f}"
            
        print()
    
    # Save updated CSV
    df.to_csv(csv_file, index=False)
    print(f"✅ Updated CSV saved: {csv_file}")
    
    print("\n=== VERIFICATION ===")
    for i, row in df.iterrows():
        input_tokens = row['INPUT_TOKENS']
        output_tokens = row['OUTPUT_TOKENS']
        total_tokens = row['TOTAL_TOKENS']
        calculated_total = input_tokens + output_tokens
        
        if total_tokens != calculated_total:
            print(f"❌ {row['Model Name']}: TOTAL_TOKENS ({total_tokens}) != INPUT+OUTPUT ({calculated_total})")
        else:
            print(f"✅ {row['Model Name']}: {input_tokens} + {output_tokens} = {total_tokens}")

if __name__ == "__main__":
    main()
