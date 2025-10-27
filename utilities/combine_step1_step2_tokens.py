#!/usr/bin/env python3
"""
Combine Step 1 and Step 2 token data to get correct total token usage
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

def get_step1_token_data(model_name):
    """Get Step 1 token data from Phase 3 files"""
    step1_file = f"output/phase3_topic_clustering/{model_name}.json"
    
    try:
        with open(step1_file, 'r') as f:
            data = json.load(f)
        
        usage = data.get('usage', {})
        cost = data.get('cost', {})
        
        return {
            'input_tokens': usage.get('prompt_tokens', 0),
            'output_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
            'input_cost': cost.get('input_cost', 0),
            'output_cost': cost.get('output_cost', 0),
            'total_cost': cost.get('total_cost', 0)
        }
    except Exception as e:
        print(f"Error reading Step 1 file for {model_name}: {e}")
        return {
            'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
            'input_cost': 0, 'output_cost': 0, 'total_cost': 0
        }

def get_step2_token_data(model_name):
    """Get Step 2 token data from Phase 4 files"""
    step2_file = f"output/phase4_balanced_refinement/{model_name}_balanced.json"
    
    try:
        with open(step2_file, 'r') as f:
            data = json.load(f)
        
        usage = data.get('usage', {})
        cost = data.get('cost', {})
        
        return {
            'input_tokens': usage.get('prompt_tokens', 0),
            'output_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
            'input_cost': cost.get('input_cost', 0),
            'output_cost': cost.get('output_cost', 0),
            'total_cost': cost.get('total_cost', 0)
        }
    except Exception as e:
        print(f"Error reading Step 2 file for {model_name}: {e}")
        return {
            'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
            'input_cost': 0, 'output_cost': 0, 'total_cost': 0
        }

def get_combined_token_data():
    """Get combined Step 1 + Step 2 token data"""
    combined_data = {}
    
    # Get all Step 2 files
    step2_files = sorted([f for f in glob.glob("output/phase4_balanced_refinement/*_balanced.json") 
                         if not f.endswith('comprehensive_results.json')])
    
    for step2_file in step2_files:
        model_name = step2_file.split('/')[-1].replace('_balanced.json', '')
        
        print(f"Processing: {model_name}")
        
        # Get Step 1 data
        step1_data = get_step1_token_data(model_name)
        print(f"  Step 1: {step1_data['input_tokens']} + {step1_data['output_tokens']} = {step1_data['total_tokens']}")
        
        # Get Step 2 data
        step2_data = get_step2_token_data(model_name)
        print(f"  Step 2: {step2_data['input_tokens']} + {step2_data['output_tokens']} = {step2_data['total_tokens']}")
        
        # Combine the data
        total_input = step1_data['input_tokens'] + step2_data['input_tokens']
        total_output = step1_data['output_tokens'] + step2_data['output_tokens']
        total_tokens = total_input + total_output
        
        # Calculate correct token cost using pricing
        pricing = get_model_pricing(model_name)
        cost_per_input = pricing['COST_PER_INPUT_TOKEN']
        cost_per_output = pricing['COST_PER_OUTPUT_TOKEN']
        token_cost = (total_input * cost_per_input) + (total_output * cost_per_output)
        
        combined_data[model_name] = {
            'INPUT_TOKENS': total_input,
            'OUTPUT_TOKENS': total_output,
            'TOTAL_TOKENS': total_tokens,
            'COST_PER_INPUT_TOKEN': cost_per_input,
            'COST_PER_OUTPUT_TOKEN': cost_per_output,
            'TOKEN_COST': token_cost,
            'STEP1_INPUT': step1_data['input_tokens'],
            'STEP1_OUTPUT': step1_data['output_tokens'],
            'STEP2_INPUT': step2_data['input_tokens'],
            'STEP2_OUTPUT': step2_data['output_tokens']
        }
        
        print(f"  Combined: {total_input} + {total_output} = {total_tokens} (${token_cost:.10f})")
        print()
    
    return combined_data

def main():
    """Update CSV with combined Step 1 + Step 2 token data"""
    
    print("=== COMBINING STEP 1 + STEP 2 TOKEN DATA ===\n")
    
    # Read current CSV
    csv_file = "output/step2_client_analysis/step2_all_models_FINAL_CORRECT.csv"
    df = pd.read_csv(csv_file)
    
    # Get combined token data
    combined_data = get_combined_token_data()
    
    print("=== UPDATING CSV WITH COMBINED TOKEN DATA ===")
    for i, row in df.iterrows():
        model_name = row['Model Name']
        if model_name in combined_data:
            data = combined_data[model_name]
            
            print(f"\n{model_name}:")
            print(f"  OLD - INPUT_TOKENS: {row['INPUT_TOKENS']} -> NEW: {data['INPUT_TOKENS']}")
            print(f"  OLD - OUTPUT_TOKENS: {row['OUTPUT_TOKENS']} -> NEW: {data['OUTPUT_TOKENS']}")
            print(f"  OLD - TOTAL_TOKENS: {row['TOTAL_TOKENS']} -> NEW: {data['TOTAL_TOKENS']}")
            print(f"  OLD - TOKEN_COST: {row['TOKEN_COST']} -> NEW: {data['TOKEN_COST']:.10f}")
            print(f"  Breakdown: Step1({data['STEP1_INPUT']}+{data['STEP1_OUTPUT']}) + Step2({data['STEP2_INPUT']}+{data['STEP2_OUTPUT']})")
            
            # Update the row
            df.at[i, 'INPUT_TOKENS'] = data['INPUT_TOKENS']
            df.at[i, 'OUTPUT_TOKENS'] = data['OUTPUT_TOKENS']
            df.at[i, 'TOTAL_TOKENS'] = data['TOTAL_TOKENS']
            df.at[i, 'COST_PER_INPUT_TOKEN'] = data['COST_PER_INPUT_TOKEN']
            df.at[i, 'COST_PER_OUTPUT_TOKEN'] = data['COST_PER_OUTPUT_TOKEN']
            df.at[i, 'TOKEN_COST'] = f"{data['TOKEN_COST']:.10f}"
    
    # Save updated CSV
    df.to_csv(csv_file, index=False)
    print(f"\n✅ Updated CSV saved: {csv_file}")
    
    print("\n=== FINAL VERIFICATION ===")
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
