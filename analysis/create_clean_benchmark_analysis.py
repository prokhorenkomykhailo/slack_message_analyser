#!/usr/bin/env python3
"""
Create Clean Benchmark Analysis
Focuses only on columns that directly answer the client's deviation question
"""

import csv
import pandas as pd

def create_clean_benchmark_analysis():
    """Create a clean analysis file with only benchmark-relevant columns"""
    
    input_file = '/home/ubuntu/deemerge/phase_evaluation_engine/clustering_analysis_enhanced_with_deviations.csv'
    output_file = '/home/ubuntu/deemerge/phase_evaluation_engine/benchmark_matching_analysis.csv'
    
    # Define the essential columns for benchmark matching
    essential_columns = [
        'MODEL',
        'SUCCESS', 
        'CLUSTER_ID',
        'CLUSTER_TITLE',
        'CLUSTER_MESSAGES',
        'CLUSTER_PARTICIPANTS',
        'TOTAL_CLUSTERS',
        'TOTAL_MESSAGES_CLUSTERED',
        'DURATION_SECONDS',
        'TOKEN_COST',
        'MESSAGE_IDS',
        'BENCHMARK_CLUSTER_ID',
        'BENCHMARK_TITLE', 
        'BENCHMARK_MESSAGE_COUNT',
        'LLM_MESSAGE_COUNT',
        'MATCHED_MESSAGES',
        'MISSING_MESSAGES',
        'EXTRA_MESSAGES',
        'MESSAGE_COUNT_DEVIATION_PERCENT',
        'COVERAGE_PERCENTAGE',
        'PRECISION_PERCENT'
    ]
    
    print("Creating clean benchmark analysis...")
    
    # Read the enhanced file and filter to essential columns
    df = pd.read_csv(input_file)
    
    # Keep only essential columns
    clean_df = df[essential_columns].copy()
    
    # Rename columns to be more client-friendly
    column_rename = {
        'BENCHMARK_CLUSTER_ID': 'BENCHMARK_TOPIC_ID',
        'BENCHMARK_TITLE': 'BENCHMARK_TOPIC_NAME',
        'BENCHMARK_MESSAGE_COUNT': 'EXPECTED_MESSAGES',
        'LLM_MESSAGE_COUNT': 'FOUND_MESSAGES', 
        'MESSAGE_COUNT_DEVIATION_PERCENT': 'DEVIATION_PERCENT',
        'COVERAGE_PERCENTAGE': 'FOUND_PERCENT',
        'PRECISION_PERCENT': 'ACCURACY_PERCENT'
    }
    
    clean_df = clean_df.rename(columns=column_rename)
    
    # Save to new file
    clean_df.to_csv(output_file, index=False)
    
    print(f"Clean benchmark analysis saved to: {output_file}")
    print(f"Total rows: {len(clean_df)}")
    print(f"Columns: {len(clean_df.columns)}")
    
    # Show sample of the clean data
    print("\n=== SAMPLE OF CLEAN DATA ===")
    print(clean_df[['MODEL', 'BENCHMARK_TOPIC_NAME', 'EXPECTED_MESSAGES', 'FOUND_MESSAGES', 'DEVIATION_PERCENT']].head(10))
    
    # Show best performing models
    print("\n=== BEST MODELS FOR BENCHMARK MATCHING ===")
    model_performance = clean_df[clean_df['SUCCESS'] == True].groupby('MODEL').agg({
        'DEVIATION_PERCENT': 'mean',
        'FOUND_PERCENT': 'mean',
        'ACCURACY_PERCENT': 'mean'
    }).round(2)
    
    # Calculate simple score (lower deviation + higher found + higher accuracy = better)
    model_performance['SCORE'] = (
        (100 - abs(model_performance['DEVIATION_PERCENT'])) * 0.4 +
        model_performance['FOUND_PERCENT'] * 0.3 +
        model_performance['ACCURACY_PERCENT'] * 0.3
    ).round(2)
    
    top_models = model_performance.sort_values('SCORE', ascending=False).head(5)
    
    for i, (model, row) in enumerate(top_models.iterrows(), 1):
        print(f"{i}. {model}")
        print(f"   Deviation: {row['DEVIATION_PERCENT']:6.2f}% (closer to 0% = better)")
        print(f"   Found:     {row['FOUND_PERCENT']:6.2f}% (higher = better)")  
        print(f"   Accuracy:  {row['ACCURACY_PERCENT']:6.2f}% (higher = better)")
        print(f"   Score:     {row['SCORE']:6.2f}/100")
        print()
    
    print("=== COLUMN EXPLANATIONS ===")
    print("• EXPECTED_MESSAGES: How many messages should be in this topic (from benchmark)")
    print("• FOUND_MESSAGES: How many messages the LLM found for this topic")
    print("• DEVIATION_PERCENT: The difference between expected and found (0% = perfect match)")
    print("• FOUND_PERCENT: What percentage of expected messages were found")
    print("• ACCURACY_PERCENT: What percentage of found messages are correct")

if __name__ == "__main__":
    create_clean_benchmark_analysis()
