#!/usr/bin/env python3
"""
Add Overall Model Score Column
Adds one column showing the overall model score to each row in the benchmark comparison file
"""

import csv
import pandas as pd

def add_overall_model_score():
    """Add overall model score column to the benchmark comparison file"""
    
    input_file = '/home/ubuntu/deemerge/phase_evaluation_engine/llm_analysis_with_benchmark_comparison.csv'
    output_file = '/home/ubuntu/deemerge/phase_evaluation_engine/llm_analysis_with_overall_model_score.csv'
    
    print("Adding overall model score column...")
    
    
    df = pd.read_csv(input_file)
    
    
    model_stats = df.groupby('MODEL').agg({
        'SUCCESS': 'first',
        'MESSAGE_COUNT_DEVIATION_PERCENT': 'mean',
        'COVERAGE_PERCENTAGE': 'mean',
        'PRECISION_PERCENT': 'mean'
    }).round(2)
    
    
    def calculate_overall_model_score(row):
        """Calculate overall score for the entire model"""
        
        
        if row['SUCCESS'] == False:
            return 0.0
        
        
        avg_deviation = abs(row['MESSAGE_COUNT_DEVIATION_PERCENT'])
        avg_coverage = row['COVERAGE_PERCENTAGE']
        avg_precision = row['PRECISION_PERCENT']
        
        
        
        deviation_score = max(0, 100 - avg_deviation)
        
        
        coverage_score = avg_coverage
        
        
        precision_score = avg_precision
        
        
        overall_model_score = (deviation_score * 0.4) + (coverage_score * 0.3) + (precision_score * 0.3)
        
        return round(overall_model_score, 2)
    
    
    model_stats['OVERALL_MODEL_SCORE'] = model_stats.apply(calculate_overall_model_score, axis=1)
    
    
    model_to_score = model_stats['OVERALL_MODEL_SCORE'].to_dict()
    
    
    df['OVERALL_MODEL_SCORE'] = df['MODEL'].map(model_to_score)
    
    
    df.to_csv(output_file, index=False)
    
    print(f"Enhanced file with overall model score saved to: {output_file}")
    print(f"Total rows: {len(df)}")
    
    
    print("\n=== OVERALL MODEL SCORES ===")
    model_summary = df.groupby('MODEL').agg({
        'OVERALL_MODEL_SCORE': 'first',
        'MESSAGE_COUNT_DEVIATION_PERCENT': 'mean',
        'COVERAGE_PERCENTAGE': 'mean',
        'PRECISION_PERCENT': 'mean'
    }).round(2)
    
    
    model_summary = model_summary.sort_values('OVERALL_MODEL_SCORE', ascending=False)
    
    print("TOP 10 MODELS:")
    print("=" * 80)
    for i, (model, row) in enumerate(model_summary.head(10).iterrows(), 1):
        print(f"{i:2d}. {model}")
        print(f"    Overall Model Score: {row['OVERALL_MODEL_SCORE']:6.2f}/100")
        print(f"    Avg Deviation:       {row['MESSAGE_COUNT_DEVIATION_PERCENT']:6.2f}%")
        print(f"    Avg Coverage:        {row['COVERAGE_PERCENTAGE']:6.2f}%")
        print(f"    Avg Precision:       {row['PRECISION_PERCENT']:6.2f}%")
        print()
    
    
    print("=== SAMPLE WITH OVERALL MODEL SCORE ===")
    sample_df = df[['MODEL', 'BENCHMARK_TITLE', 'MESSAGE_COUNT_DEVIATION_PERCENT', 'OVERALL_MODEL_SCORE']].head(10)
    print(sample_df.to_string(index=False))
    
    print("\n=== EXPLANATION ===")
    print("• OVERALL_MODEL_SCORE: Shows the overall performance of each model (0-100)")
    print("• This score is the same for all clusters within the same model")
    print("• Higher scores mean better overall model performance")
    print("• Clients can use this column to easily identify the best models")

if __name__ == "__main__":
    add_overall_model_score()
