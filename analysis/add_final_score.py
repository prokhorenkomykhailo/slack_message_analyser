#!/usr/bin/env python3
"""
Add Final Score to Benchmark Analysis
Creates a comprehensive final score that clients can easily understand
"""

import csv
import pandas as pd

def add_final_score():
    """Add final score column to the benchmark analysis"""
    
    input_file = '/home/ubuntu/deemerge/phase_evaluation_engine/llm_analysis_with_benchmark_comparison.csv'
    output_file = '/home/ubuntu/deemerge/phase_evaluation_engine/llm_analysis_with_final_score.csv'
    
    print("Adding final score to benchmark analysis...")
    
    # Read the data
    df = pd.read_csv(input_file)
    
    # Calculate final score for each row
    def calculate_final_score(row):
        """Calculate final score based on benchmark matching metrics"""
        
        # Handle failed models
        if row['SUCCESS'] == False:
            return 0.0
        
        # Get metrics
        deviation = abs(row['MESSAGE_COUNT_DEVIATION_PERCENT'])
        coverage = row['COVERAGE_PERCENTAGE']
        precision = row['PRECISION_PERCENT']
        
        # Calculate score components
        # 1. Deviation score (40% weight) - closer to 0% is better
        deviation_score = max(0, 100 - deviation)  # 0% deviation = 100 points, 50% deviation = 50 points
        
        # 2. Coverage score (30% weight) - higher is better
        coverage_score = coverage  # 100% coverage = 100 points
        
        # 3. Precision score (30% weight) - higher is better
        precision_score = precision  # 100% precision = 100 points
        
        # Calculate weighted final score
        final_score = (deviation_score * 0.4) + (coverage_score * 0.3) + (precision_score * 0.3)
        
        return round(final_score, 2)
    
    # Add final score column
    df['FINAL_SCORE'] = df.apply(calculate_final_score, axis=1)
    
    # Save the enhanced file
    df.to_csv(output_file, index=False)
    
    print(f"Enhanced file with final score saved to: {output_file}")
    print(f"Total rows: {len(df)}")
    
    # Show summary statistics
    print("\n=== FINAL SCORE SUMMARY ===")
    
    # Group by model and calculate averages
    model_stats = df.groupby('MODEL').agg({
        'FINAL_SCORE': 'mean',
        'MESSAGE_COUNT_DEVIATION_PERCENT': 'mean',
        'COVERAGE_PERCENTAGE': 'mean',
        'PRECISION_PERCENT': 'mean'
    }).round(2)
    
    # Sort by final score
    model_stats = model_stats.sort_values('FINAL_SCORE', ascending=False)
    
    print("\nTOP 10 MODELS RANKED BY FINAL SCORE:")
    print("=" * 80)
    for i, (model, row) in enumerate(model_stats.head(10).iterrows(), 1):
        print(f"{i:2d}. {model}")
        print(f"    Final Score: {row['FINAL_SCORE']:6.2f}/100")
        print(f"    Deviation:   {row['MESSAGE_COUNT_DEVIATION_PERCENT']:6.2f}%")
        print(f"    Coverage:    {row['COVERAGE_PERCENTAGE']:6.2f}%")
        print(f"    Precision:   {row['PRECISION_PERCENT']:6.2f}%")
        print()
    
    # Show sample of the data with final scores
    print("=== SAMPLE WITH FINAL SCORES ===")
    sample_df = df[['MODEL', 'BENCHMARK_TITLE', 'BENCHMARK_MESSAGE_COUNT', 'LLM_MESSAGE_COUNT', 'MESSAGE_COUNT_DEVIATION_PERCENT', 'FINAL_SCORE']].head(10)
    print(sample_df.to_string(index=False))
    
    print("\n=== FINAL SCORE EXPLANATION ===")
    print("FINAL_SCORE (0-100, higher = better):")
    print("• Combines deviation (40%), coverage (30%), and precision (30%)")
    print("• 100 = Perfect benchmark matching")
    print("• 80+ = Excellent performance")
    print("• 60-79 = Good performance")
    print("• 40-59 = Fair performance")
    print("• Below 40 = Poor performance")
    print()
    print("For clients: Look at FINAL_SCORE column - this is the single metric to compare models!")

if __name__ == "__main__":
    add_final_score()
