#!/usr/bin/env python3
"""
Create Model-Based Final Score
Calculates final score per MODEL (not per cluster) to find the best model for phase 3 clustering
"""

import csv
import pandas as pd

def create_model_based_final_score():
    """Create final score calculated per model, not per cluster"""
    
    input_file = '/home/ubuntu/deemerge/phase_evaluation_engine/llm_analysis_with_benchmark_comparison.csv'
    output_file = '/home/ubuntu/deemerge/phase_evaluation_engine/model_based_final_scores.csv'
    
    print("Creating model-based final scores for phase 3 clustering...")
    
    # Read the data
    df = pd.read_csv(input_file)
    
    # Calculate model-level statistics
    model_stats = df.groupby('MODEL').agg({
        'SUCCESS': 'first',  # Whether the model succeeded
        'MESSAGE_COUNT_DEVIATION_PERCENT': 'mean',  # Average deviation across all clusters
        'COVERAGE_PERCENTAGE': 'mean',  # Average coverage across all clusters
        'PRECISION_PERCENT': 'mean',  # Average precision across all clusters
        'TOTAL_CLUSTERS': 'first',  # Number of clusters created
        'TOTAL_MESSAGES_CLUSTERED': 'first',  # Total messages processed
        'DURATION_SECONDS': 'first',  # Processing time
        'TOKEN_COST': 'first'  # Cost
    }).round(2)
    
    # Calculate model-based final score
    def calculate_model_final_score(row):
        """Calculate final score for the entire model"""
        
        # Handle failed models
        if row['SUCCESS'] == False:
            return 0.0
        
        # Get average metrics across all clusters for this model
        avg_deviation = abs(row['MESSAGE_COUNT_DEVIATION_PERCENT'])
        avg_coverage = row['COVERAGE_PERCENTAGE']
        avg_precision = row['PRECISION_PERCENT']
        
        # Calculate score components
        # 1. Deviation score (40% weight) - closer to 0% is better
        deviation_score = max(0, 100 - avg_deviation)
        
        # 2. Coverage score (30% weight) - higher is better
        coverage_score = avg_coverage
        
        # 3. Precision score (30% weight) - higher is better
        precision_score = avg_precision
        
        # Calculate weighted final score
        model_final_score = (deviation_score * 0.4) + (coverage_score * 0.3) + (precision_score * 0.3)
        
        return round(model_final_score, 2)
    
    # Add model final score
    model_stats['MODEL_FINAL_SCORE'] = model_stats.apply(calculate_model_final_score, axis=1)
    
    # Add performance category
    def get_performance_category(score):
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Very Good"
        elif score >= 70:
            return "Good"
        elif score >= 60:
            return "Fair"
        else:
            return "Poor"
    
    model_stats['PERFORMANCE_CATEGORY'] = model_stats['MODEL_FINAL_SCORE'].apply(get_performance_category)
    
    # Sort by final score
    model_stats = model_stats.sort_values('MODEL_FINAL_SCORE', ascending=False)
    
    # Save model-based results
    model_stats.to_csv(output_file)
    
    print(f"Model-based final scores saved to: {output_file}")
    
    # Display results
    print("\n=== BEST MODELS FOR PHASE 3 CLUSTERING ===")
    print("=" * 100)
    print(f"{'Rank':<4} {'Model':<25} {'Final Score':<12} {'Category':<12} {'Avg Deviation':<15} {'Avg Coverage':<15} {'Avg Precision':<15}")
    print("=" * 100)
    
    for i, (model, row) in enumerate(model_stats.iterrows(), 1):
        print(f"{i:<4} {model:<25} {row['MODEL_FINAL_SCORE']:<12.2f} {row['PERFORMANCE_CATEGORY']:<12} {row['MESSAGE_COUNT_DEVIATION_PERCENT']:<15.2f}% {row['COVERAGE_PERCENTAGE']:<15.2f}% {row['PRECISION_PERCENT']:<15.2f}%")
    
    print("\n=== TOP 5 MODELS SUMMARY ===")
    top_5 = model_stats.head(5)
    for i, (model, row) in enumerate(top_5.iterrows(), 1):
        print(f"\n{i}. {model}")
        print(f"   Final Score: {row['MODEL_FINAL_SCORE']:.2f}/100 ({row['PERFORMANCE_CATEGORY']})")
        print(f"   Clusters: {row['TOTAL_CLUSTERS']}")
        print(f"   Messages: {row['TOTAL_MESSAGES_CLUSTERED']}")
        print(f"   Time: {row['DURATION_SECONDS']:.2f}s")
        print(f"   Cost: ${row['TOKEN_COST']:.4f}")
        print(f"   Avg Deviation: {row['MESSAGE_COUNT_DEVIATION_PERCENT']:.2f}%")
        print(f"   Avg Coverage: {row['COVERAGE_PERCENTAGE']:.2f}%")
        print(f"   Avg Precision: {row['PRECISION_PERCENT']:.2f}%")
    
    print("\n=== RECOMMENDATION FOR CLIENT ===")
    best_model = model_stats.index[0]
    best_score = model_stats.iloc[0]['MODEL_FINAL_SCORE']
    best_category = model_stats.iloc[0]['PERFORMANCE_CATEGORY']
    
    print(f"BEST MODEL FOR PHASE 3 CLUSTERING: {best_model}")
    print(f"Final Score: {best_score:.2f}/100 ({best_category})")
    print(f"This model provides the best balance of accuracy, coverage, and precision for clustering tasks.")

if __name__ == "__main__":
    create_model_based_final_score()
