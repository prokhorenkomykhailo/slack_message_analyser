#!/usr/bin/env python3
"""
Create Comprehensive Analysis with ROUGE + Benchmark
Includes clear explanations for the client
"""

import csv
import pandas as pd

def create_comprehensive_analysis():
    """Create comprehensive analysis with both benchmark and ROUGE metrics"""
    
    input_file = '/home/ubuntu/deemerge/phase_evaluation_engine/clustering_analysis_enhanced_with_deviations.csv'
    output_file = '/home/ubuntu/deemerge/phase_evaluation_engine/comprehensive_model_analysis.csv'
    
    # Define comprehensive columns
    comprehensive_columns = [
        # Basic info
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
        
        # Benchmark matching (what client asked for)
        'BENCHMARK_CLUSTER_ID',
        'BENCHMARK_TITLE', 
        'BENCHMARK_MESSAGE_COUNT',
        'LLM_MESSAGE_COUNT',
        'MATCHED_MESSAGES',
        'MISSING_MESSAGES',
        'EXTRA_MESSAGES',
        'MESSAGE_COUNT_DEVIATION_PERCENT',
        'COVERAGE_PERCENTAGE',
        'PRECISION_PERCENT',
        
        # ROUGE evaluation (for finding best overall model)
        'ROUGE_OVERALL_SCORE',
        'ROUGE_SIMILARITY',
        'ROUGE_CONSISTENCY',
        'ROUGE_COVERAGE',
        'ROUGE_RECOMMENDATION'
    ]
    
    print("Creating comprehensive analysis...")
    
    # Read and filter data
    df = pd.read_csv(input_file)
    comprehensive_df = df[comprehensive_columns].copy()
    
    # Rename columns to be client-friendly
    column_rename = {
        'BENCHMARK_CLUSTER_ID': 'BENCHMARK_TOPIC_ID',
        'BENCHMARK_TITLE': 'BENCHMARK_TOPIC_NAME',
        'BENCHMARK_MESSAGE_COUNT': 'EXPECTED_MESSAGES',
        'LLM_MESSAGE_COUNT': 'FOUND_MESSAGES', 
        'MESSAGE_COUNT_DEVIATION_PERCENT': 'DEVIATION_PERCENT',
        'COVERAGE_PERCENTAGE': 'FOUND_PERCENT',
        'PRECISION_PERCENT': 'ACCURACY_PERCENT',
        'ROUGE_OVERALL_SCORE': 'OVERALL_QUALITY_SCORE',
        'ROUGE_SIMILARITY': 'CLUSTER_SIMILARITY_SCORE',
        'ROUGE_CONSISTENCY': 'CLUSTER_CONSISTENCY_SCORE',
        'ROUGE_COVERAGE': 'COMPLETENESS_SCORE',
        'ROUGE_RECOMMENDATION': 'QUALITY_RECOMMENDATION'
    }
    
    comprehensive_df = comprehensive_df.rename(columns=column_rename)
    
    # Save to new file
    comprehensive_df.to_csv(output_file, index=False)
    
    print(f"Comprehensive analysis saved to: {output_file}")
    print(f"Total rows: {len(comprehensive_df)}")
    print(f"Columns: {len(comprehensive_df.columns)}")
    
    # Calculate comprehensive model ranking
    print("\n=== COMPREHENSIVE MODEL RANKING ===")
    model_performance = comprehensive_df[comprehensive_df['SUCCESS'] == True].groupby('MODEL').agg({
        'DEVIATION_PERCENT': 'mean',
        'FOUND_PERCENT': 'mean',
        'ACCURACY_PERCENT': 'mean',
        'OVERALL_QUALITY_SCORE': 'mean',
        'CLUSTER_SIMILARITY_SCORE': 'mean',
        'CLUSTER_CONSISTENCY_SCORE': 'mean'
    }).round(4)
    
    # Calculate comprehensive score (weighted combination)
    model_performance['COMPREHENSIVE_SCORE'] = (
        (100 - abs(model_performance['DEVIATION_PERCENT'])) * 0.25 +  # Benchmark deviation (25%)
        model_performance['FOUND_PERCENT'] * 0.20 +  # Benchmark coverage (20%)
        model_performance['ACCURACY_PERCENT'] * 0.20 +    # Benchmark precision (20%)
        model_performance['OVERALL_QUALITY_SCORE'] * 100 * 0.35  # ROUGE overall (35% - most important)
    ).round(2)
    
    top_models = model_performance.sort_values('COMPREHENSIVE_SCORE', ascending=False).head(10)
    
    print("TOP 10 MODELS (Combined Evaluation):")
    print("=" * 80)
    for i, (model, row) in enumerate(top_models.iterrows(), 1):
        print(f"{i:2d}. {model}")
        print(f"    Comprehensive Score: {row['COMPREHENSIVE_SCORE']:6.2f}/100")
        print(f"    Benchmark Match: {row['FOUND_PERCENT']:5.1f}% found, {row['ACCURACY_PERCENT']:5.1f}% accurate, {row['DEVIATION_PERCENT']:+6.1f}% deviation")
        print(f"    Quality Scores: {row['OVERALL_QUALITY_SCORE']:.4f} overall, {row['CLUSTER_SIMILARITY_SCORE']:.4f} similarity, {row['CLUSTER_CONSISTENCY_SCORE']:.4f} consistency")
        print()
    
    print("=== EXPLANATION FOR CLIENT ===")
    print()
    print("BENCHMARK MATCHING METRICS (What you asked for):")
    print("• EXPECTED_MESSAGES: How many messages should be in this topic (from your benchmark)")
    print("• FOUND_MESSAGES: How many messages the LLM found for this topic")
    print("• DEVIATION_PERCENT: The difference between expected and found (0% = perfect match)")
    print("• FOUND_PERCENT: What percentage of expected messages were found")
    print("• ACCURACY_PERCENT: What percentage of found messages are correct")
    print()
    print("QUALITY EVALUATION METRICS (For finding the best overall model):")
    print("• OVERALL_QUALITY_SCORE: How well the model clusters topics overall (0-1, higher = better)")
    print("• CLUSTER_SIMILARITY_SCORE: How similar messages within clusters are (0-1, higher = better)")
    print("• CLUSTER_CONSISTENCY_SCORE: How consistent the clustering is (0-1, higher = better)")
    print("• COMPLETENESS_SCORE: How completely the model processed all messages (0-1, higher = better)")
    print("• QUALITY_RECOMMENDATION: Overall assessment of the model's performance")
    print()
    print("COMPREHENSIVE SCORE: Combines benchmark matching (65%) + quality evaluation (35%)")
    print("This gives you the best model that both matches your benchmark AND produces high-quality clusters.")

if __name__ == "__main__":
    create_comprehensive_analysis()
