#!/usr/bin/env python3
"""
Create Correct Benchmark Analysis
Adds benchmark comparison columns to the original LLM evaluation file
"""

import json
import csv
import pandas as pd
from collections import defaultdict

def load_benchmark_data():
    """Load benchmark cluster data from JSON file"""
    with open('/home/ubuntu/deemerge/phase_evaluation_engine/phases/phase3_clusters.json', 'r') as f:
        benchmark_data = json.load(f)
    
    benchmark_clusters = {}
    for cluster in benchmark_data:
        cluster_id = cluster['cluster_id']
        message_ids = set(cluster['message_ids'])
        benchmark_clusters[cluster_id] = {
            'title': cluster['draft_title'],
            'message_ids': message_ids,
            'message_count': len(message_ids),
            'participants': cluster['participants']
        }
    
    return benchmark_clusters

def find_best_benchmark_match(llm_message_ids, benchmark_clusters):
    """Find the best matching benchmark cluster for LLM cluster"""
    if not llm_message_ids:
        return None
    
    best_match = None
    best_overlap_percentage = 0
    
    for benchmark_id, benchmark_data in benchmark_clusters.items():
        overlap = benchmark_data['message_ids'].intersection(llm_message_ids)
        overlap_count = len(overlap)
        overlap_percentage = (overlap_count / len(benchmark_data['message_ids'])) * 100 if benchmark_data['message_ids'] else 0
        
        if overlap_percentage > best_overlap_percentage:
            best_overlap_percentage = overlap_percentage
            best_match = {
                'benchmark_id': benchmark_id,
                'benchmark_title': benchmark_data['title'],
                'benchmark_count': len(benchmark_data['message_ids']),
                'overlap_count': overlap_count,
                'overlap_percentage': overlap_percentage,
                'missing_messages': benchmark_data['message_ids'] - llm_message_ids,
                'extra_messages': llm_message_ids - benchmark_data['message_ids']
            }
    
    return best_match

def create_enhanced_analysis():
    """Create enhanced analysis by adding benchmark comparison to original file"""
    
    # Load benchmark data
    benchmark_clusters = load_benchmark_data()
    print(f"Loaded {len(benchmark_clusters)} benchmark clusters")
    
    # Read original LLM results
    original_file = '/home/ubuntu/deemerge/phase_evaluation_engine/clustering_analysis_expert_complete_simple_format_fixed_final_gpt5_corrected.csv'
    output_file = '/home/ubuntu/deemerge/phase_evaluation_engine/llm_analysis_with_benchmark_comparison.csv'
    
    enhanced_results = []
    
    with open(original_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse LLM message IDs
            message_ids_str = row['MESSAGE_IDS']
            if message_ids_str and message_ids_str != '0':
                llm_message_ids = set(map(int, message_ids_str.split(';')))
            else:
                llm_message_ids = set()
            
            # Find best benchmark match
            best_match = find_best_benchmark_match(llm_message_ids, benchmark_clusters)
            
            # Calculate benchmark comparison metrics
            if best_match:
                benchmark_count = best_match['benchmark_count']
                llm_count = len(llm_message_ids)
                matched_count = best_match['overlap_count']
                missing_count = len(best_match['missing_messages'])
                extra_count = len(best_match['extra_messages'])
                
                # Calculate deviations and metrics
                message_count_deviation = ((llm_count - benchmark_count) / benchmark_count * 100) if benchmark_count > 0 else 0
                coverage_percentage = (matched_count / benchmark_count * 100) if benchmark_count > 0 else 0
                precision = (matched_count / llm_count * 100) if llm_count > 0 else 0
                recall = (matched_count / benchmark_count * 100) if benchmark_count > 0 else 0
                
                benchmark_cluster_id = best_match['benchmark_id']
                benchmark_title = best_match['benchmark_title']
            else:
                # No benchmark match found
                message_count_deviation = 0
                coverage_percentage = 0
                precision = 0
                recall = 0
                benchmark_cluster_id = 'N/A'
                benchmark_title = 'N/A'
                missing_count = 0
                extra_count = 0
                matched_count = 0
                benchmark_count = 0
            
            # Create enhanced row with original columns + new benchmark columns
            enhanced_row = {
                # Original columns
                'MODEL': row['MODEL'],
                'SUCCESS': row['SUCCESS'],
                'CLUSTER_ID': row['CLUSTER_ID'],
                'CLUSTER_TITLE': row['CLUSTER_TITLE'],
                'CLUSTER_MESSAGES': row['CLUSTER_MESSAGES'],
                'CLUSTER_PARTICIPANTS': row['CLUSTER_PARTICIPANTS'],
                'TOTAL_CLUSTERS': row['TOTAL_CLUSTERS'],
                'TOTAL_MESSAGES_CLUSTERED': row['TOTAL_MESSAGES_CLUSTERED'],
                'DURATION_SECONDS': row['DURATION_SECONDS'],
                'INPUT_TOKENS': row['INPUT_TOKENS'],
                'OUTPUT_TOKENS': row['OUTPUT_TOKENS'],
                'COST_PER_INPUT_TOKEN': row['COST_PER_INPUT_TOKEN'],
                'COST_PER_OUTPUT_TOKEN': row['COST_PER_OUTPUT_TOKEN'],
                'TOKEN_COST': row['TOKEN_COST'],
                'MESSAGE_IDS': row['MESSAGE_IDS'],
                
                # New benchmark comparison columns
                'BENCHMARK_CLUSTER_ID': benchmark_cluster_id,
                'BENCHMARK_TITLE': benchmark_title,
                'BENCHMARK_MESSAGE_COUNT': benchmark_count,
                'LLM_MESSAGE_COUNT': len(llm_message_ids),
                'MATCHED_MESSAGES': matched_count,
                'MISSING_MESSAGES': missing_count,
                'EXTRA_MESSAGES': extra_count,
                'MESSAGE_COUNT_DEVIATION_PERCENT': round(message_count_deviation, 2),
                'COVERAGE_PERCENTAGE': round(coverage_percentage, 2),
                'PRECISION_PERCENT': round(precision, 2),
                'RECALL_PERCENT': round(recall, 2)
            }
            
            enhanced_results.append(enhanced_row)
    
    # Save enhanced results
    with open(output_file, 'w', newline='') as f:
        if enhanced_results:
            fieldnames = enhanced_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enhanced_results)
    
    print(f"Enhanced analysis saved to: {output_file}")
    print(f"Total rows: {len(enhanced_results)}")
    
    # Show summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    df = pd.DataFrame(enhanced_results)
    
    # Group by model and calculate averages
    model_stats = df.groupby('MODEL').agg({
        'MESSAGE_COUNT_DEVIATION_PERCENT': 'mean',
        'COVERAGE_PERCENTAGE': 'mean',
        'PRECISION_PERCENT': 'mean',
        'RECALL_PERCENT': 'mean'
    }).round(2)
    
    # Calculate composite score (lower deviation + higher coverage + higher precision = better)
    model_stats['COMPOSITE_SCORE'] = (
        (100 - abs(model_stats['MESSAGE_COUNT_DEVIATION_PERCENT'])) * 0.4 +
        model_stats['COVERAGE_PERCENTAGE'] * 0.3 +
        model_stats['PRECISION_PERCENT'] * 0.3
    ).round(2)
    
    # Sort by composite score
    model_stats = model_stats.sort_values('COMPOSITE_SCORE', ascending=False)
    
    print("\nTOP 10 MODELS RANKED BY BENCHMARK MATCHING:")
    print("=" * 80)
    for i, (model, row) in enumerate(model_stats.head(10).iterrows(), 1):
        print(f"{i:2d}. {model}")
        print(f"    Deviation: {row['MESSAGE_COUNT_DEVIATION_PERCENT']:6.2f}% (closer to 0% = better)")
        print(f"    Coverage:  {row['COVERAGE_PERCENTAGE']:6.2f}% (higher = better)")
        print(f"    Precision: {row['PRECISION_PERCENT']:6.2f}% (higher = better)")
        print(f"    Score:     {row['COMPOSITE_SCORE']:6.2f}/100")
        print()
    
    # Show sample of the enhanced data
    print("=== SAMPLE OF ENHANCED DATA ===")
    sample_df = df[['MODEL', 'BENCHMARK_TITLE', 'BENCHMARK_MESSAGE_COUNT', 'LLM_MESSAGE_COUNT', 'MESSAGE_COUNT_DEVIATION_PERCENT']].head(10)
    print(sample_df.to_string(index=False))
    
    print("\n=== COLUMN EXPLANATIONS ===")
    print("• BENCHMARK_MESSAGE_COUNT: Expected messages from benchmark")
    print("• LLM_MESSAGE_COUNT: Messages found by LLM")
    print("• MESSAGE_COUNT_DEVIATION_PERCENT: Difference between expected and found (0% = perfect)")
    print("• COVERAGE_PERCENTAGE: How many benchmark messages were found")
    print("• PRECISION_PERCENT: How many found messages are correct")

if __name__ == "__main__":
    create_enhanced_analysis()
