#!/usr/bin/env python3
"""
Deviation Analysis Script
Compares benchmark clusters with LLM clustering results to calculate deviations
"""

import json
import csv
import pandas as pd
from collections import defaultdict

def load_benchmark_data(json_file):
    """Load benchmark cluster data from JSON file"""
    with open(json_file, 'r') as f:
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

def analyze_llm_results(csv_file):
    """Analyze LLM clustering results from CSV file"""
    llm_results = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['MODEL']
            if model not in llm_results:
                llm_results[model] = {}
            
            cluster_id = int(row['CLUSTER_ID'])
            message_ids_str = row['MESSAGE_IDS']
            if message_ids_str and message_ids_str != '0':
                message_ids = set(map(int, message_ids_str.split(';')))
            else:
                message_ids = set()
            
            llm_results[model][cluster_id] = {
                'title': row['CLUSTER_TITLE'],
                'message_ids': message_ids,
                'message_count': len(message_ids),
                'success': row['SUCCESS'] == 'TRUE',
                'total_clusters': int(row['TOTAL_CLUSTERS']) if row['TOTAL_CLUSTERS'] else 0,
                'total_messages_clustered': int(row['TOTAL_MESSAGES_CLUSTERED']) if row['TOTAL_MESSAGES_CLUSTERED'] else 0
            }
    
    return llm_results

def calculate_cluster_matching(benchmark_clusters, llm_clusters):
    """Calculate how well LLM clusters match benchmark clusters"""
    matches = {}
    
    for benchmark_id, benchmark_data in benchmark_clusters.items():
        best_match = None
        best_overlap = 0
        best_overlap_percentage = 0
        
        for llm_cluster_id, llm_data in llm_clusters.items():
            if not llm_data['message_ids']:
                continue
                
            # Calculate overlap
            overlap = benchmark_data['message_ids'].intersection(llm_data['message_ids'])
            overlap_count = len(overlap)
            overlap_percentage = (overlap_count / len(benchmark_data['message_ids'])) * 100 if benchmark_data['message_ids'] else 0
            
            if overlap_percentage > best_overlap_percentage:
                best_overlap_percentage = overlap_percentage
                best_overlap = overlap_count
                best_match = {
                    'llm_cluster_id': llm_cluster_id,
                    'llm_title': llm_data['title'],
                    'overlap_count': overlap_count,
                    'overlap_percentage': overlap_percentage,
                    'benchmark_count': len(benchmark_data['message_ids']),
                    'llm_count': len(llm_data['message_ids']),
                    'missing_messages': benchmark_data['message_ids'] - llm_data['message_ids'],
                    'extra_messages': llm_data['message_ids'] - benchmark_data['message_ids']
                }
        
        matches[benchmark_id] = best_match
    
    return matches

def create_deviation_analysis(benchmark_clusters, llm_results):
    """Create comprehensive deviation analysis"""
    results = []
    
    for model, llm_clusters in llm_results.items():
        matches = calculate_cluster_matching(benchmark_clusters, llm_clusters)
        
        for benchmark_id, match_data in matches.items():
            if match_data is None:
                # No match found
                benchmark_data = benchmark_clusters[benchmark_id]
                results.append({
                    'MODEL': model,
                    'BENCHMARK_CLUSTER_ID': benchmark_id,
                    'BENCHMARK_TITLE': benchmark_data['title'],
                    'BENCHMARK_MESSAGE_COUNT': benchmark_data['message_count'],
                    'LLM_CLUSTER_ID': 'N/A',
                    'LLM_TITLE': 'N/A',
                    'LLM_MESSAGE_COUNT': 0,
                    'MATCHED_MESSAGES': 0,
                    'MISSING_MESSAGES': benchmark_data['message_count'],
                    'EXTRA_MESSAGES': 0,
                    'OVERLAP_PERCENTAGE': 0.0,
                    'MESSAGE_COUNT_DEVIATION': -100.0,
                    'COVERAGE_PERCENTAGE': 0.0,
                    'PRECISION': 0.0,
                    'RECALL': 0.0,
                    'F1_SCORE': 0.0
                })
            else:
                # Calculate metrics
                benchmark_count = match_data['benchmark_count']
                llm_count = match_data['llm_count']
                matched_count = match_data['overlap_count']
                missing_count = len(match_data['missing_messages'])
                extra_count = len(match_data['extra_messages'])
                
                # Calculate deviations and metrics
                message_count_deviation = ((llm_count - benchmark_count) / benchmark_count * 100) if benchmark_count > 0 else 0
                coverage_percentage = (matched_count / benchmark_count * 100) if benchmark_count > 0 else 0
                precision = (matched_count / llm_count * 100) if llm_count > 0 else 0
                recall = (matched_count / benchmark_count * 100) if benchmark_count > 0 else 0
                f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
                
                results.append({
                    'MODEL': model,
                    'BENCHMARK_CLUSTER_ID': benchmark_id,
                    'BENCHMARK_TITLE': benchmark_clusters[benchmark_id]['title'],
                    'BENCHMARK_MESSAGE_COUNT': benchmark_count,
                    'LLM_CLUSTER_ID': match_data['llm_cluster_id'],
                    'LLM_TITLE': match_data['llm_title'],
                    'LLM_MESSAGE_COUNT': llm_count,
                    'MATCHED_MESSAGES': matched_count,
                    'MISSING_MESSAGES': missing_count,
                    'EXTRA_MESSAGES': extra_count,
                    'OVERLAP_PERCENTAGE': round(match_data['overlap_percentage'], 2),
                    'MESSAGE_COUNT_DEVIATION': round(message_count_deviation, 2),
                    'COVERAGE_PERCENTAGE': round(coverage_percentage, 2),
                    'PRECISION': round(precision, 2),
                    'RECALL': round(recall, 2),
                    'F1_SCORE': round(f1_score, 2)
                })
    
    return results

def main():
    # File paths
    benchmark_file = '/home/ubuntu/deemerge/phase_evaluation_engine/phases/phase3_clusters.json'
    llm_results_file = '/home/ubuntu/deemerge/phase_evaluation_engine/clustering_analysis_expert_complete_simple_format_fixed_final_gpt5_corrected.csv'
    output_file = '/home/ubuntu/deemerge/phase_evaluation_engine/clustering_deviation_analysis.csv'
    
    print("Loading benchmark data...")
    benchmark_clusters = load_benchmark_data(benchmark_file)
    print(f"Loaded {len(benchmark_clusters)} benchmark clusters")
    
    print("Analyzing LLM results...")
    llm_results = analyze_llm_results(llm_results_file)
    print(f"Loaded results for {len(llm_results)} models")
    
    print("Calculating deviations...")
    deviation_results = create_deviation_analysis(benchmark_clusters, llm_results)
    
    print("Writing results to CSV...")
    with open(output_file, 'w', newline='') as f:
        if deviation_results:
            fieldnames = deviation_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(deviation_results)
    
    print(f"Deviation analysis complete! Results saved to: {output_file}")
    print(f"Total analysis rows: {len(deviation_results)}")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    df = pd.DataFrame(deviation_results)
    
    # Group by model and calculate averages
    model_stats = df.groupby('MODEL').agg({
        'OVERLAP_PERCENTAGE': 'mean',
        'MESSAGE_COUNT_DEVIATION': 'mean',
        'COVERAGE_PERCENTAGE': 'mean',
        'PRECISION': 'mean',
        'RECALL': 'mean',
        'F1_SCORE': 'mean'
    }).round(2)
    
    print("\nAverage Performance by Model:")
    print(model_stats)
    
    # Best performing models
    best_f1 = model_stats['F1_SCORE'].idxmax()
    best_coverage = model_stats['COVERAGE_PERCENTAGE'].idxmax()
    best_precision = model_stats['PRECISION'].idxmax()
    best_recall = model_stats['RECALL'].idxmax()
    
    print(f"\nBest F1 Score: {best_f1} ({model_stats.loc[best_f1, 'F1_SCORE']:.2f}%)")
    print(f"Best Coverage: {best_coverage} ({model_stats.loc[best_coverage, 'COVERAGE_PERCENTAGE']:.2f}%)")
    print(f"Best Precision: {best_precision} ({model_stats.loc[best_precision, 'PRECISION']:.2f}%)")
    print(f"Best Recall: {best_recall} ({model_stats.loc[best_recall, 'RECALL']:.2f}%)")

if __name__ == "__main__":
    main()
