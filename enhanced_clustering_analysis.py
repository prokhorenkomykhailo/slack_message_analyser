#!/usr/bin/env python3
"""
Enhanced Clustering Analysis Script
Extends the original clustering results with deviation analysis and ROUGE metrics
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

def load_rouge_results(rouge_file):
    """Load ROUGE evaluation results"""
    rouge_data = {}
    with open(rouge_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row['model_name']
            
            if 'google_' in model_name:
                model_name = model_name.replace('google_', '')
            elif 'openai_' in model_name:
                model_name = model_name.replace('openai_', '')
            elif 'xai_' in model_name:
                model_name = model_name.replace('xai_', '')
            
            rouge_data[model_name] = {
                'overall_score': float(row['overall_score']),
                'rouge_similarity': float(row['rouge_similarity']),
                'rouge_consistency': float(row['rouge_consistency']),
                'coverage': float(row['coverage']),
                'recommendation': row['recommendation']
            }
    
    return rouge_data

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

def create_enhanced_analysis(original_csv, benchmark_clusters, rouge_data):
    """Create enhanced analysis by extending original CSV with deviation metrics"""
    enhanced_results = []
    
    with open(original_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['MODEL']
            cluster_id = int(row['CLUSTER_ID'])
            message_ids_str = row['MESSAGE_IDS']
            
            if message_ids_str and message_ids_str != '0':
                message_ids = set(map(int, message_ids_str.split(';')))
            else:
                message_ids = set()
            
            
            best_benchmark_match = None
            best_overlap_percentage = 0
            
            for benchmark_id, benchmark_data in benchmark_clusters.items():
                overlap = benchmark_data['message_ids'].intersection(message_ids)
                overlap_percentage = (len(overlap) / len(benchmark_data['message_ids'])) * 100 if benchmark_data['message_ids'] else 0
                
                if overlap_percentage > best_overlap_percentage:
                    best_overlap_percentage = overlap_percentage
                    best_benchmark_match = {
                        'benchmark_id': benchmark_id,
                        'benchmark_title': benchmark_data['title'],
                        'benchmark_count': len(benchmark_data['message_ids']),
                        'overlap_count': len(overlap),
                        'missing_count': len(benchmark_data['message_ids'] - message_ids),
                        'extra_count': len(message_ids - benchmark_data['message_ids'])
                    }
            
            
            if best_benchmark_match:
                benchmark_count = best_benchmark_match['benchmark_count']
                llm_count = len(message_ids)
                matched_count = best_benchmark_match['overlap_count']
                missing_count = best_benchmark_match['missing_count']
                extra_count = best_benchmark_match['extra_count']
                
                
                message_count_deviation = ((llm_count - benchmark_count) / benchmark_count * 100) if benchmark_count > 0 else 0
                coverage_percentage = (matched_count / benchmark_count * 100) if benchmark_count > 0 else 0
                precision = (matched_count / llm_count * 100) if llm_count > 0 else 0
                recall = (matched_count / benchmark_count * 100) if benchmark_count > 0 else 0
                
                benchmark_cluster_id = best_benchmark_match['benchmark_id']
                benchmark_title = best_benchmark_match['benchmark_title']
            else:
                
                message_count_deviation = 0
                coverage_percentage = 0
                precision = 0
                recall = 0
                benchmark_cluster_id = 'N/A'
                benchmark_title = 'N/A'
                missing_count = 0
                extra_count = 0
                matched_count = 0
            
            
            rouge_metrics = rouge_data.get(model, {
                'overall_score': 0.0,
                'rouge_similarity': 0.0,
                'rouge_consistency': 0.0,
                'coverage': 0.0,
                'recommendation': 'N/A'
            })
            
            
            enhanced_row = {
                
                'MODEL': model,
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
                
                
                'BENCHMARK_CLUSTER_ID': benchmark_cluster_id,
                'BENCHMARK_TITLE': benchmark_title,
                'BENCHMARK_MESSAGE_COUNT': best_benchmark_match['benchmark_count'] if best_benchmark_match else 0,
                'LLM_MESSAGE_COUNT': len(message_ids),
                'MATCHED_MESSAGES': matched_count,
                'MISSING_MESSAGES': missing_count,
                'EXTRA_MESSAGES': extra_count,
                'MESSAGE_COUNT_DEVIATION_PERCENT': round(message_count_deviation, 2),
                'COVERAGE_PERCENTAGE': round(coverage_percentage, 2),
                'PRECISION_PERCENT': round(precision, 2),
                'RECALL_PERCENT': round(recall, 2),
                
                
                'ROUGE_OVERALL_SCORE': round(rouge_metrics['overall_score'], 4),
                'ROUGE_SIMILARITY': round(rouge_metrics['rouge_similarity'], 4),
                'ROUGE_CONSISTENCY': round(rouge_metrics['rouge_consistency'], 4),
                'ROUGE_COVERAGE': round(rouge_metrics['coverage'], 4),
                'ROUGE_RECOMMENDATION': rouge_metrics['recommendation']
            }
            
            enhanced_results.append(enhanced_row)
    
    return enhanced_results

def main():
    
    benchmark_file = '/home/ubuntu/deemerge/phase_evaluation_engine/phases/phase3_clusters.json'
    original_csv = '/home/ubuntu/deemerge/phase_evaluation_engine/clustering_analysis_expert_complete_simple_format_fixed_final_gpt5_corrected.csv'
    rouge_file = '/home/ubuntu/deemerge/phase_evaluation_engine/enhanced_rouge_results/enhanced_all_models_comparison.csv'
    output_file = '/home/ubuntu/deemerge/phase_evaluation_engine/clustering_analysis_enhanced_with_deviations.csv'
    
    print("Loading benchmark data...")
    benchmark_clusters = load_benchmark_data(benchmark_file)
    print(f"Loaded {len(benchmark_clusters)} benchmark clusters")
    
    print("Loading ROUGE results...")
    rouge_data = load_rouge_results(rouge_file)
    print(f"Loaded ROUGE data for {len(rouge_data)} models")
    
    print("Creating enhanced analysis...")
    enhanced_results = create_enhanced_analysis(original_csv, benchmark_clusters, rouge_data)
    
    print("Writing enhanced results to CSV...")
    with open(output_file, 'w', newline='') as f:
        if enhanced_results:
            fieldnames = enhanced_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enhanced_results)
    
    print(f"Enhanced analysis complete! Results saved to: {output_file}")
    print(f"Total enhanced rows: {len(enhanced_results)}")
    
    
    print("\n=== SUMMARY STATISTICS ===")
    df = pd.DataFrame(enhanced_results)
    
    
    model_stats = df.groupby('MODEL').agg({
        'MESSAGE_COUNT_DEVIATION_PERCENT': 'mean',
        'COVERAGE_PERCENTAGE': 'mean',
        'PRECISION_PERCENT': 'mean',
        'RECALL_PERCENT': 'mean',
        'ROUGE_OVERALL_SCORE': 'mean',
        'ROUGE_SIMILARITY': 'mean',
        'ROUGE_CONSISTENCY': 'mean',
        'ROUGE_COVERAGE': 'mean'
    }).round(2)
    
    print("\nAverage Performance by Model:")
    print(model_stats)
    
    
    best_rouge = model_stats['ROUGE_OVERALL_SCORE'].idxmax()
    best_coverage = model_stats['COVERAGE_PERCENTAGE'].idxmax()
    best_precision = model_stats['PRECISION_PERCENT'].idxmax()
    best_recall = model_stats['RECALL_PERCENT'].idxmax()
    
    print(f"\nBest ROUGE Overall Score: {best_rouge} ({model_stats.loc[best_rouge, 'ROUGE_OVERALL_SCORE']:.4f})")
    print(f"Best Coverage: {best_coverage} ({model_stats.loc[best_coverage, 'COVERAGE_PERCENTAGE']:.2f}%)")
    print(f"Best Precision: {best_precision} ({model_stats.loc[best_precision, 'PRECISION_PERCENT']:.2f}%)")
    print(f"Best Recall: {best_recall} ({model_stats.loc[best_recall, 'RECALL_PERCENT']:.2f}%)")

if __name__ == "__main__":
    main()
