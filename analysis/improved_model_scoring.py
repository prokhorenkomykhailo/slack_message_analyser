#!/usr/bin/env python3
"""
Improved Model Scoring System
Addresses cluster count mismatches and missing clusters in model evaluation
"""

import csv
import pandas as pd
import numpy as np

def calculate_improved_model_score():
    """Calculate improved model scores that properly handle cluster mismatches"""
    
    input_file = '/home/ubuntu/deemerge/phase_evaluation_engine/llm_analysis_with_benchmark_comparison.csv'
    output_file = '/home/ubuntu/deemerge/phase_evaluation_engine/llm_analysis_with_improved_scores.csv'
    
    print("Calculating improved model scores...")
    
    # Read the data
    df = pd.read_csv(input_file)
    
    # Define expected benchmark clusters
    expected_benchmark_clusters = {
        'eco_bloom_campaign': 'EcoBloom Campaign',
        'fitfusion_rebrand': 'FitFusion Rebranding', 
        'technova_launch': 'TechNova Product Launch',
        'greenscape_report': 'GreenScape Sustainability Report',
        'q3_content_calendar': 'Q3 Content Calendar',
        'urbanedge_strategy': 'UrbanEdge Social Media Strategy'
    }
    
    print(f"Expected benchmark clusters: {len(expected_benchmark_clusters)}")
    
    def calculate_model_performance(model_name):
        """Calculate comprehensive performance metrics for a model"""
        
        model_data = df[df['MODEL'] == model_name]
        
        if len(model_data) == 0:
            return None
            
        # Check if model failed
        if not model_data['SUCCESS'].iloc[0]:
            return {
                'model': model_name,
                'overall_score': 0.0,
                'cluster_count_score': 0.0,
                'coverage_score': 0.0,
                'precision_score': 0.0,
                'deviation_score': 0.0,
                'missing_clusters': len(expected_benchmark_clusters),
                'extra_clusters': 0,
                'total_clusters': 0,
                'expected_clusters': len(expected_benchmark_clusters)
            }
        
        # Get unique benchmark clusters found by this model
        found_benchmark_clusters = set(model_data['BENCHMARK_CLUSTER_ID'].unique())
        expected_cluster_ids = set(expected_benchmark_clusters.keys())
        
        # Calculate cluster coverage metrics
        missing_clusters = expected_cluster_ids - found_benchmark_clusters
        extra_clusters = found_benchmark_clusters - expected_cluster_ids
        total_clusters = len(model_data)
        expected_clusters = len(expected_benchmark_clusters)
        
        # 1. Cluster Count Score (25% weight) - penalize for wrong number of clusters
        cluster_count_ratio = min(expected_clusters, total_clusters) / max(expected_clusters, total_clusters)
        cluster_count_score = cluster_count_ratio * 100
        
        # 2. Cluster Coverage Score (25% weight) - how many expected clusters were found
        coverage_ratio = len(found_benchmark_clusters & expected_cluster_ids) / len(expected_cluster_ids)
        coverage_score = coverage_ratio * 100
        
        # 3. Precision Score (25% weight) - average precision across all clusters
        avg_precision = model_data['PRECISION_PERCENT'].mean()
        precision_score = avg_precision
        
        # 4. Deviation Score (25% weight) - how close message counts are to expected
        avg_deviation = abs(model_data['MESSAGE_COUNT_DEVIATION_PERCENT']).mean()
        deviation_score = max(0, 100 - avg_deviation)
        
        # Calculate weighted overall score
        overall_score = (
            cluster_count_score * 0.25 +
            coverage_score * 0.25 + 
            precision_score * 0.25 +
            deviation_score * 0.25
        )
        
        return {
            'model': model_name,
            'overall_score': round(overall_score, 2),
            'cluster_count_score': round(cluster_count_score, 2),
            'coverage_score': round(coverage_score, 2),
            'precision_score': round(avg_precision, 2),
            'deviation_score': round(deviation_score, 2),
            'missing_clusters': len(missing_clusters),
            'extra_clusters': len(extra_clusters),
            'total_clusters': total_clusters,
            'expected_clusters': expected_clusters,
            'missing_cluster_names': [expected_benchmark_clusters[c] for c in missing_clusters],
            'extra_cluster_names': list(extra_clusters)
        }
    
    # Calculate scores for all models
    model_scores = []
    for model in df['MODEL'].unique():
        score_data = calculate_model_performance(model)
        if score_data:
            model_scores.append(score_data)
    
    # Create DataFrame for analysis
    scores_df = pd.DataFrame(model_scores)
    scores_df = scores_df.sort_values('overall_score', ascending=False)
    
    # Add improved scores to original data
    score_mapping = dict(zip(scores_df['model'], scores_df['overall_score']))
    df['IMPROVED_MODEL_SCORE'] = df['MODEL'].map(score_mapping)
    
    # Save enhanced file
    df.to_csv(output_file, index=False)
    
    print(f"Enhanced file with improved scores saved to: {output_file}")
    print(f"Total rows: {len(df)}")
    
    # Display results
    print("\n" + "="*100)
    print("IMPROVED MODEL SCORING RESULTS")
    print("="*100)
    print(f"{'Rank':<4} {'Model':<25} {'Overall':<8} {'Clusters':<9} {'Coverage':<9} {'Precision':<10} {'Deviation':<10} {'Missing':<8} {'Extra':<6}")
    print("-"*100)
    
    for i, row in scores_df.iterrows():
        print(f"{scores_df.index.get_loc(i)+1:<4} {row['model']:<25} {row['overall_score']:<8.2f} "
              f"{row['total_clusters']:<9} {row['coverage_score']:<9.2f} {row['precision_score']:<10.2f} "
              f"{row['deviation_score']:<10.2f} {row['missing_clusters']:<8} {row['extra_clusters']:<6}")
    
    print("\n" + "="*100)
    print("DETAILED ANALYSIS")
    print("="*100)
    
    for i, row in scores_df.head(5).iterrows():
        print(f"\n{row['model']} (Score: {row['overall_score']:.2f})")
        print(f"  Expected clusters: {row['expected_clusters']}, Generated: {row['total_clusters']}")
        print(f"  Missing clusters: {row['missing_clusters']} - {row['missing_cluster_names']}")
        print(f"  Extra clusters: {row['extra_clusters']} - {row['extra_cluster_names']}")
        print(f"  Component scores:")
        print(f"    Cluster Count: {row['cluster_count_score']:.2f}")
        print(f"    Coverage: {row['coverage_score']:.2f}")
        print(f"    Precision: {row['precision_score']:.2f}")
        print(f"    Deviation: {row['deviation_score']:.2f}")
    
    print("\n" + "="*100)
    print("SCORING METHODOLOGY")
    print("="*100)
    print("• Overall Score = 25% Cluster Count + 25% Coverage + 25% Precision + 25% Deviation")
    print("• Cluster Count: Penalizes models that generate wrong number of clusters")
    print("• Coverage: Rewards finding all expected benchmark clusters")
    print("• Precision: Average precision across all generated clusters")
    print("• Deviation: Penalizes message count deviations from benchmark")
    print("• Missing clusters: Expected clusters not found by the model")
    print("• Extra clusters: Clusters generated that don't match any benchmark")
    
    return scores_df

if __name__ == "__main__":
    calculate_improved_model_score()
