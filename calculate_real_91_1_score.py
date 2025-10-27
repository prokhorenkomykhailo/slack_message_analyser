#!/usr/bin/env python3
"""
Calculate the REAL 91.1 IMPROVED_MODEL_SCORE step by step
Using the actual formula from improved_model_scoring.py
"""

import pandas as pd
import json

def calculate_real_91_1_score():
    """Calculate the actual 91.1 score using the real formula"""
    
    print("=== CALCULATING REAL 91.1 IMPROVED_MODEL_SCORE ===")
    print("Using the ACTUAL formula from improved_model_scoring.py")
    print()
    
    # Load the corrected analysis data
    df = pd.read_csv('gemini_1.5_flash_corrected_analysis.csv')
    gemini_data = df[df['MODEL'] == 'gemini-1.5-flash'].copy()
    
    # Load benchmark clusters for comparison
    with open('phases/phase3_clusters.json', 'r') as f:
        clusters_data = json.load(f)
    
    # Define expected benchmark clusters (from the original scoring system)
    expected_benchmark_clusters = {
        'eco_bloom_campaign': 'EcoBloom Campaign',
        'fitfusion_rebrand': 'FitFusion Rebranding', 
        'technova_launch': 'TechNova Product Launch',
        'greenscape_report': 'GreenScape Sustainability Report',
        'q3_content_calendar': 'Q3 Content Calendar',
        'urbanedge_strategy': 'UrbanEdge Social Media Strategy'
    }
    
    print("STEP 1: Input Data Analysis")
    print(f"Expected benchmark clusters: {len(expected_benchmark_clusters)}")
    print(f"Gemini 1.5 Flash generated clusters: {len(gemini_data)}")
    print()
    
    # Get unique benchmark clusters found by Gemini
    found_benchmark_clusters = set(gemini_data['BENCHMARK_CLUSTER_ID'].unique())
    expected_cluster_ids = set(expected_benchmark_clusters.keys())
    
    print("Found benchmark clusters:", found_benchmark_clusters)
    print("Expected cluster IDs:", expected_cluster_ids)
    print()
    
    # Calculate cluster coverage metrics
    missing_clusters = expected_cluster_ids - found_benchmark_clusters
    extra_clusters = found_benchmark_clusters - expected_cluster_ids
    total_clusters = len(gemini_data)
    expected_clusters = len(expected_benchmark_clusters)
    
    print("STEP 2: Cluster Analysis")
    print(f"Missing clusters: {len(missing_clusters)} - {list(missing_clusters)}")
    print(f"Extra clusters: {len(extra_clusters)} - {list(extra_clusters)}")
    print(f"Total clusters generated: {total_clusters}")
    print(f"Expected clusters: {expected_clusters}")
    print()
    
    # COMPONENT 1: Cluster Count Score (25% weight)
    cluster_count_ratio = min(expected_clusters, total_clusters) / max(expected_clusters, total_clusters)
    cluster_count_score = cluster_count_ratio * 100
    
    print("STEP 3: Component 1 - Cluster Count Score (25% weight)")
    print(f"Formula: min(expected, total) / max(expected, total) × 100")
    print(f"Calculation: min({expected_clusters}, {total_clusters}) / max({expected_clusters}, {total_clusters}) × 100")
    print(f"Calculation: {min(expected_clusters, total_clusters)} / {max(expected_clusters, total_clusters)} × 100")
    print(f"Calculation: {cluster_count_ratio:.4f} × 100")
    print(f"Cluster Count Score: {cluster_count_score:.2f}")
    print()
    
    # COMPONENT 2: Cluster Coverage Score (25% weight)
    coverage_ratio = len(found_benchmark_clusters & expected_cluster_ids) / len(expected_cluster_ids)
    coverage_score = coverage_ratio * 100
    
    print("STEP 4: Component 2 - Cluster Coverage Score (25% weight)")
    print(f"Formula: (found ∩ expected) / expected × 100")
    print(f"Found clusters that match expected: {found_benchmark_clusters & expected_cluster_ids}")
    print(f"Calculation: {len(found_benchmark_clusters & expected_cluster_ids)} / {len(expected_cluster_ids)} × 100")
    print(f"Calculation: {coverage_ratio:.4f} × 100")
    print(f"Coverage Score: {coverage_score:.2f}")
    print()
    
    # COMPONENT 3: Precision Score (25% weight)
    avg_precision = gemini_data['PRECISION_PERCENT'].mean()
    precision_score = avg_precision
    
    print("STEP 5: Component 3 - Precision Score (25% weight)")
    print(f"Formula: Average precision across all clusters")
    print("Individual cluster precisions:")
    for _, row in gemini_data.iterrows():
        print(f"  Topic {row['CLUSTER_ID']}: {row['PRECISION_PERCENT']:.1f}%")
    print(f"Calculation: Sum of precisions / number of clusters")
    print(f"Calculation: {gemini_data['PRECISION_PERCENT'].sum():.1f} / {len(gemini_data)}")
    print(f"Precision Score: {precision_score:.2f}")
    print()
    
    # COMPONENT 4: Deviation Score (25% weight)
    avg_deviation = abs(gemini_data['MESSAGE_COUNT_DEVIATION_PERCENT']).mean()
    deviation_score = max(0, 100 - avg_deviation)
    
    print("STEP 6: Component 4 - Deviation Score (25% weight)")
    print(f"Formula: max(0, 100 - average_absolute_deviation)")
    print("Individual cluster deviations:")
    for _, row in gemini_data.iterrows():
        print(f"  Topic {row['CLUSTER_ID']}: {row['MESSAGE_COUNT_DEVIATION_PERCENT']:.1f}% (abs: {abs(row['MESSAGE_COUNT_DEVIATION_PERCENT']):.1f}%)")
    print(f"Average absolute deviation: {avg_deviation:.2f}%")
    print(f"Calculation: max(0, 100 - {avg_deviation:.2f})")
    print(f"Deviation Score: {deviation_score:.2f}")
    print()
    
    # FINAL CALCULATION: Weighted Overall Score
    overall_score = (
        cluster_count_score * 0.25 +
        coverage_score * 0.25 + 
        precision_score * 0.25 +
        deviation_score * 0.25
    )
    
    print("STEP 7: FINAL WEIGHTED CALCULATION")
    print("Formula: (Cluster Count × 0.25) + (Coverage × 0.25) + (Precision × 0.25) + (Deviation × 0.25)")
    print()
    print(f"Calculation:")
    print(f"  ({cluster_count_score:.2f} × 0.25) + ({coverage_score:.2f} × 0.25) + ({precision_score:.2f} × 0.25) + ({deviation_score:.2f} × 0.25)")
    print(f"  {cluster_count_score * 0.25:.2f} + {coverage_score * 0.25:.2f} + {precision_score * 0.25:.2f} + {deviation_score * 0.25:.2f}")
    print(f"  = {overall_score:.2f}")
    print()
    
    print("="*80)
    print(f"🎯 FINAL RESULT: IMPROVED_MODEL_SCORE = {overall_score:.1f}")
    print("="*80)
    print()
    
    # Verification
    expected_score = 91.1
    print("VERIFICATION:")
    print(f"Calculated score: {overall_score:.1f}")
    print(f"Expected score: {expected_score}")
    if abs(overall_score - expected_score) < 0.1:
        print("✅ MATCH! The calculation is correct.")
    else:
        print("❌ MISMATCH! Need to investigate further.")
        print(f"Difference: {abs(overall_score - expected_score):.2f}")
    
    return {
        'cluster_count_score': cluster_count_score,
        'coverage_score': coverage_score,
        'precision_score': precision_score,
        'deviation_score': deviation_score,
        'overall_score': overall_score
    }

if __name__ == "__main__":
    calculate_real_91_1_score()
