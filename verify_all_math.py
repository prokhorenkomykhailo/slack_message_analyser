#!/usr/bin/env python3
"""
Verify ALL mathematical values in the Excel sheet are correct
Check every calculation step by step
"""

import pandas as pd
import json

def verify_all_math():
    """Verify every mathematical calculation in the Excel analysis"""
    
    print("="*80)
    print("MATHEMATICAL VERIFICATION OF ALL VALUES")
    print("="*80)
    print()
    
    # Load the original data that produced 91.1
    original_df = pd.read_csv('llm_analysis_with_improved_scores.csv')
    gemini_data = original_df[original_df['MODEL'] == 'gemini-1.5-flash'].copy()
    
    print("STEP 1: VERIFY INPUT DATA")
    print("-" * 40)
    print(f"Number of Gemini clusters: {len(gemini_data)}")
    print()
    
    # Verify each topic's data
    print("Individual Topic Data Verification:")
    for _, row in gemini_data.iterrows():
        print(f"Topic {row['CLUSTER_ID']}:")
        print(f"  Title: {row['CLUSTER_TITLE']}")
        print(f"  Benchmark Messages: {row['BENCHMARK_MESSAGE_COUNT']}")
        print(f"  LLM Messages: {row['LLM_MESSAGE_COUNT']}")
        print(f"  Matched Messages: {row['MATCHED_MESSAGES']}")
        print(f"  Missing Messages: {row['MISSING_MESSAGES']}")
        print(f"  Extra Messages: {row['EXTRA_MESSAGES']}")
        print(f"  Deviation %: {row['MESSAGE_COUNT_DEVIATION_PERCENT']:.2f}%")
        print(f"  Coverage %: {row['COVERAGE_PERCENTAGE']:.2f}%")
        print(f"  Precision %: {row['PRECISION_PERCENT']:.2f}%")
        print(f"  Recall %: {row['RECALL_PERCENT']:.2f}%")
        print()
    
    print("STEP 2: VERIFY COMPONENT CALCULATIONS")
    print("-" * 40)
    
    # COMPONENT 1: Cluster Count Score
    expected_clusters = 6
    total_clusters = len(gemini_data)
    
    cluster_count_ratio = min(expected_clusters, total_clusters) / max(expected_clusters, total_clusters)
    cluster_count_score = cluster_count_ratio * 100
    
    print("COMPONENT 1: CLUSTER COUNT SCORE")
    print(f"Expected clusters: {expected_clusters}")
    print(f"Generated clusters: {total_clusters}")
    print(f"min({expected_clusters}, {total_clusters}) = {min(expected_clusters, total_clusters)}")
    print(f"max({expected_clusters}, {total_clusters}) = {max(expected_clusters, total_clusters)}")
    print(f"Ratio: {min(expected_clusters, total_clusters)} / {max(expected_clusters, total_clusters)} = {cluster_count_ratio:.6f}")
    print(f"Score: {cluster_count_ratio:.6f} × 100 = {cluster_count_score:.2f}")
    print()
    
    # COMPONENT 2: Coverage Score
    unique_benchmark_clusters = set(gemini_data['BENCHMARK_CLUSTER_ID'].unique())
    expected_cluster_ids = {'eco_bloom_campaign', 'fitfusion_rebrand', 'technova_launch', 
                           'greenscape_report', 'q3_content_calendar', 'urbanedge_strategy'}
    
    found_expected = len(unique_benchmark_clusters & expected_cluster_ids)
    total_expected = len(expected_cluster_ids)
    coverage_score = (found_expected / total_expected) * 100
    
    print("COMPONENT 2: COVERAGE SCORE")
    print(f"Expected cluster IDs: {expected_cluster_ids}")
    print(f"Found cluster IDs: {unique_benchmark_clusters}")
    print(f"Intersection: {unique_benchmark_clusters & expected_cluster_ids}")
    print(f"Found expected clusters: {found_expected}")
    print(f"Total expected clusters: {total_expected}")
    print(f"Coverage ratio: {found_expected} / {total_expected} = {found_expected/total_expected:.6f}")
    print(f"Coverage score: {found_expected/total_expected:.6f} × 100 = {coverage_score:.2f}")
    print()
    
    # COMPONENT 3: Precision Score
    precision_values = gemini_data['PRECISION_PERCENT'].tolist()
    precision_sum = sum(precision_values)
    precision_count = len(precision_values)
    avg_precision = precision_sum / precision_count
    
    print("COMPONENT 3: PRECISION SCORE")
    print(f"Individual precisions: {precision_values}")
    print(f"Sum: {' + '.join([f'{p:.2f}' for p in precision_values])} = {precision_sum:.2f}")
    print(f"Count: {precision_count}")
    print(f"Average: {precision_sum:.2f} / {precision_count} = {avg_precision:.6f}")
    print(f"Precision score: {avg_precision:.2f}")
    print()
    
    # COMPONENT 4: Deviation Score
    deviation_values = gemini_data['MESSAGE_COUNT_DEVIATION_PERCENT'].tolist()
    abs_deviation_values = [abs(x) for x in deviation_values]
    deviation_sum = sum(abs_deviation_values)
    deviation_count = len(abs_deviation_values)
    avg_deviation = deviation_sum / deviation_count
    deviation_score = max(0, 100 - avg_deviation)
    
    print("COMPONENT 4: DEVIATION SCORE")
    print(f"Individual deviations: {deviation_values}")
    print(f"Absolute deviations: {abs_deviation_values}")
    print(f"Sum of absolute deviations: {' + '.join([f'{d:.2f}' for d in abs_deviation_values])} = {deviation_sum:.2f}")
    print(f"Count: {deviation_count}")
    print(f"Average absolute deviation: {deviation_sum:.2f} / {deviation_count} = {avg_deviation:.6f}")
    print(f"Deviation score: max(0, 100 - {avg_deviation:.2f}) = {deviation_score:.2f}")
    print()
    
    print("STEP 3: VERIFY FINAL CALCULATION")
    print("-" * 40)
    
    # Final weighted calculation
    component_weights = [0.25, 0.25, 0.25, 0.25]
    component_scores = [cluster_count_score, coverage_score, avg_precision, deviation_score]
    weighted_scores = [score * weight for score, weight in zip(component_scores, component_weights)]
    final_score = sum(weighted_scores)
    
    print("FINAL WEIGHTED CALCULATION:")
    print(f"Components: {component_scores}")
    print(f"Weights: {component_weights}")
    print(f"Weighted components:")
    for i, (score, weight, weighted) in enumerate(zip(component_scores, component_weights, weighted_scores)):
        print(f"  Component {i+1}: {score:.2f} × {weight} = {weighted:.6f}")
    
    print(f"Sum: {' + '.join([f'{w:.6f}' for w in weighted_scores])} = {final_score:.6f}")
    print(f"Final score: {final_score:.2f}")
    print()
    
    print("STEP 4: VERIFICATION AGAINST EXPECTED")
    print("-" * 40)
    expected_score = 91.1
    difference = abs(final_score - expected_score)
    
    print(f"Calculated score: {final_score:.2f}")
    print(f"Expected score: {expected_score}")
    print(f"Difference: {difference:.6f}")
    
    if difference < 0.01:
        print("✅ PERFECT MATCH!")
    elif difference < 0.1:
        print("✅ VERY CLOSE (within rounding tolerance)")
    else:
        print("❌ SIGNIFICANT DIFFERENCE - Need investigation")
    
    print()
    
    print("STEP 5: VERIFY INDIVIDUAL METRIC CALCULATIONS")
    print("-" * 40)
    
    # Verify each topic's calculations
    for _, row in gemini_data.iterrows():
        topic_id = row['CLUSTER_ID']
        benchmark_count = row['BENCHMARK_MESSAGE_COUNT']
        llm_count = row['LLM_MESSAGE_COUNT']
        matched = row['MATCHED_MESSAGES']
        missing = row['MISSING_MESSAGES']
        extra = row['EXTRA_MESSAGES']
        
        # Verify basic arithmetic
        calculated_missing = benchmark_count - matched
        calculated_extra = llm_count - matched
        calculated_deviation = ((llm_count - benchmark_count) / benchmark_count) * 100 if benchmark_count > 0 else 0
        calculated_coverage = (matched / benchmark_count) * 100 if benchmark_count > 0 else 0
        calculated_precision = (matched / llm_count) * 100 if llm_count > 0 else 0
        
        print(f"Topic {topic_id} Verification:")
        print(f"  Missing: {missing} vs calculated {calculated_missing} {'✅' if missing == calculated_missing else '❌'}")
        print(f"  Extra: {extra} vs calculated {calculated_extra} {'✅' if extra == calculated_extra else '❌'}")
        print(f"  Deviation: {row['MESSAGE_COUNT_DEVIATION_PERCENT']:.2f}% vs calculated {calculated_deviation:.2f}% {'✅' if abs(row['MESSAGE_COUNT_DEVIATION_PERCENT'] - calculated_deviation) < 0.01 else '❌'}")
        print(f"  Coverage: {row['COVERAGE_PERCENTAGE']:.2f}% vs calculated {calculated_coverage:.2f}% {'✅' if abs(row['COVERAGE_PERCENTAGE'] - calculated_coverage) < 0.01 else '❌'}")
        print(f"  Precision: {row['PRECISION_PERCENT']:.2f}% vs calculated {calculated_precision:.2f}% {'✅' if abs(row['PRECISION_PERCENT'] - calculated_precision) < 0.01 else '❌'}")
        print()
    
    print("="*80)
    print("MATHEMATICAL VERIFICATION COMPLETE")
    print("="*80)
    
    return {
        'cluster_count_score': cluster_count_score,
        'coverage_score': coverage_score,
        'precision_score': avg_precision,
        'deviation_score': deviation_score,
        'final_score': final_score,
        'expected_score': expected_score,
        'difference': difference
    }

if __name__ == "__main__":
    verify_all_math()
