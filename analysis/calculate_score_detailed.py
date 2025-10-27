#!/usr/bin/env python3
"""
Detailed Score Calculation for google_gemini-1.5-flash-latest
Shows exactly how the 0.9426 score was computed
"""

import json
import numpy as np
from proper_clustering_evaluator import ProperClusteringEvaluator

def calculate_detailed_score():
    """Calculate and show detailed breakdown of the score"""
    
    print("🔍 DETAILED SCORE CALCULATION FOR google_gemini-1.5-flash-latest")
    print("=" * 80)
    
    # Initialize evaluator
    evaluator = ProperClusteringEvaluator(
        reference_path="phases/phase3_clusters.json",
        output_dir="detailed_calculation"
    )
    
    # Load the model clusters
    model_file = "output/phase3_topic_clustering/google_gemini-1.5-flash-latest.json"
    
    with open(model_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clusters = data.get('clusters', [])
    print(f"📊 Loaded {len(clusters)} clusters from {model_file}")
    
    # Calculate each component step by step
    print("\n📊 STEP 1: MESSAGE COVERAGE CALCULATION (35% weight)")
    print("-" * 60)
    
    coverage_metrics = evaluator.calculate_message_coverage(clusters)
    coverage_score = coverage_metrics.get('coverage', 0)
    
    print(f"   Total Reference Messages: {coverage_metrics['total_reference_messages']}")
    print(f"   Total Predicted Messages: {coverage_metrics['total_predicted_messages']}")
    print(f"   Covered Messages: {coverage_metrics['covered_messages']}")
    print(f"   Missing Messages: {coverage_metrics['missing_messages']}")
    print(f"   Coverage Score: {coverage_score:.4f}")
    print(f"   Coverage Contribution: {coverage_score * 0.35:.4f} (35% weight)")
    
    print("\n📏 STEP 2: CLUSTER SIZE SIMILARITY CALCULATION (20% weight)")
    print("-" * 60)
    
    size_metrics = evaluator.calculate_cluster_size_similarity(clusters)
    size_score = size_metrics.get('overall_size_score', 0)
    
    print(f"   Reference Mean Size: {size_metrics['reference_mean_size']:.1f}")
    print(f"   Reference Std Size: {size_metrics['reference_std_size']:.1f}")
    print(f"   Predicted Mean Size: {size_metrics['predicted_mean_size']:.1f}")
    print(f"   Predicted Std Size: {size_metrics['predicted_std_size']:.1f}")
    print(f"   Size Similarity: {size_metrics['size_similarity']:.4f}")
    print(f"   Distribution Similarity: {size_metrics['distribution_similarity']:.4f}")
    print(f"   Overall Size Score: {size_score:.4f}")
    print(f"   Size Contribution: {size_score * 0.20:.4f} (20% weight)")
    
    print("\n🔍 STEP 3: CLUSTER MATCHING QUALITY CALCULATION (25% weight)")
    print("-" * 60)
    
    matching_metrics = evaluator.calculate_cluster_matching(clusters)
    matching_score = matching_metrics.get('matching_quality', 0)
    
    print(f"   Average ROUGE-L F1: {matching_metrics['avg_rouge_l_f1']:.4f}")
    print(f"   Average Message Overlap: {matching_metrics['avg_message_overlap']:.4f}")
    print(f"   Overall Matching Quality: {matching_score:.4f}")
    print(f"   Matching Contribution: {matching_score * 0.25:.4f} (25% weight)")
    
    print("\n🏗️  STEP 4: CLUSTERING STRUCTURE CALCULATION (20% weight)")
    print("-" * 60)
    
    structure_metrics = evaluator.calculate_clustering_structure_metrics(clusters)
    
    structure_scores = [
        structure_metrics.get('adjusted_rand_index', 0),
        structure_metrics.get('v_measure', 0),
        structure_metrics.get('homogeneity', 0),
        structure_metrics.get('completeness', 0)
    ]
    
    structure_score = np.mean(structure_scores)
    
    print(f"   Adjusted Rand Index: {structure_metrics.get('adjusted_rand_index', 0):.4f}")
    print(f"   V-Measure: {structure_metrics.get('v_measure', 0):.4f}")
    print(f"   Homogeneity: {structure_metrics.get('homogeneity', 0):.4f}")
    print(f"   Completeness: {structure_metrics.get('completeness', 0):.4f}")
    print(f"   Average Structure Score: {structure_score:.4f}")
    print(f"   Structure Contribution: {structure_score * 0.20:.4f} (20% weight)")
    
    print("\n🏆 STEP 5: FINAL SCORE CALCULATION")
    print("-" * 60)
    
    # Calculate weighted components
    coverage_contribution = coverage_score * 0.35
    size_contribution = size_score * 0.20
    matching_contribution = matching_score * 0.25
    structure_contribution = structure_score * 0.20
    
    # Final score
    final_score = coverage_contribution + size_contribution + matching_contribution + structure_contribution
    
    print(f"   Coverage (35%): {coverage_score:.4f} × 0.35 = {coverage_contribution:.4f}")
    print(f"   Size (20%):     {size_score:.4f} × 0.20 = {size_contribution:.4f}")
    print(f"   Matching (25%): {matching_score:.4f} × 0.25 = {matching_contribution:.4f}")
    print(f"   Structure (20%): {structure_score:.4f} × 0.20 = {structure_contribution:.4f}")
    print(f"   " + "=" * 50)
    print(f"   FINAL SCORE: {final_score:.4f}")
    
    print(f"\n📈 SCORE BREAKDOWN SUMMARY:")
    print(f"   🎯 Coverage: {coverage_score:.4f} (35% weight) → {coverage_contribution:.4f}")
    print(f"   📏 Size Similarity: {size_score:.4f} (20% weight) → {size_contribution:.4f}")
    print(f"   🔍 Cluster Matching: {matching_score:.4f} (25% weight) → {matching_contribution:.4f}")
    print(f"   🏗️  Clustering Structure: {structure_score:.4f} (20% weight) → {structure_contribution:.4f}")
    print(f"   🏆 OVERALL: {final_score:.4f}")
    
    # Show why this model is excellent
    print(f"\n💡 WHY THIS SCORE IS EXCELLENT:")
    if coverage_score >= 0.95:
        print(f"   ✅ Coverage: {coverage_score:.4f} - Nearly perfect message coverage")
    if size_score >= 0.8:
        print(f"   ✅ Size Similarity: {size_score:.4f} - Very similar cluster sizes to reference")
    if matching_score >= 0.8:
        print(f"   ✅ Cluster Matching: {matching_score:.4f} - Excellent semantic and message matching")
    if structure_score >= 0.8:
        print(f"   ✅ Clustering Structure: {structure_score:.4f} - High quality clustering organization")
    
    if final_score >= 0.9:
        print(f"   🏆 Overall: {final_score:.4f} - EXCELLENT clustering quality!")

if __name__ == "__main__":
    calculate_detailed_score()

