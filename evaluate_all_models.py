#!/usr/bin/env python3
"""
Comprehensive ROUGE evaluation of all Phase 3 clustering models
"""

import json
import os
import glob
from typing import Dict, List, Any
from rouge_clustering_evaluator import RougeClusteringEvaluator
import pandas as pd

def load_model_clusters(file_path: str) -> List[Dict]:
    """Load clusters from a model file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different file structures
        if isinstance(data, list):
            return data
        elif 'clusters' in data:
            return data['clusters']
        elif 'results' in data:
            return data['results']
        else:
            print(f"⚠️  Unknown structure in {file_path}")
            return []
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return []

def extract_model_name(file_path: str) -> str:
    """Extract model name from file path"""
    filename = os.path.basename(file_path)
    # Remove common suffixes
    for suffix in ['_clusters.json', '.json', '_results.json']:
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)]
    return filename

def evaluate_all_models():
    """Evaluate all available models"""
    print("🚀 COMPREHENSIVE ROUGE EVALUATION OF ALL MODELS")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = RougeClusteringEvaluator(
        reference_path="phases/phase3_clusters.json",
        output_dir="comprehensive_rouge_results"
    )
    
    # Find all model files
    model_files = glob.glob("output/phase3_topic_clustering/*.json")
    
    # Filter out non-model files
    exclude_files = ['detailed_analysis.json', 'comprehensive_results.json']
    model_files = [f for f in model_files if not any(ex in f for ex in exclude_files)]
    
    print(f"📁 Found {len(model_files)} model files to evaluate")
    print()
    
    all_results = []
    
    for model_file in model_files:
        model_name = extract_model_name(model_file)
        print(f"🔍 Evaluating {model_name}...")
        
        # Load clusters
        clusters = load_model_clusters(model_file)
        if not clusters:
            print(f"   ⚠️  No clusters found, skipping")
            continue
        
        print(f"   📊 Loaded {len(clusters)} clusters")
        
        try:
            # Run evaluation
            results = evaluator.evaluate_clusters(clusters, model_name)
            
            # Save individual results
            evaluator.save_results(results, model_name)
            
            # Store for comparison
            all_results.append({
                'model_name': model_name,
                'file_path': model_file,
                'clusters_count': len(clusters),
                'adjusted_rand_index': results['clustering_metrics']['adjusted_rand_index'],
                'v_measure': results['clustering_metrics']['v_measure'],
                'homogeneity': results['clustering_metrics']['homogeneity'],
                'completeness': results['clustering_metrics']['completeness'],
                'rouge_l_f1': results['rouge_metrics'].get('avg_rougeL_fmeasure', 0),
                'rouge_1_f1': results['rouge_metrics'].get('avg_rouge1_fmeasure', 0),
                'rouge_2_f1': results['rouge_metrics'].get('avg_rouge2_fmeasure', 0),
                'overall_f1': results['message_metrics']['overall_f1'],
                'overall_precision': results['message_metrics']['overall_precision'],
                'overall_recall': results['message_metrics']['overall_recall']
            })
            
            print(f"   ✅ Evaluation completed")
            
        except Exception as e:
            print(f"   ❌ Error evaluating {model_name}: {e}")
            continue
    
    # Create comprehensive comparison
    if all_results:
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE RESULTS COMPARISON")
        print("=" * 60)
        
        # Create DataFrame for easy comparison
        df = pd.DataFrame(all_results)
        
        # Sort by overall F1 score (primary metric)
        df_sorted = df.sort_values('overall_f1', ascending=False)
        
        print("\n🏆 RANKING BY OVERALL F1 SCORE:")
        print(df_sorted[['model_name', 'overall_f1', 'overall_precision', 'overall_recall']].to_string(index=False))
        
        print("\n🏗️  CLUSTERING STRUCTURE METRICS:")
        print(df_sorted[['model_name', 'adjusted_rand_index', 'v_measure', 'homogeneity', 'completeness']].to_string(index=False))
        
        print("\n🔍 ROUGE TEXT SIMILARITY:")
        print(df_sorted[['model_name', 'rouge_l_f1', 'rouge_1_f1', 'rouge_2_f1']].to_string(index=False))
        
        # Save comprehensive results
        comprehensive_file = "comprehensive_rouge_results/all_models_comparison.csv"
        df_sorted.to_csv(comprehensive_file, index=False)
        print(f"\n💾 Comprehensive comparison saved to: {comprehensive_file}")
        
        # Find best model
        best_model = df_sorted.iloc[0]
        print(f"\n🥇 BEST MODEL: {best_model['model_name']}")
        print(f"   Overall F1: {best_model['overall_f1']:.4f}")
        print(f"   Adjusted Rand Index: {best_model['adjusted_rand_index']:.4f}")
        print(f"   ROUGE-L F1: {best_model['rouge_l_f1']:.4f}")
    
    else:
        print("❌ No models were successfully evaluated")

if __name__ == "__main__":
    evaluate_all_models()
