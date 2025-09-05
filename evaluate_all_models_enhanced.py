#!/usr/bin/env python3
"""
Enhanced evaluation of all Phase 3 clustering models
Uses semantic quality metrics rather than rigid structural matching
Better suited for unlimited topics and summary types
"""

import json
import os
import glob
from typing import Dict, List, Any
from enhanced_rouge_evaluator import EnhancedRougeEvaluator
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

def evaluate_all_models_enhanced():
    """Evaluate all models using enhanced semantic quality metrics"""
    print("🚀 ENHANCED EVALUATION: SEMANTIC QUALITY FOCUS")
    print("=" * 70)
    print("🎯 Focus: Topic discovery, summary quality, semantic richness")
    print("📊 Approach: Intrinsic quality (70%) + Reference similarity (30%)")
    print("=" * 70)
    
    # Initialize enhanced evaluator
    evaluator = EnhancedRougeEvaluator(
        reference_path="phases/phase3_clusters.json",
        output_dir="enhanced_rouge_results"
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
        print(f"🔍 Enhanced evaluation of {model_name}...")
        
        # Load clusters
        clusters = load_model_clusters(model_file)
        if not clusters:
            print(f"   ⚠️  No clusters found, skipping")
            continue
        
        print(f"   📊 Loaded {len(clusters)} clusters")
        
        try:
            # Calculate total messages (estimate from cluster data)
            total_messages = 300  # Based on your data
            
            # Run enhanced evaluation
            results = evaluator.comprehensive_evaluation(clusters, model_name, total_messages)
            
            # Save enhanced results
            evaluator.save_results(results, model_name)
            
            # Store for comparison
            combined = results['combined_score']
            intrinsic = results['intrinsic_quality']
            
            all_results.append({
                'model_name': model_name,
                'file_path': model_file,
                'clusters_count': len(clusters),
                'overall_score': combined['overall_score'],
                'intrinsic_score': combined['intrinsic_score'],
                'reference_score': combined['reference_score'],
                'summary_quality': combined['breakdown']['summary_quality'],
                'semantic_richness': combined['breakdown']['semantic_richness'],
                'cluster_balance': combined['breakdown']['cluster_balance'],
                'rouge_similarity': combined['breakdown']['rouge_similarity'],
                'rouge_consistency': combined['breakdown']['rouge_consistency'],
                'coverage': combined['breakdown']['coverage'],
                'avg_cluster_size': intrinsic.get('avg_cluster_size', 0),
                'cluster_size_std': intrinsic.get('cluster_size_std', 0),
                'recommendation': results['recommendation']
            })
            
            print(f"   ✅ Enhanced evaluation completed")
            print(f"   🏆 Overall Score: {combined['overall_score']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error evaluating {model_name}: {e}")
            continue
    
    # Create comprehensive comparison
    if all_results:
        print("\n" + "=" * 70)
        print("📊 ENHANCED EVALUATION RESULTS COMPARISON")
        print("=" * 70)
        
        # Create DataFrame for easy comparison
        df = pd.DataFrame(all_results)
        
        # Sort by overall score (primary metric)
        df_sorted = df.sort_values('overall_score', ascending=False)
        
        print("\n🏆 RANKING BY OVERALL SEMANTIC QUALITY SCORE:")
        print(df_sorted[['model_name', 'overall_score', 'intrinsic_score', 'reference_score']].to_string(index=False))
        
        print("\n🔍 INTRINSIC QUALITY METRICS:")
        print(df_sorted[['model_name', 'summary_quality', 'semantic_richness', 'cluster_balance']].to_string(index=False))
        
        print("\n📚 REFERENCE SIMILARITY METRICS:")
        print(df_sorted[['model_name', 'rouge_similarity', 'rouge_consistency']].to_string(index=False))
        
        print("\n📊 COVERAGE METRICS:")
        print(df_sorted[['model_name', 'coverage', 'clusters_count', 'avg_cluster_size']].to_string(index=False))
        
        print("\n🏗️  CLUSTERING STRUCTURE:")
        print(df_sorted[['model_name', 'clusters_count', 'avg_cluster_size', 'cluster_size_std']].to_string(index=False))
        
        # Save comprehensive results
        comprehensive_file = "enhanced_rouge_results/enhanced_all_models_comparison.csv"
        df_sorted.to_csv(comprehensive_file, index=False)
        print(f"\n💾 Enhanced comparison saved to: {comprehensive_file}")
        
        # Find best model
        best_model = df_sorted.iloc[0]
        print(f"\n🥇 BEST MODEL: {best_model['model_name']}")
        print(f"   Overall Score: {best_model['overall_score']:.4f}")
        print(f"   Intrinsic Quality: {best_model['intrinsic_score']:.4f}")
        print(f"   Reference Similarity: {best_model['reference_score']:.4f}")
        print(f"   Recommendation: {best_model['recommendation']}")
        
        # Show top 3 models
        print(f"\n🏅 TOP 3 MODELS:")
        for i, (_, model) in enumerate(df_sorted.head(3).iterrows(), 1):
            print(f"   {i}. {model['model_name']} - Score: {model['overall_score']:.4f}")
            print(f"      {model['recommendation']}")
    
    else:
        print("❌ No models were successfully evaluated")

def main():
    evaluate_all_models_enhanced()

if __name__ == "__main__":
    main()
