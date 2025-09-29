#!/usr/bin/env python3
"""
Proper evaluation of all Phase 3 clustering models
Uses actual clustering quality metrics that matter
"""

import json
import os
import glob
from typing import Dict, List, Any
from proper_clustering_evaluator import ProperClusteringEvaluator
import pandas as pd

def load_model_data(file_path: str) -> Dict[str, Any]:
    """Load model data including clusters and metadata"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        
        clusters = []
        if isinstance(data, list):
            clusters = data
        elif 'clusters' in data:
            clusters = data['clusters']
        elif 'results' in data:
            clusters = data['results']
        
        
        metadata = {
            'provider': data.get('provider', 'unknown'),
            'model': data.get('model', 'unknown'),
            'success': data.get('success', False),
            'duration': data.get('duration', 0),
            'usage': data.get('usage', {}),
            'cost': data.get('cost', {})
        }
        
        return {
            'clusters': clusters,
            'metadata': metadata
        }
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return {'clusters': [], 'metadata': {}}

def extract_model_name(file_path: str) -> str:
    """Extract model name from file path"""
    filename = os.path.basename(file_path)
    
    for suffix in ['_clusters.json', '.json', '_results.json']:
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)]
    return filename

def evaluate_all_models_proper():
    """Evaluate all models using proper clustering metrics"""
    print("🚀 PROPER CLUSTERING EVALUATION")
    print("=" * 70)
    print("🎯 Focus: Message coverage, cluster size similarity, matching quality")
    print("📊 Approach: Coverage (35%) + Size (20%) + Matching (25%) + Structure (20%)")
    print("=" * 70)
    
    
    evaluator = ProperClusteringEvaluator(
        reference_path="phases/phase3_clusters.json",
        output_dir="proper_clustering_results"
    )
    
    
    model_files = glob.glob("output/phase3_topic_clustering/*.json")
    
    
    exclude_files = ['detailed_analysis.json', 'comprehensive_results.json']
    model_files = [f for f in model_files if not any(ex in f for ex in exclude_files)]
    
    print(f"📁 Found {len(model_files)} model files to evaluate")
    print()
    
    all_results = []
    
    for model_file in model_files:
        model_name = extract_model_name(model_file)
        print(f"🔍 Proper evaluation of {model_name}...")
        
        
        model_data = load_model_data(model_file)
        clusters = model_data['clusters']
        metadata = model_data['metadata']
        
        if not clusters:
            print(f"   ⚠️  No clusters found, skipping")
            continue
        
        print(f"   📊 Loaded {len(clusters)} clusters")
        print(f"   💰 Cost: ${metadata['cost'].get('total_cost', 0):.6f}")
        
        try:
            
            results = evaluator.comprehensive_evaluation(clusters, model_name)
            
            
            evaluator.save_results(results, model_name)
            
            
            overall = results['overall_score']
            coverage = results['coverage_metrics']
            size = results['size_metrics']
            matching = results['matching_metrics']
            
            all_results.append({
                'model_name': model_name,
                'provider': metadata['provider'],
                'model': metadata['model'],
                'file_path': model_file,
                'clusters_count': len(clusters),
                'overall_score': overall['overall_score'],
                'coverage_score': overall['coverage_score'],
                'size_score': overall['size_score'],
                'matching_score': overall['matching_score'],
                'structure_score': overall['structure_score'],
                'message_coverage': coverage['coverage'],
                'covered_messages': coverage['covered_messages'],
                'total_reference_messages': coverage['total_reference_messages'],
                'missing_messages': coverage['missing_messages'],
                'avg_cluster_size': size['predicted_mean_size'],
                'size_similarity': size['overall_size_score'],
                'rouge_l_f1': matching['avg_rouge_l_f1'],
                'message_overlap': matching['avg_message_overlap'],
                'total_cost': metadata['cost'].get('total_cost', 0),
                'input_tokens': metadata['usage'].get('input_tokens', 0),
                'output_tokens': metadata['usage'].get('output_tokens', 0),
                'duration': metadata['duration'],
                'recommendation': results['recommendation']
            })
            
            print(f"   ✅ Proper evaluation completed")
            print(f"   🏆 Overall Score: {overall['overall_score']:.4f}")
            print(f"   📊 Coverage: {coverage['coverage']:.4f} ({coverage['covered_messages']}/{coverage['total_reference_messages']})")
            
        except Exception as e:
            print(f"   ❌ Error evaluating {model_name}: {e}")
            continue
    
    
    if all_results:
        print("\n" + "=" * 70)
        print("📊 PROPER CLUSTERING EVALUATION RESULTS COMPARISON")
        print("=" * 70)
        
        
        df = pd.DataFrame(all_results)
        
        
        df_sorted = df.sort_values('overall_score', ascending=False)
        
        print("\n🏆 RANKING BY OVERALL CLUSTERING QUALITY SCORE:")
        print(df_sorted[['model_name', 'overall_score', 'coverage_score', 'matching_score']].to_string(index=False))
        
        print("\n📊 MESSAGE COVERAGE METRICS (35% weight):")
        print(df_sorted[['model_name', 'message_coverage', 'covered_messages', 'missing_messages']].to_string(index=False))
        
        print("\n🔍 CLUSTER MATCHING QUALITY (25% weight):")
        print(df_sorted[['model_name', 'rouge_l_f1', 'message_overlap', 'matching_score']].to_string(index=False))
        
        print("\n📏 CLUSTER SIZE SIMILARITY (20% weight):")
        print(df_sorted[['model_name', 'size_similarity', 'avg_cluster_size', 'size_score']].to_string(index=False))
        
        print("\n🏗️  CLUSTERING STRUCTURE (20% weight):")
        print(df_sorted[['model_name', 'structure_score', 'clusters_count']].to_string(index=False))
        
        
        comprehensive_file = "proper_clustering_results/proper_all_models_comparison.csv"
        df_sorted.to_csv(comprehensive_file, index=False)
        print(f"\n💾 Proper comparison saved to: {comprehensive_file}")
        
        
        df_sorted['cost_effectiveness'] = df_sorted['overall_score'] / (df_sorted['total_cost'] + 0.000001)  
        
        
        print(f"\n💰 COST-EFFECTIVENESS RANKING (Score per Dollar):")
        print("=" * 80)
        cost_effective_df = df_sorted.sort_values('cost_effectiveness', ascending=False)
        
        for i, (_, model) in enumerate(cost_effective_df.head(10).iterrows(), 1):
            print(f"{i:<3} {model['model_name'][:40]:<40} Score: {model['overall_score']:.4f}, Cost: ${model['total_cost']:.6f}, Ratio: {model['cost_effectiveness']:.2f}")
        
        
        best_model = df_sorted.iloc[0]
        print(f"\n🥇 BEST OVERALL MODEL: {best_model['model_name']}")
        print(f"   Overall Score: {best_model['overall_score']:.4f}")
        print(f"   Coverage: {best_model['message_coverage']:.4f} ({best_model['covered_messages']}/{best_model['total_reference_messages']})")
        print(f"   Matching Quality: {best_model['matching_score']:.4f}")
        print(f"   Cost: ${best_model['total_cost']:.6f}")
        print(f"   Recommendation: {best_model['recommendation']}")
        
        
        most_cost_effective = cost_effective_df.iloc[0]
        print(f"\n💡 MOST COST-EFFECTIVE MODEL: {most_cost_effective['model_name']}")
        print(f"   Cost-Effectiveness: {most_cost_effective['cost_effectiveness']:.2f} score per dollar")
        print(f"   Overall Score: {most_cost_effective['overall_score']:.4f}")
        print(f"   Cost: ${most_cost_effective['total_cost']:.6f}")
        
        
        print(f"\n🏅 TOP 10 MODELS:")
        print("=" * 100)
        print(f"{'Rank':<4} {'Model':<35} {'Score':<8} {'Coverage':<10} {'Cost':<12} {'Provider':<15} {'Duration':<10}")
        print("-" * 100)
        
        for i, (_, model) in enumerate(df_sorted.head(10).iterrows(), 1):
            print(f"{i:<4} {model['model_name'][:34]:<35} {model['overall_score']:<8.4f} {model['message_coverage']:<10.4f} ${model['total_cost']:<11.6f} {model['provider'][:14]:<15} {model['duration']:<10.2f}s")
        
        print("\n📊 DETAILED TOP 10 BREAKDOWN:")
        print("=" * 100)
        
        for i, (_, model) in enumerate(df_sorted.head(10).iterrows(), 1):
            print(f"\n{i}. {model['model_name']}")
            print(f"   🏆 Overall Score: {model['overall_score']:.4f}")
            print(f"   📊 Coverage: {model['message_coverage']:.4f} ({model['covered_messages']}/{model['total_reference_messages']})")
            print(f"   🔍 Matching Quality: {model['matching_score']:.4f}")
            print(f"   💰 Cost: ${model['total_cost']:.6f}")
            print(f"   🚀 Provider: {model['provider']} ({model['model']})")
            print(f"   ⏱️  Duration: {model['duration']:.2f}s")
            print(f"   📈 Tokens: {model['input_tokens']:,} input, {model['output_tokens']:,} output")
            print(f"   💡 Recommendation: {model['recommendation']}")
        
        
        print(f"\n⚠️  MODELS WITH ISSUES (Bottom 5):")
        print("=" * 80)
        for i, (_, model) in enumerate(df_sorted.tail(5).iterrows(), 1):
            rank = len(df_sorted) - i + 1
            print(f"   {rank}. {model['model_name']} - Score: {model['overall_score']:.4f}")
            print(f"      Coverage: {model['message_coverage']:.4f}, Missing: {model['missing_messages']}, Cost: ${model['total_cost']:.6f}")
    
    else:
        print("❌ No models were successfully evaluated")

def main():
    evaluate_all_models_proper()

if __name__ == "__main__":
    main()
