#!/usr/bin/env python3
"""
Run ROUGE evaluation on all existing model results
"""

import json
import os
import glob
from typing import Dict, List, Any
import pandas as pd
from rouge_clustering_evaluator import RougeClusteringEvaluator

def load_model_results(model_file: str) -> List[Dict]:
    """Load clusters from a model result file"""
    try:
        with open(model_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        
        if isinstance(data, list):
            return data  
        elif 'clusters' in data:
            return data['clusters']  
        else:
            print(f"⚠️  Unexpected format in {model_file}")
            return []
            
    except Exception as e:
        print(f"❌ Error loading {model_file}: {e}")
        return []

def find_model_files(base_dir: str) -> List[str]:
    """Find all model result files"""
    pattern = os.path.join(base_dir, "google_gemini-*.json")
    files = glob.glob(pattern)
    
    
    model_files = [f for f in files if not f.endswith('_clusters.json') and 'lite' not in f]
    
    return sorted(model_files)

def run_comprehensive_evaluation():
    """Run ROUGE evaluation on all models"""
    
    print("🎯 COMPREHENSIVE ROUGE EVALUATION")
    print("=" * 60)
    
    
    evaluator = RougeClusteringEvaluator(
        reference_path="../phases/phase3_clusters.json",
        output_dir="rouge_results"
    )
    
    
    base_dir = "../output/phase3_topic_clustering"
    model_files = find_model_files(base_dir)
    
    print(f"📁 Found {len(model_files)} model files to evaluate")
    print()
    
    
    all_results = []
    
    
    for model_file in model_files:
        model_name = os.path.basename(model_file).replace('.json', '')
        print(f"🔍 Evaluating {model_name}...")
        
        
        clusters = load_model_results(model_file)
        if not clusters:
            print(f"   ⚠️  No clusters found, skipping")
            continue
        
        print(f"   📊 Loaded {len(clusters)} clusters")
        
        
        try:
            results = evaluator.evaluate_clusters(clusters, model_name)
            
            
            evaluator.save_results(results, model_name)
            
            
            all_results.append(results)
            
            
            evaluator.print_summary(results)
            
        except Exception as e:
            print(f"   ❌ Error evaluating {model_name}: {e}")
            continue
        
        print("-" * 60)
    
    
    if all_results:
        create_comparison_report(all_results)
    
    print("✅ Evaluation complete!")

def create_comparison_report(all_results: List[Dict[str, Any]]):
    """Create a comprehensive comparison report"""
    
    print("\n📊 CREATING COMPREHENSIVE COMPARISON REPORT")
    print("=" * 60)
    
    
    comparison_data = []
    
    for results in all_results:
        model_name = results['model_name']
        clustering = results['clustering_metrics']
        message = results['message_metrics']
        rouge = results['rouge_metrics']
        
        comparison_data.append({
            'Model': model_name,
            'Reference_Clusters': clustering['reference_clusters'],
            'Predicted_Clusters': clustering['predicted_clusters'],
            'ARI': clustering['adjusted_rand_index'],
            'V_Measure': clustering['v_measure'],
            'Homogeneity': clustering['homogeneity'],
            'Completeness': clustering['completeness'],
            'Message_Precision': message['overall_precision'],
            'Message_Recall': message['overall_recall'],
            'Message_F1': message['overall_f1'],
            'ROUGE1_F1': rouge.get('avg_rouge1_fmeasure', 0),
            'ROUGE2_F1': rouge.get('avg_rouge2_fmeasure', 0),
            'ROUGEL_F1': rouge.get('avg_rougeL_fmeasure', 0),
            'Total_Messages': message['total_predicted_messages'],
            'Message_Overlap': message['total_overlap']
        })
    
    
    df = pd.DataFrame(comparison_data)
    
    
    df_sorted = df.sort_values('ROUGEL_F1', ascending=False)
    
    
    comparison_file = "rouge_results/model_comparison.csv"
    df_sorted.to_csv(comparison_file, index=False)
    print(f"💾 Comparison report saved to: {comparison_file}")
    
    
    print("\n🏆 TOP PERFORMING MODELS BY ROUGE-L F1:")
    print("-" * 50)
    for i, row in df_sorted.head(5).iterrows():
        print(f"{i+1}. {row['Model']}")
        print(f"   ROUGE-L F1: {row['ROUGEL_F1']:.4f}")
        print(f"   Message F1: {row['Message_F1']:.4f}")
        print(f"   ARI: {row['ARI']:.4f}")
        print(f"   Clusters: {row['Predicted_Clusters']}")
        print()
    
    
    print("📈 METRIC CORRELATIONS:")
    print("-" * 30)
    correlations = df[['ROUGEL_F1', 'Message_F1', 'ARI', 'V_Measure']].corr()
    print(correlations)
    
    
    corr_file = "rouge_results/metric_correlations.csv"
    correlations.to_csv(corr_file)
    print(f"💾 Correlations saved to: {corr_file}")

def main():
    """Main execution"""
    print("🚀 Starting ROUGE-based clustering evaluation...")
    
    
    if not os.path.exists("../phases/phase3_clusters.json"):
        print("❌ Error: phase3_clusters.json not found!")
        print("   Please run this script from the rouge_evaluation_engine directory")
        return
    
    if not os.path.exists("../output/phase3_topic_clustering"):
        print("❌ Error: model output directory not found!")
        print("   Please run this script from the rouge_evaluation_engine directory")
        return
    
    
    run_comprehensive_evaluation()

if __name__ == "__main__":
    main()
