#!/usr/bin/env python3
"""
Test ROUGE evaluation with a single model
"""

import json
import os
from rouge_clustering_evaluator import RougeClusteringEvaluator

def test_single_model():
    """Test ROUGE evaluation with Google Gemini 1.5 Flash"""
    
    print("🧪 TESTING ROUGE EVALUATION")
    print("=" * 50)
    
    # Check paths
    reference_path = "../phases/phase3_clusters.json"
    model_path = "../output/phase3_topic_clustering/google_gemini-1.5-flash.json"
    
    if not os.path.exists(reference_path):
        print(f"❌ Reference file not found: {reference_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    print("✅ Files found, starting evaluation...")
    
    # Initialize evaluator
    evaluator = RougeClusteringEvaluator(
        reference_path=reference_path,
        output_dir="test_results"
    )
    
    # Load model clusters
    try:
        with open(model_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        if 'clusters' in model_data:
            clusters = model_data['clusters']
        else:
            clusters = model_data
        
        print(f"📊 Loaded {len(clusters)} clusters from model")
        
        # Run evaluation
        results = evaluator.evaluate_clusters(clusters, "google_gemini-1.5-flash")
        
        # Save results
        evaluator.save_results(results, "google_gemini-1.5-flash")
        
        # Print summary
        evaluator.print_summary(results)
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

def main():
    test_single_model()

if __name__ == "__main__":
    main()
