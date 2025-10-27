#!/usr/bin/env python3
"""
Evaluate alternative open models for Phase 3 clustering (skip Cohere entirely)
"""

import os
import sys
import json
import time
import pandas as pd
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from the same directory
try:
    from cohere_clustering import CohereCommandRPlusClustering
except ImportError:
    print("❌ Could not import CohereCommandRPlusClustering")
    print("Make sure you have installed the required dependencies:")
    print("pip install transformers torch accelerate bitsandbytes")
    sys.exit(1)

def load_messages(file_path: str) -> List[Dict]:
    """Load messages from CSV file"""
    
    if not os.path.exists(file_path):
        print(f"❌ Messages file not found: {file_path}")
        return []
    
    try:
        df = pd.read_csv(file_path)
        
        # Convert to the expected format
        messages = []
        for _, row in df.iterrows():
            message = {
                "id": str(row.get('id', len(messages))),
                "content": str(row.get('content', row.get('message', ''))),
                "timestamp": str(row.get('timestamp', '')),
                "user": str(row.get('user', row.get('author', 'unknown')))
            }
            messages.append(message)
        
        print(f"📊 Loaded {len(messages)} messages")
        return messages
        
    except Exception as e:
        print(f"❌ Error loading messages: {e}")
        return []

def evaluate_alternative_models(messages: List[Dict], output_dir: str):
    """Evaluate alternative open models for clustering"""
    
    print("🔄 Using alternative open models for clustering evaluation...")
    
    # Alternative models that don't require special access
    alternative_models = [
        {
            "name": "microsoft_dialoGPT_large",
            "path": "microsoft/DialoGPT-large",
            "quantization": None
        },
        {
            "name": "google_flan_t5_large",
            "path": "google/flan-t5-large", 
            "quantization": None
        },
        {
            "name": "facebook_blenderbot_400M",
            "path": "facebook/blenderbot-400M-distill",
            "quantization": None
        }
    ]
    
    results = {}
    
    for model_config in alternative_models:
        print(f"\n🚀 Evaluating {model_config['name']}...")
        
        try:
            # Initialize clustering engine with alternative model
            clustering_engine = CohereCommandRPlusClustering(
                model_path=model_config["path"],
                quantization=model_config["quantization"]
            )
            
            # Load model
            if not clustering_engine.load_model():
                print(f"❌ Failed to load {model_config['name']}")
                results[model_config['name']] = {
                    "success": False,
                    "error": "Model loading failed",
                    "clusters": [],
                    "duration": 0
                }
                continue
            
            # Perform clustering
            start_time = time.time()
            clustering_result = clustering_engine.cluster_messages(messages)
            total_time = time.time() - start_time
            
            # Prepare results
            model_result = {
                "success": True,
                "model_name": model_config['name'],
                "model_path": model_config['path'],
                "quantization": model_config['quantization'],
                "clusters": clustering_result.get("clusters", []),
                "generation_time": clustering_result.get("generation_time", 0),
                "total_time": total_time,
                "duration": total_time,
                "num_clusters": len(clustering_result.get("clusters", [])),
                "response_text": clustering_result.get("response_text", "")
            }
            
            results[model_config['name']] = model_result
            
            # Save individual result
            output_file = os.path.join(output_dir, f"{model_config['name']}.json")
            with open(output_file, 'w') as f:
                json.dump(model_result, f, indent=2)
            
            print(f"✅ {model_config['name']} completed successfully")
            print(f"   Clusters: {len(clustering_result.get('clusters', []))}")
            print(f"   Duration: {total_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_config['name']}: {e}")
            results[model_config['name']] = {
                "success": False,
                "error": str(e),
                "clusters": [],
                "duration": 0
            }
    
    return results

def main():
    """Main evaluation function"""
    
    print("🚀 Phase 3 Topic Clustering Evaluation with Alternative Models")
    print("=" * 70)
    
    # Configuration
    messages_file = "data/Synthetic_Slack_Messages.csv"
    output_dir = "output/phase3_topic_clustering"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load messages
    print(f"📁 Loading messages from: {messages_file}")
    messages = load_messages(messages_file)
    
    if not messages:
        print("❌ No messages loaded. Exiting.")
        return
    
    # Evaluate alternative models
    print(f"\n🔍 Evaluating alternative models...")
    cohere_results = evaluate_alternative_models(messages, output_dir)
    
    # Save comprehensive results
    comprehensive_results = {
        "evaluation_type": "phase3_topic_clustering_alternative_models",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_messages": len(messages),
        "models_evaluated": len(cohere_results),
        "results": cohere_results
    }
    
    comprehensive_file = os.path.join(output_dir, "comprehensive_results.json")
    with open(comprehensive_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\n💾 Comprehensive results saved to: {comprehensive_file}")
    
    # Summary
    successful = sum(1 for r in cohere_results.values() if r.get("success", False))
    total = len(cohere_results)
    
    print(f"\n📊 Alternative Models Evaluation Summary:")
    print(f"   Total Models: {total}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {total - successful}")
    if total > 0:
        print(f"   Success Rate: {(successful/total*100):.1f}%")
    else:
        print(f"   Success Rate: 0.0% (No models could be loaded)")
    
    # Show successful models
    if successful > 0:
        print(f"\n✅ Successfully evaluated models:")
        for name, result in cohere_results.items():
            if result.get("success", False):
                clusters = len(result.get("clusters", []))
                duration = result.get("duration", 0)
                print(f"   - {name}: {clusters} clusters in {duration:.2f}s")

if __name__ == "__main__":
    main()
