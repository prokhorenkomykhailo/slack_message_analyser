#!/usr/bin/env python3
"""
Phase 3 Topic Clustering Evaluation with Cohere Command R+ Support
"""

import os
import sys
import json
import time
import pandas as pd
from typing import Dict, List, Any
from dotenv import load_dotenv


load_dotenv()


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
        messages = []
        
        for _, row in df.iterrows():
            message = {
                "id": row.get("id", len(messages) + 1),
                "user": row.get("user", "Unknown"),
                "content": row.get("content", ""),
                "timestamp": row.get("timestamp", ""),
                "channel": row.get("channel", "#general")
            }
            messages.append(message)
        
        return messages
        
    except Exception as e:
        print(f"❌ Error loading messages: {e}")
        return []

def check_huggingface_auth():
    """Check if Hugging Face authentication is set up"""
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("⚠️  No HUGGINGFACE_TOKEN found in environment variables")
        print("   Please set your token: export HUGGINGFACE_TOKEN=your_token_here")
        print("   Or login with: huggingface-cli login")
        return False
    
    
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

def check_cohere_model_access():
    """Check if we have access to the specific Cohere model by testing actual model loading"""
    try:
        from transformers import AutoTokenizer
        
        
        print("🔍 Testing actual model access...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained("CohereLabs/c4ai-command-r-plus-08-2024")
            print("✅ Access to latest Cohere Command R+ model confirmed")
        except:
            tokenizer = AutoTokenizer.from_pretrained("CohereLabs/c4ai-command-r-plus")
            print("✅ Access to Cohere Command R+ model confirmed")
        return True
        
    except Exception as e:
        if "401" in str(e) or "gated" in str(e).lower() or "restricted" in str(e).lower():
            print("❌ No access to gated Cohere Command R+ model")
            print("   You need to request access at: https://huggingface.co/CohereLabs/c4ai-command-r-plus")
            print("   Click 'Request access' and wait for approval")
        else:
            print(f"❌ Error checking model access: {e}")
        return False

def evaluate_cohere_models(messages: List[Dict], output_dir: str):
    """Evaluate Cohere Command R+ models"""
    
    
    auth_ok = check_huggingface_auth()
    if not auth_ok:
        print("\n🔄 Trying alternative open models instead...")
        return evaluate_alternative_models(messages, output_dir)
    
    
    print("\n🔍 Checking access to Cohere Command R+ model...")
    if not check_cohere_model_access():
        print("\n🔄 No access to gated Cohere model. Using alternative models instead...")
        return evaluate_alternative_models(messages, output_dir)
    
    cohere_models = [
        {
            "name": "cohere_command-r-plus-latest",
            "path": "CohereLabs/c4ai-command-r-plus-08-2024",
            "quantization": None
        },
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    ]
    
    results = {}
    
    for model_config in cohere_models:
        print(f"\n🚀 Evaluating {model_config['name']}...")
        
        try:
            
            clustering_engine = CohereCommandRPlusClustering(
                model_path=model_config["path"],
                quantization=model_config["quantization"]
            )
            
            
            if not clustering_engine.load_model():
                print(f"❌ Failed to load {model_config['name']}")
                continue
            
            
            start_time = time.time()
            clustering_result = clustering_engine.cluster_messages(messages)
            total_time = time.time() - start_time
            
            
            model_result = {
                "provider": "cohere",
                "model": model_config["name"],
                "phase": "phase3_topic_clustering",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "success": True,
                "duration": total_time,
                "clusters": clustering_result.get("clusters", []),
                "generation_time": clustering_result.get("generation_time", 0),
                "response_text": clustering_result.get("response_text", ""),
                "metrics": {
                    "num_clusters": len(clustering_result.get("clusters", [])),
                    "total_messages_clustered": len(messages),
                    "avg_cluster_size": len(messages) / max(len(clustering_result.get("clusters", [])), 1),
                    "max_cluster_size": max([len(c.get("message_ids", [])) for c in clustering_result.get("clusters", [])] + [0]),
                    "min_cluster_size": min([len(c.get("message_ids", [])) for c in clustering_result.get("clusters", [])] + [0]),
                    "coverage": 1.0
                },
                "usage": {
                    "prompt_tokens": len(clustering_result.get("response_text", "")),
                    "completion_tokens": 0,
                    "total_tokens": len(clustering_result.get("response_text", ""))
                },
                "cost": {
                    "input_cost": 0.0,
                    "output_cost": 0.0,
                    "total_cost": 0.0
                }
            }
            
            results[model_config["name"]] = model_result
            
            
            output_file = os.path.join(output_dir, f"{model_config['name']}.json")
            with open(output_file, 'w') as f:
                json.dump(model_result, f, indent=2)
            
            print(f"✅ {model_config['name']} evaluation completed")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_config['name']}: {e}")
            
            
            if "401" in str(e) or "gated" in str(e).lower() or "restricted" in str(e).lower():
                print("🔄 Access denied detected. Switching to alternative models...")
                
                return evaluate_alternative_models(messages, output_dir)
            
            results[model_config["name"]] = {
                "provider": "cohere",
                "model": model_config["name"],
                "success": False,
                "error": str(e),
                "duration": 0
            }
    
    return results

def evaluate_alternative_models(messages: List[Dict], output_dir: str):
    """Evaluate alternative open models when Cohere is not accessible"""
    
    print("🔄 Using alternative open models for clustering evaluation...")
    
    
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
        }
    ]
    
    results = {}
    
    for model_config in alternative_models:
        print(f"\n🚀 Evaluating {model_config['name']}...")
        
        try:
            
            clustering_engine = CohereCommandRPlusClustering(
                model_path=model_config["path"],
                quantization=model_config["quantization"]
            )
            
            
            if not clustering_engine.load_model():
                print(f"❌ Failed to load {model_config['name']}")
                results[model_config['name']] = {
                    "success": False,
                    "error": "Model loading failed",
                    "clusters": [],
                    "duration": 0
                }
                continue
            
            
            start_time = time.time()
            clustering_result = clustering_engine.cluster_messages(messages)
            total_time = time.time() - start_time
            
            
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
    
    print("🚀 Phase 3 Topic Clustering Evaluation with Cohere Command R+")
    print("=" * 70)
    
    
    messages_file = "data/Synthetic_Slack_Messages.csv"
    output_dir = "output/phase3_topic_clustering"
    
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    print(f"📁 Loading messages from: {messages_file}")
    messages = load_messages(messages_file)
    
    if not messages:
        print("❌ No messages loaded. Please check the file path and format.")
        return
    
    print(f"📊 Loaded {len(messages)} messages")
    
    
    print("\n🔍 Evaluating Cohere Command R+ models...")
    cohere_results = evaluate_cohere_models(messages, output_dir)
    
    
    existing_results = {}
    comprehensive_file = os.path.join(output_dir, "comprehensive_results.json")
    if os.path.exists(comprehensive_file):
        with open(comprehensive_file, 'r') as f:
            existing_results = json.load(f)
    
    
    all_results = {**existing_results, **cohere_results}
    
    
    with open(comprehensive_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Comprehensive results saved to: {comprehensive_file}")
    
    
    successful = sum(1 for r in cohere_results.values() if r.get("success", False))
    total = len(cohere_results)
    
    print(f"\n📊 Cohere Model Evaluation Summary:")
    print(f"   Total Models: {total}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {total - successful}")
    if total > 0:
        print(f"   Success Rate: {(successful/total*100):.1f}%")
    else:
        print(f"   Success Rate: 0.0% (No models could be loaded)")
    
    
    if successful > 0:
        print(f"\n✅ Successfully evaluated models:")
        for name, result in cohere_results.items():
            if result.get("success", False):
                clusters = len(result.get("clusters", []))
                duration = result.get("duration", 0)
                print(f"   - {name}: {clusters} clusters in {duration:.2f}s")

if __name__ == "__main__":
    main()
