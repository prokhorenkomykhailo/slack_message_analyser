#!/usr/bin/env python3
"""
Test script for Cohere Command R+ Step 1 evaluation
This tests the model specifically for corpus-wide topic clustering
"""

import os
import sys
import json
import pandas as pd
from typing import List, Dict, Any

def load_test_messages() -> List[Dict]:
    """Load the 200 synthetic messages for testing"""
    
    messages_file = "data/Synthetic_Slack_Messages.csv"
    
    if not os.path.exists(messages_file):
        print(f"❌ Messages file not found: {messages_file}")
        return []
    
    try:
        df = pd.read_csv(messages_file)
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
        
        print(f"✅ Loaded {len(messages)} messages")
        return messages
        
    except Exception as e:
        print(f"❌ Error loading messages: {e}")
        return []

def test_cohere_access():
    """Test if we can access the Cohere model"""
    
    try:
        from transformers import AutoTokenizer
        
        print("🔍 Testing Cohere Command R+ access...")
        
        # Try to load tokenizer (lightweight test)
        tokenizer = AutoTokenizer.from_pretrained("CohereLabs/c4ai-command-r-plus")
        print("✅ Successfully loaded Cohere tokenizer")
        return True
        
    except Exception as e:
        if "401" in str(e) or "gated" in str(e).lower():
            print("❌ Access denied to Cohere Command R+")
            print("   You need to request access at: https://huggingface.co/CohereLabs/c4ai-command-r-plus")
            print("   Click 'Request access' and wait for approval")
            return False
        else:
            print(f"❌ Error testing access: {e}")
            return False

def run_cohere_step1_test():
    """Run the Cohere Step 1 test"""
    
    print("🚀 Cohere Command R+ Step 1 Test")
    print("=" * 50)
    
    # Step 1: Load messages
    messages = load_test_messages()
    if not messages:
        print("❌ No messages loaded")
        return
    
    # Step 2: Test model access
    if not test_cohere_access():
        print("\\n🔄 No access to Cohere model. Testing with alternative approach...")
        return test_alternative_models(messages)
    
    # Step 3: Run Cohere evaluation
    try:
        from cohere_step1_evaluator import CohereStep1Evaluator
        
        evaluator = CohereStep1Evaluator()
        
        if not evaluator.load_model():
            print("❌ Failed to load Cohere model")
            return
        
        print("\\n🔍 Running Step 1 clustering evaluation...")
        result = evaluator.evaluate_step1_clustering(messages)
        
        if result["success"]:
            print("\\n✅ Step 1 evaluation completed successfully!")
            print(f"   📊 Clusters created: {result['num_clusters']}")
            print(f"   ⏱️  Generation time: {result['generation_time']:.2f}s")
            print(f"   📝 Messages analyzed: {result['num_messages_analyzed']}")
            
            # Save results
            output_file = "output/cohere_step1_results.json"
            os.makedirs("output", exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\\n💾 Results saved to: {output_file}")
            
            # Show sample clusters
            print("\\n📋 Sample clusters:")
            for i, cluster in enumerate(result["clusters"][:3]):
                print(f"   {i+1}. {cluster.get('title', 'Untitled')} ({len(cluster.get('message_ids', []))} messages)")
            
        else:
            print(f"❌ Step 1 evaluation failed: {result.get('error', 'Unknown error')}")
            
    except ImportError:
        print("❌ Cohere evaluator not found. Please run setup first:")
        print("   python setup_cohere_step1.py")
    except Exception as e:
        print(f"❌ Error running Cohere evaluation: {e}")

def test_alternative_models(messages: List[Dict]):
    """Test with alternative models if Cohere is not accessible"""
    
    print("\\n🔄 Testing with alternative models...")
    
    # Test with the fixed clustering implementation
    try:
        from cohere_clustering import CohereCommandRPlusClustering
        
        # Test with a smaller, accessible model
        test_models = [
            {
                "name": "microsoft_dialoGPT_large",
                "path": "microsoft/DialoGPT-large"
            }
        ]
        
        for model_config in test_models:
            print(f"\\n🚀 Testing {model_config['name']}...")
            
            clustering_engine = CohereCommandRPlusClustering(model_config["path"])
            
            if clustering_engine.load_model():
                result = clustering_engine.cluster_messages(messages[:50])  # Test with first 50 messages
                
                if result.get("clusters"):
                    print(f"   ✅ {model_config['name']} completed successfully")
                    print(f"   📊 Clusters: {len(result['clusters'])}")
                    print(f"   ⏱️  Time: {result.get('generation_time', 0):.2f}s")
                else:
                    print(f"   ❌ {model_config['name']} failed to generate clusters")
            else:
                print(f"   ❌ {model_config['name']} failed to load")
                
    except Exception as e:
        print(f"❌ Error testing alternative models: {e}")

def main():
    """Main test function"""
    
    print("🎯 Cohere Step 1 Evaluation Test")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("data/Synthetic_Slack_Messages.csv"):
        print("❌ Please run this from the project root directory")
        print("   Expected file: data/Synthetic_Slack_Messages.csv")
        return
    
    # Run the test
    run_cohere_step1_test()
    
    print("\\n📋 Summary:")
    print("   - If you see 'Access denied', request access to Cohere Command R+")
    print("   - If you see successful results, the model is working")
    print("   - Check output/cohere_step1_results.json for detailed results")

if __name__ == "__main__":
    main()
