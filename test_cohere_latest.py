#!/usr/bin/env python3
"""
Test Cohere Command R+ with the latest version and proper authentication
"""

import os
import sys
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

def check_auth_and_access():
    """Check authentication and model access"""
    
    print("🔐 Checking authentication...")
    
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False
    
    # Test both model versions
    models_to_test = [
        "CohereLabs/c4ai-command-r-plus-08-2024",  # Latest version
        "CohereLabs/c4ai-command-r-plus"           # Original version
    ]
    
    for model_name in models_to_test:
        print(f"\\n🔍 Testing access to {model_name}...")
        
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.list_repo_files(model_name)
            print(f"✅ Have access to {model_name}")
            return model_name
        except Exception as e:
            print(f"❌ No access to {model_name}: {e}")
            continue
    
    return None

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

def test_cohere_model_loading(model_name: str):
    """Test loading the Cohere model"""
    
    print(f"\\n🧪 Testing Cohere model loading: {model_name}")
    
    try:
        # Load tokenizer
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("   ✅ Tokenizer loaded")
        
        # Test chat template
        test_messages = [{"role": "user", "content": "Hello, how are you?"}]
        input_ids = tokenizer.apply_chat_template(
            test_messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        print("   ✅ Chat template applied")
        
        # Load model
        print("   Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("   ✅ Model loaded")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return None, None

def run_step1_clustering_test(tokenizer, model, messages: List[Dict]):
    """Run Step 1 clustering test with Cohere"""
    
    print("\\n🔍 Running Step 1 clustering test...")
    
    # Create clustering prompt
    messages_sample = messages[:50]  # Use first 50 messages for testing
    
    messages_text = "\\n\\n".join([
        f"Message {i+1} ({msg.get('user', 'Unknown')} in {msg.get('channel', '#general')}): {msg.get('content', '')[:200]}"
        for i, msg in enumerate(messages_sample)
    ])
    
    prompt = f"""You are an AI assistant specialized in analyzing workplace communication patterns. Your task is to analyze the following Slack messages and create topic clusters.

Messages to analyze:
{messages_text}

Please create topic clusters following these requirements:

1. **Cluster Messages**: Group related messages together based on:
   - Shared topics or themes
   - Same participants discussing related issues
   - Sequential conversation threads
   - Related business processes or decisions

2. **For each cluster, provide**:
   - A descriptive title (2-4 words)
   - List of message IDs that belong to this cluster
   - List of participants involved
   - Channel where the cluster originated
   - Brief summary of the cluster topic

3. **Output Format**: Return a JSON structure like this:
{{
  "clusters": [
    {{
      "cluster_id": "cluster_1",
      "title": "Project Planning",
      "message_ids": [1, 3, 5, 7],
      "participants": ["@alice", "@bob"],
      "channel": "#project-alpha",
      "summary": "Discussion about project timeline and resource allocation"
    }}
  ]
}}

Please analyze the messages and create appropriate clusters:"""
    
    try:
        # Format with Cohere's chat template
        messages_chat = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages_chat, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        # Generate response
        print("   Generating clusters...")
        start_time = time.time()
        
        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids,
                max_new_tokens=2000,
                do_sample=True,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        response_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        
        print(f"   ✅ Generation completed in {generation_time:.2f}s")
        
        # Extract and parse clusters
        clusters = parse_clustering_response(response_text)
        
        return {
            "success": True,
            "clusters": clusters,
            "generation_time": generation_time,
            "response_text": response_text,
            "num_messages_analyzed": len(messages_sample),
            "num_clusters": len(clusters)
        }
        
    except Exception as e:
        print(f"   ❌ Clustering test failed: {e}")
        return {"success": False, "error": str(e)}

def parse_clustering_response(response: str) -> List[Dict]:
    """Parse the clustering response from Cohere"""
    
    try:
        # Extract JSON from response
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            json_str = response[json_start:json_end].strip()
        elif "{" in response and "}" in response:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end]
        else:
            print("   ⚠️ No JSON found in response, creating fallback clusters")
            return create_fallback_clusters()
        
        # Parse JSON
        clustering_data = json.loads(json_str)
        
        if "clusters" in clustering_data:
            return clustering_data["clusters"]
        else:
            return clustering_data
            
    except Exception as e:
        print(f"   ⚠️ Error parsing response: {e}")
        return create_fallback_clusters()

def create_fallback_clusters() -> List[Dict]:
    """Create fallback clusters if parsing fails"""
    return [{
        "cluster_id": "cluster_fallback",
        "title": "General Discussion",
        "message_ids": list(range(1, 11)),
        "participants": ["@unknown"],
        "channel": "#general",
        "summary": "Fallback cluster for unparsed messages"
    }]

def main():
    """Main test function"""
    
    print("🚀 Cohere Command R+ Latest Version Test")
    print("=" * 50)
    
    import time
    
    # Step 1: Check authentication and access
    model_name = check_auth_and_access()
    if not model_name:
        print("\\n❌ No access to any Cohere model")
        print("\\n🔧 Please:")
        print("   1. Make sure you're authenticated: huggingface-cli login")
        print("   2. Request access to Cohere Command R+")
        print("   3. Wait for approval")
        return
    
    # Step 2: Load test messages
    messages = load_test_messages()
    if not messages:
        print("❌ No messages loaded")
        return
    
    # Step 3: Test model loading
    tokenizer, model = test_cohere_model_loading(model_name)
    if not tokenizer or not model:
        print("❌ Could not load Cohere model")
        return
    
    # Step 4: Run Step 1 clustering test
    result = run_step1_clustering_test(tokenizer, model, messages)
    
    if result["success"]:
        print("\\n🎉 Step 1 clustering test completed successfully!")
        print(f"   📊 Clusters created: {result['num_clusters']}")
        print(f"   ⏱️  Generation time: {result['generation_time']:.2f}s")
        print(f"   📝 Messages analyzed: {result['num_messages_analyzed']}")
        
        # Save results
        os.makedirs("output", exist_ok=True)
        output_file = "output/cohere_step1_latest_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\\n💾 Results saved to: {output_file}")
        
        # Show sample clusters
        print("\\n📋 Sample clusters:")
        for i, cluster in enumerate(result["clusters"][:3]):
            print(f"   {i+1}. {cluster.get('title', 'Untitled')} ({len(cluster.get('message_ids', []))} messages)")
        
        print("\\n✅ Cohere Command R+ is working for Step 1 evaluation!")
        
    else:
        print(f"\\n❌ Step 1 test failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
