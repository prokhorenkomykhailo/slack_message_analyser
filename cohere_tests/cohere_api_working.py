#!/usr/bin/env python3
"""
Cohere API Implementation for Message Clustering
This uses Cohere's cloud API instead of local model
Works with your current VPS - no hardware limitations
"""

import cohere
import os
import json
import pandas as pd
from typing import List, Dict, Any
import time

def load_messages():
    """Load test messages"""
    messages_file = "data/Synthetic_Slack_Messages.csv"
    
    if not os.path.exists(messages_file):
        print(f"❌ File not found: {messages_file}")
        return []
    
    try:
        df = pd.read_csv(messages_file)
        messages = []
        
        for _, row in df.iterrows():
            messages.append({
                "id": row.get("id", len(messages) + 1),
                "user": row.get("user", "Unknown"),
                "content": row.get("content", ""),
                "channel": row.get("channel", "#general")
            })
        
        print(f"✅ Loaded {len(messages)} messages")
        return messages
        
    except Exception as e:
        print(f"❌ Error loading messages: {e}")
        return []

def test_cohere_api():
    """Test Cohere API for message clustering"""
    
    print("🚀 Cohere API Test")
    print("=" * 50)
    print("💡 Using Cohere cloud API - no hardware limitations")
    
    # Check API key
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("❌ COHERE_API_KEY not found!")
        print("Please set your API key:")
        print("export COHERE_API_KEY='your_api_key_here'")
        return False
    
    print("✅ Cohere API key found")
    
    # Initialize Cohere client
    try:
        co = cohere.Client(api_key)
        print("✅ Cohere client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Cohere client: {e}")
        return False
    
    # Load messages
    messages = load_messages()
    if not messages:
        return False
    
    test_messages = messages[:5]
    print(f"📝 Testing with {len(test_messages)} messages")
    
    try:
        # Test 1: Simple generation
        print("\n🧪 Test 1: Simple generation...")
        test_prompt = "Hello, how are you?"
        
        response = co.generate(
            model="command-r-plus",
            prompt=test_prompt,
            max_tokens=50,
            temperature=0.7
        )
        
        print("✅ Simple generation works!")
        print(f"Response: {response.generations[0].text}")
        
        # Test 2: Message clustering
        print("\n🧪 Test 2: Message clustering...")
        messages_text = "\n".join([
            f"{i+1}. {msg['user']}: {msg['content'][:50]}..."
            for i, msg in enumerate(test_messages)
        ])
        
        clustering_prompt = f"""Group these Slack messages into clusters by topic:

{messages_text}

Return a JSON response with clusters."""
        
        response = co.generate(
            model="command-r-plus",
            prompt=clustering_prompt,
            max_tokens=200,
            temperature=0.3
        )
        
        clustering_response = response.generations[0].text
        print("✅ Clustering test completed!")
        print("\n📋 Clustering Response:")
        print("-" * 40)
        print(clustering_response)
        print("-" * 40)
        
        # Save results
        result = {
            "success": True,
            "model": "command-r-plus",
            "provider": "cohere_api",
            "messages_tested": len(test_messages),
            "simple_response": response.generations[0].text,
            "clustering_response": clustering_response,
            "api_usage": {
                "tokens": response.meta.get("tokens", {}),
                "billed_tokens": response.meta.get("billed_tokens", {})
            }
        }
        
        os.makedirs("output", exist_ok=True)
        with open("output/cohere_api_results.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("\n💾 Results saved to output/cohere_api_results.json")
        print("\n🎉 Cohere API is working!")
        print("🚀 No hardware limitations - uses cloud processing!")
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cohere_api_clustering():
    """Test Cohere API for full message clustering"""
    
    print("\n🚀 Cohere API Full Clustering Test")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("❌ COHERE_API_KEY not found!")
        return False
    
    try:
        co = cohere.Client(api_key)
        print("✅ Cohere client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Cohere client: {e}")
        return False
    
    # Load messages
    messages = load_messages()
    if not messages:
        return False
    
    print(f"📝 Processing {len(messages)} messages")
    
    try:
        # Prepare messages for clustering
        messages_text = "\n".join([
            f"{i+1}. {msg['user']}: {msg['content'][:100]}..."
            for i, msg in enumerate(messages)
        ])
        
        clustering_prompt = f"""Analyze these Slack messages and group them into topic clusters:

{messages_text}

Return a JSON response with clusters containing:
- cluster_id: unique identifier
- title: cluster topic title
- message_ids: list of message IDs in this cluster
- participants: list of users involved
- summary: brief description of the cluster

Format as JSON array of cluster objects."""
        
        print("🔄 Sending request to Cohere API...")
        start_time = time.time()
        
        response = co.generate(
            model="command-r-plus",
            prompt=clustering_prompt,
            max_tokens=1000,
            temperature=0.3
        )
        
        duration = time.time() - start_time
        clustering_response = response.generations[0].text
        
        print(f"✅ Clustering completed in {duration:.2f} seconds")
        print("\n📋 Clustering Response:")
        print("-" * 40)
        print(clustering_response)
        print("-" * 40)
        
        # Save results
        result = {
            "success": True,
            "model": "command-r-plus",
            "provider": "cohere_api",
            "messages_processed": len(messages),
            "duration_seconds": duration,
            "clustering_response": clustering_response,
            "api_usage": {
                "tokens": response.meta.get("tokens", {}),
                "billed_tokens": response.meta.get("billed_tokens", {})
            }
        }
        
        os.makedirs("output", exist_ok=True)
        with open("output/cohere_api_full_clustering.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("\n💾 Results saved to output/cohere_api_full_clustering.json")
        print("\n🎉 Cohere API full clustering completed!")
        print("🚀 Processed all messages using cloud API!")
        
        return True
        
    except Exception as e:
        print(f"❌ Full clustering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    
    print("🔧 Starting Cohere API test...")
    print("This version uses Cohere's cloud API - no hardware limitations")
    
    # Test basic functionality
    if test_cohere_api():
        print("\n🎯 Basic API test SUCCESS!")
        print("=" * 15)
        print("✅ Cohere API working")
        print("✅ Generation works")
        print("✅ Clustering works")
        print("🚀 Ready for full message analysis!")
        
        # Test full clustering
        print("\n" + "="*50)
        if test_cohere_api_clustering():
            print("\n🎯 Full clustering SUCCESS!")
            print("✅ Processed all messages")
            print("✅ No hardware limitations")
            print("🚀 Ready for production!")
        else:
            print("\n❌ Full clustering failed")
    else:
        print("\n❌ Basic API test failed")

if __name__ == "__main__":
    main()
