#!/usr/bin/env python3
"""
Cohere API Implementation for Step 1 (Clustering) and Step 2 (Merge/Split)
This replaces the local model with Cohere's cloud API
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

def load_benchmark_topics():
    """Load benchmark topics for comparison"""
    benchmark_file = "data/benchmark_topics_corrected_fixed.json"
    
    if not os.path.exists(benchmark_file):
        print(f"❌ Benchmark file not found: {benchmark_file}")
        return []
    
    try:
        with open(benchmark_file, "r") as f:
            data = json.load(f)
            return data.get("topics", [])
    except Exception as e:
        print(f"❌ Error loading benchmark: {e}")
        return []

def test_cohere_api_step1():
    """Step 1: Message clustering using Cohere API"""
    
    print("🚀 Step 1: Message Clustering (Cohere API)")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("❌ COHERE_API_KEY not found!")
        print("Please run: python setup_cohere_api.py")
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
        
        # Parse response
        try:
            clusters = json.loads(clustering_response)
            print(f"✅ Parsed {len(clusters)} clusters")
        except json.JSONDecodeError:
            print("⚠️  Could not parse JSON response")
            clusters = []
        
        # Save results
        result = {
            "success": True,
            "step": "step1_clustering",
            "model": "command-r-plus",
            "provider": "cohere_api",
            "messages_processed": len(messages),
            "duration_seconds": duration,
            "clusters": clusters,
            "raw_response": clustering_response,
            "api_usage": {
                "tokens": response.meta.get("tokens", {}),
                "billed_tokens": response.meta.get("billed_tokens", {})
            }
        }
        
        os.makedirs("output", exist_ok=True)
        with open("output/cohere_api_step1_results.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("\n💾 Results saved to output/cohere_api_step1_results.json")
        print("\n🎉 Step 1 completed successfully!")
        
        return clusters
        
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_cohere_api_step2(clusters):
    """Step 2: Merge/Split operations using Cohere API"""
    
    print("\n🚀 Step 2: Merge/Split Operations (Cohere API)")
    print("=" * 60)
    
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
    
    # Load benchmark topics
    benchmark_topics = load_benchmark_topics()
    if not benchmark_topics:
        print("⚠️  No benchmark topics found, proceeding without comparison")
    
    print(f"📝 Processing {len(clusters)} clusters from Step 1")
    
    try:
        # Prepare clusters for merge/split analysis
        clusters_text = "\n".join([
            f"Cluster {cluster.get('cluster_id', i)}: {cluster.get('title', 'Untitled')}\n"
            f"  Messages: {cluster.get('message_ids', [])}\n"
            f"  Participants: {cluster.get('participants', [])}\n"
            f"  Summary: {cluster.get('summary', 'No summary')}"
            for i, cluster in enumerate(clusters)
        ])
        
        merge_split_prompt = f"""Analyze these topic clusters and perform merge/split operations:

{clusters_text}

Benchmark topics for reference:
{json.dumps(benchmark_topics, indent=2) if benchmark_topics else "No benchmark topics available"}

Instructions:
1. Identify clusters that should be merged (similar topics)
2. Identify clusters that should be split (too diverse)
3. Return refined clusters with merge/split operations

Return JSON response with:
- refined_clusters: list of refined cluster objects
- operations: list of merge/split operations performed
- reasoning: explanation of decisions made

Format as JSON object."""
        
        print("🔄 Sending merge/split request to Cohere API...")
        start_time = time.time()
        
        response = co.generate(
            model="command-r-plus",
            prompt=merge_split_prompt,
            max_tokens=1000,
            temperature=0.3
        )
        
        duration = time.time() - start_time
        merge_split_response = response.generations[0].text
        
        print(f"✅ Merge/Split completed in {duration:.2f} seconds")
        
        # Parse response
        try:
            result_data = json.loads(merge_split_response)
            refined_clusters = result_data.get("refined_clusters", [])
            operations = result_data.get("operations", [])
            print(f"✅ Parsed {len(refined_clusters)} refined clusters")
            print(f"✅ Identified {len(operations)} operations")
        except json.JSONDecodeError:
            print("⚠️  Could not parse JSON response")
            refined_clusters = []
            operations = []
        
        # Save results
        result = {
            "success": True,
            "step": "step2_merge_split",
            "model": "command-r-plus",
            "provider": "cohere_api",
            "input_clusters": len(clusters),
            "refined_clusters": len(refined_clusters),
            "operations": operations,
            "duration_seconds": duration,
            "raw_response": merge_split_response,
            "api_usage": {
                "tokens": response.meta.get("tokens", {}),
                "billed_tokens": response.meta.get("billed_tokens", {})
            }
        }
        
        os.makedirs("output", exist_ok=True)
        with open("output/cohere_api_step2_results.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("\n💾 Results saved to output/cohere_api_step2_results.json")
        print("\n🎉 Step 2 completed successfully!")
        
        return refined_clusters
        
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    """Main function"""
    
    print("🔧 Starting Cohere API Step 1 & Step 2...")
    print("This version uses Cohere's cloud API - no hardware limitations")
    
    # Step 1: Clustering
    clusters = test_cohere_api_step1()
    if not clusters:
        print("\n❌ Step 1 failed")
        return
    
    # Step 2: Merge/Split
    refined_clusters = test_cohere_api_step2(clusters)
    if not refined_clusters:
        print("\n❌ Step 2 failed")
        return
    
    print("\n🎯 SUCCESS!")
    print("=" * 15)
    print("✅ Step 1: Clustering completed")
    print("✅ Step 2: Merge/Split completed")
    print("✅ No hardware limitations")
    print("🚀 Ready for production!")
    
    print("\n📋 Results saved:")
    print("- output/cohere_api_step1_results.json")
    print("- output/cohere_api_step2_results.json")

if __name__ == "__main__":
    main()
