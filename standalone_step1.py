#!/usr/bin/env python3
"""
Standalone Step 1: Topic Clustering
Complete script with everything embedded - no external dependencies
"""

import os
import json
import csv
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_messages():
    """Load synthetic Slack messages from CSV"""
    messages = []
    csv_path = "data/Synthetic_Slack_Messages.csv"
    
    try:
        with open(csv_path, "r", encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                message = {
                    "id": i + 1,
                    "channel": row["channel"],
                    "user": row["user_name"],
                    "user_id": row["user_id"],
                    "timestamp": row["timestamp"],
                    "text": row["text"],
                    "thread_ts": row["thread_id"] if row["thread_id"] != "None" else None
                }
                messages.append(message)
        print(f"✅ Loaded {len(messages)} messages")
        return messages
    except Exception as e:
        print(f"❌ Error loading messages: {e}")
        return []


def format_messages_for_prompt(messages, max_tokens=80000):
    """Format messages for the prompt"""
    formatted = []
    total_tokens = 0
    
    for msg in messages:
        msg_text = (
            f"ID: {msg['id']} | Channel: {msg['channel']} | User: {msg['user']} | "
            f"Thread: {msg.get('thread_ts', 'None')} | Text: {msg['text'][:300]}..."
        )
        
        msg_tokens = len(msg_text) // 4  # Rough token estimation
        
        if total_tokens + msg_tokens > max_tokens:
            print(f"⚠️  Token limit reached. Using {len(formatted)} messages out of {len(messages)}")
            break
        
        formatted.append(msg_text)
        total_tokens += msg_tokens
    
    return "\n".join(formatted)

def get_clustering_prompt():
    """Get the topic clustering prompt"""
    return """
You are an expert at analyzing Slack conversations and grouping messages into coherent topics.

Given a set of Slack messages, your task is to:

1. **Group messages into topic clusters** based on:
   - Shared thread relationships (same thread_id)
   - Same participants
   - Semantic similarity (same subject matter)
   - Temporal proximity
   - Channel context

2. **For each cluster, provide:**
   - cluster_id: unique identifier (e.g., "cluster_001")
   - message_ids: list of message IDs in this cluster
   - draft_title: brief descriptive title
   - participants: list of users involved
   - channel: primary channel for this topic
   - thread_id: if messages belong to a specific thread

**Messages to analyze:**
{messages}

**Instructions:**
- Group related messages together based on meaningful relationships
- Each message should belong to exactly one cluster
- Consider both explicit relationships (threads) and implicit ones (same topic)
- Merge related threads and topics into cohesive clusters
- Provide clear, descriptive titles for each cluster
- Pay attention to project names
- Consider temporal relationships and deadlines mentioned
- Focus on creating logical, meaningful groupings rather than arbitrary cluster counts

**Output Format (JSON):**
{{
  "clusters": [
    {{
      "cluster_id": "cluster_001",
      "message_ids": [1, 2, 3, 4, 5],
      "draft_title": "EcoBloom Summer Campaign Planning",
      "participants": ["Devon", "Sam", "Leah", "Jordan"],
      "channel": "#campaign-briefs",
      "thread_id": "thread_001"
    }}
  ]
}}

Analyze the messages and provide the clustering results in the specified JSON format.
"""

def call_gemini(prompt):
    """Call Gemini API"""
    try:
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        start_time = time.time()
        response = model.generate_content(prompt)
        duration = time.time() - start_time
        
        return {
            "success": True,
            "response": response.text,
            "duration": duration
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "duration": 0
        }

def parse_clustering_response(response_text):
    """Parse the clustering response from the model"""
    try:
        # Find JSON block
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            json_str = json_str.replace('\n', ' ').replace('\r', ' ')
            json_str = json_str.replace(',}', '}').replace(',]', ']')
            parsed = json.loads(json_str)
            return parsed.get("clusters", [])
        else:
            print("⚠️  No JSON found in response")
            return []
    except Exception as e:
        print(f"⚠️  JSON parsing error: {e}")
        return []

def calculate_metrics(clusters):
    """Calculate basic metrics"""
    if not clusters:
        return {"num_clusters": 0, "total_messages": 0}
    
    total_messages = sum(len(cluster.get("message_ids", [])) for cluster in clusters)
    
    return {
        "num_clusters": len(clusters),
        "total_messages": total_messages,
        "avg_cluster_size": total_messages / len(clusters) if clusters else 0
    }

def main():
    print("🎯 STANDALONE STEP 1: TOPIC CLUSTERING")
    print("=" * 50)
    
    # Load data
    messages = load_messages()
    if not messages:
        return
    
    # Prepare prompt
    messages_str = format_messages_for_prompt(messages)
    prompt = get_clustering_prompt().replace("{messages}", messages_str)
    
    print(f"📝 Prompt tokens: ~{len(prompt) // 4:,}")
    
    # Call Gemini
    print("\n🧪 Calling Gemini 2.5 Flash...")
    result = call_gemini(prompt)
    
    if result['success']:
        print(f"✅ SUCCESS!")
        print(f"Duration: {result['duration']:.2f}s")
        
        # Parse clusters
        clusters = parse_clustering_response(result['response'])
        
        if clusters:
            metrics = calculate_metrics(clusters)
            
            print(f"\n📊 RESULTS:")
            print(f"Clusters found: {metrics['num_clusters']}")
            print(f"Messages clustered: {metrics['total_messages']}")
            print(f"Average cluster size: {metrics['avg_cluster_size']:.1f}")
            
            # Save results
            output_data = {
                "clusters": clusters,
                "metrics": metrics,
                "duration": result['duration'],
                "raw_response": result['response']
            }
            
            with open("step1_results.json", "w") as f:
                json.dump(output_data, f, indent=2)
            
            with open("step1_clusters.json", "w") as f:
                json.dump(clusters, f, indent=2)
            
            print(f"\n📁 Results saved to:")
            print(f"   - step1_results.json")
            print(f"   - step1_clusters.json")
            
            # Show sample clusters
            print(f"\n📋 Sample clusters:")
            for i, cluster in enumerate(clusters[:3]):
                print(f"   {i+1}. {cluster.get('draft_title', 'No title')}")
                print(f"      Messages: {len(cluster.get('message_ids', []))}")
                print(f"      Participants: {cluster.get('participants', [])}")
        else:
            print("❌ No clusters found in response")
    else:
        print(f"❌ FAILED!")
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()
