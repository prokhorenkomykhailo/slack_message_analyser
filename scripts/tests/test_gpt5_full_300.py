#!/usr/bin/env python3
"""
GPT-5 Full 300 Messages Clustering Test
Processes all 300 messages for comprehensive evaluation
"""

import os
import json
import time
import csv
import re
from typing import Dict, List, Any
from datetime import datetime
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_clients import call_model_with_retry

class GPT5FullClusteringTest:
    """GPT-5 clustering test with all 300 messages"""
    
    def __init__(self):
        self.output_dir = "output/phase3_topic_clustering"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data
        self.messages = self.load_messages_from_csv()
        self.reference_clusters = self.load_reference_clusters()
        
        print(f"🚀 GPT-5 Full 300 Messages Clustering Test")
        print(f"📊 Loaded {len(self.messages)} messages")
        print(f"🎯 Loaded {len(self.reference_clusters)} reference clusters")
    
    def load_reference_clusters(self) -> List[Dict]:
        """Load reference clusters"""
        reference_file = os.path.join("phases", "phase3_clusters.json")
        try:
            with open(reference_file, "r", encoding='utf-8') as f:
                clusters = json.load(f)
            return clusters
        except Exception as e:
            print(f"❌ Error loading reference clusters: {e}")
            return []
    
    def load_messages_from_csv(self) -> List[Dict]:
        """Load message dataset"""
        csv_path = os.path.join("data", "Synthetic_Slack_Messages.csv")
        try:
            messages = []
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
            return messages
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return []
    
    def get_clustering_prompt(self) -> str:
        """Get the clustering prompt"""
        return """You are an expert at analyzing Slack conversations and grouping messages into coherent topics.

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
- Pay attention to project names (EcoBloom, FitFusion, TechNova, GreenScape, UrbanEdge)
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
    
    def test_gpt5_full_clustering(self):
        """Test GPT-5 clustering with all 300 messages"""
        print(f"\n🔍 Testing GPT-5 clustering with ALL 300 messages...")
        
        # Use all 300 messages for comprehensive evaluation
        all_messages = self.messages[:300]
        print(f"   📊 Processing all {len(all_messages)} messages for comprehensive evaluation")
        print(f"   ⚠️  This will be more expensive but provide complete results")
        
        # Prepare messages for prompt
        messages_text = ""
        for msg in all_messages:
            messages_text += f"ID: {msg['id']} | Channel: {msg['channel']} | User: {msg['user']} | Thread: {msg.get('thread_ts', 'None')} | Text: {msg['text'][:200]}...\n"
        
        # Full prompt with placeholder replacement
        full_prompt = self.get_clustering_prompt().replace("{messages}", messages_text)
        
        start_time = time.time()
        
        try:
            # Call GPT-5 with all messages
            print(f"   📤 Calling GPT-5 with 300 messages...")
            response = call_model_with_retry(
                provider="openai",
                model_name="gpt-5",
                prompt=full_prompt,
                # GPT-5 specific parameters are handled in the client
            )
            
            duration = time.time() - start_time
            
            if response and response.get('success'):
                print(f"   ✅ GPT-5 response received in {duration:.2f}s")
                
                # Parse response
                clusters = self.parse_gpt5_response(response['response'])
                
                # Save results
                results = {
                    "provider": "openai",
                    "model": "gpt-5",
                    "success": True,
                    "duration": duration,
                    "usage": {
                        "input_tokens": response.get('usage', {}).get('prompt_tokens', 0),
                        "output_tokens": response.get('usage', {}).get('completion_tokens', 0),
                        "total_tokens": response.get('usage', {}).get('total_tokens', 0)
                    },
                    "cost": self.calculate_cost(response.get('usage', {})),
                    "clusters": clusters,
                    "timestamp": datetime.now().isoformat(),
                    "messages_processed": len(all_messages),
                    "test_type": "full_300_messages"
                }
                
                # Save to file
                output_file = os.path.join(self.output_dir, "openai_gpt-5_full_300.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                
                print(f"   💾 Results saved to: {output_file}")
                print(f"   📊 Generated {len(clusters)} clusters")
                print(f"   💰 Cost: ${results['cost']['total_cost']:.6f}")
                print(f"   📈 Coverage: {len(all_messages)} messages processed")
                
                return results
                
            else:
                print(f"   ❌ GPT-5 call failed: {response.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"   ❌ Error testing GPT-5: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def parse_gpt5_response(self, content: str) -> List[Dict]:
        """Parse GPT-5 response to extract clusters"""
        try:
            # Try to extract JSON from response
            response_text = content
            
            # Find JSON block
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                # Try to clean up common JSON issues
                json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                # Remove any trailing commas before closing braces
                json_str = json_str.replace(',}', '}').replace(',]', ']')
                parsed = json.loads(json_str)
                return parsed.get("clusters", [])
            else:
                print(f"   ⚠️  No JSON found in response")
                return []
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"   ⚠️  JSON parsing error: {e}")
            # Try to extract partial clusters if possible
            try:
                # Look for cluster patterns in the response
                clusters = []
                lines = response_text.split('\n')
                current_cluster = None
                
                for line in lines:
                    if '"cluster_id"' in line or '"clusterId"' in line:
                        if current_cluster:
                            clusters.append(current_cluster)
                        current_cluster = {}
                    elif '"message_ids"' in line or '"messageIds"' in line:
                        # Extract message IDs from the line
                        ids = re.findall(r'\d+', line)
                        if current_cluster:
                            current_cluster['message_ids'] = [int(id) for id in ids]
                    elif '"draft_title"' in line:
                        # Extract title
                        title_match = re.search(r'"draft_title":\s*"([^"]+)"', line)
                        if title_match and current_cluster:
                            current_cluster['draft_title'] = title_match.group(1)
                
                if current_cluster:
                    clusters.append(current_cluster)
                
                if clusters:
                    print(f"   ⚠️  Extracted {len(clusters)} clusters from malformed JSON")
                    return clusters
            except Exception:
                pass
            
            return []
    
    def calculate_cost(self, usage: Dict) -> Dict[str, float]:
        """Calculate cost based on GPT-5 pricing"""
        # GPT-5 pricing (official): $1.5625 per 1M input, $12.50 per 1M output
        input_cost_per_1M = 1.5625  # $1.5625 per 1M input tokens
        output_cost_per_1M = 12.50  # $12.50 per 1M output tokens
        
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        
        input_cost = (input_tokens / 1000000) * input_cost_per_1M
        output_cost = (output_tokens / 1000000) * output_cost_per_1M
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

def main():
    """Main function"""
    print("🚀 Starting GPT-5 Full 300 Messages Clustering Test")
    print("=" * 70)
    
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in .env file")
        print("   Please add your OpenAI API key to continue")
        return
    
    # Run test
    tester = GPT5FullClusteringTest()
    results = tester.test_gpt5_full_clustering()
    
    if results:
        print(f"\n🎉 GPT-5 Full 300 Messages Test Completed Successfully!")
        print(f"   📁 Results saved to: output/phase3_topic_clustering/openai_gpt-5_full_300.json")
        print(f"   ⏱️  Duration: {results['duration']:.2f}s")
        print(f"   💰 Total cost: ${results['cost']['total_cost']:.6f}")
        print(f"   📊 Messages processed: {results['messages_processed']}")
        print(f"   🎯 Test type: {results['test_type']}")
    else:
        print(f"\n❌ GPT-5 Full 300 Messages Test Failed")

if __name__ == "__main__":
    main()
