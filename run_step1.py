
import os
import json
import time
import csv
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model context limits and token management
MODEL_LIMITS = {
    "gemini-2.0-flash-001": {"max_tokens": 8192, "context_length": 1000000},
}

# Cost per 1K tokens (Gemini 2.0 Flash pricing)
COST_PER_1K_TOKENS = {
    "gemini-2.0-flash-001": {"input": 0.000075, "output": 0.0003},  # $0.075 per 1M input, $0.30 per 1M output
}

def get_model_config(provider: str, model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model"""
    return MODEL_LIMITS.get(model_name, {})

def get_model_cost(provider: str, model_name: str) -> Dict[str, float]:
    """Get cost configuration for a specific model"""
    return COST_PER_1K_TOKENS.get(model_name, {"input": 0.0, "output": 0.0})

def call_model_with_retry(provider: str, model_name: str, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    """Call model with retry logic"""
    import google.generativeai as genai
    
    for attempt in range(max_retries):
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(model_name)
            
            start_time = time.time()
            response = model.generate_content(prompt)
            duration = time.time() - start_time
            
            # Simple token estimation since usage_metadata is causing issues
            prompt_tokens = len(prompt) // 4
            response_tokens = len(response.text) // 4
            
            return {
                "success": True,
                "response": response.text,
                "duration": duration,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": response_tokens,
                    "total_tokens": prompt_tokens + response_tokens
                }
            }
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "success": False,
                    "error": str(e),
                    "duration": 0,
                    "usage": {}
                }
            time.sleep(1)  # Wait before retry


class Phase3Evaluator:
    """Evaluates models on topic clustering task using synthetic Slack messages"""
    
    def __init__(self):
        self.phase_name = "phase3_topic_clustering"
        self.output_dir = os.path.join("output", self.phase_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data from CSV
        self.messages = self.load_messages_from_csv()
        
        # Topic clustering prompt
        self.prompt_template = self.get_clustering_prompt()
        
        print(f"📊 Loaded {len(self.messages)} messages from CSV")
    
    def load_reference_clusters(self) -> List[Dict]:
        """Load reference clusters from phase3_clusters.json"""
        reference_file = os.path.join("phases", "phase3_clusters.json")
        try:
            with open(reference_file, "r", encoding='utf-8') as f:
                clusters = json.load(f)
            print(f"✅ Loaded reference clusters from {reference_file}")
            return clusters
        except FileNotFoundError:
            print(f"❌ {reference_file} not found. Please ensure the reference clusters file exists.")
            return []
        except Exception as e:
            print(f"❌ Error loading reference clusters: {e}")
            return []
    
    def load_messages_from_csv(self) -> List[Dict]:
        """Load message dataset from CSV file"""
        csv_path = os.path.join("data", "Synthetic_Slack_Messages.csv")
        try:
            messages = []
            with open(csv_path, "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    # Create a message object with an ID
                    message = {
                        "id": i + 1,  # Add sequential ID
                        "channel": row["channel"],
                        "user": row["user_name"],
                        "user_id": row["user_id"],
                        "timestamp": row["timestamp"],
                        "text": row["text"],
                        "thread_ts": row["thread_id"] if row["thread_id"] != "None" else None
                    }
                    messages.append(message)
            return messages
        except FileNotFoundError:
            print(f"❌ {csv_path} not found. Please ensure the CSV file exists.")
            return []
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return []
    
    def get_clustering_prompt(self) -> str:
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
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)"""
        return len(text) // 4
    
    def format_messages_for_prompt(self, max_tokens: int = 80000) -> str:
        """Format messages for the prompt with token management"""
        formatted = []
        total_tokens = 0
        
        for msg in self.messages:
            # Format each message
            msg_text = (
                f"ID: {msg['id']} | Channel: {msg['channel']} | User: {msg['user']} | "
                f"Thread: {msg.get('thread_ts', 'None')} | Text: {msg['text'][:300]}..."
            )
            
            # Estimate tokens for this message
            msg_tokens = self.estimate_tokens(msg_text)
            
            # Check if adding this message would exceed the limit
            if total_tokens + msg_tokens > max_tokens:
                print(f"⚠️  Token limit reached. Using {len(formatted)} messages out of {len(self.messages)}")
                break
            
            formatted.append(msg_text)
            total_tokens += msg_tokens
        
        return "\n".join(formatted)
    
    def evaluate_model(self, provider: str, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model on topic clustering"""
        print(f"  🧪 Testing {provider}/{model_name}...")
        
        # Get model limits
        model_config = get_model_config(provider, model_name)
        max_tokens = MODEL_LIMITS.get(model_name, {}).get("max_tokens", 4096)
        context_length = MODEL_LIMITS.get(model_name, {}).get("context_length", 8192)
        
        # Calculate available tokens for input (reserve some for output)
        available_tokens = min(context_length * 0.8, 80000)  # Conservative limit
        
        # Prepare prompt
        messages_str = self.format_messages_for_prompt(int(available_tokens))
        prompt = self.prompt_template.replace("{messages}", messages_str)
        
        # Estimate prompt tokens
        prompt_tokens = self.estimate_tokens(prompt)
        print(f"    📝 Estimated prompt tokens: {prompt_tokens:,}")
        
        # Call model
        result = call_model_with_retry(provider, model_name, prompt, max_retries=3)
        
        # Parse response
        clusters = self.parse_clustering_response(result)
        
        # Calculate basic metrics
        metrics = self.calculate_clustering_metrics(clusters)
        
        # Calculate cost
        cost = self.calculate_cost(provider, model_name, result)
        
        return {
            "provider": provider,
            "model": model_name,
            "phase": self.phase_name,
            "timestamp": datetime.now().isoformat(),
            "success": result["success"],
            "duration": result["duration"],
            "usage": result["usage"],
            "cost": cost,
            "clusters": clusters,
            "metrics": metrics,
            "error": result.get("error", ""),
            "prompt_tokens_estimated": prompt_tokens,
            "messages_used": len(messages_str.split('\n'))
        }
    
    def parse_clustering_response(self, result: Dict[str, Any]) -> List[Dict]:
        """Parse the clustering response from the model"""
        if not result["success"]:
            return []
        
        try:
            # Try to extract JSON from response
            response_text = result["response"]
            
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
                print("    ⚠️  No JSON found in response")
                return []
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    ⚠️  JSON parsing error: {e}")
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
                        import re
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
                    print(f"    ⚠️  Extracted {len(clusters)} clusters from malformed JSON")
                    return clusters
            except Exception:
                pass
            
            return []
    
    
    def calculate_clustering_metrics(self, clusters: List[Dict]) -> Dict[str, Any]:
        """Calculate clustering quality metrics"""
        if not clusters:
            return {
                "num_clusters": 0,
                "total_messages_clustered": 0,
                "avg_cluster_size": 0,
                "coverage": 0.0,
                "thread_coherence": 0.0
            }
        
        # Count messages in clusters
        clustered_messages = set()
        thread_clusters = {}
        
        for cluster in clusters:
            message_ids = cluster.get("message_ids", [])
            clustered_messages.update(message_ids)
            
            # Track thread coherence
            thread_id = cluster.get("thread_id")
            if thread_id:
                if thread_id not in thread_clusters:
                    thread_clusters[thread_id] = []
                thread_clusters[thread_id].append(cluster["cluster_id"])
        
        total_messages = len(self.messages)
        coverage = len(clustered_messages) / total_messages if total_messages > 0 else 0
        
        cluster_sizes = [len(cluster.get("message_ids", [])) for cluster in clusters]
        
        # Calculate thread coherence (how well threads are kept together)
        thread_coherence = 0.0
        if thread_clusters:
            coherent_threads = sum(1 for thread_cluster_list in thread_clusters.values() 
                                 if len(thread_cluster_list) == 1)
            thread_coherence = coherent_threads / len(thread_clusters)
        
        return {
            "num_clusters": len(clusters),
            "total_messages_clustered": len(clustered_messages),
            "avg_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
            "cluster_titles": [cluster.get("draft_title", "") for cluster in clusters]
        }
    
    def calculate_cost(self, provider: str, model_name: str, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the cost of the API call"""
        cost_config = get_model_cost(provider, model_name)
        usage = result.get("usage", {})
        
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        input_cost = (input_tokens / 1000) * cost_config.get("input", 0)
        output_cost = (output_tokens / 1000) * cost_config.get("output", 0)
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }


def main():
    print("🎯 STEP 1: TOPIC CLUSTERING")
    
    evaluator = Phase3Evaluator()
    print("Running topic clustering...")
    result = evaluator.evaluate_model('google', 'gemini-2.0-flash-001')
    
    if result['success']:
        print(f"✅ SUCCESS! Clusters: {len(result['clusters'])}, Duration: {result['duration']:.2f}s")
        with open("step1_results.json", "w") as f:
            json.dump(result, f, indent=2)
        print("Results saved to: step1_results.json")
    else:
        print(f"❌ FAILED: {result['error']}")


if __name__ == "__main__":
    main()
