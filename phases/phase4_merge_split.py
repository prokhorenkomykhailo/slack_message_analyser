#!/usr/bin/env python3
"""
Phase 4: Merge/Split Topics Evaluation
Evaluates all models on topic refinement using cosine similarity
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import sys
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_clients import call_model_with_retry
from config.model_config import get_available_models, get_model_config, get_model_cost

class Phase4Evaluator:
    """Evaluates models on topic merge/split operations"""
    
    def __init__(self):
        self.phase_name = "phase4_merge_split"
        self.output_dir = os.path.join("output", self.phase_name)
    os.makedirs(self.output_dir, exist_ok=True)
        
        # Load previous phase results
        self.initial_clusters = self.load_initial_clusters()
        self.messages = self.load_messages()
        
        # Merge/split prompt
        self.prompt_template = self.get_merge_split_prompt()
    
    def load_initial_clusters(self) -> List[Dict]:
        """Load initial clusters from Phase 3"""
        try:
            # Try to load from Phase 3 results
            phase3_dir = os.path.join("output", "phase3_topic_clustering")
            comprehensive_file = os.path.join(phase3_dir, "comprehensive_results.json")
            
            if os.path.exists(comprehensive_file):
                with open(comprehensive_file, "r") as f:
                    results = json.load(f)
                
                # Get the best performing model's clusters
                best_model = self.get_best_phase3_model(results)
                if best_model:
                    return results[best_model]["clusters"]
            
            # Fallback: create dummy clusters for testing
            return self.create_dummy_clusters()
            
        except Exception as e:
            print(f"⚠️  Could not load Phase 3 results: {e}")
            return self.create_dummy_clusters()
    
    def get_best_phase3_model(self, results: Dict) -> str:
        """Get the best performing model from Phase 3"""
        successful_results = {k: v for k, v in results.items() if v["success"]}
        
        if not successful_results:
            return None
        
        # Find model with highest coverage
        best_model = max(successful_results.items(), 
                        key=lambda x: x[1]["metrics"]["coverage"])
        return best_model[0]
    
    def create_dummy_clusters(self) -> List[Dict]:
        """Create dummy clusters for testing"""
        return [
            {
                "cluster_id": "cluster_001",
                "message_ids": [1, 2, 3, 4, 5],
                "draft_title": "Project Planning",
                "participants": ["@alice", "@bob"],
                "channel": "#general"
            },
            {
                "cluster_id": "cluster_002", 
                "message_ids": [6, 7, 8, 9, 10],
                "draft_title": "Technical Discussion",
                "participants": ["@charlie", "@david"],
                "channel": "#tech"
            },
            {
                "cluster_id": "cluster_003",
                "message_ids": [11, 12, 13, 14, 15],
                "draft_title": "Meeting Scheduling",
                "participants": ["@alice", "@eve"],
                "channel": "#general"
            }
        ]
    
    def load_messages(self) -> List[Dict]:
        """Load message dataset"""
        try:
            with open("../message_dataset.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print("❌ message_dataset.json not found. Please run Phase 1 first.")
            return []
    
    def get_merge_split_prompt(self) -> str:
        """Get the merge/split prompt"""
        return """
You are an expert at refining topic clusters by analyzing their similarity and determining when to merge or split them.

Given a set of initial topic clusters, your task is to:

1. **Analyze cluster similarity** based on:
   - Semantic overlap in titles and content
   - Participant overlap
   - Channel relationships
   - Temporal proximity

2. **Make merge/split decisions:**
   - If clusters are very similar (similarity > 0.85), merge them
   - If a cluster is too large or diverse, split it into smaller clusters
   - Keep clusters that are well-defined and distinct

3. **For each refined cluster, provide:**
   - cluster_id: unique identifier
   - message_ids: list of message IDs
   - draft_title: refined title
   - participants: list of users involved
   - channel: primary channel
   - merge_reason: explanation if merged
   - split_reason: explanation if split

**Initial Clusters:**
{clusters}

**Messages:**
{messages}

**Instructions:**
- Use cosine similarity threshold of 0.85 for merging
- Split clusters that have more than 20 messages or cover multiple distinct topics
- Provide clear reasoning for each merge/split decision
- Maintain cluster quality and coherence

**Output Format (JSON):**
{{
  "refined_clusters": [
    {{
      "cluster_id": "cluster_001_refined",
      "message_ids": [1, 2, 3, 4, 5, 6, 7],
      "draft_title": "Project Planning & Scheduling",
      "participants": ["@alice", "@bob", "@eve"],
      "channel": "#general",
      "merge_reason": "Merged planning and scheduling clusters due to high similarity",
      "split_reason": null
    }}
  ],
  "operations": [
    {{
      "operation": "merge",
      "clusters_involved": ["cluster_001", "cluster_003"],
      "reason": "High semantic similarity in project-related discussions"
    }}
  ]
}}

Analyze the clusters and provide the refined results in the specified JSON format.
"""
    
    def format_clusters_for_prompt(self) -> str:
        """Format clusters for the prompt"""
        formatted = []
        for cluster in self.initial_clusters:
            formatted.append(
                f"Cluster {cluster['cluster_id']}: {cluster['draft_title']}\n"
                f"  Messages: {cluster['message_ids']}\n"
                f"  Participants: {cluster['participants']}\n"
                f"  Channel: {cluster['channel']}"
            )
        return "\n".join(formatted)
    
    def format_messages_for_prompt(self) -> str:
        """Format messages for the prompt"""
        formatted = []
        for msg in self.messages[:50]:  # Limit for prompt size
            formatted.append(
                f"ID: {msg['id']} | Channel: {msg['channel']} | User: {msg['user']} | "
                f"Text: {msg['text'][:150]}..."
            )
        return "\n".join(formatted)
    
    def evaluate_model(self, provider: str, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model on merge/split operations"""
        print(f"  🧪 Testing {provider}/{model_name}...")
        
        # Prepare prompt
        clusters_str = self.format_clusters_for_prompt()
        messages_str = self.format_messages_for_prompt()
        prompt = self.prompt_template.replace("{clusters}", clusters_str).replace("{messages}", messages_str)
        
        # Call model
        result = call_model_with_retry(provider, model_name, prompt, max_retries=3)
        
        # Parse response
        refined_clusters = self.parse_merge_split_response(result)
        
        # Calculate metrics
        metrics = self.calculate_merge_split_metrics(refined_clusters)
        
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
            "initial_clusters": self.initial_clusters,
            "refined_clusters": refined_clusters,
            "metrics": metrics,
            "raw_response": result.get("response", ""),
            "error": result.get("error", "")
        }
    
    def parse_merge_split_response(self, result: Dict[str, Any]) -> List[Dict]:
        """Parse the merge/split response from the model"""
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
                parsed = json.loads(json_str)
                return parsed.get("refined_clusters", [])
            else:
                return []
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    ⚠️  JSON parsing error: {e}")
            return []
    
    def calculate_merge_split_metrics(self, refined_clusters: List[Dict]) -> Dict[str, Any]:
        """Calculate merge/split quality metrics"""
        if not refined_clusters:
            return {
                "num_initial_clusters": len(self.initial_clusters),
                "num_refined_clusters": 0,
                "merge_operations": 0,
                "split_operations": 0,
                "avg_cluster_size": 0,
                "coverage": 0.0
            }
        
        # Count messages in refined clusters
        refined_messages = set()
        for cluster in refined_clusters:
            message_ids = cluster.get("message_ids", [])
            refined_messages.update(message_ids)
        
        # Count operations
        merge_ops = sum(1 for cluster in refined_clusters if cluster.get("merge_reason"))
        split_ops = sum(1 for cluster in refined_clusters if cluster.get("split_reason"))
        
        # Calculate metrics
        total_messages = len(self.messages)
        coverage = len(refined_messages) / total_messages if total_messages > 0 else 0
        
        cluster_sizes = [len(cluster.get("message_ids", [])) for cluster in refined_clusters]
        
        return {
            "num_initial_clusters": len(self.initial_clusters),
            "num_refined_clusters": len(refined_clusters),
            "merge_operations": merge_ops,
            "split_operations": split_ops,
            "total_operations": merge_ops + split_ops,
            "avg_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
            "coverage": coverage,
            "cluster_titles": [cluster.get("draft_title", "") for cluster in refined_clusters]
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
    
    def run_evaluation(self):
        """Run evaluation on all available models"""
        print(f"🎯 {self.phase_name.upper()} EVALUATION")
        print("=" * 60)
        
        available_models = get_available_models()
        if not available_models:
            print("❌ No models available. Please check your API keys.")
            return
        
        results = {}
        total_models = 0
        successful_models = 0
        
        for provider, models in available_models.items():
            print(f"\n{'='*20} Testing {provider.upper()} Models {'='*20}")
            
            for model_name in models:
                total_models += 1
                result = self.evaluate_model(provider, model_name)
                results[f"{provider}_{model_name}"] = result
                
                # Save individual result
                output_file = os.path.join(self.output_dir, f"{provider}_{model_name}.json")
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)
                
                # Print status
                status = "✅" if result["success"] else "❌"
                if result["success"]:
                    successful_models += 1
                    metrics = result["metrics"]
                    print(f"  {status} {model_name}: {result['duration']:.2f}s, "
                          f"Operations: {metrics['total_operations']}, "
                          f"Clusters: {metrics['num_refined_clusters']}, "
                          f"Cost: ${result['cost']['total_cost']:.6f}")
                else:
                    print(f"  {status} {model_name}: {result['duration']:.2f}s, "
                          f"Error: {result['error'][:100]}...")
        
        # Save comprehensive results
        comprehensive_output = os.path.join(self.output_dir, "comprehensive_results.json")
        with open(comprehensive_output, "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate summary
        self.generate_summary(results, total_models, successful_models)
    
    def generate_summary(self, results: Dict, total_models: int, successful_models: int):
        """Generate evaluation summary"""
        print(f"\n{'='*60}")
        print("📊 EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total models tested: {total_models}")
        print(f"Successful models: {successful_models}")
        print(f"Success rate: {(successful_models/total_models*100):.1f}%" if total_models > 0 else "0%")
        
        # Find best performing models
        successful_results = {k: v for k, v in results.items() if v["success"]}
        
        if successful_results:
            # Best by operations performed
            best_operations = max(successful_results.items(), 
                                key=lambda x: x[1]["metrics"]["total_operations"])
            
            # Best by cost efficiency
            best_cost_efficiency = min(successful_results.items(),
                                     key=lambda x: x[1]["cost"]["total_cost"])
            
            # Best by speed
            best_speed = min(successful_results.items(),
                           key=lambda x: x[1]["duration"])
            
            print(f"\n🏆 Best Performance:")
            print(f"  Operations: {best_operations[0]} ({best_operations[1]['metrics']['total_operations']} ops)")
            print(f"  Cost Efficiency: {best_cost_efficiency[0]} (${best_cost_efficiency[1]['cost']['total_cost']:.6f})")
            print(f"  Speed: {best_speed[0]} ({best_speed[1]['duration']:.2f}s)")
        
        print(f"\nResults saved to: {self.output_dir}")

if __name__ == "__main__":
    evaluator = Phase4Evaluator()
    evaluator.run_evaluation()
