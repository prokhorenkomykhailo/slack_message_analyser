#!/usr/bin/env python3
"""
Phase 5: Metadata Generation Evaluation
Evaluates all models on generating detailed topic metadata
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_clients import call_model_with_retry
from config.model_config import get_available_models, get_model_config, get_model_cost

class Phase5Evaluator:
    """Evaluates models on metadata generation task"""
    
    def __init__(self):
        self.phase_name = "phase5_metadata_generation"
        self.output_dir = os.path.join("output", self.phase_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load refined clusters from Phase 4
        self.refined_clusters = self.load_refined_clusters()
        self.messages = self.load_messages()
        
        # Metadata generation prompt
        self.prompt_template = self.get_metadata_prompt()
    
    def load_refined_clusters(self) -> List[Dict]:
        """Load refined clusters from Phase 4"""
        try:
            # Try to load from Phase 4 results
            phase4_dir = os.path.join("output", "phase4_merge_split")
            comprehensive_file = os.path.join(phase4_dir, "comprehensive_results.json")
            
            if os.path.exists(comprehensive_file):
                with open(comprehensive_file, "r") as f:
                    results = json.load(f)
                
                # Get the best performing model's clusters
                best_model = self.get_best_phase4_model(results)
                if best_model:
                    return results[best_model]["refined_clusters"]
            
            # Fallback: create dummy clusters for testing
            return self.create_dummy_clusters()
            
        except Exception as e:
            print(f"⚠️  Could not load Phase 4 results: {e}")
            return self.create_dummy_clusters()
    
    def get_best_phase4_model(self, results: Dict) -> str:
        """Get the best performing model from Phase 4"""
        successful_results = {k: v for k, v in results.items() if v["success"]}
        
        if not successful_results:
            return None
        
        # Find model with most operations performed
        best_model = max(successful_results.items(), 
                        key=lambda x: x[1]["metrics"]["total_operations"])
        return best_model[0]
    
    def create_dummy_clusters(self) -> List[Dict]:
        """Create dummy clusters for testing"""
        return [
            {
                "cluster_id": "cluster_001_refined",
                "message_ids": [1, 2, 3, 4, 5, 6, 7],
                "draft_title": "Project Planning & Scheduling",
                "participants": ["@alice", "@bob", "@eve"],
                "channel": "#general"
            },
            {
                "cluster_id": "cluster_002_refined",
                "message_ids": [8, 9, 10, 11, 12],
                "draft_title": "Technical Architecture Discussion",
                "participants": ["@charlie", "@david"],
                "channel": "#tech"
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
    
    def get_metadata_prompt(self) -> str:
        """Get the metadata generation prompt"""
        return """
You are an expert at analyzing Slack conversations and extracting comprehensive metadata for topics.

Given a topic cluster with its messages, your task is to generate detailed metadata including:

1. **Title**: Clear, descriptive title for the topic
2. **Summary**: Concise summary of the discussion
3. **Action Items**: List of tasks, decisions, or next steps
4. **Participants**: All users involved in the discussion
5. **Urgency**: Low/Medium/High based on content
6. **Deadline**: Any mentioned deadlines or time constraints
7. **Status**: Current status (active, completed, pending, etc.)
8. **Tags**: Relevant tags for categorization
9. **Thread Root**: ID of the main thread message

**Topic Cluster:**
{cluster_info}

**Messages in this topic:**
{messages}

**Instructions:**
- Extract all action items with owners and due dates
- Identify urgency based on language and context
- Determine status from conversation flow
- Add relevant tags for easy categorization
- Provide comprehensive but concise summary

**Output Format (JSON):**
{{
  "title": "Project Planning & Sprint Setup",
  "summary": "Team discussed Q2 project planning, sprint structure, and resource allocation. Decided on 2-week sprints with daily standups.",
  "action_items": [
    {{
      "task": "Create sprint board",
      "owner": "@alice",
      "due_date": "2024-04-15",
      "priority": "high"
    }}
  ],
  "participants": ["@alice", "@bob", "@charlie"],
  "urgency": "medium",
  "deadline": "2024-04-20",
  "status": "active",
  "tags": ["planning", "sprint", "project"],
  "thread_root": 1,
  "channel": "#general"
}}

Generate comprehensive metadata for this topic cluster in the specified JSON format.
"""
    
    def format_cluster_for_prompt(self, cluster: Dict) -> str:
        """Format a single cluster for the prompt"""
        return f"""
Cluster ID: {cluster['cluster_id']}
Title: {cluster['draft_title']}
Participants: {cluster['participants']}
Channel: {cluster['channel']}
Message IDs: {cluster['message_ids']}
"""
    
    def format_messages_for_prompt(self, message_ids: List[int]) -> str:
        """Format messages for the prompt"""
        formatted = []
        for msg in self.messages:
            if msg['id'] in message_ids:
                formatted.append(
                    f"ID: {msg['id']} | User: {msg['user']} | "
                    f"Thread: {msg.get('thread_ts', 'None')} | "
                    f"Text: {msg['text']}"
                )
        return "\n".join(formatted)
    
    def evaluate_model(self, provider: str, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model on metadata generation"""
        print(f"  🧪 Testing {provider}/{model_name}...")
        
        all_metadata = []
        total_cost = 0
        total_duration = 0
        total_tokens = 0
        
        # Process each cluster
        for cluster in self.refined_clusters:
            # Prepare prompt for this cluster
            cluster_info = self.format_cluster_for_prompt(cluster)
            messages_str = self.format_messages_for_prompt(cluster['message_ids'])
            prompt = self.prompt_template.replace("{cluster_info}", cluster_info).replace("{messages}", messages_str)
            
            # Call model
            result = call_model_with_retry(provider, model_name, prompt, max_retries=3)
            
            # Parse response
            metadata = self.parse_metadata_response(result, cluster['cluster_id'])
            all_metadata.append(metadata)
            
            # Accumulate costs and timing
            total_cost += self.calculate_cost(provider, model_name, result)["total_cost"]
            total_duration += result["duration"]
            total_tokens += result["usage"].get("total_tokens", 0)
        
        # Calculate overall metrics
        metrics = self.calculate_metadata_metrics(all_metadata)
        
        return {
            "provider": provider,
            "model": model_name,
            "phase": self.phase_name,
            "timestamp": datetime.now().isoformat(),
            "success": all(all(m["success"] for m in all_metadata)),
            "duration": total_duration,
            "usage": {"total_tokens": total_tokens},
            "cost": {"total_cost": total_cost},
            "clusters_processed": len(self.refined_clusters),
            "metadata_results": all_metadata,
            "metrics": metrics
        }
    
    def parse_metadata_response(self, result: Dict[str, Any], cluster_id: str) -> Dict[str, Any]:
        """Parse the metadata response from the model"""
        if not result["success"]:
            return {
                "cluster_id": cluster_id,
                "success": False,
                "error": result.get("error", ""),
                "metadata": {}
            }
        
        try:
            # Try to extract JSON from response
            response_text = result["response"]
            
            # Find JSON block
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                metadata = json.loads(json_str)
                return {
                    "cluster_id": cluster_id,
                    "success": True,
                    "metadata": metadata,
                    "raw_response": response_text
                }
            else:
                return {
                    "cluster_id": cluster_id,
                    "success": False,
                    "error": "No JSON found in response",
                    "metadata": {}
                }
        except (json.JSONDecodeError, KeyError) as e:
            return {
                "cluster_id": cluster_id,
                "success": False,
                "error": f"JSON parsing error: {e}",
                "metadata": {}
            }
    
    def calculate_metadata_metrics(self, metadata_results: List[Dict]) -> Dict[str, Any]:
        """Calculate metadata generation quality metrics"""
        successful_results = [m for m in metadata_results if m["success"]]
        
        if not successful_results:
            return {
                "clusters_processed": len(metadata_results),
                "successful_generations": 0,
                "success_rate": 0.0,
                "avg_action_items": 0,
                "avg_tags": 0
            }
        
        # Calculate metrics
        success_rate = len(successful_results) / len(metadata_results)
        
        action_items_counts = []
        tags_counts = []
        
        for result in successful_results:
            metadata = result["metadata"]
            
            # Count action items
            action_items = metadata.get("action_items", [])
            action_items_counts.append(len(action_items))
            
            # Count tags
            tags = metadata.get("tags", [])
            tags_counts.append(len(tags))
        
        return {
            "clusters_processed": len(metadata_results),
            "successful_generations": len(successful_results),
            "success_rate": success_rate,
            "avg_action_items": np.mean(action_items_counts) if action_items_counts else 0,
            "avg_tags": np.mean(tags_counts) if tags_counts else 0,
            "total_action_items": sum(action_items_counts),
            "total_tags": sum(tags_counts)
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
            "total_cost": total_cost
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
                          f"Success Rate: {metrics['success_rate']:.1%}, "
                          f"Action Items: {metrics['total_action_items']}, "
                          f"Cost: ${result['cost']['total_cost']:.6f}")
                else:
                    print(f"  {status} {model_name}: {result['duration']:.2f}s, "
                          f"Clusters: {result['clusters_processed']}")
        
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
            # Best by success rate
            best_success_rate = max(successful_results.items(), 
                                  key=lambda x: x[1]["metrics"]["success_rate"])
            
            # Best by action items generated
            best_action_items = max(successful_results.items(),
                                  key=lambda x: x[1]["metrics"]["total_action_items"])
            
            # Best by cost efficiency
            best_cost_efficiency = min(successful_results.items(),
                                     key=lambda x: x[1]["cost"]["total_cost"])
            
            print(f"\n🏆 Best Performance:")
            print(f"  Success Rate: {best_success_rate[0]} ({best_success_rate[1]['metrics']['success_rate']:.1%})")
            print(f"  Action Items: {best_action_items[0]} ({best_action_items[1]['metrics']['total_action_items']} items)")
            print(f"  Cost Efficiency: {best_cost_efficiency[0]} (${best_cost_efficiency[1]['cost']['total_cost']:.6f})")
        
        print(f"\nResults saved to: {self.output_dir}")

if __name__ == "__main__":
    evaluator = Phase5Evaluator()
    evaluator.run_evaluation()
