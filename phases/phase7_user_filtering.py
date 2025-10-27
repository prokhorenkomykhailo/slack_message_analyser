#!/usr/bin/env python3
"""
Phase 7: User Topic Filtering Evaluation
Evaluates all models on filtering topics per user
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

class Phase7Evaluator:
    """Evaluates models on user-based topic filtering"""
    
    def __init__(self):
        self.phase_name = "phase7_user_filtering"
        self.output_dir = os.path.join("output", self.phase_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load topics from Phase 5
        self.topics = self.load_topics()
        
        # Define test users
        self.test_users = [
            {"name": "alice", "channels": ["#general", "#planning"]},
            {"name": "bob", "channels": ["#general", "#tech"]},
            {"name": "charlie", "channels": ["#tech", "#development"]},
            {"name": "david", "channels": ["#general", "#marketing"]}
        ]
        
        # User filtering prompt
        self.prompt_template = self.get_filtering_prompt()
    
    def load_topics(self) -> List[Dict]:
        """Load topics from Phase 5"""
        try:
            # Try to load from Phase 5 results
            phase5_dir = os.path.join("output", "phase5_metadata_generation")
            comprehensive_file = os.path.join(phase5_dir, "comprehensive_results.json")
            
            if os.path.exists(comprehensive_file):
                with open(comprehensive_file, "r") as f:
                    results = json.load(f)
                
                # Get the best performing model's topics
                best_model = self.get_best_phase5_model(results)
                if best_model:
                    return results[best_model]["metadata_results"]
            
            # Fallback: create dummy topics for testing
            return self.create_dummy_topics()
            
        except Exception as e:
            print(f"⚠️  Could not load Phase 5 results: {e}")
            return self.create_dummy_topics()
    
    def get_best_phase5_model(self, results: Dict) -> str:
        """Get the best performing model from Phase 5"""
        successful_results = {k: v for k, v in results.items() if v["success"]}
        
        if not successful_results:
            return None
        
        # Find model with highest success rate
        best_model = max(successful_results.items(), 
                        key=lambda x: x[1]["metrics"]["success_rate"])
        return best_model[0]
    
    def create_dummy_topics(self) -> List[Dict]:
        """Create dummy topics for testing"""
        return [
            {
                "cluster_id": "cluster_001",
                "success": True,
                "metadata": {
                    "title": "Project Planning & Sprint Setup",
                    "summary": "Team discussed Q2 project planning and sprint structure",
                    "action_items": [
                        {"task": "Create sprint board", "owner": "@alice", "due_date": "2024-04-15"}
                    ],
                    "participants": ["@alice", "@bob"],
                    "urgency": "medium",
                    "tags": ["planning", "sprint"],
                    "channel": "#general"
                }
            },
            {
                "cluster_id": "cluster_002",
                "success": True,
                "metadata": {
                    "title": "Technical Architecture Discussion",
                    "summary": "Discussed system architecture and technical decisions",
                    "action_items": [
                        {"task": "Review architecture docs", "owner": "@charlie", "due_date": "2024-04-20"}
                    ],
                    "participants": ["@charlie", "@david"],
                    "urgency": "high",
                    "tags": ["architecture", "technical"],
                    "channel": "#tech"
                }
            }
        ]
    
    def get_filtering_prompt(self) -> str:
        """Get the user filtering prompt"""
        return """
You are an expert at filtering topics based on user relevance. Your task is to determine which topics should be visible to a specific user based on their channel memberships and involvement.

Given a user and a set of topics, filter topics where:
1. The user is in the topic's channel AND
2. The user is either:
   - A participant in the topic, OR
   - An owner of an open action item in the topic, OR
   - The topic is highly relevant to their role

**User Information:**
{user_info}

**Available Topics:**
{topics}

**Instructions:**
- Only include topics where the user has legitimate access
- Consider channel memberships and topic relevance
- Prioritize topics where the user has action items
- Exclude topics that are not relevant to the user's role

**Output Format (JSON):**
{{
  "user": "alice",
  "visible_topics": [
    {{
      "topic_id": "cluster_001",
      "title": "Project Planning & Sprint Setup",
      "relevance_score": 0.95,
      "reason": "User is participant and has action items",
      "action_items": [
        {{
          "task": "Create sprint board",
          "owner": "@alice",
          "due_date": "2024-04-15"
        }}
      ]
    }}
  ],
  "filtered_out": [
    {{
      "topic_id": "cluster_002",
      "reason": "User not in #tech channel"
    }}
  ]
}}

Filter the topics for this user and provide the results in the specified JSON format.
"""
    
    def format_user_info(self, user: Dict) -> str:
        """Format user information for the prompt"""
        return f"""
Name: {user['name']}
Channels: {', '.join(user['channels'])}
"""
    
    def format_topics_for_prompt(self) -> str:
        """Format topics for the prompt"""
        formatted = []
        for topic in self.topics:
            if not topic["success"]:
                continue
                
            metadata = topic.get("metadata", {})
            formatted.append(f"""
Topic ID: {topic['cluster_id']}
Title: {metadata.get('title', 'N/A')}
Channel: {metadata.get('channel', 'N/A')}
Participants: {metadata.get('participants', [])}
Action Items: {metadata.get('action_items', [])}
Urgency: {metadata.get('urgency', 'N/A')}
Tags: {metadata.get('tags', [])}
""")
        return "\n".join(formatted)
    
    def evaluate_model(self, provider: str, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model on user filtering"""
        print(f"  🧪 Testing {provider}/{model_name}...")
        
        all_user_results = []
        total_cost = 0
        total_duration = 0
        total_tokens = 0
        
        # Process each user
        for user in self.test_users:
            # Prepare prompt for this user
            user_info = self.format_user_info(user)
            topics_str = self.format_topics_for_prompt()
            prompt = self.prompt_template.replace("{user_info}", user_info).replace("{topics}", topics_str)
            
            # Call model
            result = call_model_with_retry(provider, model_name, prompt, max_retries=3)
            
            # Parse response
            user_result = self.parse_filtering_response(result, user['name'])
            all_user_results.append(user_result)
            
            # Accumulate costs and timing
            total_cost += self.calculate_cost(provider, model_name, result)["total_cost"]
            total_duration += result["duration"]
            total_tokens += result["usage"].get("total_tokens", 0)
        
        # Calculate overall metrics
        metrics = self.calculate_filtering_metrics(all_user_results)
        
        return {
            "provider": provider,
            "model": model_name,
            "phase": self.phase_name,
            "timestamp": datetime.now().isoformat(),
            "success": all(r["success"] for r in all_user_results),
            "duration": total_duration,
            "usage": {"total_tokens": total_tokens},
            "cost": {"total_cost": total_cost},
            "users_processed": len(self.test_users),
            "user_results": all_user_results,
            "metrics": metrics
        }
    
    def parse_filtering_response(self, result: Dict[str, Any], user_name: str) -> Dict[str, Any]:
        """Parse the filtering response from the model"""
        if not result["success"]:
            return {
                "user": user_name,
                "success": False,
                "error": result.get("error", ""),
                "visible_topics": [],
                "filtered_out": []
            }
        
        try:
            # Try to extract JSON from response
            response_text = result["response"]
            
            # Find JSON block
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                return {
                    "user": user_name,
                    "success": True,
                    "visible_topics": parsed.get("visible_topics", []),
                    "filtered_out": parsed.get("filtered_out", []),
                    "raw_response": response_text
                }
            else:
                return {
                    "user": user_name,
                    "success": False,
                    "error": "No JSON found in response",
                    "visible_topics": [],
                    "filtered_out": []
                }
        except (json.JSONDecodeError, KeyError) as e:
            return {
                "user": user_name,
                "success": False,
                "error": f"JSON parsing error: {e}",
                "visible_topics": [],
                "filtered_out": []
            }
    
    def calculate_filtering_metrics(self, user_results: List[Dict]) -> Dict[str, Any]:
        """Calculate filtering quality metrics"""
        successful_results = [r for r in user_results if r["success"]]
        
        if not successful_results:
            return {
                "users_processed": len(user_results),
                "successful_filters": 0,
                "success_rate": 0.0,
                "avg_visible_topics": 0,
                "avg_filtered_out": 0
            }
        
        # Calculate metrics
        success_rate = len(successful_results) / len(user_results)
        
        visible_counts = [len(r["visible_topics"]) for r in successful_results]
        filtered_counts = [len(r["filtered_out"]) for r in successful_results]
        
        return {
            "users_processed": len(user_results),
            "successful_filters": len(successful_results),
            "success_rate": success_rate,
            "avg_visible_topics": np.mean(visible_counts) if visible_counts else 0,
            "avg_filtered_out": np.mean(filtered_counts) if filtered_counts else 0,
            "total_visible_topics": sum(visible_counts),
            "total_filtered_out": sum(filtered_counts)
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
                          f"Avg Visible: {metrics['avg_visible_topics']:.1f}, "
                          f"Cost: ${result['cost']['total_cost']:.6f}")
                else:
                    print(f"  {status} {model_name}: {result['duration']:.2f}s, "
                          f"Users: {result['users_processed']}")
        
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
            
            # Best by filtering precision
            best_precision = max(successful_results.items(),
                               key=lambda x: x[1]["metrics"]["avg_visible_topics"])
            
            # Best by cost efficiency
            best_cost_efficiency = min(successful_results.items(),
                                     key=lambda x: x[1]["cost"]["total_cost"])
            
            print(f"\n🏆 Best Performance:")
            print(f"  Success Rate: {best_success_rate[0]} ({best_success_rate[1]['metrics']['success_rate']:.1%})")
            print(f"  Filtering Precision: {best_precision[0]} ({best_precision[1]['metrics']['avg_visible_topics']:.1f} topics)")
            print(f"  Cost Efficiency: {best_cost_efficiency[0]} (${best_cost_efficiency[1]['cost']['total_cost']:.6f})")
        
        print(f"\nResults saved to: {self.output_dir}")

if __name__ == "__main__":
    evaluator = Phase7Evaluator()
    evaluator.run_evaluation()
