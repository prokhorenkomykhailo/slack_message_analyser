#!/usr/bin/env python3
"""
Phase 8: New Message Processing Evaluation
Evaluates all models on processing new messages and updating topics
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

class Phase8Evaluator:
    """Evaluates models on new message processing"""
    
    def __init__(self):
        self.phase_name = "phase8_new_message_processing"
        self.output_dir = os.path.join("output", self.phase_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load existing topics from Phase 5
        self.existing_topics = self.load_existing_topics()
        
        # Create new test messages
        self.new_messages = self.create_new_messages()
        
        # New message processing prompt
        self.prompt_template = self.get_processing_prompt()
    
    def load_existing_topics(self) -> List[Dict]:
        """Load existing topics from Phase 5"""
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
            }
        ]
    
    def create_new_messages(self) -> List[Dict]:
        """Create new test messages"""
        return [
            {
                "id": 201,
                "user": "@alice",
                "channel": "#general",
                "text": "I've completed the sprint board setup. Should we schedule the sprint planning meeting for tomorrow?",
                "timestamp": "2024-04-14T10:30:00Z",
                "thread_ts": None
            },
            {
                "id": 202,
                "user": "@bob",
                "channel": "#general",
                "text": "Great work Alice! Let's schedule it for 2 PM tomorrow. I'll send out the calendar invite.",
                "timestamp": "2024-04-14T10:35:00Z",
                "thread_ts": None
            },
            {
                "id": 203,
                "user": "@charlie",
                "channel": "#tech",
                "text": "I found a critical bug in the authentication system. We need to fix this before the release.",
                "timestamp": "2024-04-14T11:00:00Z",
                "thread_ts": None
            },
            {
                "id": 204,
                "user": "@david",
                "channel": "#marketing",
                "text": "The new marketing campaign is ready for review. Can someone from the team take a look?",
                "timestamp": "2024-04-14T11:15:00Z",
                "thread_ts": None
            },
            {
                "id": 205,
                "user": "@alice",
                "channel": "#general",
                "text": "Perfect! I'll prepare the agenda for the sprint planning meeting.",
                "timestamp": "2024-04-14T11:30:00Z",
                "thread_ts": None
            }
        ]
    
    def get_processing_prompt(self) -> str:
        """Get the new message processing prompt"""
        return """
You are an expert at processing new messages and determining how they relate to existing topics.

Given a new message and existing topics, your task is to:

1. **Analyze the new message** for:
   - Content and context
   - Participants involved
   - Channel and thread relationships
   - Potential action items or decisions

2. **Compare to existing topics** and determine:
   - If it matches an existing topic (similarity > 0.8)
   - If it should create a new topic
   - How to update existing topic metadata

3. **Provide processing decision** with:
   - Topic assignment (existing or new)
   - Updated metadata for affected topics
   - New action items or changes

**New Message:**
{new_message}

**Existing Topics:**
{existing_topics}

**Instructions:**
- Use semantic similarity to match messages to topics
- Update topic metadata when adding new messages
- Create new topics for unrelated content
- Maintain topic coherence and relevance

**Output Format (JSON):**
{{
  "message_id": 201,
  "processing_decision": "update_existing", // or "create_new"
  "matched_topic_id": "cluster_001", // if updating existing
  "new_topic": null, // if creating new
  "updated_metadata": {{
    "title": "Updated title",
    "summary": "Updated summary",
    "action_items": [
      {{
        "task": "Schedule sprint planning meeting",
        "owner": "@bob",
        "due_date": "2024-04-15"
      }}
    ],
    "participants": ["@alice", "@bob"],
    "urgency": "medium"
  }},
  "similarity_score": 0.95,
  "reasoning": "Message relates to sprint planning and involves same participants"
}}

Process the new message and provide the results in the specified JSON format.
"""
    
    def format_new_message(self, message: Dict) -> str:
        """Format a new message for the prompt"""
        return f"""
Message ID: {message['id']}
User: {message['user']}
Channel: {message['channel']}
Text: {message['text']}
Timestamp: {message['timestamp']}
Thread: {message.get('thread_ts', 'None')}
"""
    
    def format_existing_topics(self) -> str:
        """Format existing topics for the prompt"""
        formatted = []
        for topic in self.existing_topics:
            if not topic["success"]:
                continue
                
            metadata = topic.get("metadata", {})
            formatted.append(f"""
Topic ID: {topic['cluster_id']}
Title: {metadata.get('title', 'N/A')}
Summary: {metadata.get('summary', 'N/A')}
Channel: {metadata.get('channel', 'N/A')}
Participants: {metadata.get('participants', [])}
Action Items: {metadata.get('action_items', [])}
Urgency: {metadata.get('urgency', 'N/A')}
Tags: {metadata.get('tags', [])}
""")
        return "\n".join(formatted)
    
    def evaluate_model(self, provider: str, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model on new message processing"""
        print(f"  🧪 Testing {provider}/{model_name}...")
        
        all_message_results = []
        total_cost = 0
        total_duration = 0
        total_tokens = 0
        
        # Process each new message
        for message in self.new_messages:
            # Prepare prompt for this message
            message_info = self.format_new_message(message)
            topics_str = self.format_existing_topics()
            prompt = self.prompt_template.replace("{new_message}", message_info).replace("{existing_topics}", topics_str)
            
            # Call model
            result = call_model_with_retry(provider, model_name, prompt, max_retries=3)
            
            # Parse response
            message_result = self.parse_processing_response(result, message['id'])
            all_message_results.append(message_result)
            
            # Accumulate costs and timing
            total_cost += self.calculate_cost(provider, model_name, result)["total_cost"]
            total_duration += result["duration"]
            total_tokens += result["usage"].get("total_tokens", 0)
        
        # Calculate overall metrics
        metrics = self.calculate_processing_metrics(all_message_results)
        
        return {
            "provider": provider,
            "model": model_name,
            "phase": self.phase_name,
            "timestamp": datetime.now().isoformat(),
            "success": all(r["success"] for r in all_message_results),
            "duration": total_duration,
            "usage": {"total_tokens": total_tokens},
            "cost": {"total_cost": total_cost},
            "messages_processed": len(self.new_messages),
            "message_results": all_message_results,
            "metrics": metrics
        }
    
    def parse_processing_response(self, result: Dict[str, Any], message_id: int) -> Dict[str, Any]:
        """Parse the processing response from the model"""
        if not result["success"]:
            return {
                "message_id": message_id,
                "success": False,
                "error": result.get("error", ""),
                "processing_decision": None,
                "updated_metadata": {}
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
                    "message_id": message_id,
                    "success": True,
                    "processing_decision": parsed.get("processing_decision"),
                    "matched_topic_id": parsed.get("matched_topic_id"),
                    "new_topic": parsed.get("new_topic"),
                    "updated_metadata": parsed.get("updated_metadata", {}),
                    "similarity_score": parsed.get("similarity_score", 0),
                    "reasoning": parsed.get("reasoning", ""),
                    "raw_response": response_text
                }
            else:
                return {
                    "message_id": message_id,
                    "success": False,
                    "error": "No JSON found in response",
                    "processing_decision": None,
                    "updated_metadata": {}
                }
        except (json.JSONDecodeError, KeyError) as e:
            return {
                "message_id": message_id,
                "success": False,
                "error": f"JSON parsing error: {e}",
                "processing_decision": None,
                "updated_metadata": {}
            }
    
    def calculate_processing_metrics(self, message_results: List[Dict]) -> Dict[str, Any]:
        """Calculate processing quality metrics"""
        successful_results = [r for r in message_results if r["success"]]
        
        if not successful_results:
            return {
                "messages_processed": len(message_results),
                "successful_processing": 0,
                "success_rate": 0.0,
                "avg_similarity_score": 0.0,
                "update_decisions": 0,
                "new_topic_decisions": 0
            }
        
        # Calculate metrics
        success_rate = len(successful_results) / len(message_results)
        
        similarity_scores = [r.get("similarity_score", 0) for r in successful_results]
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
        
        update_decisions = sum(1 for r in successful_results if r.get("processing_decision") == "update_existing")
        new_topic_decisions = sum(1 for r in successful_results if r.get("processing_decision") == "create_new")
        
        return {
            "messages_processed": len(message_results),
            "successful_processing": len(successful_results),
            "success_rate": success_rate,
            "avg_similarity_score": avg_similarity,
            "update_decisions": update_decisions,
            "new_topic_decisions": new_topic_decisions,
            "total_decisions": update_decisions + new_topic_decisions
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
                          f"Avg Similarity: {metrics['avg_similarity_score']:.3f}, "
                          f"Cost: ${result['cost']['total_cost']:.6f}")
                else:
                    print(f"  {status} {model_name}: {result['duration']:.2f}s, "
                          f"Messages: {result['messages_processed']}")
        
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
            
            # Best by similarity score
            best_similarity = max(successful_results.items(),
                                key=lambda x: x[1]["metrics"]["avg_similarity_score"])
            
            # Best by cost efficiency
            best_cost_efficiency = min(successful_results.items(),
                                     key=lambda x: x[1]["cost"]["total_cost"])
            
            print(f"\n🏆 Best Performance:")
            print(f"  Success Rate: {best_success_rate[0]} ({best_success_rate[1]['metrics']['success_rate']:.1%})")
            print(f"  Similarity Score: {best_similarity[0]} ({best_similarity[1]['metrics']['avg_similarity_score']:.3f})")
            print(f"  Cost Efficiency: {best_cost_efficiency[0]} (${best_cost_efficiency[1]['cost']['total_cost']:.6f})")
        
        print(f"\nResults saved to: {self.output_dir}")

if __name__ == "__main__":
    evaluator = Phase8Evaluator()
    evaluator.run_evaluation()
