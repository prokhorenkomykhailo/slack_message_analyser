#!/usr/bin/env python3
"""
Phase 6: Topic Embedding Evaluation
Evaluates all models on generating topic embeddings
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

class Phase6Evaluator:
    """Evaluates models on topic embedding generation"""
    
    def __init__(self):
        self.phase_name = "phase6_embedding"
        self.output_dir = os.path.join("output", self.phase_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load topics from Phase 5
        self.topics = self.load_topics()
        
        # Embedding generation prompt
        self.prompt_template = self.get_embedding_prompt()
    
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
                    "tags": ["planning", "sprint"]
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
                    "tags": ["architecture", "technical"]
                }
            }
        ]
    
    def get_embedding_prompt(self) -> str:
        """Get the embedding generation prompt"""
        return """
You are an expert at generating semantic embeddings for topics. Your task is to create a 768-dimensional embedding vector that represents the semantic meaning of a topic.

Given a topic with its metadata, generate an embedding that captures:
- The main subject matter
- The urgency and importance
- The type of discussion (planning, technical, etc.)
- The participants involved
- The action items and outcomes

**Topic Information:**
{topic_info}

**Instructions:**
- Generate a 768-dimensional embedding vector
- Use values between -1.0 and 1.0
- Ensure the embedding captures semantic meaning
- Make similar topics have similar embeddings
- Consider all metadata fields in the embedding

**Output Format (JSON):**
{{
  "topic_id": "cluster_001",
  "embedding": [0.123, -0.456, 0.789, ...], // 768 values
  "embedding_norm": 0.987, // L2 norm of the embedding
  "semantic_keywords": ["planning", "sprint", "project"]
}}

Generate the embedding vector for this topic in the specified JSON format.
"""
    
    def format_topic_for_prompt(self, topic: Dict) -> str:
        """Format a topic for the prompt"""
        metadata = topic.get("metadata", {})
        return f"""
Topic ID: {topic['cluster_id']}
Title: {metadata.get('title', 'N/A')}
Summary: {metadata.get('summary', 'N/A')}
Participants: {metadata.get('participants', [])}
Urgency: {metadata.get('urgency', 'N/A')}
Tags: {metadata.get('tags', [])}
Action Items: {metadata.get('action_items', [])}
"""
    
    def evaluate_model(self, provider: str, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model on embedding generation"""
        print(f"  🧪 Testing {provider}/{model_name}...")
        
        all_embeddings = []
        total_cost = 0
        total_duration = 0
        total_tokens = 0
        
        # Process each topic
        for topic in self.topics:
            if not topic["success"]:
                continue
                
            # Prepare prompt for this topic
            topic_info = self.format_topic_for_prompt(topic)
            prompt = self.prompt_template.replace("{topic_info}", topic_info)
            
            # Call model
            result = call_model_with_retry(provider, model_name, prompt, max_retries=3)
            
            # Parse response
            embedding = self.parse_embedding_response(result, topic['cluster_id'])
            all_embeddings.append(embedding)
            
            # Accumulate costs and timing
            total_cost += self.calculate_cost(provider, model_name, result)["total_cost"]
            total_duration += result["duration"]
            total_tokens += result["usage"].get("total_tokens", 0)
        
        # Calculate overall metrics
        metrics = self.calculate_embedding_metrics(all_embeddings)
        
        return {
            "provider": provider,
            "model": model_name,
            "phase": self.phase_name,
            "timestamp": datetime.now().isoformat(),
            "success": len(all_embeddings) > 0,
            "duration": total_duration,
            "usage": {"total_tokens": total_tokens},
            "cost": {"total_cost": total_cost},
            "topics_processed": len(self.topics),
            "embeddings_generated": len(all_embeddings),
            "embedding_results": all_embeddings,
            "metrics": metrics
        }
    
    def parse_embedding_response(self, result: Dict[str, Any], topic_id: str) -> Dict[str, Any]:
        """Parse the embedding response from the model"""
        if not result["success"]:
            return {
                "topic_id": topic_id,
                "success": False,
                "error": result.get("error", ""),
                "embedding": []
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
                
                embedding = parsed.get("embedding", [])
                if len(embedding) != 768:
                    # Pad or truncate to 768 dimensions
                    if len(embedding) < 768:
                        embedding.extend([0.0] * (768 - len(embedding)))
                    else:
                        embedding = embedding[:768]
                
                return {
                    "topic_id": topic_id,
                    "success": True,
                    "embedding": embedding,
                    "embedding_norm": parsed.get("embedding_norm", np.linalg.norm(embedding)),
                    "semantic_keywords": parsed.get("semantic_keywords", []),
                    "raw_response": response_text
                }
            else:
                return {
                    "topic_id": topic_id,
                    "success": False,
                    "error": "No JSON found in response",
                    "embedding": []
                }
        except (json.JSONDecodeError, KeyError) as e:
            return {
                "topic_id": topic_id,
                "success": False,
                "error": f"JSON parsing error: {e}",
                "embedding": []
            }
    
    def calculate_embedding_metrics(self, embedding_results: List[Dict]) -> Dict[str, Any]:
        """Calculate embedding generation quality metrics"""
        successful_results = [e for e in embedding_results if e["success"]]
        
        if not successful_results:
            return {
                "topics_processed": len(embedding_results),
                "successful_embeddings": 0,
                "success_rate": 0.0,
                "avg_embedding_norm": 0.0,
                "embedding_diversity": 0.0
            }
        
        # Calculate metrics
        success_rate = len(successful_results) / len(embedding_results)
        
        # Calculate embedding norms
        norms = [e.get("embedding_norm", 0) for e in successful_results]
        avg_norm = np.mean(norms) if norms else 0
        
        # Calculate embedding diversity (if we have multiple embeddings)
        if len(successful_results) > 1:
            embeddings = [e["embedding"] for e in successful_results]
            embedding_matrix = np.array(embeddings)
            
            # Calculate pairwise cosine similarities
            similarities = []
            for i in range(len(embedding_matrix)):
                for j in range(i + 1, len(embedding_matrix)):
                    cos_sim = np.dot(embedding_matrix[i], embedding_matrix[j]) / (
                        np.linalg.norm(embedding_matrix[i]) * np.linalg.norm(embedding_matrix[j])
                    )
                    similarities.append(cos_sim)
            
            # Diversity is 1 - average similarity
            embedding_diversity = 1 - np.mean(similarities) if similarities else 0
        else:
            embedding_diversity = 0
        
        return {
            "topics_processed": len(embedding_results),
            "successful_embeddings": len(successful_results),
            "success_rate": success_rate,
            "avg_embedding_norm": avg_norm,
            "embedding_diversity": embedding_diversity,
            "embedding_dimensions": 768
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
                          f"Embeddings: {metrics['successful_embeddings']}, "
                          f"Diversity: {metrics['embedding_diversity']:.3f}, "
                          f"Cost: ${result['cost']['total_cost']:.6f}")
                else:
                    print(f"  {status} {model_name}: {result['duration']:.2f}s, "
                          f"Topics: {result['topics_processed']}")
        
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
            
            # Best by embedding diversity
            best_diversity = max(successful_results.items(),
                               key=lambda x: x[1]["metrics"]["embedding_diversity"])
            
            # Best by cost efficiency
            best_cost_efficiency = min(successful_results.items(),
                                     key=lambda x: x[1]["cost"]["total_cost"])
            
            print(f"\n🏆 Best Performance:")
            print(f"  Success Rate: {best_success_rate[0]} ({best_success_rate[1]['metrics']['success_rate']:.1%})")
            print(f"  Embedding Diversity: {best_diversity[0]} ({best_diversity[1]['metrics']['embedding_diversity']:.3f})")
            print(f"  Cost Efficiency: {best_cost_efficiency[0]} (${best_cost_efficiency[1]['cost']['total_cost']:.6f})")
        
        print(f"\nResults saved to: {self.output_dir}")

if __name__ == "__main__":
    evaluator = Phase6Evaluator()
    evaluator.run_evaluation()
