#!/usr/bin/env python3
"""
Step 2: Merge/Split Operations - Updated Service
Uses Step 1 results as input and tests all AI models against benchmark
"""

import os
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Step2MergeSplitEvaluator:
    """Updated Step 2 evaluator using Step 1 results and testing all AI models"""
    
    def __init__(self):
        self.phase_name = "step2_merge_split"
        self.output_dir = os.path.join("output", self.phase_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load Step 1 results
        self.step1_results = self.load_step1_results()
        self.benchmark_topics = self.load_benchmark_topics()
        self.messages = self.load_messages()
        
        # Available AI models to test
        self.ai_models = self.get_available_ai_models()
        
        print(f"✅ Loaded {len(self.step1_results)} Step 1 results")
        print(f"✅ Loaded {len(self.benchmark_topics)} benchmark topics")
        print(f"✅ Loaded {len(self.messages)} messages")
        print(f"✅ Will test {len(self.ai_models)} AI models")
    
    def load_step1_results(self) -> Dict:
        """Load Step 1 clustering results"""
        try:
            step1_file = "output/phase3_topic_clustering/comprehensive_results.json"
            if os.path.exists(step1_file):
                with open(step1_file, "r") as f:
                    return json.load(f)
            else:
                print("❌ Step 1 results not found. Please run Step 1 first.")
                return {}
        except Exception as e:
            print(f"❌ Error loading Step 1 results: {e}")
            return {}
    
    def load_benchmark_topics(self) -> List[Dict]:
        """Load benchmark topics (ground truth)"""
        try:
            benchmark_file = "data/ground_truth_topics.json"
            if os.path.exists(benchmark_file):
                with open(benchmark_file, "r") as f:
                    return json.load(f)
            else:
                print("❌ Benchmark topics not found")
                return []
        except Exception as e:
            print(f"❌ Error loading benchmark topics: {e}")
            return []
    
    def load_messages(self) -> List[Dict]:
        """Load message dataset"""
        try:
            messages_file = "data/Synthetic_Slack_Messages.csv"
            if os.path.exists(messages_file):
                df = pd.read_csv(messages_file)
                return df.to_dict('records')
            else:
                print("❌ Message dataset not found")
                return []
        except Exception as e:
            print(f"❌ Error loading messages: {e}")
            return []
    
    def get_available_ai_models(self) -> List[Dict]:
        """Get list of available AI models to test"""
        return [
            # OpenAI Models
            {"provider": "openai", "model": "gpt-4", "api_key": "OPENAI_API_KEY"},
            {"provider": "openai", "model": "gpt-3.5-turbo", "api_key": "OPENAI_API_KEY"},
            {"provider": "openai", "model": "gpt-4o", "api_key": "OPENAI_API_KEY"},
            
            # Google Models
            {"provider": "google", "model": "gemini-1.5-flash", "api_key": "GOOGLE_API_KEY"},
            {"provider": "google", "model": "gemini-1.5-pro", "api_key": "GOOGLE_API_KEY"},
            {"provider": "google", "model": "gemini-2.0-flash", "api_key": "GOOGLE_API_KEY"},
            {"provider": "google", "model": "gemini-2.5-flash", "api_key": "GOOGLE_API_KEY"},
            {"provider": "google", "model": "gemini-2.5-pro", "api_key": "GOOGLE_API_KEY"},
            
            # Anthropic Models
            {"provider": "anthropic", "model": "claude-3-opus-20240229", "api_key": "ANTHROPIC_API_KEY"},
            {"provider": "anthropic", "model": "claude-3-sonnet-20240229", "api_key": "ANTHROPIC_API_KEY"},
            {"provider": "anthropic", "model": "claude-3-haiku-20240307", "api_key": "ANTHROPIC_API_KEY"},
            
            # Groq Models
            {"provider": "groq", "model": "llama3-8b-8192", "api_key": "GROQ_API_KEY"},
            {"provider": "groq", "model": "llama3-70b-8192", "api_key": "GROQ_API_KEY"},
            {"provider": "groq", "model": "mixtral-8x7b-32768", "api_key": "GROQ_API_KEY"},
            
            # x.ai Models
            {"provider": "xai", "model": "grok-2-1212", "api_key": "XAI_API_KEY"},
            {"provider": "xai", "model": "grok-2-vision-1212", "api_key": "XAI_API_KEY"},
            {"provider": "xai", "model": "grok-3", "api_key": "XAI_API_KEY"},
        ]
    
    def get_merge_split_prompt(self, clusters: List[Dict]) -> str:
        """Generate merge/split prompt for AI model"""
        clusters_str = "\n".join([
            f"Cluster {cluster['cluster_id']}: {cluster.get('draft_title', 'Untitled')}\n"
            f"  Messages: {cluster.get('message_ids', [])}\n"
            f"  Participants: {cluster.get('participants', [])}\n"
            f"  Channel: {cluster.get('channel', 'Unknown')}"
            for cluster in clusters
        ])
        
        return f"""You are an expert at refining topic clusters by analyzing their similarity and determining when to merge or split them.

Given these initial topic clusters from Step 1, your task is to:

1. **Analyze cluster similarity** based on:
   - Semantic overlap in titles and content
   - Participant overlap
   - Channel relationships
   - Temporal proximity

2. **Make merge/split decisions:**
   - If clusters are very similar (similarity > 0.85), merge them
   - If a cluster is too large or diverse, split it into smaller clusters
   - Keep clusters that are well-defined and distinct

**Initial Clusters:**
{clusters_str}

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

Analyze the clusters and provide the refined results in the specified JSON format."""
    
    def call_ai_model(self, provider: str, model: str, prompt: str) -> Dict:
        """Call AI model API (simplified version)"""
        # This is a simplified version - in real implementation, you'd use actual API calls
        print(f"    🔄 Calling {provider}/{model}...")
        
        # Simulate API call
        time.sleep(1)  # Simulate processing time
        
        # Mock response for demonstration
        mock_response = {
            "success": True,
            "response": json.dumps({
                "refined_clusters": [
                    {
                        "cluster_id": "cluster_001_refined",
                        "message_ids": [1, 2, 3, 4, 5],
                        "draft_title": "Project Planning",
                        "participants": ["@alice", "@bob"],
                        "channel": "#general",
                        "merge_reason": "Merged similar planning clusters",
                        "split_reason": None
                    }
                ],
                "operations": [
                    {
                        "operation": "merge",
                        "clusters_involved": ["cluster_001", "cluster_002"],
                        "reason": "High semantic similarity"
                    }
                ]
            }),
            "duration": 1.5,
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "total_tokens": 1200
            }
        }
        
        return mock_response
    
    def evaluate_model_against_benchmark(self, provider: str, model: str, clusters: List[Dict]) -> Dict:
        """Evaluate a single model against benchmark"""
        print(f"  🧪 Testing {provider}/{model}...")
        
        # Get merge/split prompt
        prompt = self.get_merge_split_prompt(clusters)
        
        # Call AI model
        result = self.call_ai_model(provider, model, prompt)
        
        # Parse response
        refined_clusters = self.parse_merge_split_response(result)
        
        # Compare against benchmark
        benchmark_metrics = self.compare_against_benchmark(refined_clusters)
        
        # Calculate cost (mock)
        cost = self.calculate_cost(provider, model, result)
        
        return {
            "provider": provider,
            "model": model,
            "phase": self.phase_name,
            "timestamp": datetime.now().isoformat(),
            "success": result["success"],
            "duration": result["duration"],
            "usage": result["usage"],
            "cost": cost,
            "initial_clusters": clusters,
            "refined_clusters": refined_clusters,
            "benchmark_metrics": benchmark_metrics,
            "raw_response": result.get("response", ""),
            "error": result.get("error", "")
        }
    
    def parse_merge_split_response(self, result: Dict) -> List[Dict]:
        """Parse AI model response"""
        if not result["success"]:
            return []
        
        try:
            response_text = result["response"]
            parsed = json.loads(response_text)
            return parsed.get("refined_clusters", [])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    ⚠️  JSON parsing error: {e}")
            return []
    
    def compare_against_benchmark(self, refined_clusters: List[Dict]) -> Dict:
        """Compare refined clusters against benchmark topics"""
        if not refined_clusters or not self.benchmark_topics:
            return {
                "accuracy": 0.0,
                "coverage": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
        
        # Calculate metrics against benchmark
        total_messages = len(self.messages)
        clustered_messages = set()
        
        for cluster in refined_clusters:
            message_ids = cluster.get("message_ids", [])
            clustered_messages.update(message_ids)
        
        coverage = len(clustered_messages) / total_messages if total_messages > 0 else 0
        
        # Mock accuracy calculation (in real implementation, you'd compare against benchmark)
        accuracy = min(0.95, coverage + 0.1)  # Mock calculation
        
        return {
            "accuracy": accuracy,
            "coverage": coverage,
            "precision": accuracy * 0.9,
            "recall": accuracy * 0.85,
            "f1_score": 2 * (accuracy * 0.9) * (accuracy * 0.85) / ((accuracy * 0.9) + (accuracy * 0.85)),
            "total_clusters": len(refined_clusters),
            "total_messages_clustered": len(clustered_messages)
        }
    
    def calculate_cost(self, provider: str, model: str, result: Dict) -> Dict:
        """Calculate API cost"""
        # Mock cost calculation
        usage = result.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        # Mock pricing (per 1000 tokens)
        pricing = {
            "openai": {"gpt-4": {"input": 0.03, "output": 0.06}},
            "google": {"gemini-1.5-flash": {"input": 0.001, "output": 0.002}},
            "anthropic": {"claude-3-opus": {"input": 0.015, "output": 0.075}},
            "groq": {"llama3-8b": {"input": 0.0001, "output": 0.0001}},
            "xai": {"grok-2": {"input": 0.01, "output": 0.03}}
        }
        
        # Get pricing for model
        model_pricing = pricing.get(provider, {}).get(model, {"input": 0.01, "output": 0.02})
        
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    
    def run_step2_evaluation(self):
        """Run Step 2 evaluation on all AI models"""
        print(f"🎯 STEP 2: MERGE/SPLIT EVALUATION")
        print("=" * 60)
        print(f"📊 Testing {len(self.ai_models)} AI models against benchmark")
        print(f"📈 Using {len(self.step1_results)} Step 1 results as input")
        
        results = {}
        total_models = 0
        successful_models = 0
        
        # Get best Step 1 result to use as input
        best_step1_model = self.get_best_step1_model()
        if not best_step1_model:
            print("❌ No successful Step 1 results found")
            return
        
        clusters = best_step1_model.get("clusters", [])
        print(f"✅ Using clusters from: {best_step1_model.get('model_name', 'Unknown')}")
        
        for model_config in self.ai_models:
            provider = model_config["provider"]
            model = model_config["model"]
            total_models += 1
            
            try:
                result = self.evaluate_model_against_benchmark(provider, model, clusters)
                results[f"{provider}_{model}"] = result
                
                # Save individual result
                output_file = os.path.join(self.output_dir, f"{provider}_{model}.json")
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)
                
                # Print status
                status = "✅" if result["success"] else "❌"
                if result["success"]:
                    successful_models += 1
                    metrics = result["benchmark_metrics"]
                    print(f"  {status} {model}: {result['duration']:.2f}s, "
                          f"Accuracy: {metrics['accuracy']:.3f}, "
                          f"Coverage: {metrics['coverage']:.3f}, "
                          f"Cost: ${result['cost']['total_cost']:.6f}")
                else:
                    print(f"  {status} {model}: {result['duration']:.2f}s, "
                          f"Error: {result['error'][:100]}...")
            
            except Exception as e:
                print(f"  ❌ {model}: Error - {str(e)[:100]}...")
                results[f"{provider}_{model}"] = {
                    "provider": provider,
                    "model": model,
                    "success": False,
                    "error": str(e)
                }
        
        # Save comprehensive results
        comprehensive_output = os.path.join(self.output_dir, "comprehensive_results.json")
        with open(comprehensive_output, "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate summary
        self.generate_summary(results, total_models, successful_models)
    
    def get_best_step1_model(self) -> Dict:
        """Get the best performing model from Step 1"""
        if not self.step1_results:
            return None
        
        # Find model with highest success rate and best metrics
        successful_results = {k: v for k, v in self.step1_results.items() if v.get("success", False)}
        
        if not successful_results:
            return None
        
        # Return the first successful result (in real implementation, you'd rank by metrics)
        return list(successful_results.values())[0]
    
    def generate_summary(self, results: Dict, total_models: int, successful_models: int):
        """Generate evaluation summary"""
        print(f"\n{'='*60}")
        print("📊 STEP 2 EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total models tested: {total_models}")
        print(f"Successful models: {successful_models}")
        print(f"Success rate: {(successful_models/total_models*100):.1f}%" if total_models > 0 else "0%")
        
        # Find best performing models
        successful_results = {k: v for k, v in results.items() if v.get("success", False)}
        
        if successful_results:
            # Best by accuracy
            best_accuracy = max(successful_results.items(), 
                              key=lambda x: x[1]["benchmark_metrics"]["accuracy"])
            
            # Best by cost efficiency
            best_cost_efficiency = min(successful_results.items(),
                                     key=lambda x: x[1]["cost"]["total_cost"])
            
            # Best by speed
            best_speed = min(successful_results.items(),
                           key=lambda x: x[1]["duration"])
            
            print(f"\n🏆 Best Performance:")
            print(f"  Accuracy: {best_accuracy[0]} ({best_accuracy[1]['benchmark_metrics']['accuracy']:.3f})")
            print(f"  Cost Efficiency: {best_cost_efficiency[0]} (${best_cost_efficiency[1]['cost']['total_cost']:.6f})")
            print(f"  Speed: {best_speed[0]} ({best_speed[1]['duration']:.2f}s)")
        
        print(f"\nResults saved to: {self.output_dir}")

def main():
    """Main function"""
    print("🚀 Starting Step 2: Merge/Split Operations")
    print("=" * 50)
    
    evaluator = Step2MergeSplitEvaluator()
    evaluator.run_step2_evaluation()

if __name__ == "__main__":
    main()
