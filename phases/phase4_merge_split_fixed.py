#!/usr/bin/env python3
"""
Phase 4: Fixed Merge/Split Topics Evaluation
Only runs on successful Step 1 results, skips failed models
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import sys
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_clients import call_model_with_retry
from config.model_config import get_available_models, get_model_config, get_model_cost

class Phase4EvaluatorFixed:
    """Fixed evaluator that only processes successful Step 1 results"""
    
    def __init__(self):
        self.phase_name = "phase4_merge_split_fixed"
        self.output_dir = os.path.join("output", self.phase_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load the 6-cluster benchmark as input
        self.initial_clusters = self.load_benchmark_clusters()
        self.messages = self.load_messages()
        self.prompt_template = self.get_merge_split_prompt()
    
    def load_benchmark_clusters(self) -> List[Dict]:
        """Load the 6-cluster benchmark from phases/phase3_clusters.json"""
        try:
            benchmark_path = "phases/phase3_clusters.json"
            with open(benchmark_path, "r") as f:
                clusters = json.load(f)
            
            print(f"✅ Loaded {len(clusters)} benchmark clusters from {benchmark_path}")
            return clusters
            
        except Exception as e:
            print(f"❌ Could not load benchmark clusters: {e}")
            return []
    
    def load_messages(self) -> List[Dict]:
        """Load message dataset from CSV file"""
        try:
            df = pd.read_csv("data/Synthetic_Slack_Messages.csv")
            messages = []
            for idx, row in df.iterrows():
                messages.append({
                    "message_id": idx + 1,
                    "text": row["text"],
                    "user": row["user"],
                    "timestamp": row["timestamp"]
                })
            print(f"✅ Loaded {len(messages)} messages")
            return messages
        except Exception as e:
            print(f"❌ Could not load messages: {e}")
            return []
    
    def get_merge_split_prompt(self) -> str:
        """Get the merge/split prompt template"""
        return """
You are an expert at analyzing topic clusters and determining which ones should be merged or split for better organization.

Given the following initial clusters, analyze them and determine:
1. Which clusters should be MERGED (high similarity, redundant topics)
2. Which clusters should be SPLIT (too broad, multiple distinct topics)
3. Provide refined clusters with clear reasoning

Initial Clusters:
{clusters}

Please provide your analysis in JSON format with:
- refined_clusters: List of final clusters after merge/split operations
- operations: List of merge/split operations performed
- reasoning: Explanation of decisions made
"""
    
    def check_step1_success(self, model_name: str) -> bool:
        """Check if Step 1 was successful for this model"""
        try:
            # Check if model has successful results in Step 1
            step1_file = f"output/phase3_topic_clustering/{model_name}.json"
            if os.path.exists(step1_file):
                with open(step1_file, "r") as f:
                    result = json.load(f)
                return result.get("success", False)
            return False
        except:
            return False
    
    def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model on merge/split operations"""
        print(f"\n🔍 Evaluating {model_name}...")
        
        # Check if Step 1 was successful
        if not self.check_step1_success(model_name):
            print(f"⏭️  Skipping {model_name} - Step 1 failed")
            return {
                "provider": model_name.split("_")[0],
                "model": model_name,
                "phase": "step2_merge_split",
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "duration": 0,
                "error": "Step 1 failed - skipping Step 2",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cost": {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
            }
        
        try:
            # Prepare prompt
            clusters_text = json.dumps(self.initial_clusters, indent=2)
            prompt = self.prompt_template.format(clusters=clusters_text)
            
            # Call model
            start_time = time.time()
            response = call_model_with_retry(model_name, prompt)
            duration = time.time() - start_time
            
            # Parse response
            result = self.parse_model_response(response, model_name, duration)
            return result
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {str(e)}")
            return {
                "provider": model_name.split("_")[0],
                "model": model_name,
                "phase": "step2_merge_split",
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "duration": 0,
                "error": str(e),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cost": {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
            }
    
    def parse_model_response(self, response: str, model_name: str, duration: float) -> Dict[str, Any]:
        """Parse model response and extract clusters"""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                raise ValueError("No valid JSON found in response")
            
            result_data = json.loads(json_str)
            
            # Calculate metrics
            initial_count = len(self.initial_clusters)
            refined_count = len(result_data.get("refined_clusters", []))
            
            return {
                "provider": model_name.split("_")[0],
                "model": model_name,
                "phase": "step2_merge_split",
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "duration": duration,
                "initial_clusters": self.initial_clusters,
                "refined_clusters": result_data.get("refined_clusters", []),
                "operations": result_data.get("operations", []),
                "reasoning": result_data.get("reasoning", ""),
                "metrics": {
                    "initial_clusters": initial_count,
                    "refined_clusters": refined_count,
                    "reduction_ratio": (initial_count - refined_count) / initial_count if initial_count > 0 else 0
                },
                "raw_response": response,
                "error": ""
            }
            
        except Exception as e:
            return {
                "provider": model_name.split("_")[0],
                "model": model_name,
                "phase": "step2_merge_split",
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "duration": duration,
                "error": f"Failed to parse response: {str(e)}",
                "raw_response": response
            }
    
    def run_evaluation(self):
        """Run evaluation on all available models"""
        print("🚀 Starting Phase 4: Fixed Merge/Split Evaluation")
        print("=" * 60)
        
        # Get available models
        available_models = get_available_models()
        print(f"📊 Found {len(available_models)} models to evaluate")
        
        results = {}
        successful_count = 0
        
        for model_name in available_models:
            print(f"\n🔍 Processing {model_name}...")
            result = self.evaluate_model(model_name)
            results[model_name] = result
            
            if result["success"]:
                successful_count += 1
                print(f"✅ {model_name} completed successfully")
            else:
                print(f"❌ {model_name} failed: {result.get('error', 'Unknown error')}")
        
        # Save results
        self.save_results(results)
        
        print(f"\n📊 Evaluation Complete:")
        print(f"✅ Successful: {successful_count}/{len(available_models)}")
        print(f"❌ Failed: {len(available_models) - successful_count}/{len(available_models)}")
    
    def save_results(self, results: Dict[str, Any]):
        """Save evaluation results"""
        for model_name, result in results.items():
            output_file = os.path.join(self.output_dir, f"{model_name}_step2.json")
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"💾 Saved results for {model_name}")
        
        # Save comprehensive results
        comprehensive_file = os.path.join(self.output_dir, "comprehensive_results.json")
        with open(comprehensive_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Saved comprehensive results")

def main():
    """Main execution function"""
    evaluator = Phase4EvaluatorFixed()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
