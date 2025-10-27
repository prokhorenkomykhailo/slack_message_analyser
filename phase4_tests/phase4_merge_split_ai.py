#!/usr/bin/env python3
"""
Phase 4: Merge/Split Operations - AI-Powered Analysis
Uses actual AI models to analyze cluster similarity and suggest merge/split operations
"""

import os
import json
import time
import csv
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_clients import call_model_with_retry
from config.model_config import get_available_models, get_model_config, get_model_cost, MODEL_LIMITS

class Phase4MergeSplitEvaluator:
    """Evaluates models on merge/split operations using AI analysis"""
    
    def __init__(self):
        self.phase_name = "phase4_merge_split"
        self.output_dir = os.path.join("output", self.phase_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load Phase 3 results (successful models only)
        self.phase3_results = self.load_successful_phase3_results()
        self.messages = self.load_messages_from_csv()
        
        print(f"📊 Loaded {len(self.phase3_results)} successful Phase 3 results")
        print(f"📝 Loaded {len(self.messages)} messages from CSV")
    
    def load_successful_phase3_results(self) -> List[Dict]:
        """Load only successful Phase 3 results"""
        phase3_dir = "output/phase3_topic_clustering"
        successful_results = []
        
        # Get all JSON files from Phase 3
        for filename in os.listdir(phase3_dir):
            if filename.endswith('.json') and not filename.startswith('comprehensive_') and not filename.startswith('detailed_') and not filename.startswith('best_'):
                filepath = os.path.join(phase3_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        result = json.load(f)
                    
                    # Only include successful results
                    if result.get('success', False) and result.get('clusters'):
                        successful_results.append(result)
                        print(f"✅ Loaded {result['provider']}/{result['model']} - {len(result['clusters'])} clusters")
                        
                except Exception as e:
                    print(f"⚠️ Skipped {filename}: {e}")
        
        return successful_results
    
    def load_messages_from_csv(self) -> List[Dict]:
        """Load messages from CSV file"""
        csv_path = os.path.join("data", "Synthetic_Slack_Messages.csv")
        messages = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                messages.append({
                    "id": idx + 1,
                    "channel": row["channel"],
                    "user": row["user_name"],
                    "text": row["text"],
                    "timestamp": row["timestamp"],
                    "thread_id": row["thread_id"] if row["thread_id"] else None
                })
        
        return messages
    
    def get_merge_split_prompt(self, clusters: List[Dict], messages_str: str) -> str:
        """Get the merge/split analysis prompt"""
        return f"""
You are an expert at analyzing topic clusters and determining whether they should be merged or split.

Given a set of topic clusters from Slack conversations, your task is to:

1. **Analyze cluster similarity** using:
   - Participant overlap (same users involved)
   - Channel overlap (same channels)
   - Thread relationships (same thread_id)
   - Semantic similarity (similar topics/subjects)
   - Temporal proximity (similar timeframes)

2. **Determine merge operations** for clusters that are:
   - Too similar (high participant/channel/thread overlap)
   - Covering the same topic from different angles
   - Redundant or duplicate content

3. **Determine split operations** for clusters that are:
   - Too large (more than 15-20 messages)
   - Covering multiple distinct topics
   - Too diverse in participants/channels

**Current Clusters to Analyze:**
{clusters}

**Relevant Messages:**
{messages_str}

**Instructions:**
- Analyze each cluster pair for potential merging
- Identify clusters that need splitting
- Consider project names (EcoBloom, FitFusion, TechNova, GreenScape, UrbanEdge)
- Focus on meaningful relationships, not arbitrary groupings
- Provide clear reasoning for each operation

**Output Format (JSON):**
{{
  "merge_operations": [
    {{
      "operation": "merge",
      "clusters": ["cluster_001", "cluster_002"],
      "reason": "High participant overlap (80%) and similar topics - both discuss EcoBloom campaign planning",
      "similarity_score": 0.85
    }}
  ],
  "split_operations": [
    {{
      "operation": "split",
      "cluster": "cluster_003",
      "reason": "Large cluster (25 messages) covering both EcoBloom and FitFusion topics - should be split by project",
      "suggested_clusters": ["EcoBloom Planning", "FitFusion Development"]
    }}
  ],
  "refined_clusters": [
    {{
      "cluster_id": "cluster_001",
      "message_ids": [1, 2, 3, 4, 5],
      "draft_title": "EcoBloom Summer Campaign Planning",
      "participants": ["Devon", "Sam", "Leah"],
      "channel": "#campaign-briefs",
      "thread_id": "thread_001",
      "merge_reason": null,
      "split_reason": null
    }}
  ]
}}

Analyze the clusters and provide merge/split recommendations in the specified JSON format.
"""
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)"""
        return len(text) // 4
    
    def format_messages_for_prompt(self, message_ids: List[int], max_tokens: int = 20000) -> str:
        """Format relevant messages for the prompt with token management"""
        formatted = []
        total_tokens = 0
        
        # Create message lookup
        message_lookup = {msg['id']: msg for msg in self.messages}
        
        for msg_id in message_ids:
            if msg_id in message_lookup:
                msg = message_lookup[msg_id]
                msg_text = (
                    f"ID: {msg['id']} | Channel: {msg['channel']} | User: {msg['user']} | "
                    f"Thread: {msg.get('thread_id', 'None')} | Text: {msg['text'][:200]}..."
                )
                
                msg_tokens = self.estimate_tokens(msg_text)
                
                if total_tokens + msg_tokens > max_tokens:
                    print(f"⚠️  Token limit reached. Using {len(formatted)} messages out of {len(message_ids)}")
                    break
                
                formatted.append(msg_text)
                total_tokens += msg_tokens
        
        return "\n".join(formatted)
    
    def evaluate_model(self, provider: str, model_name: str, phase3_result: Dict) -> Dict[str, Any]:
        """Evaluate a single model on merge/split operations"""
        print(f"  🧪 Testing {provider}/{model_name}...")
        
        # Get model configuration
        model_config = get_model_config(provider, model_name)
        max_tokens = MODEL_LIMITS.get(model_name, {}).get("max_tokens", 4096)
        context_length = MODEL_LIMITS.get(model_name, {}).get("context_length", 8192)
        
        # Calculate available tokens
        available_tokens = min(context_length * 0.8, 80000)
        
        # Get all message IDs from clusters
        all_message_ids = []
        for cluster in phase3_result['clusters']:
            all_message_ids.extend(cluster.get('message_ids', []))
        
        # Format messages for prompt
        messages_str = self.format_messages_for_prompt(all_message_ids, int(available_tokens))
        
        # Create prompt
        prompt = self.get_merge_split_prompt(phase3_result['clusters'], messages_str)
        
        # Estimate tokens
        prompt_tokens = self.estimate_tokens(prompt)
        print(f"    📝 Estimated prompt tokens: {prompt_tokens:,}")
        
        # Call model
        result = call_model_with_retry(provider, model_name, prompt, max_retries=3)
        
        # Parse response
        merge_split_analysis = self.parse_merge_split_response(result)
        
        # Calculate metrics
        metrics = self.calculate_merge_split_metrics(merge_split_analysis)
        
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
            "merge_operations": merge_split_analysis.get("merge_operations", []),
            "split_operations": merge_split_analysis.get("split_operations", []),
            "refined_clusters": merge_split_analysis.get("refined_clusters", []),
            "metrics": metrics,
            "raw_response": result.get("response", ""),
            "error": result.get("error", ""),
            "prompt_tokens_estimated": prompt_tokens,
            "messages_used": len(messages_str.split('\n'))
        }
    
    def parse_merge_split_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the merge/split response from the model"""
        if not result["success"]:
            return {"merge_operations": [], "split_operations": [], "refined_clusters": []}
        
        try:
            # Extract JSON from response
            response_text = result["response"]
            
            # Find JSON boundaries
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                
                # Clean up JSON
                json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                json_str = json_str.replace(',}', '}').replace(',]', ']')
                
                parsed = json.loads(json_str)
                return {
                    "merge_operations": parsed.get("merge_operations", []),
                    "split_operations": parsed.get("split_operations", []),
                    "refined_clusters": parsed.get("refined_clusters", [])
                }
            else:
                print("    ⚠️  No JSON found in response")
                return {"merge_operations": [], "split_operations": [], "refined_clusters": []}
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    ⚠️  JSON parsing error: {e}")
            return {"merge_operations": [], "split_operations": [], "refined_clusters": []}
    
    def calculate_merge_split_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate merge/split operation metrics"""
        merge_ops = analysis.get("merge_operations", [])
        split_ops = analysis.get("split_operations", [])
        refined_clusters = analysis.get("refined_clusters", [])
        
        # Calculate similarity scores for merge operations
        similarity_scores = [op.get("similarity_score", 0) for op in merge_ops]
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
        
        # Calculate cluster size distribution
        cluster_sizes = [len(cluster.get("message_ids", [])) for cluster in refined_clusters]
        
        return {
            "num_merge_operations": len(merge_ops),
            "num_split_operations": len(split_ops),
            "num_refined_clusters": len(refined_clusters),
            "avg_similarity_score": avg_similarity,
            "max_similarity_score": max(similarity_scores) if similarity_scores else 0,
            "min_similarity_score": min(similarity_scores) if similarity_scores else 0,
            "avg_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
            "total_messages": sum(cluster_sizes),
            "merge_operation_reasons": [op.get("reason", "") for op in merge_ops],
            "split_operation_reasons": [op.get("reason", "") for op in split_ops]
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
        """Run evaluation on all successful Phase 3 models"""
        print(f"🎯 {self.phase_name.upper()} EVALUATION")
        print("=" * 60)
        print(f"📊 Phase 3 results: {len(self.phase3_results)} successful models")
        print(f"📝 Messages: {len(self.messages)}")
        print(f"📁 Output directory: {self.output_dir}")
        
        if not self.phase3_results:
            print("❌ No successful Phase 3 results found. Please run Phase 3 first.")
            return
        
        results = {}
        total_models = 0
        successful_models = 0
        
        for phase3_result in self.phase3_results:
            provider = phase3_result['provider']
            model_name = phase3_result['model']
            
            print(f"\n{'='*20} Testing {provider.upper()}/{model_name.upper()} {'='*20}")
            
            total_models += 1
            try:
                result = self.evaluate_model(provider, model_name, phase3_result)
                results[f"{provider}_{model_name}"] = result
                
                # Save individual result
                output_file = os.path.join(self.output_dir, f"{provider}_{model_name}_step2.json")
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)
                
                if result["success"]:
                    successful_models += 1
                    metrics = result["metrics"]
                    print(f"  ✅ {model_name}: {result['duration']:.2f}s, "
                          f"Merges: {metrics['num_merge_operations']}, "
                          f"Splits: {metrics['num_split_operations']}, "
                          f"Clusters: {metrics['num_refined_clusters']}, "
                          f"Avg Similarity: {metrics['avg_similarity_score']:.3f}, "
                          f"Cost: ${result['cost']['total_cost']:.6f}")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    print(f"  ❌ {model_name}: {result['duration']:.2f}s, "
                          f"Error: {error_msg[:100]}...")
                
            except Exception as e:
                print(f"  ❌ {model_name}: Evaluation failed with error: {str(e)}")
                result = {
                    "provider": provider,
                    "model": model_name,
                    "success": False,
                    "error": str(e),
                    "duration": 0,
                    "cost": {"total_cost": 0},
                    "merge_operations": [],
                    "split_operations": [],
                    "refined_clusters": [],
                    "metrics": {}
                }
                results[f"{provider}_{model_name}"] = result
        
        # Save comprehensive results
        comprehensive_output = os.path.join(self.output_dir, "comprehensive_results.json")
        with open(comprehensive_output, "w") as f:
            json.dump(results, f, indent=2)
        
        # Find best model
        self.find_and_save_best_model(results)
        
        # Generate summary
        self.generate_summary(results, total_models, successful_models)
    
    def find_and_save_best_model(self, results: Dict):
        """Find the best model based on merge/split analysis quality"""
        successful_results = {k: v for k, v in results.items() if v["success"]}
        
        if not successful_results:
            print("⚠️  No successful results to analyze")
            return
        
        # Find best model based on number of meaningful operations
        best_model = None
        best_score = -1
        
        for model_name, result in successful_results.items():
            metrics = result["metrics"]
            # Score based on meaningful operations and similarity quality
            score = (
                metrics["num_merge_operations"] * 0.4 +
                metrics["num_split_operations"] * 0.3 +
                metrics["avg_similarity_score"] * 0.3
            )
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        if best_model:
            best_result = successful_results[best_model]
            
            # Save best model results
            best_file = os.path.join(self.output_dir, "best_model_step2.json")
            with open(best_file, "w") as f:
                json.dump(best_result, f, indent=2)
            
            print(f"\n🏆 Best Model: {best_model}")
            print(f"   Score: {best_score:.3f}")
            print(f"   Merge Operations: {best_result['metrics']['num_merge_operations']}")
            print(f"   Split Operations: {best_result['metrics']['num_split_operations']}")
            print(f"   Avg Similarity: {best_result['metrics']['avg_similarity_score']:.3f}")
            print(f"   Cost: ${best_result['cost']['total_cost']:.6f}")
            print(f"   Duration: {best_result['duration']:.2f}s")
            print(f"   Best results saved to: {best_file}")
    
    def generate_summary(self, results: Dict, total_models: int, successful_models: int):
        """Generate evaluation summary"""
        print(f"\n{'='*60}")
        print("📊 EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total models tested: {total_models}")
        print(f"Successful models: {successful_models}")
        print(f"Success rate: {(successful_models/total_models*100):.1f}%" if total_models > 0 else "0%")
        
        # Analyze successful results
        successful_results = {k: v for k, v in results.items() if v["success"]}
        
        if successful_results:
            # Find best performers
            best_merges = max(successful_results.items(), 
                           key=lambda x: x[1]["metrics"]["num_merge_operations"])
            
            best_splits = max(successful_results.items(),
                           key=lambda x: x[1]["metrics"]["num_split_operations"])
            
            best_similarity = max(successful_results.items(),
                               key=lambda x: x[1]["metrics"]["avg_similarity_score"])
            
            best_cost_efficiency = min(successful_results.items(),
                                     key=lambda x: x[1]["cost"]["total_cost"])
            
            print(f"\n🏆 Best Performance:")
            print(f"  Most Merges: {best_merges[0]} ({best_merges[1]['metrics']['num_merge_operations']})")
            print(f"  Most Splits: {best_splits[0]} ({best_splits[1]['metrics']['num_split_operations']})")
            print(f"  Best Similarity: {best_similarity[0]} ({best_similarity[1]['metrics']['avg_similarity_score']:.3f})")
            print(f"  Cost Efficiency: {best_cost_efficiency[0]} (${best_cost_efficiency[1]['cost']['total_cost']:.6f})")
        
        print(f"\nResults saved to: {self.output_dir}")

if __name__ == "__main__":
    evaluator = Phase4MergeSplitEvaluator()
    evaluator.run_evaluation()
