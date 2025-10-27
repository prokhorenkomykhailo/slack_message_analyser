#!/usr/bin/env python3
"""
Phase 3: Comprehensive Topic Clustering Evaluation
Evaluates all models on topic clustering task with detailed metrics and analysis
"""

import os
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_clients import call_model_with_retry
from config.model_config import get_available_models, get_model_config, get_model_cost

# Model configurations for comprehensive testing
MODELS = {
    "openai": {
        "gpt-4": {"provider": "openai", "model": "gpt-4"},
        "gpt-3.5-turbo": {"provider": "openai", "model": "gpt-3.5-turbo"},
        "gpt-4o": {"provider": "openai", "model": "gpt-4o"}
    },
    "google": {
        "gemini-1.5-flash": {"provider": "google", "model": "gemini-1.5-flash"},
        "gemini-1.5-pro": {"provider": "google", "model": "gemini-1.5-pro"},
        "gemini-2.0-flash": {"provider": "google", "model": "gemini-2.0-flash"},
        "gemini-2.5-flash": {"provider": "google", "model": "gemini-2.5-flash"},
        "gemini-2.5-pro": {"provider": "google", "model": "gemini-2.5-pro"},
        "gemma-3-1b-it": {"provider": "google", "model": "gemma-3-1b-it"},
        "gemma-3-4b-it": {"provider": "google", "model": "gemma-3-4b-it"},
        "gemma-3-12b-it": {"provider": "google", "model": "gemma-3-12b-it"},
        "gemma-3-27b-it": {"provider": "google", "model": "gemma-3-27b-it"}
    },
    "groq": {
        "llama3-8b": {"provider": "groq", "model": "llama3-8b-8192"},
        "llama3-70b": {"provider": "groq", "model": "llama3-70b-8192"},
        "mixtral-8x7b": {"provider": "groq", "model": "mixtral-8x7b-32768"}
    },
    "anthropic": {
        "claude-3-opus": {"provider": "anthropic", "model": "claude-3-opus-20240229"},
        "claude-3-sonnet": {"provider": "anthropic", "model": "claude-3-sonnet-20240229"},
        "claude-3-haiku": {"provider": "anthropic", "model": "claude-3-haiku-20240307"}
    },
    "xai": {
        "grok-2": {"provider": "xai", "model": "grok-2-1212"},
        "grok-2-vision": {"provider": "xai", "model": "grok-2-vision-1212"},
        "grok-3": {"provider": "xai", "model": "grok-3"}
    }
}

class Phase3ComprehensiveEvaluator:
    """Comprehensive evaluator for Phase 3 topic clustering"""
    
    def __init__(self):
        self.phase_name = "phase3_topic_clustering"
        self.output_dir = os.path.join("output", self.phase_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data
        self.messages = self.load_messages()
        self.benchmark_topics = self.load_benchmark_topics()
        
        # Topic clustering prompt
        self.prompt_template = self.get_clustering_prompt()
        
        # Results storage
        self.results = {}
        self.summary_stats = {}
    
    def load_messages(self) -> List[Dict]:
        """Load message dataset"""
        try:
            # Try multiple possible locations for the CSV file
            possible_paths = [
                "deemerge_test/data/Synthetic_Slack_Messages.csv",
                "../deemerge_test/data/Synthetic_Slack_Messages.csv",
                "data/Synthetic_Slack_Messages.csv",
                "../data/Synthetic_Slack_Messages.csv",
                "message_dataset.json",
                "../message_dataset.json"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"📁 Loading messages from: {path}")
                    if path.endswith('.csv'):
                        messages = self.load_csv_messages(path)
                        print(f"✅ Loaded {len(messages)} messages from CSV")
                        return messages
                    else:
                        with open(path, "r") as f:
                            messages = json.load(f)
                            print(f"✅ Loaded {len(messages)} messages from JSON")
                            return messages
            
            # If not found, create dummy data for testing
            print("⚠️  Synthetic_Slack_Messages.csv not found. Creating dummy data for testing.")
            return self.create_dummy_messages()
            
        except Exception as e:
            print(f"❌ Error loading messages: {e}")
            return self.create_dummy_messages()
    
    def load_csv_messages(self, csv_path: str) -> List[Dict]:
        """Load messages from CSV format"""
        import pandas as pd
        df = pd.read_csv(csv_path)
        messages = []
        for idx, row in df.iterrows():
            messages.append({
                "id": idx + 1,
                "channel": row['channel'],
                "user": row['user_name'],
                "text": row['text'],
                "thread_ts": row['thread_id'],
                "timestamp": row['timestamp']
            })
        print(f"📊 CSV contains {len(df)} rows, created {len(messages)} message objects")
        return messages
    
    def create_dummy_messages(self) -> List[Dict]:
        """Create dummy messages for testing"""
        messages = []
        for i in range(1, 51):
            messages.append({
                "id": i,
                "channel": "#general",
                "user": f"user_{i % 4 + 1}",
                "text": f"This is message {i} for testing purposes.",
                "thread_ts": f"thread_{i // 5 + 1}"
            })
        return messages
    
    def load_benchmark_topics(self) -> List[Dict]:
        """Load benchmark topics"""
        try:
            # Try multiple possible locations
            possible_paths = [
                "../benchmark_topics.json",
                "benchmark_topics.json",
                "data/benchmark_topics.json",
                "deemerge_test/data/benchmark_topics_corrected_fixed.json"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        data = json.load(f)
                        return data.get("topics", [])
            
            print("⚠️  benchmark_topics.json not found. Using empty benchmark.")
            return []
            
        except Exception as e:
            print(f"❌ Error loading benchmark topics: {e}")
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

**Messages to analyze:**
{messages}

**Instructions:**
- Group related messages together based on meaningful relationships
- Each message should belong to exactly one cluster
- Consider both explicit relationships (threads) and implicit ones (same topic)
- Merge related threads and topics into cohesive clusters
- Provide clear, descriptive titles for each cluster
- Focus on project discussions, client work, and team coordination
- Focus on creating logical, meaningful groupings rather than arbitrary cluster counts
- Look for patterns across threads that discuss the same projects or clients

**Output Format (JSON):**
{{
  "clusters": [
    {{
      "cluster_id": "cluster_001",
      "message_ids": [1, 2, 3, 4, 5],
      "draft_title": "Project Planning Discussion",
      "participants": ["@alice", "@bob", "@charlie"],
      "channel": "#general"
    }}
  ]
}}

Analyze the messages and provide the clustering results in the specified JSON format.
"""
    
    def format_messages_for_prompt(self) -> str:
        """Format messages for the prompt"""
        formatted = []
        
        # If we have more than 200 messages, sample them to avoid token limits
        messages_to_process = self.messages
        if len(self.messages) > 200:
            print(f"⚠️  Large dataset detected ({len(self.messages)} messages). Sampling 200 messages for analysis.")
            # Sample every nth message to get a representative subset
            step = len(self.messages) // 200
            messages_to_process = self.messages[::step][:200]
        
        for msg in messages_to_process:
            # Truncate text to avoid token limits
            text_preview = msg['text'][:150] + "..." if len(msg['text']) > 150 else msg['text']
            formatted.append(
                f"ID: {msg['id']} | Channel: {msg['channel']} | User: {msg['user']} | "
                f"Thread: {msg.get('thread_ts', 'None')} | Text: {text_preview}"
            )
        
        print(f"📝 Formatted {len(formatted)} messages for prompt")
        return "\n".join(formatted)
    
    def evaluate_model(self, provider: str, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model on topic clustering"""
        print(f"  🧪 Testing {provider}/{model_name}...")
        
        start_time = time.time()
        
        try:
            # Prepare prompt
            messages_str = self.format_messages_for_prompt()
            prompt = self.prompt_template.replace("{messages}", messages_str)
            
            # Call model
            result = call_model_with_retry(provider, model_name, prompt, max_retries=3)
            
            # Parse response
            clusters = self.parse_clustering_response(result)
            
            # Calculate metrics
            metrics = self.calculate_clustering_metrics(clusters)
            
            # Calculate cost
            cost = self.calculate_cost(provider, model_name, result)
            
            duration = time.time() - start_time
            
            return {
                "provider": provider,
                "model": model_name,
                "phase": self.phase_name,
                "timestamp": datetime.now().isoformat(),
                "success": result["success"],
                "duration": duration,
                "usage": result.get("usage", {}),
                "cost": cost,
                "clusters": clusters,
                "metrics": metrics,
                "raw_response": result.get("response", ""),
                "error": result.get("error", "")
            }
            
        except Exception as e:
            duration = time.time() - start_time
            return {
                "provider": provider,
                "model": model_name,
                "phase": self.phase_name,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "duration": duration,
                "usage": {},
                "cost": {"total_cost": 0, "input_cost": 0, "output_cost": 0},
                "clusters": [],
                "metrics": {
                    "num_clusters": 0,
                    "total_messages_clustered": 0,
                    "avg_cluster_size": 0,
                    "coverage": 0.0
                },
                "raw_response": "",
                "error": str(e)
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
                parsed = json.loads(json_str)
                return parsed.get("clusters", [])
            else:
                return []
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    ⚠️  JSON parsing error: {e}")
            return []
    
    def calculate_clustering_metrics(self, clusters: List[Dict]) -> Dict[str, Any]:
        """Calculate clustering quality metrics"""
        if not clusters:
            return {
                "num_clusters": 0,
                "total_messages_clustered": 0,
                "avg_cluster_size": 0,
                "max_cluster_size": 0,
                "min_cluster_size": 0,
                "coverage": 0.0,
                "cluster_titles": []
            }
        
        # Count messages in clusters
        clustered_messages = set()
        for cluster in clusters:
            message_ids = cluster.get("message_ids", [])
            clustered_messages.update(message_ids)
        
        total_messages = len(self.messages)
        coverage = len(clustered_messages) / total_messages if total_messages > 0 else 0
        
        cluster_sizes = [len(cluster.get("message_ids", [])) for cluster in clusters]
        
        return {
            "num_clusters": len(clusters),
            "total_messages_clustered": len(clustered_messages),
            "avg_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
            "coverage": coverage,
            "cluster_titles": [cluster.get("draft_title", "") for cluster in clusters]
        }
    
    def calculate_cost(self, provider: str, model_name: str, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the cost of the API call"""
        try:
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
        except Exception as e:
            print(f"    ⚠️  Cost calculation error: {e}")
            return {
                "input_cost": 0,
                "output_cost": 0,
                "total_cost": 0,
                "input_tokens": 0,
                "output_tokens": 0
            }
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation on all models"""
        print(f"🎯 PHASE 3 COMPREHENSIVE TOPIC CLUSTERING EVALUATION")
        print("=" * 80)
        print(f"Total messages to analyze: {len(self.messages)}")
        print(f"Benchmark topics available: {len(self.benchmark_topics)}")
        print("=" * 80)
        
        total_models = 0
        successful_models = 0
        
        # Test each provider and model
        for provider, provider_models in MODELS.items():
            print(f"\n{'='*30} Testing {provider.upper()} Models {'='*30}")
            
            for model_key, model_config in provider_models.items():
                total_models += 1
                model_name = model_config["model"]
                
                print(f"\n📊 Testing {provider}/{model_name}...")
                
                # Evaluate model
                result = self.evaluate_model(provider, model_name)
                self.results[f"{provider}_{model_name}"] = result
                
                # Save individual result
                output_file = os.path.join(self.output_dir, f"{provider}_{model_name}.json")
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)
                
                # Print status
                status = "✅" if result["success"] else "❌"
                if result["success"]:
                    successful_models += 1
                    metrics = result["metrics"]
                    cost = result["cost"]
                    print(f"  {status} Success: {result['duration']:.2f}s")
                    print(f"     Clusters: {metrics['num_clusters']}")
                    print(f"     Coverage: {metrics['coverage']:.2%}")
                    print(f"     Avg Cluster Size: {metrics['avg_cluster_size']:.1f}")
                    print(f"     Cost: ${cost['total_cost']:.6f}")
                    print(f"     Tokens: {cost['input_tokens']} input, {cost['output_tokens']} output")
                else:
                    print(f"  {status} Failed: {result['duration']:.2f}s")
                    print(f"     Error: {result['error'][:100]}...")
        
        # Save comprehensive results
        comprehensive_output = os.path.join(self.output_dir, "comprehensive_results.json")
        with open(comprehensive_output, "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Generate detailed analysis
        self.generate_detailed_analysis()
        
        # Generate summary
        self.generate_summary(total_models, successful_models)
    
    def generate_detailed_analysis(self):
        """Generate detailed analysis of results"""
        print(f"\n{'='*80}")
        print("📈 DETAILED ANALYSIS")
        print(f"{'='*80}")
        
        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() if v["success"]}
        
        if not successful_results:
            print("❌ No successful results to analyze")
            return
        
        # Create analysis data
        analysis_data = []
        for model_key, result in successful_results.items():
            analysis_data.append({
                "model": model_key,
                "provider": result["provider"],
                "model_name": result["model"],
                "duration": result["duration"],
                "num_clusters": result["metrics"]["num_clusters"],
                "coverage": result["metrics"]["coverage"],
                "avg_cluster_size": result["metrics"]["avg_cluster_size"],
                "total_cost": result["cost"]["total_cost"],
                "input_tokens": result["cost"]["input_tokens"],
                "output_tokens": result["cost"]["output_tokens"]
            })
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(analysis_data)
        
        # Performance rankings
        print("\n🏆 PERFORMANCE RANKINGS:")
        
        # Best coverage
        best_coverage = df.loc[df['coverage'].idxmax()]
        print(f"  📊 Best Coverage: {best_coverage['model']} ({best_coverage['coverage']:.2%})")
        
        # Best cost efficiency
        best_cost = df.loc[df['total_cost'].idxmin()]
        print(f"  💰 Best Cost Efficiency: {best_cost['model']} (${best_cost['total_cost']:.6f})")
        
        # Fastest
        fastest = df.loc[df['duration'].idxmin()]
        print(f"  ⚡ Fastest: {fastest['model']} ({fastest['duration']:.2f}s)")
        
        # Most clusters
        most_clusters = df.loc[df['num_clusters'].idxmax()]
        print(f"  📋 Most Clusters: {most_clusters['model']} ({most_clusters['num_clusters']})")
        
        # Provider analysis
        print(f"\n🏢 PROVIDER ANALYSIS:")
        provider_stats = df.groupby('provider').agg({
            'coverage': ['mean', 'max'],
            'total_cost': ['mean', 'min'],
            'duration': ['mean', 'min']
        }).round(4)
        print(provider_stats)
        
        # Save analysis
        analysis_output = os.path.join(self.output_dir, "detailed_analysis.json")
        
        # Convert provider_stats to a JSON-serializable format
        provider_stats_dict = {}
        for provider in provider_stats.index:
            provider_stats_dict[provider] = {}
            for col in provider_stats.columns:
                # Convert tuple column names to strings
                col_name = "_".join(col) if isinstance(col, tuple) else str(col)
                provider_stats_dict[provider][col_name] = float(provider_stats.loc[provider, col])
        
        analysis_summary = {
            "total_models_tested": len(self.results),
            "successful_models": len(successful_results),
            "success_rate": len(successful_results) / len(self.results),
            "best_performers": {
                "coverage": best_coverage.to_dict(),
                "cost_efficiency": best_cost.to_dict(),
                "speed": fastest.to_dict(),
                "cluster_count": most_clusters.to_dict()
            },
            "provider_stats": provider_stats_dict
        }
        
        with open(analysis_output, "w") as f:
            json.dump(analysis_summary, f, indent=2)
        
        # Save DataFrame as CSV
        csv_output = os.path.join(self.output_dir, "results_analysis.csv")
        df.to_csv(csv_output, index=False)
        print(f"\n📁 Analysis saved to: {analysis_output}")
        print(f"📁 CSV results saved to: {csv_output}")
    
    def generate_summary(self, total_models: int, successful_models: int):
        """Generate evaluation summary"""
        print(f"\n{'='*80}")
        print("📊 EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total models tested: {total_models}")
        print(f"Successful models: {successful_models}")
        print(f"Success rate: {(successful_models/total_models*100):.1f}%" if total_models > 0 else "0%")
        
        # Calculate total cost
        total_cost = sum(result["cost"]["total_cost"] for result in self.results.values())
        print(f"Total evaluation cost: ${total_cost:.6f}")
        
        # Provider breakdown
        provider_counts = {}
        for result in self.results.values():
            provider = result["provider"]
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        print(f"\n📊 Provider Breakdown:")
        for provider, count in provider_counts.items():
            print(f"  {provider}: {count} models")
        
        print(f"\n📁 Results saved to: {self.output_dir}")
        print(f"📁 Comprehensive results: {os.path.join(self.output_dir, 'comprehensive_results.json')}")
        print(f"📁 Detailed analysis: {os.path.join(self.output_dir, 'detailed_analysis.json')}")

if __name__ == "__main__":
    evaluator = Phase3ComprehensiveEvaluator()
    evaluator.run_comprehensive_evaluation()
