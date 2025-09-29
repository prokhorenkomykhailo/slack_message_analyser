#!/usr/bin/env python3
"""
Phase 3: Topic Clustering Evaluation
Evaluates all models on topic clustering task using synthetic Slack messages
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

class Phase3Evaluator:
    """Evaluates models on topic clustering task using synthetic Slack messages"""
    
    def __init__(self):
        self.phase_name = "phase3_topic_clustering"
        self.output_dir = os.path.join("output", self.phase_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        
        self.messages = self.load_messages_from_csv()
        
        
        self.reference_clusters = self.load_reference_clusters()
        
        
        self.prompt_template = self.get_clustering_prompt()
        
        print(f"📊 Loaded {len(self.messages)} messages from CSV")
        print(f"🎯 Loaded {len(self.reference_clusters)} reference clusters")
    
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
                    
                    message = {
                        "id": i + 1,  
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
            
            msg_text = (
                f"ID: {msg['id']} | Channel: {msg['channel']} | User: {msg['user']} | "
                f"Thread: {msg.get('thread_ts', 'None')} | Text: {msg['text'][:300]}..."
            )
            
            
            msg_tokens = self.estimate_tokens(msg_text)
            
            
            if total_tokens + msg_tokens > max_tokens:
                print(f"⚠️  Token limit reached. Using {len(formatted)} messages out of {len(self.messages)}")
                break
            
            formatted.append(msg_text)
            total_tokens += msg_tokens
        
        return "\n".join(formatted)
    
    def evaluate_model(self, provider: str, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model on topic clustering"""
        print(f"  🧪 Testing {provider}/{model_name}...")
        
        
        model_config = get_model_config(provider, model_name)
        max_tokens = MODEL_LIMITS.get(model_name, {}).get("max_tokens", 4096)
        context_length = MODEL_LIMITS.get(model_name, {}).get("context_length", 8192)
        
        
        available_tokens = min(context_length * 0.8, 80000)  
        
        
        messages_str = self.format_messages_for_prompt(int(available_tokens))
        prompt = self.prompt_template.replace("{messages}", messages_str)
        
        
        prompt_tokens = self.estimate_tokens(prompt)
        print(f"    📝 Estimated prompt tokens: {prompt_tokens:,}")
        
        
        result = call_model_with_retry(provider, model_name, prompt, max_retries=3)
        
        
        clusters = self.parse_clustering_response(result)
        
        
        metrics = self.calculate_clustering_metrics(clusters)
        
        
        comparison_metrics = self.compare_with_reference(clusters)
        
        
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
            "comparison_metrics": comparison_metrics,
            "raw_response": result.get("response", ""),
            "error": result.get("error", ""),
            "prompt_tokens_estimated": prompt_tokens,
            "messages_used": len(messages_str.split('\n'))
        }
    
    def parse_clustering_response(self, result: Dict[str, Any]) -> List[Dict]:
        """Parse the clustering response from the model"""
        if not result["success"]:
            return []
        
        try:
            
            response_text = result["response"]
            
            
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                
                json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                
                json_str = json_str.replace(',}', '}').replace(',]', ']')
                parsed = json.loads(json_str)
                return parsed.get("clusters", [])
            else:
                print("    ⚠️  No JSON found in response")
                return []
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    ⚠️  JSON parsing error: {e}")
            
            try:
                
                clusters = []
                lines = response_text.split('\n')
                current_cluster = None
                
                for line in lines:
                    if '"cluster_id"' in line or '"clusterId"' in line:
                        if current_cluster:
                            clusters.append(current_cluster)
                        current_cluster = {}
                    elif '"message_ids"' in line or '"messageIds"' in line:
                        
                        import re
                        ids = re.findall(r'\d+', line)
                        if current_cluster:
                            current_cluster['message_ids'] = [int(id) for id in ids]
                    elif '"draft_title"' in line:
                        
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
    
    def compare_with_reference(self, predicted_clusters: List[Dict]) -> Dict[str, Any]:
        """Compare predicted clusters with reference clusters"""
        if not self.reference_clusters or not predicted_clusters:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "cluster_similarity": 0.0,
                "title_similarity": 0.0,
                "participant_accuracy": 0.0
            }
        
        
        reference_sets = {}
        predicted_sets = {}
        
        for i, cluster in enumerate(self.reference_clusters):
            reference_sets[f"ref_{i}"] = set(cluster.get("message_ids", []))
        
        for i, cluster in enumerate(predicted_clusters):
            predicted_sets[f"pred_{i}"] = set(cluster.get("message_ids", []))
        
        
        total_intersection = 0
        total_predicted = 0
        total_reference = 0
        
        for pred_key, pred_set in predicted_sets.items():
            total_predicted += len(pred_set)
            best_intersection = 0
            for ref_key, ref_set in reference_sets.items():
                intersection = len(pred_set & ref_set)
                if intersection > best_intersection:
                    best_intersection = intersection
            total_intersection += best_intersection
        
        for ref_key, ref_set in reference_sets.items():
            total_reference += len(ref_set)
        
        precision = total_intersection / total_predicted if total_predicted > 0 else 0
        recall = total_intersection / total_reference if total_reference > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        
        title_similarities = []
        for pred_cluster in predicted_clusters:
            pred_title = pred_cluster.get("draft_title", "").lower()
            best_similarity = 0
            for ref_cluster in self.reference_clusters:
                ref_title = ref_cluster.get("draft_title", "").lower()
                pred_words = set(pred_title.split())
                ref_words = set(ref_title.split())
                if pred_words and ref_words:
                    similarity = len(pred_words & ref_words) / len(pred_words | ref_words)
                    best_similarity = max(best_similarity, similarity)
            title_similarities.append(best_similarity)
        
        avg_title_similarity = np.mean(title_similarities) if title_similarities else 0
        
        
        participant_accuracies = []
        for pred_cluster in predicted_clusters:
            pred_participants = set(pred_cluster.get("participants", []))
            best_accuracy = 0
            for ref_cluster in self.reference_clusters:
                ref_participants = set(ref_cluster.get("participants", []))
                if pred_participants and ref_participants:
                    accuracy = len(pred_participants & ref_participants) / len(pred_participants | ref_participants)
                    best_accuracy = max(best_accuracy, accuracy)
            participant_accuracies.append(best_accuracy)
        
        avg_participant_accuracy = np.mean(participant_accuracies) if participant_accuracies else 0
        
        
        ref_cluster_count = len(self.reference_clusters)
        pred_cluster_count = len(predicted_clusters)
        cluster_count_similarity = 1.0 - abs(ref_cluster_count - pred_cluster_count) / max(ref_cluster_count, pred_cluster_count)
        
        
        ref_avg_size = np.mean([len(cluster.get("message_ids", [])) for cluster in self.reference_clusters])
        pred_avg_size = np.mean([len(cluster.get("message_ids", [])) for cluster in predicted_clusters])
        size_similarity = 1.0 - abs(ref_avg_size - pred_avg_size) / max(ref_avg_size, pred_avg_size) if max(ref_avg_size, pred_avg_size) > 0 else 0
        
        
        overall_similarity = (
            f1_score * 0.4 +  
            cluster_count_similarity * 0.2 +  
            size_similarity * 0.2 +  
            avg_title_similarity * 0.15 +  
            avg_participant_accuracy * 0.05  
        )
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "cluster_similarity": overall_similarity,
            "title_similarity": avg_title_similarity,
            "participant_accuracy": avg_participant_accuracy,
            "cluster_count_similarity": cluster_count_similarity,
            "size_similarity": size_similarity,
            "ref_cluster_count": ref_cluster_count,
            "pred_cluster_count": pred_cluster_count
        }
    
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
        
        
        clustered_messages = set()
        thread_clusters = {}
        
        for cluster in clusters:
            message_ids = cluster.get("message_ids", [])
            clustered_messages.update(message_ids)
            
            
            thread_id = cluster.get("thread_id")
            if thread_id:
                if thread_id not in thread_clusters:
                    thread_clusters[thread_id] = []
                thread_clusters[thread_id].append(cluster["cluster_id"])
        
        total_messages = len(self.messages)
        coverage = len(clustered_messages) / total_messages if total_messages > 0 else 0
        
        cluster_sizes = [len(cluster.get("message_ids", [])) for cluster in clusters]
        
        
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
            "coverage": coverage,
            "thread_coherence": thread_coherence,
            "cluster_titles": [cluster.get("draft_title", "") for cluster in clusters],
            "unique_threads": len(thread_clusters)
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
        print(f"📊 Dataset: {len(self.messages)} synthetic Slack messages")
        print(f"🎯 Reference clusters: {len(self.reference_clusters)}")
        print(f"📁 Output directory: {self.output_dir}")
        
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
                try:
                    result = self.evaluate_model(provider, model_name)
                    results[f"{provider}_{model_name}"] = result
                except Exception as e:
                    print(f"  ❌ {model_name}: Evaluation failed with error: {str(e)}")
                    result = {
                        "provider": provider,
                        "model": model_name,
                        "success": False,
                        "error": str(e),
                        "duration": 0,
                        "cost": {"total_cost": 0},
                        "clusters": [],
                        "metrics": {},
                        "comparison_metrics": {}
                    }
                    results[f"{provider}_{model_name}"] = result
                
                
                output_file = os.path.join(self.output_dir, f"{provider}_{model_name}.json")
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)
                
                
                if result["success"] and result["clusters"]:
                    clusters_file = os.path.join(self.output_dir, f"{provider}_{model_name}_clusters.json")
                    with open(clusters_file, "w") as f:
                        json.dump(result["clusters"], f, indent=2)
                
                
                status = "✅" if result["success"] else "❌"
                if result["success"]:
                    successful_models += 1
                    try:
                        metrics = result["metrics"]
                        comparison = result["comparison_metrics"]
                        print(f"  {status} {model_name}: {result['duration']:.2f}s, "
                              f"Clusters: {metrics['num_clusters']} (ref: {comparison['ref_cluster_count']}), "
                              f"Overall: {comparison['cluster_similarity']:.3f}, "
                              f"F1: {comparison['f1_score']:.3f}, "
                              f"Title: {comparison['title_similarity']:.3f}, "
                              f"Cost: ${result['cost']['total_cost']:.6f}")
                    except (KeyError, TypeError) as e:
                        print(f"  {status} {model_name}: {result['duration']:.2f}s, "
                              f"Error in metrics: {str(e)}")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    print(f"  {status} {model_name}: {result['duration']:.2f}s, "
                          f"Error: {error_msg[:100]}...")
        
        
        comprehensive_output = os.path.join(self.output_dir, "comprehensive_results.json")
        with open(comprehensive_output, "w") as f:
            json.dump(results, f, indent=2)
        
        
        self.find_and_save_best_model(results)
        
        
        self.generate_summary(results, total_models, successful_models)
    
    def find_and_save_best_model(self, results: Dict):
        """Find the best model and save its clusters"""
        successful_results = {k: v for k, v in results.items() if v["success"]}
        
        if not successful_results:
            print("⚠️  No successful results to analyze")
            return
        
        
        best_model = None
        best_f1 = -1
        
        for model_name, result in successful_results.items():
            f1_score = result["comparison_metrics"]["f1_score"]
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model = model_name
        
        if best_model:
            best_result = successful_results[best_model]
            
            
            best_clusters_file = os.path.join(self.output_dir, "best_model_clusters.json")
            with open(best_clusters_file, "w") as f:
                json.dump(best_result["clusters"], f, indent=2)
            
            print(f"\n🏆 Best Model: {best_model}")
            print(f"   F1 Score: {best_f1:.3f}")
            print(f"   Precision: {best_result['comparison_metrics']['precision']:.3f}")
            print(f"   Recall: {best_result['comparison_metrics']['recall']:.3f}")
            print(f"   Title Similarity: {best_result['comparison_metrics']['title_similarity']:.3f}")
            print(f"   Cost: ${best_result['cost']['total_cost']:.6f}")
            print(f"   Duration: {best_result['duration']:.2f}s")
            print(f"   Best clusters saved to: {best_clusters_file}")
    
    def generate_summary(self, results: Dict, total_models: int, successful_models: int):
        """Generate evaluation summary"""
        print(f"\n{'='*60}")
        print("📊 EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total models tested: {total_models}")
        print(f"Successful models: {successful_models}")
        print(f"Success rate: {(successful_models/total_models*100):.1f}%" if total_models > 0 else "0%")
        
        
        successful_results = {k: v for k, v in results.items() if v["success"]}
        
        if successful_results:
            
            best_f1 = max(successful_results.items(), 
                         key=lambda x: x[1]["comparison_metrics"]["f1_score"])
            
            
            best_title = max(successful_results.items(),
                           key=lambda x: x[1]["comparison_metrics"]["title_similarity"])
            
            
            best_cost_efficiency = min(successful_results.items(),
                                     key=lambda x: x[1]["cost"]["total_cost"])
            
            
            best_speed = min(successful_results.items(),
                           key=lambda x: x[1]["duration"])
            
            print(f"\n🏆 Best Performance:")
            print(f"  F1 Score: {best_f1[0]} ({best_f1[1]['comparison_metrics']['f1_score']:.3f})")
            print(f"  Title Similarity: {best_title[0]} ({best_title[1]['comparison_metrics']['title_similarity']:.3f})")
            print(f"  Cost Efficiency: {best_cost_efficiency[0]} (${best_cost_efficiency[1]['cost']['total_cost']:.6f})")
            print(f"  Speed: {best_speed[0]} ({best_speed[1]['duration']:.2f}s)")
        
        print(f"\nResults saved to: {self.output_dir}")

if __name__ == "__main__":
    evaluator = Phase3Evaluator()
    evaluator.run_evaluation()
