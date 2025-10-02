#!/usr/bin/env python3
"""
Phase 4: Fixed AI Embeddings Merge/Split Service
Uses the 6-cluster benchmark and only processes successful models
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_clients import call_model_with_retry
from config.model_config import get_available_models, get_model_config, get_model_cost

class Phase4AIEmbeddingsFixed:
    """Fixed AI embeddings evaluator for merge/split operations"""
    
    def __init__(self):
        self.phase_name = "phase4_ai_embeddings_fixed"
        self.output_dir = os.path.join("output", self.phase_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load the 6-cluster benchmark
        self.initial_clusters = self.load_benchmark_clusters()
        self.messages = self.load_messages()
        
        # Initialize embedding model
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
    
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
    
    def get_cluster_embeddings(self, clusters: List[Dict]) -> np.ndarray:
        """Get embeddings for cluster content"""
        cluster_texts = []
        for cluster in clusters:
            # Get messages for this cluster
            cluster_messages = [msg for msg in self.messages 
                              if msg["message_id"] in cluster["message_ids"]]
            
            # Combine all message text
            combined_text = " ".join([msg["text"] for msg in cluster_messages])
            cluster_texts.append(combined_text)
        
        # Get embeddings
        embeddings = self.embedding_model.encode(cluster_texts)
        return embeddings
    
    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity matrix"""
        return cosine_similarity(embeddings)
    
    def find_merge_candidates(self, similarity_matrix: np.ndarray, threshold: float = 0.7) -> List[tuple]:
        """Find clusters that should be merged based on similarity"""
        merge_candidates = []
        n_clusters = len(similarity_matrix)
        
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                if similarity_matrix[i][j] >= threshold:
                    merge_candidates.append((i, j, similarity_matrix[i][j]))
        
        # Sort by similarity (highest first)
        merge_candidates.sort(key=lambda x: x[2], reverse=True)
        return merge_candidates
    
    def create_merge_split_prompt(self, clusters: List[Dict], similarity_analysis: Dict) -> str:
        """Create prompt for AI model to perform merge/split operations"""
        clusters_text = json.dumps(clusters, indent=2)
        similarity_text = json.dumps(similarity_analysis, indent=2)
        
        return f"""
You are an expert at analyzing topic clusters and determining merge/split operations.

INITIAL CLUSTERS (6 clusters):
{clusters_text}

SIMILARITY ANALYSIS:
{similarity_text}

Based on the similarity analysis above, determine which clusters should be:
1. MERGED (high similarity > 0.7)
2. SPLIT (too broad, multiple topics)
3. KEPT AS-IS (well-organized)

Provide your analysis in this JSON format:
{{
    "refined_clusters": [
        {{
            "cluster_id": "refined_001",
            "message_ids": [1, 2, 3, 4, 5],
            "draft_title": "Merged Topic Title",
            "participants": ["@user1", "@user2"],
            "merge_reason": "High similarity (0.85) - both about campaign planning",
            "split_reason": null
        }}
    ],
    "operations": [
        {{
            "operation": "merge",
            "clusters_involved": ["cluster_001", "cluster_002"],
            "reason": "High cosine similarity (0.85) based on campaign planning content",
            "similarity_score": 0.85
        }}
    ],
    "reasoning": "Used embedding + similarity approach: merged clusters with >0.7 similarity, split overloaded clusters"
}}
"""
    
    def check_step1_success(self, model_name: str) -> bool:
        """Check if Step 1 was successful for this model"""
        try:
            step1_file = f"output/phase3_topic_clustering/{model_name}.json"
            if os.path.exists(step1_file):
                with open(step1_file, "r") as f:
                    result = json.load(f)
                return result.get("success", False)
            return False
        except:
            return False
    
    def evaluate_model_with_embeddings(self, model_name: str) -> Dict[str, Any]:
        """Evaluate model using embedding-based similarity analysis"""
        print(f"\n🔍 Evaluating {model_name} with AI embeddings...")
        
        # Check if Step 1 was successful
        if not self.check_step1_success(model_name):
            print(f"⏭️  Skipping {model_name} - Step 1 failed")
            return {
                "provider": model_name.split("_")[0],
                "model": model_name,
                "phase": "step2_ai_embeddings",
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "duration": 0,
                "error": "Step 1 failed - skipping Step 2",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cost": {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
            }
        
        try:
            # Get cluster embeddings
            print("🔄 Calculating cluster embeddings...")
            embeddings = self.get_cluster_embeddings(self.initial_clusters)
            
            # Calculate similarity matrix
            print("🔄 Calculating similarity matrix...")
            similarity_matrix = self.calculate_similarity_matrix(embeddings)
            
            # Find merge candidates
            merge_candidates = self.find_merge_candidates(similarity_matrix, threshold=0.7)
            
            # Prepare similarity analysis
            similarity_analysis = {
                "similarity_matrix": similarity_matrix.tolist(),
                "merge_candidates": merge_candidates,
                "threshold": 0.7,
                "total_clusters": len(self.initial_clusters)
            }
            
            # Create prompt with similarity analysis
            prompt = self.create_merge_split_prompt(self.initial_clusters, similarity_analysis)
            
            # Call AI model
            print(f"🤖 Calling {model_name}...")
            start_time = time.time()
            response = call_model_with_retry(model_name, prompt)
            duration = time.time() - start_time
            
            # Parse response
            result = self.parse_model_response(response, model_name, duration, similarity_analysis)
            return result
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {str(e)}")
            return {
                "provider": model_name.split("_")[0],
                "model": model_name,
                "phase": "step2_ai_embeddings",
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "duration": 0,
                "error": str(e),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cost": {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
            }
    
    def parse_model_response(self, response: str, model_name: str, duration: float, similarity_analysis: Dict) -> Dict[str, Any]:
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
                "phase": "step2_ai_embeddings",
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "duration": duration,
                "initial_clusters": self.initial_clusters,
                "refined_clusters": result_data.get("refined_clusters", []),
                "operations": result_data.get("operations", []),
                "reasoning": result_data.get("reasoning", ""),
                "similarity_analysis": similarity_analysis,
                "metrics": {
                    "initial_clusters": initial_count,
                    "refined_clusters": refined_count,
                    "reduction_ratio": (initial_count - refined_count) / initial_count if initial_count > 0 else 0,
                    "merge_candidates_found": len(similarity_analysis["merge_candidates"])
                },
                "raw_response": response,
                "error": ""
            }
            
        except Exception as e:
            return {
                "provider": model_name.split("_")[0],
                "model": model_name,
                "phase": "step2_ai_embeddings",
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "duration": duration,
                "error": f"Failed to parse response: {str(e)}",
                "raw_response": response,
                "similarity_analysis": similarity_analysis
            }
    
    def run_evaluation(self):
        """Run evaluation on all available models"""
        print("🚀 Starting Phase 4: Fixed AI Embeddings Merge/Split")
        print("=" * 60)
        print(f"📊 Using {len(self.initial_clusters)} benchmark clusters as input")
        
        # Get available models
        available_models = get_available_models()
        print(f"📊 Found {len(available_models)} models to evaluate")
        
        results = {}
        successful_count = 0
        
        for model_name in available_models:
            print(f"\n�� Processing {model_name}...")
            result = self.evaluate_model_with_embeddings(model_name)
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
            output_file = os.path.join(self.output_dir, f"{model_name}_embeddings_step2.json")
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
    evaluator = Phase4AIEmbeddingsFixed()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
