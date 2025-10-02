#!/usr/bin/env python3
"""
Evaluate Phase 4 Balanced Refinement Output Against Benchmark
Compares the refined clusters from phase4_balanced_refinement.py against the benchmark topics
"""

import os
import json
import time
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

class Phase4BalancedRefinementEvaluator:
    """Evaluates Phase 4 Balanced Refinement results against benchmark"""
    
    def __init__(self):
        self.output_dir = "output/phase4_balanced_refinement_evaluation"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load benchmark topics
        self.benchmark_topics = self.load_benchmark_topics()
        print(f"✅ Loaded {len(self.benchmark_topics)} benchmark topics")
        
        # Initialize embedding model for similarity
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
        
        # Load phase4 results
        self.phase4_results = self.load_phase4_results()
        print(f"✅ Loaded {len(self.phase4_results)} Phase 4 results")
    
    def load_benchmark_topics(self) -> List[Dict]:
        """Load benchmark topics from the corrected fixed file"""
        try:
            benchmark_path = "data/benchmark_topics_corrected_fixed.json"
            with open(benchmark_path, "r") as f:
                data = json.load(f)
            
            topics = data.get("topics", [])
            print(f"📊 Benchmark topics structure: {len(topics)} topics")
            return topics
            
        except Exception as e:
            print(f"❌ Error loading benchmark: {e}")
            return []
    
    def load_phase4_results(self) -> Dict[str, Dict]:
        """Load all Phase 4 results"""
        results = {}
        phase4_dir = "output/phase4_balanced_refinement"
        
        for filename in os.listdir(phase4_dir):
            if filename.endswith('.json') and not filename.startswith('comprehensive_'):
                filepath = os.path.join(phase4_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        result = json.load(f)
                    
                    if result.get('success', False):
                        model_key = f"{result['provider']}_{result['model']}"
                        results[model_key] = result
                        print(f"✅ Loaded {model_key}")
                        
                except Exception as e:
                    print(f"⚠️ Error loading {filename}: {e}")
        
        return results
    
    def calculate_similarity_score(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using embeddings"""
        try:
            embeddings = self.embedding_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"⚠️ Error calculating similarity: {e}")
            return 0.0
    
    def evaluate_clusters_against_benchmark(self, refined_clusters: List[Dict]) -> Dict[str, Any]:
        """Evaluate refined clusters against benchmark topics"""
        
        if not refined_clusters or not self.benchmark_topics:
            return {"error": "No clusters or benchmark topics available"}
        
        # Extract cluster titles and summaries
        cluster_titles = [cluster.get('draft_title', '') for cluster in refined_clusters]
        cluster_summaries = [cluster.get('summary', '') for cluster in refined_clusters]
        
        # Extract benchmark titles and summaries
        benchmark_titles = [topic.get('title', '') for topic in self.benchmark_topics]
        benchmark_summaries = [topic.get('summary', '') for topic in self.benchmark_topics]
        
        # Calculate similarity matrices
        title_similarities = []
        summary_similarities = []
        
        for cluster_title in cluster_titles:
            cluster_title_sims = []
            cluster_summary_sims = []
            
            for i, benchmark_title in enumerate(benchmark_titles):
                # Title similarity
                title_sim = self.calculate_similarity_score(cluster_title, benchmark_title)
                cluster_title_sims.append(title_sim)
                
                # Summary similarity (if available)
                cluster_summary = cluster_summaries[cluster_titles.index(cluster_title)] if cluster_title in cluster_titles else ""
                benchmark_summary = benchmark_summaries[i]
                summary_sim = self.calculate_similarity_score(cluster_summary, benchmark_summary)
                cluster_summary_sims.append(summary_sim)
            
            title_similarities.append(cluster_title_sims)
            summary_similarities.append(cluster_summary_sims)
        
        # Find best matches
        best_matches = []
        for i, cluster_title in enumerate(cluster_titles):
            max_title_sim = max(title_similarities[i]) if title_similarities[i] else 0
            max_summary_sim = max(summary_similarities[i]) if summary_similarities[i] else 0
            best_match_idx = title_similarities[i].index(max_title_sim)
            
            best_matches.append({
                "cluster_title": cluster_title,
                "best_benchmark_match": benchmark_titles[best_match_idx],
                "title_similarity": max_title_sim,
                "summary_similarity": max_summary_sim,
                "combined_similarity": (max_title_sim + max_summary_sim) / 2
            })
        
        # Calculate overall metrics
        avg_title_similarity = np.mean([match["title_similarity"] for match in best_matches])
        avg_summary_similarity = np.mean([match["summary_similarity"] for match in best_matches])
        avg_combined_similarity = np.mean([match["combined_similarity"] for match in best_matches])
        
        # Count high-quality matches (similarity > 0.7)
        high_quality_matches = len([m for m in best_matches if m["combined_similarity"] > 0.7])
        quality_rate = high_quality_matches / len(best_matches) if best_matches else 0
        
        return {
            "num_refined_clusters": len(refined_clusters),
            "num_benchmark_topics": len(self.benchmark_topics),
            "avg_title_similarity": avg_title_similarity,
            "avg_summary_similarity": avg_summary_similarity,
            "avg_combined_similarity": avg_combined_similarity,
            "high_quality_matches": high_quality_matches,
            "quality_rate": quality_rate,
            "best_matches": best_matches,
            "title_similarities": title_similarities,
            "summary_similarities": summary_similarities
        }
    
    def evaluate_model_result(self, model_key: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single model result"""
        
        if not result.get('success', False):
            return {
                "model": model_key,
                "success": False,
                "error": "Model execution failed"
            }
        
        refined_clusters = result.get('refined_clusters', [])
        if not refined_clusters:
            return {
                "model": model_key,
                "success": False,
                "error": "No refined clusters found"
            }
        
        # Evaluate against benchmark
        benchmark_evaluation = self.evaluate_clusters_against_benchmark(refined_clusters)
        
        # Calculate additional metrics
        cluster_sizes = [len(cluster.get('message_ids', [])) for cluster in refined_clusters]
        operations = result.get('operations', [])
        merge_ops = len([op for op in operations if op.get('operation') == 'merge'])
        split_ops = len([op for op in operations if op.get('operation') == 'split'])
        
        return {
            "model": model_key,
            "success": True,
            "provider": result.get('provider', 'unknown'),
            "model_name": result.get('model', 'unknown'),
            "duration": result.get('duration', 0),
            "num_refined_clusters": len(refined_clusters),
            "avg_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
            "merge_operations": merge_ops,
            "split_operations": split_ops,
            "total_operations": len(operations),
            "benchmark_evaluation": benchmark_evaluation,
            "cluster_titles": [cluster.get('draft_title', '') for cluster in refined_clusters],
            "timestamp": result.get('timestamp', '')
        }
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation on all Phase 4 results"""
        
        print("🎯 PHASE 4 BALANCED REFINEMENT EVALUATION")
        print("=" * 60)
        
        evaluation_results = {}
        
        for model_key, result in self.phase4_results.items():
            print(f"\n📊 Evaluating {model_key}...")
            
            evaluation = self.evaluate_model_result(model_key, result)
            evaluation_results[model_key] = evaluation
            
            if evaluation['success']:
                benchmark_eval = evaluation['benchmark_evaluation']
                print(f"  ✅ Success: {evaluation['num_refined_clusters']} clusters")
                print(f"  📈 Avg Similarity: {benchmark_eval['avg_combined_similarity']:.3f}")
                print(f"  🎯 Quality Rate: {benchmark_eval['quality_rate']:.1%}")
                print(f"  ⚡ Duration: {evaluation['duration']:.2f}s")
            else:
                print(f"  ❌ Failed: {evaluation.get('error', 'Unknown error')}")
        
        # Save detailed results
        detailed_output = os.path.join(self.output_dir, "detailed_evaluation_results.json")
        with open(detailed_output, "w") as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Generate summary report
        self.generate_summary_report(evaluation_results)
        
        print(f"\n💾 Detailed results saved to: {detailed_output}")
        print("🎉 Evaluation complete!")
    
    def generate_summary_report(self, evaluation_results: Dict[str, Dict]):
        """Generate a summary report of the evaluation"""
        
        successful_models = [eval for eval in evaluation_results.values() if eval['success']]
        
        if not successful_models:
            print("❌ No successful models to evaluate")
            return
        
        # Calculate summary statistics
        avg_similarity = np.mean([eval['benchmark_evaluation']['avg_combined_similarity'] 
                                for eval in successful_models])
        avg_quality_rate = np.mean([eval['benchmark_evaluation']['quality_rate'] 
                                  for eval in successful_models])
        avg_clusters = np.mean([eval['num_refined_clusters'] for eval in successful_models])
        avg_duration = np.mean([eval['duration'] for eval in successful_models])
        
        # Find best performing models
        best_similarity = max(successful_models, 
                            key=lambda x: x['benchmark_evaluation']['avg_combined_similarity'])
        best_quality = max(successful_models, 
                         key=lambda x: x['benchmark_evaluation']['quality_rate'])
        
        summary = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_models_evaluated": len(evaluation_results),
            "successful_models": len(successful_models),
            "summary_statistics": {
                "avg_similarity_score": avg_similarity,
                "avg_quality_rate": avg_quality_rate,
                "avg_clusters_generated": avg_clusters,
                "avg_duration_seconds": avg_duration
            },
            "best_performers": {
                "highest_similarity": {
                    "model": best_similarity['model'],
                    "similarity_score": best_similarity['benchmark_evaluation']['avg_combined_similarity']
                },
                "highest_quality_rate": {
                    "model": best_quality['model'],
                    "quality_rate": best_quality['benchmark_evaluation']['quality_rate']
                }
            },
            "model_rankings": self.rank_models(successful_models)
        }
        
        # Save summary
        summary_output = os.path.join(self.output_dir, "evaluation_summary.json")
        with open(summary_output, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n📋 EVALUATION SUMMARY")
        print("=" * 40)
        print(f"✅ Successful Models: {len(successful_models)}/{len(evaluation_results)}")
        print(f"📊 Avg Similarity Score: {avg_similarity:.3f}")
        print(f"�� Avg Quality Rate: {avg_quality_rate:.1%}")
        print(f"📈 Avg Clusters Generated: {avg_clusters:.1f}")
        print(f"⚡ Avg Duration: {avg_duration:.2f}s")
        print(f"\n🏆 Best Similarity: {best_similarity['model']} ({best_similarity['benchmark_evaluation']['avg_combined_similarity']:.3f})")
        print(f"🏆 Best Quality: {best_quality['model']} ({best_quality['benchmark_evaluation']['quality_rate']:.1%})")
        
        print(f"\n💾 Summary saved to: {summary_output}")
    
    def rank_models(self, successful_models: List[Dict]) -> List[Dict]:
        """Rank models by combined performance score"""
        
        def calculate_score(eval_result):
            similarity = eval_result['benchmark_evaluation']['avg_combined_similarity']
            quality = eval_result['benchmark_evaluation']['quality_rate']
            # Combined score: 70% similarity + 30% quality
            return 0.7 * similarity + 0.3 * quality
        
        ranked_models = sorted(successful_models, key=calculate_score, reverse=True)
        
        return [
            {
                "rank": i + 1,
                "model": model['model'],
                "combined_score": calculate_score(model),
                "similarity_score": model['benchmark_evaluation']['avg_combined_similarity'],
                "quality_rate": model['benchmark_evaluation']['quality_rate'],
                "clusters_generated": model['num_refined_clusters'],
                "duration": model['duration']
            }
            for i, model in enumerate(ranked_models)
        ]

def main():
    """Main evaluation function"""
    
    print("🚀 Starting Phase 4 Balanced Refinement Evaluation")
    print("=" * 60)
    
    evaluator = Phase4BalancedRefinementEvaluator()
    evaluator.run_comprehensive_evaluation()

if __name__ == "__main__":
    main()
