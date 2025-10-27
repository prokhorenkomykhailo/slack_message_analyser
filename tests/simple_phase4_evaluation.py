#!/usr/bin/env python3
"""
Simple Phase 4 Balanced Refinement Evaluation
Evaluates the output against benchmark without external dependencies
"""

import os
import json
from typing import Dict, List, Any
from datetime import datetime

class SimplePhase4Evaluator:
    """Simple evaluator for Phase 4 results"""
    
    def __init__(self):
        self.output_dir = "output/phase4_balanced_refinement_evaluation"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load benchmark topics
        self.benchmark_topics = self.load_benchmark_topics()
        print(f"✅ Loaded {len(self.benchmark_topics)} benchmark topics")
        
        # Load phase4 results
        self.phase4_results = self.load_phase4_results()
        print(f"✅ Loaded {len(self.phase4_results)} Phase 4 results")
    
    def load_benchmark_topics(self) -> List[Dict]:
        """Load benchmark topics"""
        try:
            benchmark_path = "data/benchmark_topics_corrected_fixed.json"
            with open(benchmark_path, "r") as f:
                data = json.load(f)
            
            topics = data.get("topics", [])
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
    
    def simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def evaluate_clusters_against_benchmark(self, refined_clusters: List[Dict]) -> Dict[str, Any]:
        """Evaluate refined clusters against benchmark topics"""
        
        if not refined_clusters or not self.benchmark_topics:
            return {"error": "No clusters or benchmark topics available"}
        
        # Extract cluster titles
        cluster_titles = [cluster.get('draft_title', '') for cluster in refined_clusters]
        benchmark_titles = [topic.get('title', '') for topic in self.benchmark_topics]
        
        # Calculate similarities
        similarities = []
        best_matches = []
        
        for cluster_title in cluster_titles:
            cluster_similarities = []
            for benchmark_title in benchmark_titles:
                similarity = self.simple_text_similarity(cluster_title, benchmark_title)
                cluster_similarities.append(similarity)
            
            max_similarity = max(cluster_similarities) if cluster_similarities else 0
            best_match_idx = cluster_similarities.index(max_similarity)
            
            similarities.append(max_similarity)
            best_matches.append({
                "cluster_title": cluster_title,
                "best_benchmark_match": benchmark_titles[best_match_idx],
                "similarity": max_similarity
            })
        
        # Calculate metrics
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        high_quality_matches = len([s for s in similarities if s > 0.3])  # 30% threshold
        quality_rate = high_quality_matches / len(similarities) if similarities else 0
        
        return {
            "num_refined_clusters": len(refined_clusters),
            "num_benchmark_topics": len(self.benchmark_topics),
            "avg_similarity": avg_similarity,
            "high_quality_matches": high_quality_matches,
            "quality_rate": quality_rate,
            "best_matches": best_matches,
            "similarities": similarities
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
            "avg_cluster_size": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
            "merge_operations": merge_ops,
            "split_operations": split_ops,
            "total_operations": len(operations),
            "benchmark_evaluation": benchmark_evaluation,
            "cluster_titles": [cluster.get('draft_title', '') for cluster in refined_clusters],
            "timestamp": result.get('timestamp', '')
        }
    
    def run_evaluation(self):
        """Run evaluation on all Phase 4 results"""
        
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
                print(f"  📈 Avg Similarity: {benchmark_eval['avg_similarity']:.3f}")
                print(f"  �� Quality Rate: {benchmark_eval['quality_rate']:.1%}")
                print(f"  ⚡ Duration: {evaluation['duration']:.2f}s")
                print(f"  🔄 Operations: {evaluation['merge_operations']} merge, {evaluation['split_operations']} split")
            else:
                print(f"  ❌ Failed: {evaluation.get('error', 'Unknown error')}")
        
        # Save detailed results
        detailed_output = os.path.join(self.output_dir, "detailed_evaluation_results.json")
        with open(detailed_output, "w") as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Generate summary
        self.generate_summary(evaluation_results)
        
        print(f"\n💾 Detailed results saved to: {detailed_output}")
        print("🎉 Evaluation complete!")
    
    def generate_summary(self, evaluation_results: Dict[str, Dict]):
        """Generate summary report"""
        
        successful_models = [eval for eval in evaluation_results.values() if eval['success']]
        
        if not successful_models:
            print("❌ No successful models to evaluate")
            return
        
        # Calculate summary statistics
        avg_similarity = sum(eval['benchmark_evaluation']['avg_similarity'] for eval in successful_models) / len(successful_models)
        avg_quality_rate = sum(eval['benchmark_evaluation']['quality_rate'] for eval in successful_models) / len(successful_models)
        avg_clusters = sum(eval['num_refined_clusters'] for eval in successful_models) / len(successful_models)
        avg_duration = sum(eval['duration'] for eval in successful_models) / len(successful_models)
        
        # Find best performing models
        best_similarity = max(successful_models, key=lambda x: x['benchmark_evaluation']['avg_similarity'])
        best_quality = max(successful_models, key=lambda x: x['benchmark_evaluation']['quality_rate'])
        
        # Rank models
        ranked_models = sorted(successful_models, 
                             key=lambda x: x['benchmark_evaluation']['avg_similarity'], 
                             reverse=True)
        
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
                    "similarity_score": best_similarity['benchmark_evaluation']['avg_similarity']
                },
                "highest_quality_rate": {
                    "model": best_quality['model'],
                    "quality_rate": best_quality['benchmark_evaluation']['quality_rate']
                }
            },
            "model_rankings": [
                {
                    "rank": i + 1,
                    "model": model['model'],
                    "similarity_score": model['benchmark_evaluation']['avg_similarity'],
                    "quality_rate": model['benchmark_evaluation']['quality_rate'],
                    "clusters_generated": model['num_refined_clusters'],
                    "duration": model['duration']
                }
                for i, model in enumerate(ranked_models)
            ]
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
        print(f"🎯 Avg Quality Rate: {avg_quality_rate:.1%}")
        print(f"📈 Avg Clusters Generated: {avg_clusters:.1f}")
        print(f"⚡ Avg Duration: {avg_duration:.2f}s")
        print(f"\n🏆 Best Similarity: {best_similarity['model']} ({best_similarity['benchmark_evaluation']['avg_similarity']:.3f})")
        print(f"🏆 Best Quality: {best_quality['model']} ({best_quality['benchmark_evaluation']['quality_rate']:.1%})")
        
        print(f"\n📊 TOP 5 MODELS BY SIMILARITY:")
        for i, model in enumerate(ranked_models[:5]):
            print(f"  {i+1}. {model['model']}: {model['benchmark_evaluation']['avg_similarity']:.3f}")
        
        print(f"\n💾 Summary saved to: {summary_output}")

def main():
    """Main evaluation function"""
    
    print("🚀 Starting Simple Phase 4 Balanced Refinement Evaluation")
    print("=" * 60)
    
    evaluator = SimplePhase4Evaluator()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
