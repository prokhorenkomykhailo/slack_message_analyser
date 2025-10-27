#!/usr/bin/env python3
"""
Phase 4 Comprehensive Evaluation Engine
Evaluates all Phase 4 balanced refinement results and generates comprehensive analysis
Similar to phase3_comprehensive_evaluation.py but for Step 2 (Phase 4)
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

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Phase4ComprehensiveEvaluator:
    """Comprehensive evaluator for Phase 4 balanced refinement results"""
    
    def __init__(self):
        self.phase_name = "phase4_comprehensive_evaluation"
        self.output_dir = os.path.join("output", self.phase_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load benchmark topics
        self.benchmark_topics = self.load_benchmark_topics()
        print(f"✅ Loaded {len(self.benchmark_topics)} benchmark topics")
        
        # Initialize embedding model for similarity
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
        
        # Load Phase 4 results
        self.phase4_results = self.load_phase4_results()
        print(f"✅ Loaded {len(self.phase4_results)} Phase 4 results")
    
    def load_benchmark_topics(self) -> List[Dict]:
        """Load benchmark topics from Phase 4 clusters refined file"""
        try:
            benchmark_path = "phases/phase4_clusters_refined.json"
            with open(benchmark_path, "r") as f:
                topics = json.load(f)
            
            print(f"📊 Benchmark topics structure: {len(topics)} topics")
            return topics
            
        except Exception as e:
            print(f"❌ Error loading benchmark: {e}")
            return []
    
    def load_phase4_results(self) -> Dict[str, Dict]:
        """Load all Phase 4 results from balanced refinement"""
        results = {}
        phase4_dir = "output/phase4_balanced_refinement"
        
        if not os.path.exists(phase4_dir):
            print(f"❌ Phase 4 directory not found: {phase4_dir}")
            return results
        
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
            if not text1 or not text2:
                return 0.0
            embeddings = self.embedding_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"⚠️ Error calculating similarity: {e}")
            return 0.0
    
    def evaluate_refined_clusters_against_benchmark(self, refined_clusters: List[Dict]) -> Dict[str, Any]:
        """Evaluate refined clusters against benchmark topics"""
        
        if not refined_clusters or not self.benchmark_topics:
            return {"error": "No clusters or benchmark topics available"}
        
        # Extract cluster titles
        cluster_titles = [cluster.get('draft_title', '') for cluster in refined_clusters]
        
        # Extract benchmark titles (Phase 4 benchmark uses draft_title)
        benchmark_titles = [topic.get('draft_title', '') for topic in self.benchmark_topics]
        
        # Calculate similarity matrix
        title_similarities = []
        
        for cluster_title in cluster_titles:
            cluster_title_sims = []
            
            for benchmark_title in benchmark_titles:
                # Title similarity
                title_sim = self.calculate_similarity_score(cluster_title, benchmark_title)
                cluster_title_sims.append(title_sim)
            
            title_similarities.append(cluster_title_sims)
        
        # Find best matches
        best_matches = []
        for i, cluster_title in enumerate(cluster_titles):
            max_title_sim = max(title_similarities[i]) if title_similarities[i] else 0
            best_match_idx = title_similarities[i].index(max_title_sim)
            
            best_matches.append({
                "cluster_title": cluster_title,
                "best_benchmark_match": benchmark_titles[best_match_idx],
                "title_similarity": max_title_sim,
                "combined_similarity": max_title_sim  # Only title similarity for Phase 4
            })
        
        # Calculate overall metrics
        avg_title_similarity = np.mean([match["title_similarity"] for match in best_matches])
        avg_combined_similarity = avg_title_similarity  # Same as title similarity for Phase 4
        
        # Count high-quality matches (similarity > 0.7)
        high_quality_matches = len([m for m in best_matches if m["combined_similarity"] > 0.7])
        quality_rate = high_quality_matches / len(best_matches) if best_matches else 0
        
        return {
            "num_refined_clusters": len(refined_clusters),
            "num_benchmark_topics": len(self.benchmark_topics),
            "avg_title_similarity": avg_title_similarity,
            "avg_summary_similarity": 0.0,  # Not applicable for Phase 4
            "avg_combined_similarity": avg_combined_similarity,
            "high_quality_matches": high_quality_matches,
            "quality_rate": quality_rate,
            "best_matches": best_matches,
            "title_similarities": title_similarities,
            "summary_similarities": []  # Not applicable for Phase 4
        }
    
    def analyze_refinement_operations(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the refinement operations performed by the model"""
        
        operations = result.get('operations', [])
        refined_clusters = result.get('refined_clusters', [])
        step1_clusters = result.get('step1_clusters', [])
        
        # Count operation types
        merge_ops = len([op for op in operations if op.get('operation') == 'merge'])
        split_ops = len([op for op in operations if op.get('operation') == 'split'])
        refine_ops = len([op for op in operations if op.get('operation') == 'refine'])
        
        # Calculate cluster size statistics
        cluster_sizes = [len(cluster.get('message_ids', [])) for cluster in refined_clusters]
        step1_sizes = [len(cluster.get('message_ids', [])) for cluster in step1_clusters]
        
        # Calculate refinement effectiveness
        size_reduction = np.mean(step1_sizes) - np.mean(cluster_sizes) if step1_sizes and cluster_sizes else 0
        size_std_reduction = np.std(step1_sizes) - np.std(cluster_sizes) if step1_sizes and cluster_sizes else 0
        
        return {
            "total_operations": len(operations),
            "merge_operations": merge_ops,
            "split_operations": split_ops,
            "refine_operations": refine_ops,
            "avg_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
            "avg_step1_size": np.mean(step1_sizes) if step1_sizes else 0,
            "size_reduction": size_reduction,
            "size_std_reduction": size_std_reduction,
            "cluster_count_reduction": len(step1_clusters) - len(refined_clusters)
        }
    
    def evaluate_model_result(self, model_key: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single model result comprehensively"""
        
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
        benchmark_evaluation = self.evaluate_refined_clusters_against_benchmark(refined_clusters)
        
        # Analyze refinement operations
        operations_analysis = self.analyze_refinement_operations(result)
        
        # Extract basic metrics
        cluster_titles = [cluster.get('draft_title', '') for cluster in refined_clusters]
        cluster_participants = [cluster.get('participants', []) for cluster in refined_clusters]
        total_participants = len(set([p for participants in cluster_participants for p in participants]))
        
        return {
            "model": model_key,
            "success": True,
            "provider": result.get('provider', 'unknown'),
            "model_name": result.get('model', 'unknown'),
            "duration": result.get('duration', 0),
            "timestamp": result.get('timestamp', ''),
            
            # Cluster metrics
            "num_refined_clusters": len(refined_clusters),
            "num_step1_clusters": len(result.get('step1_clusters', [])),
            "cluster_titles": cluster_titles,
            "total_participants": total_participants,
            
            # Refinement operations
            "total_operations": operations_analysis['total_operations'],
            "merge_operations": operations_analysis['merge_operations'],
            "split_operations": operations_analysis['split_operations'],
            "refine_operations": operations_analysis['refine_operations'],
            "cluster_count_reduction": operations_analysis['cluster_count_reduction'],
            
            # Size analysis
            "avg_cluster_size": operations_analysis['avg_cluster_size'],
            "avg_step1_size": operations_analysis['avg_step1_size'],
            "size_reduction": operations_analysis['size_reduction'],
            "size_std_reduction": operations_analysis['size_std_reduction'],
            
            # Benchmark evaluation
            "benchmark_evaluation": benchmark_evaluation,
            "avg_title_similarity": benchmark_evaluation.get('avg_title_similarity', 0),
            "avg_summary_similarity": benchmark_evaluation.get('avg_summary_similarity', 0),
            "avg_combined_similarity": benchmark_evaluation.get('avg_combined_similarity', 0),
            "high_quality_matches": benchmark_evaluation.get('high_quality_matches', 0),
            "quality_rate": benchmark_evaluation.get('quality_rate', 0)
        }
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation on all Phase 4 results"""
        
        print("🎯 PHASE 4 COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        evaluation_results = []
        
        for model_key, result in self.phase4_results.items():
            print(f"\n📊 Evaluating {model_key}...")
            
            evaluation = self.evaluate_model_result(model_key, result)
            evaluation_results.append(evaluation)
            
            if evaluation['success']:
                print(f"  ✅ Success: {evaluation['num_refined_clusters']} clusters")
                print(f"  📈 Avg Similarity: {evaluation['avg_combined_similarity']:.3f}")
                print(f"  🎯 Quality Rate: {evaluation['quality_rate']:.1%}")
                print(f"  ⚡ Duration: {evaluation['duration']:.2f}s")
                print(f"  🔧 Operations: {evaluation['total_operations']} (M:{evaluation['merge_operations']}, S:{evaluation['split_operations']})")
            else:
                print(f"  ❌ Failed: {evaluation.get('error', 'Unknown error')}")
        
        # Save detailed results
        detailed_output = os.path.join(self.output_dir, "detailed_evaluation_results.json")
        with open(detailed_output, "w") as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Generate CSV analysis
        self.generate_csv_analysis(evaluation_results)
        
        # Generate summary report
        self.generate_summary_report(evaluation_results)
        
        print(f"\n💾 Detailed results saved to: {detailed_output}")
        print("🎉 Comprehensive evaluation complete!")
    
    def generate_csv_analysis(self, evaluation_results: List[Dict]):
        """Generate comprehensive CSV analysis similar to Phase 3 format"""
        
        # Flatten the results for CSV
        csv_data = []
        
        for result in evaluation_results:
            if not result['success']:
                continue
            
            # Basic model info
            base_row = {
                'MODEL': result['model'],
                'PROVIDER': result['provider'],
                'SUCCESS': result['success'],
                'DURATION_SECONDS': result['duration'],
                'TIMESTAMP': result['timestamp'],
                
                # Cluster metrics
                'NUM_REFINED_CLUSTERS': result['num_refined_clusters'],
                'NUM_STEP1_CLUSTERS': result['num_step1_clusters'],
                'CLUSTER_COUNT_REDUCTION': result['cluster_count_reduction'],
                'AVG_CLUSTER_SIZE': result['avg_cluster_size'],
                'AVG_STEP1_SIZE': result['avg_step1_size'],
                'SIZE_REDUCTION': result['size_reduction'],
                'SIZE_STD_REDUCTION': result['size_std_reduction'],
                'TOTAL_PARTICIPANTS': result['total_participants'],
                
                # Operations
                'TOTAL_OPERATIONS': result['total_operations'],
                'MERGE_OPERATIONS': result['merge_operations'],
                'SPLIT_OPERATIONS': result['split_operations'],
                'REFINE_OPERATIONS': result['refine_operations'],
                
                # Benchmark evaluation
                'AVG_TITLE_SIMILARITY': result['avg_title_similarity'],
                'AVG_COMBINED_SIMILARITY': result['avg_combined_similarity'],
                'HIGH_QUALITY_MATCHES': result['high_quality_matches'],
                'QUALITY_RATE': result['quality_rate']
            }
            
            # Add cluster titles as comma-separated string
            base_row['CLUSTER_TITLES'] = '; '.join(result['cluster_titles'])
            
            csv_data.append(base_row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        
        # Save CSV
        csv_output = os.path.join(self.output_dir, "phase4_comprehensive_analysis.csv")
        df.to_csv(csv_output, index=False)
        
        print(f"📊 CSV analysis saved to: {csv_output}")
        
        # Generate model rankings
        self.generate_model_rankings(df)
    
    def generate_model_rankings(self, df: pd.DataFrame):
        """Generate model rankings and performance analysis"""
        
        if df.empty:
            print("❌ No successful models to rank")
            return
        
        # Calculate composite scores
        df['SIMILARITY_SCORE'] = df['AVG_COMBINED_SIMILARITY'] * 100
        df['OPERATION_EFFICIENCY'] = (df['TOTAL_OPERATIONS'] / df['NUM_REFINED_CLUSTERS']).fillna(0)
        df['SIZE_OPTIMIZATION'] = (100 - abs(df['SIZE_REDUCTION'])).clip(0, 100)
        
        # Combined score (weighted)
        df['COMBINED_SCORE'] = (
            df['SIMILARITY_SCORE'] * 0.4 +           # 40% - similarity to benchmark
            df['QUALITY_RATE'] * 100 * 0.3 +         # 30% - quality rate
            df['SIZE_OPTIMIZATION'] * 0.2 +          # 20% - size optimization
            (100 - df['OPERATION_EFFICIENCY']) * 0.1 # 10% - operation efficiency
        )
        
        # Sort by combined score
        df_sorted = df.sort_values('COMBINED_SCORE', ascending=False)
        
        # Save rankings
        rankings_output = os.path.join(self.output_dir, "model_rankings.csv")
        df_sorted.to_csv(rankings_output, index=False)
        
        print(f"🏆 Model rankings saved to: {rankings_output}")
        
        # Print top 10 models
        print(f"\n🏆 TOP 10 MODELS BY COMBINED SCORE:")
        print("=" * 80)
        for i, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['MODEL']}")
            print(f"    Combined Score: {row['COMBINED_SCORE']:.2f}")
            print(f"    Similarity: {row['SIMILARITY_SCORE']:.2f}")
            print(f"    Quality Rate: {row['QUALITY_RATE']:.1%}")
            print(f"    Operations: {row['TOTAL_OPERATIONS']}")
            print(f"    Clusters: {row['NUM_REFINED_CLUSTERS']}")
            print()
    
    def generate_summary_report(self, evaluation_results: List[Dict]):
        """Generate a comprehensive summary report"""
        
        successful_models = [eval for eval in evaluation_results if eval['success']]
        
        if not successful_models:
            print("❌ No successful models to evaluate")
            return
        
        # Calculate summary statistics
        summary_stats = {
            "total_models_evaluated": len(evaluation_results),
            "successful_models": len(successful_models),
            "success_rate": len(successful_models) / len(evaluation_results),
            "avg_similarity_score": np.mean([eval['avg_combined_similarity'] for eval in successful_models]),
            "avg_quality_rate": np.mean([eval['quality_rate'] for eval in successful_models]),
            "avg_clusters_generated": np.mean([eval['num_refined_clusters'] for eval in successful_models]),
            "avg_duration_seconds": np.mean([eval['duration'] for eval in successful_models]),
            "avg_operations": np.mean([eval['total_operations'] for eval in successful_models])
        }
        
        # Find best performing models
        best_similarity = max(successful_models, key=lambda x: x['avg_combined_similarity'])
        best_quality = max(successful_models, key=lambda x: x['quality_rate'])
        best_speed = min(successful_models, key=lambda x: x['duration'])
        most_operations = max(successful_models, key=lambda x: x['total_operations'])
        
        summary = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "summary_statistics": summary_stats,
            "best_performers": {
                "highest_similarity": {
                    "model": best_similarity['model'],
                    "similarity_score": best_similarity['avg_combined_similarity']
                },
                "highest_quality_rate": {
                    "model": best_quality['model'],
                    "quality_rate": best_quality['quality_rate']
                },
                "fastest_execution": {
                    "model": best_speed['model'],
                    "duration": best_speed['duration']
                },
                "most_operations": {
                    "model": most_operations['model'],
                    "operations": most_operations['total_operations']
                }
            }
        }
        
        # Save summary
        summary_output = os.path.join(self.output_dir, "evaluation_summary.json")
        with open(summary_output, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n📋 EVALUATION SUMMARY")
        print("=" * 40)
        print(f"✅ Successful Models: {summary_stats['successful_models']}/{summary_stats['total_models_evaluated']}")
        print(f"📊 Avg Similarity Score: {summary_stats['avg_similarity_score']:.3f}")
        print(f"🎯 Avg Quality Rate: {summary_stats['avg_quality_rate']:.1%}")
        print(f"📈 Avg Clusters Generated: {summary_stats['avg_clusters_generated']:.1f}")
        print(f"⚡ Avg Duration: {summary_stats['avg_duration_seconds']:.2f}s")
        print(f"🔧 Avg Operations: {summary_stats['avg_operations']:.1f}")
        
        print(f"\n🏆 Best Performers:")
        print(f"  Similarity: {best_similarity['model']} ({best_similarity['avg_combined_similarity']:.3f})")
        print(f"  Quality: {best_quality['model']} ({best_quality['quality_rate']:.1%})")
        print(f"  Speed: {best_speed['model']} ({best_speed['duration']:.2f}s)")
        print(f"  Operations: {most_operations['model']} ({most_operations['total_operations']})")
        
        print(f"\n💾 Summary saved to: {summary_output}")

def main():
    """Main evaluation function"""
    
    print("🚀 Starting Phase 4 Comprehensive Evaluation")
    print("=" * 60)
    
    evaluator = Phase4ComprehensiveEvaluator()
    evaluator.run_comprehensive_evaluation()

if __name__ == "__main__":
    main()
