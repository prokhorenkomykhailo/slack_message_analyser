#!/usr/bin/env python3
"""
Phase 3: Topic Clustering Results Analyzer
Specifically analyzes clustering results from Phase 3
"""

import json
import os
from typing import Dict, List, Any

def load_phase3_results() -> Dict[str, Any]:
    """Load Phase 3 clustering results"""
    results_file = "output/phase3_topic_clustering/comprehensive_results.json"
    
    if not os.path.exists(results_file):
        print(f"❌ Phase 3 results not found: {results_file}")
        print("Please run Phase 3 evaluation first: python run_phase3.py")
        return {}
    
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading Phase 3 results: {e}")
        return {}

def load_existing_phase3_clusters() -> List[Dict]:
    """Load existing phase3_clusters.json if it exists"""
    clusters_file = "phases/phase3_clusters.json"
    
    if os.path.exists(clusters_file):
        try:
            with open(clusters_file, 'r') as f:
                clusters = json.load(f)
                print(f"📁 Found existing clusters file: {clusters_file}")
                print(f"   Contains {len(clusters)} clusters")
                return clusters
        except Exception as e:
            print(f"⚠️  Error loading existing clusters: {e}")
    
    return []

def analyze_clustering_models(results: Dict[str, Any]):
    """Analyze and rank clustering models"""
    
    # Filter successful models
    successful_models = {}
    for model_name, result in results.items():
        if result.get("success", False):
            successful_models[model_name] = result
    
    print(f"📊 Analyzing {len(successful_models)} successful clustering models out of {len(results)} total")
    print("🎯 Phase 3: Topic Clustering Evaluation")
    print("=" * 80)
    
    if not successful_models:
        print("❌ No successful clustering models found!")
        return
    
    # Rank by Coverage
    print("\n🏆 **BEST COVERAGE** (Percentage of messages clustered)")
    coverage_ranked = sorted(successful_models.items(), 
                            key=lambda x: x[1]["metrics"]["coverage"], reverse=True)
    for i, (model, result) in enumerate(coverage_ranked[:5]):
        coverage = result["metrics"]["coverage"] * 100
        print(f"{i+1}. {model}: {coverage:.1f}%")
    
    # Rank by Thread Coherence
    print("\n🏆 **BEST THREAD COHERENCE** (How well threads are preserved)")
    coherence_ranked = sorted(successful_models.items(), 
                             key=lambda x: x[1]["metrics"]["thread_coherence"], reverse=True)
    for i, (model, result) in enumerate(coherence_ranked[:5]):
        coherence = result["metrics"]["thread_coherence"] * 100
        print(f"{i+1}. {model}: {coherence:.1f}%")
    
    # Rank by Cluster Quality
    print("\n🏆 **BEST CLUSTER QUALITY** (Optimal number of clusters)")
    cluster_quality = []
    for model, result in successful_models.items():
        num_clusters = result["metrics"]["num_clusters"]
        # Optimal range: 10-15 clusters for 300 messages
        quality_score = 1.0 - abs(num_clusters - 12.5) / 12.5  # Distance from optimal
        cluster_quality.append((model, quality_score, num_clusters))
    
    cluster_quality.sort(key=lambda x: x[1], reverse=True)
    for i, (model, score, num_clusters) in enumerate(cluster_quality[:5]):
        print(f"{i+1}. {model}: {num_clusters} clusters (quality: {score:.3f})")
    
    # Rank by Cost Efficiency
    print("\n🏆 **MOST COST EFFICIENT** (Lowest cost per evaluation)")
    cost_ranked = sorted(successful_models.items(), 
                        key=lambda x: x[1]["cost"]["total_cost"])
    for i, (model, result) in enumerate(cost_ranked[:5]):
        cost = result["cost"]["total_cost"]
        print(f"{i+1}. {model}: ${cost:.6f}")
    
    # Rank by Speed
    print("\n🏆 **FASTEST** (Lowest response time)")
    speed_ranked = sorted(successful_models.items(), 
                         key=lambda x: x[1]["duration"])
    for i, (model, result) in enumerate(speed_ranked[:5]):
        duration = result["duration"]
        print(f"{i+1}. {model}: {duration:.2f}s")
    
    # Rank by Overall Score (combined metrics)
    print("\n🏆 **OVERALL BEST** (Combined score: coverage + coherence - normalized cost - normalized time)")
    overall_scores = {}
    for model, result in successful_models.items():
        coverage = result["metrics"]["coverage"]
        coherence = result["metrics"]["thread_coherence"]
        cost = result["cost"]["total_cost"]
        duration = result["duration"]
        
        # Normalize cost and duration (lower is better)
        max_cost = max(r["cost"]["total_cost"] for r in successful_models.values())
        max_duration = max(r["duration"] for r in successful_models.values())
        
        normalized_cost = cost / max_cost if max_cost > 0 else 0
        normalized_duration = duration / max_duration if max_duration > 0 else 0
        
        # Overall score (higher is better)
        overall_score = (coverage + coherence) - normalized_cost - normalized_duration
        overall_scores[model] = overall_score
    
    overall_ranked = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (model, score) in enumerate(overall_ranked[:10]):
        result = successful_models[model]
        coverage = result["metrics"]["coverage"] * 100
        coherence = result["metrics"]["thread_coherence"] * 100
        cost = result["cost"]["total_cost"]
        duration = result["duration"]
        print(f"{i+1}. {model}: Score {score:.3f} (Coverage: {coverage:.1f}%, Coherence: {coherence:.1f}%, Cost: ${cost:.6f}, Time: {duration:.2f}s)")
    
    # Best by provider
    print("\n🏆 **BEST BY PROVIDER**")
    providers = {}
    for model, result in successful_models.items():
        provider = result["provider"]
        if provider not in providers:
            providers[provider] = []
        providers[provider].append((model, overall_scores[model]))
    
    for provider, models in providers.items():
        best_model = max(models, key=lambda x: x[1])
        print(f"{provider.upper()}: {best_model[0]} (Score: {best_model[1]:.3f})")
    
    # Detailed analysis of top 3
    print("\n🔍 **DETAILED ANALYSIS OF TOP 3 MODELS**")
    for i, (model, score) in enumerate(overall_ranked[:3]):
        result = successful_models[model]
        print(f"\n{i+1}. {model}")
        print(f"   Provider: {result['provider']}")
        print(f"   Coverage: {result['metrics']['coverage']*100:.1f}%")
        print(f"   Thread Coherence: {result['metrics']['thread_coherence']*100:.1f}%")
        print(f"   Clusters: {result['metrics']['num_clusters']}")
        print(f"   Cost: ${result['cost']['total_cost']:.6f}")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   Tokens: {result['usage']['total_tokens']:,}")
        
        # Show sample cluster titles
        if result['clusters']:
            titles = [cluster.get('draft_title', 'No title') for cluster in result['clusters'][:3]]
            print(f"   Sample Clusters: {', '.join(titles)}")
    
    # Save best clusters to file (in correct format - direct array)
    if overall_ranked:
        best_model_name = overall_ranked[0][0]
        best_result = successful_models[best_model_name]
        
        # Convert clusters to the correct format
        clusters_array = []
        for cluster in best_result["clusters"]:
            # Ensure all required fields are present
            formatted_cluster = {
                "cluster_id": cluster.get("cluster_id", f"cluster_{len(clusters_array)+1:03d}"),
                "message_ids": cluster.get("message_ids", []),
                "draft_title": cluster.get("draft_title", "Untitled Cluster"),
                "participants": cluster.get("participants", []),
                "channel": cluster.get("channel", ""),
                "thread_id": cluster.get("thread_id", "")
            }
            clusters_array.append(formatted_cluster)
        
        output_file = "phases/phase3_clusters.json"
        with open(output_file, 'w') as f:
            json.dump(clusters_array, f, indent=2)
        
        print(f"\n💾 Best clustering results saved to: {output_file}")
        print(f"   Format: Direct array of {len(clusters_array)} clusters")
        print(f"   Best model: {best_model_name} (Score: {overall_ranked[0][1]:.3f})")

def compare_with_existing_clusters():
    """Compare new results with existing clusters"""
    existing_clusters = load_existing_phase3_clusters()
    
    if existing_clusters:
        print(f"\n📊 **EXISTING CLUSTERS ANALYSIS**")
        print(f"   Total clusters: {len(existing_clusters)}")
        
        # Analyze existing clusters
        total_messages = 0
        unique_participants = set()
        
        for cluster in existing_clusters:
            message_ids = cluster.get("message_ids", [])
            total_messages += len(message_ids)
            participants = cluster.get("participants", [])
            unique_participants.update(participants)
        
        print(f"   Total messages: {total_messages}")
        print(f"   Unique participants: {len(unique_participants)}")
        print(f"   Average cluster size: {total_messages/len(existing_clusters):.1f}")
        
        # Show cluster titles
        titles = [cluster.get("draft_title", "Untitled") for cluster in existing_clusters]
        print(f"   Cluster titles: {', '.join(titles)}")

def main():
    """Main function"""
    print("🎯 Phase 3: Topic Clustering Results Analyzer")
    print("=" * 60)
    
    # Check for existing clusters first
    compare_with_existing_clusters()
    
    # Load Phase 3 results
    results = load_phase3_results()
    if results:
        analyze_clustering_models(results)
        print("\n✅ Phase 3 clustering analysis complete!")
    else:
        print("\n⚠️  No evaluation results found. Using existing clusters only.")

if __name__ == "__main__":
    main()
