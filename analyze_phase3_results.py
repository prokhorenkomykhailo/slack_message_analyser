#!/usr/bin/env python3
"""
Analyze Phase 3 Results and Rank Models
"""

import json
import os
from typing import Dict, List, Any

def load_results():
    """Load comprehensive results"""
    results_file = "output/phase3_topic_clustering/comprehensive_results.json"
    with open(results_file, 'r') as f:
        return json.load(f)

def analyze_models(results: Dict[str, Any]):
    """Analyze and rank models by different criteria"""
    
    # Filter successful models
    successful_models = {}
    for model_name, result in results.items():
        if result.get("success", False):
            successful_models[model_name] = result
    
    print(f"📊 Analyzing {len(successful_models)} successful models out of {len(results)} total")
    print("=" * 80)
    
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

if __name__ == "__main__":
    results = load_results()
    analyze_models(results)
