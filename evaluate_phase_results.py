#!/usr/bin/env python3
"""
Comprehensive Phase Evaluation Results Analyzer
Analyzes results from any phase and provides detailed rankings
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

def load_phase_results(phase_name: str) -> Dict[str, Any]:
    """Load results from a specific phase"""
    results_file = f"output/{phase_name}/comprehensive_results.json"
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        print(f"Available phases:")
        for item in os.listdir("output"):
            if os.path.isdir(os.path.join("output", item)):
                print(f"  - {item}")
        return {}
    
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return {}

def analyze_phase_results(results: Dict[str, Any], phase_name: str):
    """Analyze and rank models for a specific phase"""
    
    # Filter successful models
    successful_models = {}
    for model_name, result in results.items():
        if result.get("success", False):
            successful_models[model_name] = result
    
    print(f"📊 Analyzing {len(successful_models)} successful models out of {len(results)} total")
    print(f"🎯 Phase: {phase_name.upper()}")
    print("=" * 80)
    
    if not successful_models:
        print("❌ No successful models found!")
        return
    
    # Phase-specific analysis
    if "topic_clustering" in phase_name:
        analyze_clustering_results(successful_models)
    elif "summarization" in phase_name:
        analyze_summarization_results(successful_models)
    elif "extraction" in phase_name:
        analyze_extraction_results(successful_models)
    else:
        analyze_general_results(successful_models)
    
    # General analysis for all phases
    analyze_general_metrics(successful_models)

def analyze_clustering_results(successful_models: Dict[str, Any]):
    """Analyze topic clustering specific metrics"""
    
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

def analyze_summarization_results(successful_models: Dict[str, Any]):
    """Analyze summarization specific metrics"""
    
    # Rank by Summary Quality (if available)
    if "summary_quality" in successful_models[list(successful_models.keys())[0]]["metrics"]:
        print("\n🏆 **BEST SUMMARY QUALITY**")
        quality_ranked = sorted(successful_models.items(), 
                               key=lambda x: x[1]["metrics"].get("summary_quality", 0), reverse=True)
        for i, (model, result) in enumerate(quality_ranked[:5]):
            quality = result["metrics"].get("summary_quality", 0)
            print(f"{i+1}. {model}: {quality:.3f}")
    
    # Rank by Summary Length (if available)
    if "summary_length" in successful_models[list(successful_models.keys())[0]]["metrics"]:
        print("\n🏆 **OPTIMAL SUMMARY LENGTH**")
        length_ranked = sorted(successful_models.items(), 
                              key=lambda x: abs(x[1]["metrics"].get("summary_length", 0) - 200))  # Target 200 words
        for i, (model, result) in enumerate(length_ranked[:5]):
            length = result["metrics"].get("summary_length", 0)
            print(f"{i+1}. {model}: {length} words")

def analyze_extraction_results(successful_models: Dict[str, Any]):
    """Analyze information extraction specific metrics"""
    
    # Rank by Extraction Accuracy (if available)
    if "extraction_accuracy" in successful_models[list(successful_models.keys())[0]]["metrics"]:
        print("\n🏆 **BEST EXTRACTION ACCURACY**")
        accuracy_ranked = sorted(successful_models.items(), 
                                key=lambda x: x[1]["metrics"].get("extraction_accuracy", 0), reverse=True)
        for i, (model, result) in enumerate(accuracy_ranked[:5]):
            accuracy = result["metrics"].get("extraction_accuracy", 0) * 100
            print(f"{i+1}. {model}: {accuracy:.1f}%")
    
    # Rank by Entities Extracted (if available)
    if "entities_extracted" in successful_models[list(successful_models.keys())[0]]["metrics"]:
        print("\n🏆 **MOST ENTITIES EXTRACTED**")
        entities_ranked = sorted(successful_models.items(), 
                                key=lambda x: x[1]["metrics"].get("entities_extracted", 0), reverse=True)
        for i, (model, result) in enumerate(entities_ranked[:5]):
            entities = result["metrics"].get("entities_extracted", 0)
            print(f"{i+1}. {model}: {entities} entities")

def analyze_general_results(successful_models: Dict[str, Any]):
    """Analyze general metrics for any phase"""
    print("\n🏆 **GENERAL PERFORMANCE METRICS**")
    
    # Show available metrics
    sample_model = list(successful_models.values())[0]
    available_metrics = list(sample_model["metrics"].keys())
    print(f"Available metrics: {', '.join(available_metrics)}")

def analyze_general_metrics(successful_models: Dict[str, Any]):
    """Analyze general performance metrics for all phases"""
    
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
    
    # Rank by Token Efficiency
    print("\n🏆 **MOST TOKEN EFFICIENT** (Lowest tokens per evaluation)")
    token_ranked = sorted(successful_models.items(), 
                         key=lambda x: x[1]["usage"]["total_tokens"])
    for i, (model, result) in enumerate(token_ranked[:5]):
        tokens = result["usage"]["total_tokens"]
        print(f"{i+1}. {model}: {tokens:,} tokens")
    
    # Rank by Overall Score
    print("\n🏆 **OVERALL BEST** (Combined score: performance - normalized cost - normalized time)")
    overall_scores = {}
    for model, result in successful_models.items():
        # Get primary performance metric (varies by phase)
        metrics = result["metrics"]
        if "coverage" in metrics:
            performance = metrics["coverage"] + metrics.get("thread_coherence", 0)
        elif "summary_quality" in metrics:
            performance = metrics["summary_quality"]
        elif "extraction_accuracy" in metrics:
            performance = metrics["extraction_accuracy"]
        else:
            # Fallback to first available metric
            performance = list(metrics.values())[0] if metrics else 0
        
        cost = result["cost"]["total_cost"]
        duration = result["duration"]
        
        # Normalize cost and duration (lower is better)
        max_cost = max(r["cost"]["total_cost"] for r in successful_models.values())
        max_duration = max(r["duration"] for r in successful_models.values())
        
        normalized_cost = cost / max_cost if max_cost > 0 else 0
        normalized_duration = duration / max_duration if max_duration > 0 else 0
        
        # Overall score (higher is better)
        overall_score = performance - normalized_cost - normalized_duration
        overall_scores[model] = overall_score
    
    overall_ranked = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (model, score) in enumerate(overall_ranked[:10]):
        result = successful_models[model]
        cost = result["cost"]["total_cost"]
        duration = result["duration"]
        tokens = result["usage"]["total_tokens"]
        print(f"{i+1}. {model}: Score {score:.3f} (Cost: ${cost:.6f}, Time: {duration:.2f}s, Tokens: {tokens:,})")
    
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
        print(f"   Cost: ${result['cost']['total_cost']:.6f}")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   Tokens: {result['usage']['total_tokens']:,}")
        print(f"   Metrics: {result['metrics']}")

def list_available_phases():
    """List all available phases with results"""
    print("📁 Available phases with results:")
    print("=" * 40)
    
    if not os.path.exists("output"):
        print("❌ No output directory found")
        return []
    
    phases = []
    for item in os.listdir("output"):
        phase_dir = os.path.join("output", item)
        if os.path.isdir(phase_dir):
            results_file = os.path.join(phase_dir, "comprehensive_results.json")
            if os.path.exists(results_file):
                phases.append(item)
                print(f"✅ {item}")
            else:
                print(f"⚠️  {item} (no results file)")
    
    return phases

def main():
    """Main function"""
    print("🎯 Phase Evaluation Results Analyzer")
    print("=" * 50)
    
    # List available phases
    available_phases = list_available_phases()
    
    if not available_phases:
        print("\n❌ No phases with results found!")
        return
    
    # If phase specified as argument, use it
    if len(sys.argv) > 1:
        phase_name = sys.argv[1]
        if phase_name not in available_phases:
            print(f"❌ Phase '{phase_name}' not found in available phases: {available_phases}")
            return
    else:
        # Use the first available phase (usually phase3)
        phase_name = available_phases[0]
        print(f"\n📊 Analyzing phase: {phase_name}")
    
    # Load and analyze results
    results = load_phase_results(phase_name)
    if results:
        analyze_phase_results(results, phase_name)
    
    print(f"\n✅ Analysis complete for {phase_name}")

if __name__ == "__main__":
    main()
