#!/usr/bin/env python3
"""
Phase Evaluation Engine - Main Runner
Runs all phase evaluations across all models
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.model_config import get_available_models
from phases.phase3_topic_clustering import Phase3Evaluator
from phases.phase4_merge_split import Phase4Evaluator
from phases.phase5_metadata_generation import Phase5Evaluator
from phases.phase6_embedding import Phase6Evaluator
from phases.phase7_user_filtering import Phase7Evaluator
from phases.phase8_new_message_processing import Phase8Evaluator

class PhaseEvaluationRunner:
    """Main runner for all phase evaluations"""
    
    def __init__(self):
        self.phases = [
            ("phase3", "Topic Clustering", Phase3Evaluator),
            ("phase4", "Merge/Split Operations", Phase4Evaluator),
            ("phase5", "Metadata Generation", Phase5Evaluator),
            ("phase6", "Embedding Generation", Phase6Evaluator),
            ("phase7", "User Filtering", Phase7Evaluator),
            ("phase8", "New Message Processing", Phase8Evaluator)
        ]
        
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Check for required data files
        self.check_prerequisites()
    
    def check_prerequisites(self):
        """Check if required data files exist"""
        required_files = [
            "../message_dataset.json",
            "../benchmark_topics.json"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ Missing required data files:")
            for file_path in missing_files:
                print(f"  - {file_path}")
            print("\nPlease run Phase 1 and Phase 2 first to generate these files.")
            sys.exit(1)
    
    def run_single_phase(self, phase_id: str, phase_name: str, evaluator_class) -> Dict[str, Any]:
        """Run evaluation for a single phase"""
        print(f"\n{'='*80}")
        print(f"🚀 RUNNING {phase_name.upper()}")
        print(f"{'='*80}")
        
        try:
            evaluator = evaluator_class()
            evaluator.run_evaluation()
            
            # Load results
            results_file = os.path.join("output", phase_id, "comprehensive_results.json")
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    results = json.load(f)
                return {
                    "phase_id": phase_id,
                    "phase_name": phase_name,
                    "status": "completed",
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "phase_id": phase_id,
                    "phase_name": phase_name,
                    "status": "failed",
                    "error": "Results file not found",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "phase_id": phase_id,
                "phase_name": phase_name,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_all_phases(self):
        """Run all phase evaluations"""
        print("🎯 PHASE EVALUATION ENGINE")
        print("=" * 80)
        print("Comprehensive evaluation of all models across all phases")
        print("=" * 80)
        
        # Check available models
        available_models = get_available_models()
        if not available_models:
            print("❌ No models available. Please check your API keys.")
            return
        
        print(f"\n📋 Available Models:")
        total_models = 0
        for provider, models in available_models.items():
            print(f"  {provider.upper()}: {len(models)} models")
            total_models += len(models)
        print(f"  Total: {total_models} models")
        
        # Run all phases
        phase_results = []
        start_time = time.time()
        
        for phase_id, phase_name, evaluator_class in self.phases:
            print(f"\n⏳ Starting {phase_name}...")
            phase_result = self.run_single_phase(phase_id, phase_name, evaluator_class)
            phase_results.append(phase_result)
            
            if phase_result["status"] == "completed":
                print(f"✅ {phase_name} completed successfully")
            else:
                print(f"❌ {phase_name} failed: {phase_result.get('error', 'Unknown error')}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Generate comprehensive report
        self.generate_comprehensive_report(phase_results, total_duration, available_models)
    
    def generate_comprehensive_report(self, phase_results: List[Dict], total_duration: float, available_models: Dict):
        """Generate comprehensive evaluation report"""
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE EVALUATION REPORT")
        print(f"{'='*80}")
        
        # Summary statistics
        completed_phases = [p for p in phase_results if p["status"] == "completed"]
        failed_phases = [p for p in phase_results if p["status"] == "failed"]
        
        print(f"\n📈 Phase Completion:")
        print(f"  Completed: {len(completed_phases)}/{len(phase_results)} phases")
        print(f"  Failed: {len(failed_phases)} phases")
        print(f"  Success Rate: {(len(completed_phases)/len(phase_results)*100):.1f}%")
        print(f"  Total Duration: {total_duration:.2f} seconds")
        
        # Model performance across phases
        if completed_phases:
            print(f"\n🏆 Model Performance Summary:")
            self.analyze_model_performance(completed_phases, available_models)
        
        # Phase-specific results
        print(f"\n📋 Phase Results:")
        for phase_result in phase_results:
            status_icon = "✅" if phase_result["status"] == "completed" else "❌"
            print(f"  {status_icon} {phase_result['phase_name']}: {phase_result['status']}")
        
        # Save comprehensive results
        comprehensive_results = {
            "evaluation_summary": {
                "total_phases": len(phase_results),
                "completed_phases": len(completed_phases),
                "failed_phases": len(failed_phases),
                "success_rate": len(completed_phases) / len(phase_results),
                "total_duration": total_duration,
                "timestamp": datetime.now().isoformat()
            },
            "available_models": available_models,
            "phase_results": phase_results
        }
        
        output_file = os.path.join(self.output_dir, "comprehensive_evaluation_report.json")
        with open(output_file, "w") as f:
            json.dump(comprehensive_results, f, indent=2)
        
        print(f"\n📄 Comprehensive report saved to: {output_file}")
    
    def analyze_model_performance(self, completed_phases: List[Dict], available_models: Dict):
        """Analyze model performance across all phases"""
        # Collect all model results
        all_model_results = {}
        
        for phase in completed_phases:
            if "results" in phase:
                for model_key, result in phase["results"].items():
                    if model_key not in all_model_results:
                        all_model_results[model_key] = {
                            "phases_completed": 0,
                            "total_success": 0,
                            "total_duration": 0,
                            "total_cost": 0,
                            "total_tokens": 0
                        }
                    
                    all_model_results[model_key]["phases_completed"] += 1
                    if result.get("success", False):
                        all_model_results[model_key]["total_success"] += 1
                    
                    all_model_results[model_key]["total_duration"] += result.get("duration", 0)
                    all_model_results[model_key]["total_cost"] += result.get("cost", {}).get("total_cost", 0)
                    all_model_results[model_key]["total_tokens"] += result.get("usage", {}).get("total_tokens", 0)
        
        # Find best performing models
        if all_model_results:
            # Best by success rate
            best_success = max(all_model_results.items(), 
                             key=lambda x: x[1]["total_success"])
            
            # Best by speed
            best_speed = min(all_model_results.items(),
                           key=lambda x: x[1]["total_duration"])
            
            # Best by cost efficiency
            best_cost = min(all_model_results.items(),
                          key=lambda x: x[1]["total_cost"])
            
            print(f"  🥇 Best Success Rate: {best_success[0]} ({best_success[1]['total_success']} successful phases)")
            print(f"  ⚡ Fastest: {best_speed[0]} ({best_speed[1]['total_duration']:.2f}s)")
            print(f"  💰 Most Cost Efficient: {best_cost[0]} (${best_cost[1]['total_cost']:.6f})")

def main():
    """Main entry point"""
    runner = PhaseEvaluationRunner()
    runner.run_all_phases()

if __name__ == "__main__":
    main()
