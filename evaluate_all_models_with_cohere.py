#!/usr/bin/env python3
"""
Evaluate ALL models including Cohere Command R+ for Phase 3 clustering
This integrates Cohere models with your existing evaluation system
"""

import os
import sys
import json
import time
import subprocess
from typing import Dict, List, Any

def run_existing_evaluation():
    """Run the existing Phase 3 evaluation"""
    print("🔄 Running existing Phase 3 evaluation...")
    
    try:
        # Run the existing evaluation script
        result = subprocess.run([sys.executable, "evaluate_all_models.py"], 
                              capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✅ Existing evaluation completed successfully")
            return True
        else:
            print(f"⚠️ Existing evaluation had issues: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running existing evaluation: {e}")
        return False

def run_cohere_evaluation():
    """Run the Cohere Command R+ evaluation"""
    print("🔄 Running Cohere Command R+ evaluation...")
    
    try:
        # Run the Cohere evaluation script
        result = subprocess.run([sys.executable, "evaluate_cohere_models.py"], 
                              capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✅ Cohere evaluation completed successfully")
            return True
        else:
            print(f"⚠️ Cohere evaluation had issues: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running Cohere evaluation: {e}")
        return False

def merge_results():
    """Merge all evaluation results"""
    print("🔄 Merging all evaluation results...")
    
    try:
        # Load existing comprehensive results
        comprehensive_file = "output/phase3_topic_clustering/comprehensive_results.json"
        
        if os.path.exists(comprehensive_file):
            with open(comprehensive_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {}
        
        # Count models
        total_models = len(all_results)
        successful_models = sum(1 for r in all_results.values() if r.get("success", False))
        
        print(f"📊 Total models evaluated: {total_models}")
        print(f"✅ Successful models: {successful_models}")
        print(f"❌ Failed models: {total_models - successful_models}")
        
        # Show model breakdown by provider
        providers = {}
        for model_name, result in all_results.items():
            provider = result.get("provider", "unknown").title()
            if provider not in providers:
                providers[provider] = {"total": 0, "successful": 0}
            providers[provider]["total"] += 1
            if result.get("success", False):
                providers[provider]["successful"] += 1
        
        print(f"\n📈 Results by Provider:")
        for provider, stats in providers.items():
            success_rate = (stats["successful"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"   {provider}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error merging results: {e}")
        return False

def generate_updated_csv():
    """Generate updated CSV with all models including Cohere"""
    print("🔄 Generating updated comprehensive CSV...")
    
    try:
        # Run the CSV generation script
        result = subprocess.run([sys.executable, "generate_comprehensive_csv.py"], 
                              capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✅ Updated CSV generated successfully")
            return True
        else:
            print(f"⚠️ CSV generation had issues: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error generating CSV: {e}")
        return False

def main():
    """Main evaluation function"""
    
    print("🚀 Complete Phase 3 Evaluation with Cohere Command R+")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("evaluate_all_models.py"):
        print("❌ Please run this script from the phase_evaluation_engine directory")
        return
    
    # Step 1: Run existing evaluation
    print("\n📋 Step 1: Running existing model evaluations...")
    existing_success = run_existing_evaluation()
    
    # Step 2: Run Cohere evaluation
    print("\n📋 Step 2: Running Cohere Command R+ evaluation...")
    cohere_success = run_cohere_evaluation()
    
    # Step 3: Merge results
    print("\n📋 Step 3: Merging all results...")
    merge_success = merge_results()
    
    # Step 4: Generate updated CSV
    print("\n📋 Step 4: Generating updated CSV...")
    csv_success = generate_updated_csv()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    
    steps = [
        ("Existing Models", existing_success),
        ("Cohere Models", cohere_success),
        ("Results Merge", merge_success),
        ("CSV Generation", csv_success)
    ]
    
    for step_name, success in steps:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {step_name}: {status}")
    
    all_success = all(success for _, success in steps)
    
    if all_success:
        print("\n🎉 All evaluations completed successfully!")
        print("\n📁 Results available in:")
        print("   - output/phase3_topic_clustering/comprehensive_results.json")
        print("   - clustering_analysis_expert_complete.csv")
        print("   - Individual model JSON files")
        
        print("\n🔍 Next steps:")
        print("   1. Review the comprehensive results")
        print("   2. Analyze performance comparisons")
        print("   3. Generate final reports")
    else:
        print("\n⚠️ Some evaluations failed. Please check the errors above.")
        print("You may need to:")
        print("   - Install Cohere dependencies: python install_cohere_dependencies.py")
        print("   - Set up Hugging Face token")
        print("   - Check system requirements")

if __name__ == "__main__":
    main()
