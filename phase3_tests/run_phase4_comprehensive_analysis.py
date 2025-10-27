#!/usr/bin/env python3
"""
Run Phase 4 Comprehensive Analysis
Complete evaluation pipeline for Phase 4 results similar to Phase 3
"""

import os
import sys
import subprocess
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_phase4_comprehensive_analysis():
    """Run complete Phase 4 comprehensive analysis pipeline"""
    
    print("🚀 PHASE 4 COMPREHENSIVE ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Run comprehensive evaluation
    print("📊 Step 1: Running Phase 4 Comprehensive Evaluation...")
    print("-" * 50)
    
    try:
        from phase4_comprehensive_evaluation import Phase4ComprehensiveEvaluator
        evaluator = Phase4ComprehensiveEvaluator()
        evaluator.run_comprehensive_evaluation()
        print("✅ Step 1 completed successfully")
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        return False
    
    print()
    
    # Step 2: Generate Excel analysis
    print("📈 Step 2: Generating Excel Analysis...")
    print("-" * 50)
    
    try:
        from create_phase4_excel_analysis import Phase4ExcelAnalyzer
        analyzer = Phase4ExcelAnalyzer()
        analyzer.run_full_analysis()
        print("✅ Step 2 completed successfully")
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        return False
    
    print()
    
    # Step 3: Generate summary
    print("📋 Step 3: Generating Final Summary...")
    print("-" * 50)
    
    try:
        generate_final_summary()
        print("✅ Step 3 completed successfully")
    except Exception as e:
        print(f"❌ Step 3 failed: {e}")
        return False
    
    print()
    print("🎉 PHASE 4 COMPREHENSIVE ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("📁 Output Files:")
    print("  - output/phase4_comprehensive_evaluation/detailed_evaluation_results.json")
    print("  - output/phase4_comprehensive_evaluation/phase4_comprehensive_analysis.csv")
    print("  - output/phase4_comprehensive_evaluation/model_rankings.csv")
    print("  - output/phase4_excel_analysis/phase4_comprehensive_analysis.xlsx")
    print("  - output/phase4_excel_analysis/phase4_client_analysis.xlsx")
    print("  - output/phase4_excel_analysis/analysis_summary.txt")
    
    return True

def generate_final_summary():
    """Generate final summary of the analysis"""
    
    import json
    import pandas as pd
    
    # Load evaluation results
    eval_file = "output/phase4_comprehensive_evaluation/detailed_evaluation_results.json"
    if not os.path.exists(eval_file):
        print("⚠️ Evaluation results not found")
        return
    
    with open(eval_file, 'r') as f:
        results = json.load(f)
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        print("❌ No successful results found")
        return
    
    # Create final summary
    summary = {
        "analysis_timestamp": datetime.now().isoformat(),
        "total_models_evaluated": len(results),
        "successful_models": len(successful_results),
        "success_rate": len(successful_results) / len(results),
        "best_performing_model": max(successful_results, key=lambda x: x['avg_combined_similarity'])['model'],
        "fastest_model": min(successful_results, key=lambda x: x['duration'])['model'],
        "most_operations_model": max(successful_results, key=lambda x: x['total_operations'])['model']
    }
    
    # Save summary
    summary_file = "output/phase4_excel_analysis/final_analysis_summary.json"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Final summary saved to: {summary_file}")

def check_prerequisites():
    """Check if prerequisites are met"""
    
    print("🔍 Checking Prerequisites...")
    print("-" * 30)
    
    # Check if Phase 4 results exist
    phase4_dir = "output/phase4_balanced_refinement"
    if not os.path.exists(phase4_dir):
        print(f"❌ Phase 4 results directory not found: {phase4_dir}")
        print("Please run Phase 4 balanced refinement first")
        return False
    
    # Check if benchmark topics exist
    benchmark_file = "phases/phase4_clusters_refined.json"
    if not os.path.exists(benchmark_file):
        print(f"❌ Benchmark topics file not found: {benchmark_file}")
        return False
    
    # Check if there are any Phase 4 result files
    phase4_files = [f for f in os.listdir(phase4_dir) if f.endswith('.json') and not f.startswith('comprehensive_')]
    if not phase4_files:
        print(f"❌ No Phase 4 result files found in {phase4_dir}")
        return False
    
    print(f"✅ Found {len(phase4_files)} Phase 4 result files")
    print(f"✅ Benchmark topics file exists")
    print("✅ Prerequisites met")
    
    return True

def main():
    """Main function"""
    
    print("🎯 Phase 4 Comprehensive Analysis Runner")
    print("=" * 60)
    print("This script will:")
    print("1. Run comprehensive evaluation on all Phase 4 results")
    print("2. Generate Excel analysis similar to Phase 3 format")
    print("3. Create client-friendly reports")
    print("4. Generate model rankings and performance metrics")
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return
    
    print()
    
    # Run analysis
    success = run_phase4_comprehensive_analysis()
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        print("You can now find the results in the output directories.")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
