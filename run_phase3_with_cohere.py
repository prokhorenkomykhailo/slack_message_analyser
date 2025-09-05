#!/usr/bin/env python3
"""
Run Phase 3: Topic Clustering Evaluation with Cohere Command R+
Enhanced script that includes Cohere models in the evaluation
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_cohere_setup():
    """Check if Cohere dependencies are installed"""
    try:
        import transformers
        import torch
        import accelerate
        import bitsandbytes
        return True
    except ImportError:
        return False

def install_cohere_dependencies():
    """Install Cohere dependencies if needed"""
    print("🔄 Installing Cohere dependencies...")
    try:
        result = subprocess.run([sys.executable, "install_cohere_dependencies.py"], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def run_standard_phase3():
    """Run the standard Phase 3 evaluation"""
    print("🔄 Running standard Phase 3 evaluation...")
    try:
        from phases.phase3_topic_clustering import Phase3Evaluator
        
        evaluator = Phase3Evaluator()
        evaluator.run_evaluation()
        return True
    except Exception as e:
        print(f"❌ Error in standard Phase 3: {e}")
        return False

def run_cohere_evaluation():
    """Run Cohere Command R+ evaluation"""
    print("🔄 Running Cohere Command R+ evaluation...")
    try:
        result = subprocess.run([sys.executable, "evaluate_cohere_models.py"], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error in Cohere evaluation: {e}")
        return False

def main():
    """Main function"""
    
    print("🚀 Phase 3 Topic Clustering with Cohere Command R+")
    print("=" * 55)
    
    # Step 1: Check and install Cohere dependencies
    print("\n📋 Step 1: Checking Cohere setup...")
    if not check_cohere_setup():
        print("⚠️ Cohere dependencies not found. Installing...")
        if not install_cohere_dependencies():
            print("❌ Failed to install Cohere dependencies")
            print("Continuing with standard evaluation only...")
            cohere_available = False
        else:
            print("✅ Cohere dependencies installed successfully")
            cohere_available = True
    else:
        print("✅ Cohere dependencies already installed")
        cohere_available = True
    
    # Step 2: Run standard Phase 3 evaluation
    print("\n📋 Step 2: Running standard Phase 3 evaluation...")
    standard_success = run_standard_phase3()
    
    # Step 3: Run Cohere evaluation (if available)
    cohere_success = False
    if cohere_available:
        print("\n📋 Step 3: Running Cohere Command R+ evaluation...")
        cohere_success = run_cohere_evaluation()
    else:
        print("\n📋 Step 3: Skipping Cohere evaluation (dependencies not available)")
    
    # Summary
    print("\n" + "=" * 55)
    print("📊 EVALUATION SUMMARY")
    print("=" * 55)
    
    print(f"   Standard Phase 3: {'✅ SUCCESS' if standard_success else '❌ FAILED'}")
    if cohere_available:
        print(f"   Cohere Command R+: {'✅ SUCCESS' if cohere_success else '❌ FAILED'}")
    else:
        print("   Cohere Command R+: ⏭️ SKIPPED (dependencies not available)")
    
    if standard_success:
        print("\n🎉 Phase 3 evaluation completed!")
        print("\n📁 Results available in:")
        print("   - output/phase3_topic_clustering/")
        print("   - Individual model JSON files")
        
        if cohere_available and cohere_success:
            print("\n🔍 Cohere models included in comprehensive results")
            print("   - cohere_command-r-plus")
            print("   - cohere_command-r-plus-8bit") 
            print("   - cohere_command-r-plus-4bit")
        
        print("\n📈 Next steps:")
        print("   1. Review clustering results")
        print("   2. Analyze model performance")
        print("   3. Generate comparison reports")
    else:
        print("\n❌ Evaluation failed. Please check the errors above.")

if __name__ == "__main__":
    main()
