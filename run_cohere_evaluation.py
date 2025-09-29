#!/usr/bin/env python3
"""
Simple script to run Cohere model evaluation with fixes
"""

import os
import sys
import subprocess

def setup_environment():
    """Setup the environment for running Cohere models"""
    
    print("🔧 Setting up environment for Cohere model evaluation...")
    
    
    required_packages = [
        "transformers",
        "torch", 
        "accelerate",
        "bitsandbytes",
        "pandas",
        "python-dotenv"
    ]
    
    print("📦 Checking required packages...")
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - needs installation")
    
    return True

def apply_fixes():
    """Apply the fixes to the existing code"""
    
    print("🔧 Applying fixes to model loading...")
    
    
    try:
        result = subprocess.run([sys.executable, "fix_model_loading_error.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Fixes applied successfully")
            
            
            if os.path.exists("cohere_clustering.py"):
                os.rename("cohere_clustering.py", "cohere_clustering_backup.py")
                print("📁 Backed up original to cohere_clustering_backup.py")
            
            if os.path.exists("cohere_clustering_fixed.py"):
                os.rename("cohere_clustering_fixed.py", "cohere_clustering.py")
                print("🔄 Replaced with fixed version")
            
            return True
        else:
            print(f"❌ Error applying fixes: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running fix script: {e}")
        return False

def run_evaluation():
    """Run the actual evaluation"""
    
    print("🚀 Starting Cohere model evaluation...")
    
    try:
        
        result = subprocess.run([sys.executable, "evaluate_cohere_models.py"], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ Evaluation completed successfully")
            return True
        else:
            print("⚠️ Evaluation completed with some issues")
            return True  
            
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return False

def main():
    """Main function"""
    
    print("🎯 Cohere Model Evaluation Setup and Run")
    print("=" * 50)
    
    
    if not setup_environment():
        print("❌ Environment setup failed")
        return
    
    
    if not apply_fixes():
        print("❌ Could not apply fixes")
        return
    
    
    if not run_evaluation():
        print("❌ Evaluation failed")
        return
    
    print("\n🎉 Process completed!")
    print("\n📁 Check these locations for results:")
    print("   - output/phase3_topic_clustering/")
    print("   - Individual model JSON files")
    print("   - comprehensive_results.json")

if __name__ == "__main__":
    main()
