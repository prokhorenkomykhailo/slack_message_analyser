#!/usr/bin/env python3
"""
Test script to verify Cohere Command R+ setup
"""

import sys
import os
import torch

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    try:
        import transformers
        print(f"✅ transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ transformers: {e}")
        return False
    
    try:
        import torch
        print(f"✅ torch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ torch: {e}")
        return False
    
    try:
        import accelerate
        print(f"✅ accelerate: {accelerate.__version__}")
    except ImportError as e:
        print(f"❌ accelerate: {e}")
        return False
    
    try:
        import bitsandbytes
        print(f"✅ bitsandbytes: {bitsandbytes.__version__}")
    except ImportError as e:
        print(f"❌ bitsandbytes: {e}")
        return False
    
    try:
        import pandas
        print(f"✅ pandas: {pandas.__version__}")
    except ImportError as e:
        print(f"❌ pandas: {e}")
        return False
    
    return True

def test_cuda():
    """Test CUDA availability"""
    print("\n🔍 Testing CUDA availability...")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("⚠️  CUDA not available - will use CPU (slower)")
        return False

def test_model_access():
    """Test if we can access the Cohere model"""
    print("\n🔍 Testing model access...")
    
    try:
        from transformers import AutoTokenizer
        
        # Test tokenizer loading (lightweight test)
        tokenizer = AutoTokenizer.from_pretrained("CohereLabs/c4ai-command-r-plus")
        print("✅ Successfully loaded Cohere Command R+ tokenizer")
        
        # Test chat template
        messages = [{"role": "user", "content": "Hello"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("✅ Chat template working")
        
        return True
        
    except Exception as e:
        print(f"❌ Model access failed: {e}")
        print("   This might be due to:")
        print("   - Missing Hugging Face token")
        print("   - Network connectivity issues")
        print("   - Model access restrictions")
        return False

def test_clustering_engine():
    """Test if our clustering engine can be imported"""
    print("\n🔍 Testing clustering engine...")
    
    try:
        from cohere_clustering import CohereCommandRPlusClustering
        print("✅ CohereCommandRPlusClustering imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Clustering engine import failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 Cohere Command R+ Setup Test")
    print("=" * 40)
    
    tests = [
        ("Package Imports", test_imports),
        ("CUDA Availability", test_cuda),
        ("Model Access", test_model_access),
        ("Clustering Engine", test_clustering_engine)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! You're ready to run Cohere Command R+ evaluation.")
        print("\nNext steps:")
        print("1. python phase3_evaluation_with_cohere.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before running evaluation.")
        print("\nCommon solutions:")
        print("- Run: python install_cohere_dependencies.py")
        print("- Set Hugging Face token: export HUGGINGFACE_TOKEN=your_token")
        print("- Check internet connection")
        print("- Ensure you have enough GPU memory (8GB+ recommended)")

if __name__ == "__main__":
    main()
