#!/usr/bin/env python3
"""
Test script to verify the fixed Cohere evaluation
"""

import sys
import os

def test_imports():
    """Test if required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import pandas
        print("✅ pandas imported successfully")
    except ImportError:
        print("❌ pandas not available")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✅ dotenv imported successfully")
    except ImportError:
        print("❌ dotenv not available")
        return False
    
    try:
        from huggingface_hub import HfApi
        print("✅ huggingface_hub imported successfully")
    except ImportError:
        print("❌ huggingface_hub not available")
        return False
    
    return True

def test_script_syntax():
    """Test if the main script has valid syntax"""
    print("\n🔍 Testing script syntax...")
    
    try:
        # Try to import the main module
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import evaluate_cohere_models
        print("✅ evaluate_cohere_models imported successfully")
        return True
    except Exception as e:
        print(f"❌ Script syntax error: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 Testing Fixed Cohere Evaluation Script")
    print("=" * 45)
    
    tests = [
        ("Package Imports", test_imports),
        ("Script Syntax", test_script_syntax)
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
    print("=" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The script should work now.")
        print("\nNext steps:")
        print("1. Set up authentication: python setup_huggingface_auth.py")
        print("2. Run evaluation: python evaluate_cohere_models.py")
    else:
        print("\n⚠️  Some tests failed. Please install missing packages:")
        print("pip install pandas python-dotenv huggingface_hub")

if __name__ == "__main__":
    main()
