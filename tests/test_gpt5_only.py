#!/usr/bin/env python3
"""
Test GPT-5 Only from Main Phase3 System
Tests GPT-5 integration with the main evaluation framework
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phases.phase3_topic_clustering import Phase3Evaluator

def test_gpt5_only():
    """Test GPT-5 only from the main system"""
    print("🚀 Testing GPT-5 Integration with Main Phase3 System")
    print("=" * 60)
    
    # Create evaluator
    evaluator = Phase3Evaluator()
    
    # Test GPT-5 specifically
    print("\n🧪 Testing GPT-5 specifically...")
    try:
        result = evaluator.evaluate_model("openai", "gpt-5")
        
        if result["success"]:
            print(f"✅ GPT-5 test successful!")
            print(f"   Duration: {result['duration']:.2f}s")
            print(f"   Clusters: {len(result['clusters'])}")
            print(f"   Cost: ${result['cost']['total_cost']:.6f}")
            
            # Save result
            output_file = os.path.join(evaluator.output_dir, "gpt-5_test_result.json")
            import json
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"   Results saved to: {output_file}")
            
        else:
            print(f"❌ GPT-5 test failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error testing GPT-5: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpt5_only()
