#!/usr/bin/env python3
"""
Test GPT-5 Integration
Verifies that GPT-5 is properly integrated into the main system
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import get_available_models, get_model_config

def test_gpt5_integration():
    """Test if GPT-5 is properly integrated"""
    print("🔍 Testing GPT-5 Integration")
    print("=" * 40)
    
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in .env file")
        return False
    
    print("✅ OPENAI_API_KEY found")
    
    # Get available models
    available_models = get_available_models()
    print(f"📊 Available models: {available_models}")
    
    # Check if GPT-5 is available
    if "openai" in available_models and "gpt-5" in available_models["openai"]:
        print("✅ GPT-5 is available in OpenAI models")
        
        # Get model config
        config = get_model_config("openai", "gpt-5")
        print(f"🔧 GPT-5 config: {config}")
        
        return True
    else:
        print("❌ GPT-5 is NOT available in OpenAI models")
        return False

if __name__ == "__main__":
    success = test_gpt5_integration()
    if success:
        print("\n🎉 GPT-5 Integration Test PASSED!")
    else:
        print("\n❌ GPT-5 Integration Test FAILED!")
