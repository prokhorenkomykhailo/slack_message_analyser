#!/usr/bin/env python3
"""
Debug script to troubleshoot Hugging Face access issues
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def debug_huggingface_access():
    """Debug Hugging Face access issues"""
    
    print("🔍 Debugging Hugging Face Access Issues")
    print("=" * 50)
    
    # Check token
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("❌ No HUGGINGFACE_TOKEN found")
        return False
    
    print(f"✅ Token found: {token[:10]}...")
    
    # Test token validity
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user_info = api.whoami()
        print(f"✅ Token is valid. Authenticated as: {user_info['name']}")
        print(f"   User ID: {user_info['id']}")
        print(f"   Type: {user_info.get('type', 'unknown')}")
    except Exception as e:
        print(f"❌ Token validation failed: {e}")
        return False
    
    # Check specific model access
    models_to_check = [
        "CohereLabs/c4ai-command-r-plus",
        "CohereLabs/c4ai-command-r-plus-08-2024"
    ]
    
    for model_id in models_to_check:
        print(f"\n🔍 Checking access to {model_id}...")
        
        try:
            # Check repository info
            repo_info = api.repo_info(model_id)
            print(f"✅ Repository info accessible")
            print(f"   Private: {repo_info.private}")
            print(f"   Gated: {getattr(repo_info, 'gated', 'unknown')}")
            
            # Check if user has access
            try:
                # Try to get model info
                model_info = api.model_info(model_id)
                print(f"✅ Model info accessible")
                
                # Try to list files (this requires access)
                files = api.list_repo_files(model_id)
                print(f"✅ File listing accessible ({len(files)} files)")
                
            except Exception as e:
                print(f"❌ Model access failed: {e}")
                if "401" in str(e) or "gated" in str(e).lower():
                    print("   This suggests you don't have access to this specific model")
                    print("   Even though the web interface shows access, API access might be different")
                
        except Exception as e:
            print(f"❌ Repository check failed: {e}")
    
    # Test with transformers
    print(f"\n🔍 Testing with transformers library...")
    try:
        from transformers import AutoTokenizer
        
        # Try to load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("CohereLabs/c4ai-command-r-plus")
        print("✅ Transformers can access the model")
        return True
        
    except Exception as e:
        print(f"❌ Transformers access failed: {e}")
        
        # Check if it's a token issue
        if "401" in str(e):
            print("\n💡 Possible solutions:")
            print("1. Try logging in with huggingface-cli:")
            print("   huggingface-cli login")
            print("2. Check if your token has the right permissions")
            print("3. Try refreshing your token at: https://huggingface.co/settings/tokens")
            print("4. Make sure you've accepted the model's terms of use")
        
        return False

def main():
    """Main debug function"""
    
    if debug_huggingface_access():
        print("\n🎉 Access debugging completed - everything looks good!")
    else:
        print("\n❌ Access issues detected")
        print("\n🔧 Recommended actions:")
        print("1. Run: huggingface-cli login")
        print("2. Re-enter your token when prompted")
        print("3. Make sure to accept the model's terms of use")
        print("4. Try the alternative models script: python evaluate_alternative_models_only.py")

if __name__ == "__main__":
    main()
