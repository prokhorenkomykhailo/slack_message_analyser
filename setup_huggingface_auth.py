#!/usr/bin/env python3
"""
Setup Hugging Face authentication for Cohere model access
"""

import os
import sys
from dotenv import load_dotenv

def setup_huggingface_auth():
    """Setup Hugging Face authentication"""
    
    print("🔐 Hugging Face Authentication Setup")
    print("=" * 40)
    
    # Load existing .env file
    load_dotenv()
    
    # Check if token already exists
    existing_token = os.getenv('HUGGINGFACE_TOKEN')
    if existing_token:
        print(f"✅ HUGGINGFACE_TOKEN already set: {existing_token[:10]}...")
        
        # Test the token
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=existing_token)
            user_info = api.whoami()
            print(f"✅ Token is valid. Authenticated as: {user_info['name']}")
            return True
        except Exception as e:
            print(f"❌ Token is invalid: {e}")
            print("Please get a new token from: https://huggingface.co/settings/tokens")
            return False
    
    # Get token from user
    print("\n📝 To access the Cohere Command R+ model, you need a Hugging Face token.")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'Read' permissions")
    print("3. Request access to: https://huggingface.co/CohereLabs/c4ai-command-r-plus")
    print("4. Paste your token below:")
    
    token = input("\nEnter your Hugging Face token: ").strip()
    
    if not token:
        print("❌ No token provided")
        return False
    
    # Test the token
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user_info = api.whoami()
        print(f"✅ Token is valid. Authenticated as: {user_info['name']}")
        
        # Save to .env file
        env_file = '.env'
        with open(env_file, 'a') as f:
            f.write(f"\nHUGGINGFACE_TOKEN={token}\n")
        
        print(f"✅ Token saved to {env_file}")
        return True
        
    except Exception as e:
        print(f"❌ Token is invalid: {e}")
        print("Please check your token and try again.")
        return False

def main():
    """Main function"""
    
    if setup_huggingface_auth():
        print("\n🎉 Authentication setup complete!")
        print("You can now run: python evaluate_cohere_models.py")
    else:
        print("\n❌ Authentication setup failed.")
        print("You can still run the script with alternative models.")
        print("Run: python evaluate_cohere_models.py")

if __name__ == "__main__":
    main()
