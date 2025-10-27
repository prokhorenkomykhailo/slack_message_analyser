#!/usr/bin/env python3
"""
Script to help request access to the gated Cohere Command R+ model
"""

import webbrowser
import os
from dotenv import load_dotenv

def request_cohere_access():
    """Help user request access to Cohere model"""
    
    print("🔐 Cohere Command R+ Access Request Helper")
    print("=" * 45)
    
    # Load environment variables
    load_dotenv()
    
    # Check if user is authenticated
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("❌ No HUGGINGFACE_TOKEN found")
        print("Please run: python setup_huggingface_auth.py")
        return False
    
    print("✅ Hugging Face token found")
    
    # Open the model page
    model_url = "https://huggingface.co/CohereLabs/c4ai-command-r-plus"
    
    print(f"\n📝 To request access to the Cohere Command R+ model:")
    print(f"1. Go to: {model_url}")
    print("2. Click the 'Request access' button")
    print("3. Fill out the access request form")
    print("4. Wait for approval (usually takes a few hours to days)")
    print("5. Once approved, run the evaluation script again")
    
    # Ask if user wants to open the page
    response = input("\n🌐 Would you like to open the model page in your browser? (y/n): ").strip().lower()
    
    if response in ['y', 'yes']:
        try:
            webbrowser.open(model_url)
            print("✅ Opening model page in your browser...")
        except Exception as e:
            print(f"❌ Could not open browser: {e}")
            print(f"Please manually visit: {model_url}")
    
    print(f"\n🔗 Direct link: {model_url}")
    
    return True

def main():
    """Main function"""
    
    if request_cohere_access():
        print("\n💡 While waiting for approval, you can:")
        print("1. Run the evaluation with alternative models: python evaluate_cohere_models.py")
        print("2. The script will automatically use open models as fallback")
        print("3. Once you get access, the script will use Cohere models automatically")
    else:
        print("\n❌ Please set up authentication first")

if __name__ == "__main__":
    main()
