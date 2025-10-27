#!/usr/bin/env python3
"""
Setup script for Cohere API
Installs dependencies and sets up API key
"""

import subprocess
import os
import sys

def install_dependencies():
    """Install required dependencies"""
    print("🔧 Installing Cohere API dependencies...")
    
    try:
        # Install cohere package
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cohere"])
        print("✅ Cohere package installed")
        
        # Install other dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy"])
        print("✅ Additional dependencies installed")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_api_key():
    """Setup API key"""
    print("\n🔑 Setting up Cohere API key...")
    
    # Check if API key already exists
    api_key = os.getenv("COHERE_API_KEY")
    if api_key:
        print("✅ COHERE_API_KEY already set")
        return True
    
    print("\n📝 To get your Cohere API key:")
    print("1. Go to https://dashboard.cohere.com/")
    print("2. Sign up or log in")
    print("3. Go to API Keys section")
    print("4. Create a new API key")
    print("5. Copy the API key")
    
    # Get API key from user
    api_key = input("\nEnter your Cohere API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return False
    
    # Set environment variable
    os.environ["COHERE_API_KEY"] = api_key
    
    # Add to bashrc for persistence
    bashrc_path = os.path.expanduser("~/.bashrc")
    with open(bashrc_path, "a") as f:
        f.write(f'\nexport COHERE_API_KEY="{api_key}"\n')
    
    print("✅ API key set successfully")
    print("✅ Added to ~/.bashrc for persistence")
    
    return True

def test_setup():
    """Test the setup"""
    print("\n🧪 Testing Cohere API setup...")
    
    try:
        import cohere
        print("✅ Cohere package imported successfully")
        
        # Test API key
        api_key = os.getenv("COHERE_API_KEY")
        if api_key:
            print("✅ API key found")
            
            # Test client initialization
            co = cohere.Client(api_key)
            print("✅ Cohere client initialized successfully")
            
            return True
        else:
            print("❌ API key not found")
            return False
            
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Cohere API Setup")
    print("=" * 40)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return
    
    # Setup API key
    if not setup_api_key():
        print("❌ Failed to setup API key")
        return
    
    # Test setup
    if not test_setup():
        print("❌ Setup test failed")
        return
    
    print("\n🎉 Cohere API setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run: python cohere_api_working.py")
    print("2. Check output/cohere_api_results.json for results")
    print("3. Use Cohere API for all your message clustering needs")
    
    print("\n💡 Benefits of Cohere API:")
    print("✅ No hardware limitations")
    print("✅ Always latest model")
    print("✅ No maintenance")
    print("✅ Pay per use")
    print("✅ Works on any VPS")

if __name__ == "__main__":
    main()
