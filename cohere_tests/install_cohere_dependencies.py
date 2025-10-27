#!/usr/bin/env python3
"""
Installation script for Cohere Command R+ dependencies
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def main():
    """Main installation function"""
    
    print("🚀 Installing Cohere Command R+ Dependencies")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  No virtual environment detected. Consider using one:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # Linux/Mac")
        print("   venv\\Scripts\\activate     # Windows")
    
    # Install basic requirements
    commands = [
        ("pip install -r requirements_cohere.txt", "Installing basic requirements"),
        ("pip install 'git+https://github.com/huggingface/transformers.git'", "Installing transformers from source"),
        ("pip install bitsandbytes accelerate", "Installing quantization support"),
    ]
    
    success_count = 0
    for command, description in commands:
        if run_command(command, description):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"   Successful: {success_count}/{len(commands)}")
    print(f"   Failed: {len(commands) - success_count}/{len(commands)}")
    
    if success_count == len(commands):
        print("\n🎉 All dependencies installed successfully!")
        print("\nNext steps:")
        print("1. Make sure you have a Hugging Face token:")
        print("   - Go to https://huggingface.co/settings/tokens")
        print("   - Create a token with read access")
        print("   - Set it as environment variable: export HUGGINGFACE_TOKEN=your_token")
        print("\n2. Run the evaluation:")
        print("   python phase3_evaluation_with_cohere.py")
    else:
        print("\n❌ Some installations failed. Please check the errors above.")
        print("You may need to:")
        print("- Update pip: pip install --upgrade pip")
        print("- Install build tools for your system")
        print("- Check your Python version (3.8+ required)")

if __name__ == "__main__":
    main()
