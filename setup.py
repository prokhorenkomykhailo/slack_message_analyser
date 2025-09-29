#!/usr/bin/env python3
"""
Setup script for Phase Evaluation Engine
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def create_virtual_environment():
    """Create virtual environment if not exists"""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment already exists")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Not in a virtual environment. Please activate it first:")
        print("   source venv/bin/activate  # On Linux/Mac")
        print("   venv\\Scripts\\activate     # On Windows")
        return False
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        
        dependencies = [
            "numpy>=1.21.0",
            "pandas>=1.3.0", 
            "scikit-learn>=1.0.0",
            "openai>=1.0.0",
            "google-generativeai>=0.3.0",
            "groq>=0.4.0",
            "anthropic>=0.7.0",
            "requests>=2.28.0",
            "python-dotenv>=0.19.0",
            "tqdm>=4.64.0"
        ]
        
        for dep in dependencies:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
        
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "output",
        "output/phase3_topic_clustering",
        "output/phase4_merge_split", 
        "output/phase5_metadata_generation",
        "output/phase6_embedding",
        "output/phase7_user_filtering",
        "output/phase8_new_message_processing"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        "../message_dataset.json",
        "../benchmark_topics.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Missing required data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease run Phase 1 and Phase 2 first to generate these files.")
        return False
    else:
        print("✅ Required data files found")
        return True

def check_api_keys():
    """Check if API keys are set"""
    api_keys = [
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY", 
        "GROQ_API_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY"
    ]
    
    available_keys = []
    for key in api_keys:
        if os.getenv(key):
            available_keys.append(key.replace("_API_KEY", "").lower())
    
    if available_keys:
        print(f"✅ API keys found for: {', '.join(available_keys)}")
        return True
    else:
        print("⚠️  No API keys found. Please set environment variables:")
        for key in api_keys:
            print(f"   export {key}=\"your-api-key\"")
        return False

def main():
    """Main setup function"""
    print("🚀 Phase Evaluation Engine Setup")
    print("=" * 50)
    
    
    check_python_version()
    
    
    create_virtual_environment()
    
    
    create_directories()
    
    
    data_ok = check_data_files()
    
    
    api_ok = check_api_keys()
    
    print("\n" + "=" * 50)
    print("📋 Setup Summary:")
    print(f"   Python version: ✅")
    print(f"   Virtual environment: ✅")
    print(f"   Directories: ✅")
    print(f"   Data files: {'✅' if data_ok else '❌'}")
    print(f"   API keys: {'✅' if api_ok else '❌'}")
    
    if not data_ok:
        print("\n⚠️  Please run Phase 1 and Phase 2 first to generate required data files.")
    
    if not api_ok:
        print("\n⚠️  Please set API keys to test models.")
    
    if data_ok and api_ok:
        print("\n🎉 Setup complete! You can now run:")
        print("   python run_all_phases.py")
    else:
        print("\n⚠️  Setup incomplete. Please address the issues above.")

if __name__ == "__main__":
    main()
