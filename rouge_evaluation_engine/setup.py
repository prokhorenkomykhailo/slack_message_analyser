#!/usr/bin/env python3
"""
Setup script for ROUGE Evaluation Engine
"""

import os
import subprocess
import sys

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print("✅ Python version:", sys.version)
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_dependencies():
    """Check if all dependencies are available"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        'rouge_score', 'nltk', 'scikit-learn', 'numpy', 'pandas', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies available!")
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded!")
        return True
    except Exception as e:
        print(f"⚠️  Warning: Could not download NLTK data: {e}")
        return True  # Not critical

def check_file_structure():
    """Check if required files exist"""
    print("\n📁 Checking file structure...")
    
    required_files = [
        "../phases/phase3_clusters.json",
        "../output/phase3_topic_clustering/"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            return False
    
    return True

def run_test():
    """Run a quick test to verify everything works"""
    print("\n🧪 Running quick test...")
    
    try:
        from rouge_clustering_evaluator import RougeClusteringEvaluator
        print("✅ ROUGE evaluator imported successfully!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 ROUGE Evaluation Engine Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed at requirements installation")
        return
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Setup failed at dependency check")
        return
    
    # Download NLTK data
    download_nltk_data()
    
    # Check file structure
    if not check_file_structure():
        print("\n❌ Setup failed at file structure check")
        print("   Please ensure you're running this from the rouge_evaluation_engine directory")
        return
    
    # Run test
    if not run_test():
        print("\n❌ Setup failed at test run")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\n📝 Next steps:")
    print("   1. Test with single model: python test_rouge_evaluation.py")
    print("   2. Run full evaluation: python run_rouge_evaluation.py")
    print("   3. Check results in rouge_results/ directory")
    print("\n📖 For more information, see README.md")

if __name__ == "__main__":
    main()
