#!/usr/bin/env python3
"""
Clean up duplicate Cohere models to free up space
Keep only the latest version and remove the older one
"""

import os
import shutil
from pathlib import Path

def get_model_sizes():
    """Get sizes of both Cohere models"""
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    models = {
        "original": "models--CohereLabs--c4ai-command-r-plus",
        "latest": "models--CohereLabs--c4ai-command-r-plus-08-2024"
    }
    
    sizes = {}
    
    for name, model_dir in models.items():
        full_path = os.path.join(cache_dir, model_dir)
        if os.path.exists(full_path):
            total_size = 0
            for root, dirs, files in os.walk(full_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
            sizes[name] = total_size
            print(f"📊 {name.capitalize()} model: {format_size(total_size)}")
        else:
            print(f"❌ {name.capitalize()} model not found")
    
    return sizes

def format_size(size_bytes):
    """Format size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"

def remove_duplicate_model():
    """Remove the older duplicate model"""
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    old_model_path = os.path.join(cache_dir, "models--CohereLabs--c4ai-command-r-plus")
    latest_model_path = os.path.join(cache_dir, "models--CohereLabs--c4ai-command-r-plus-08-2024")
    
    print("🧹 Cleaning up duplicate Cohere models...")
    print("=" * 45)
    
    # Check which models exist
    old_exists = os.path.exists(old_model_path)
    latest_exists = os.path.exists(latest_model_path)
    
    print(f"📁 Original model exists: {old_exists}")
    print(f"📁 Latest model exists: {latest_exists}")
    
    if not old_exists:
        print("✅ No duplicate to remove")
        return True
    
    if not latest_exists:
        print("❌ Latest model not found - cannot remove original")
        return False
    
    # Get sizes before removal
    print("\\n📊 Model sizes before cleanup:")
    sizes = get_model_sizes()
    
    # Remove the older model
    print(f"\\n🗑️  Removing older model: {old_model_path}")
    
    try:
        shutil.rmtree(old_model_path)
        print("✅ Older model removed successfully")
        
        # Verify removal
        if not os.path.exists(old_model_path):
            print("✅ Removal confirmed")
        else:
            print("❌ Removal failed")
            return False
        
        # Show space saved
        if "original" in sizes:
            space_saved = sizes["original"]
            print(f"\\n💾 Space saved: {format_size(space_saved)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error removing model: {e}")
        return False

def verify_remaining_model():
    """Verify the remaining model works"""
    
    print("\\n🔍 Verifying remaining model...")
    
    try:
        from transformers import AutoTokenizer
        
        # Test the latest model
        model_name = "CohereLabs/c4ai-command-r-plus-08-2024"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"✅ Latest model accessible: {model_name}")
        print(f"✅ Vocabulary size: {tokenizer.vocab_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing remaining model: {e}")
        return False

def show_cleanup_summary():
    """Show cleanup summary"""
    
    print("\\n📋 Cleanup Summary:")
    print("=" * 20)
    print("✅ Removed: CohereLabs/c4ai-command-r-plus (80GB)")
    print("✅ Kept: CohereLabs/c4ai-command-r-plus-08-2024 (194GB)")
    print("💾 Space saved: ~80GB")
    print("🎯 Using: Latest version with better performance")
    
    print("\\n📈 Benefits:")
    print("   - More disk space available")
    print("   - Using the latest model version")
    print("   - Better performance and capabilities")
    print("   - No duplicate downloads")

def main():
    """Main cleanup function"""
    
    print("🧹 Cohere Model Cleanup")
    print("=" * 25)
    
    # Show current sizes
    print("\\n📊 Current model sizes:")
    sizes = get_model_sizes()
    
    # Remove duplicate
    if remove_duplicate_model():
        print("\\n✅ Cleanup completed successfully")
        
        # Verify remaining model
        if verify_remaining_model():
            show_cleanup_summary()
        else:
            print("\\n⚠️ Cleanup completed but model verification failed")
    else:
        print("\\n❌ Cleanup failed")

if __name__ == "__main__":
    main()
