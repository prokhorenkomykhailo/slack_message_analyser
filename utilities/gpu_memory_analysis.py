#!/usr/bin/env python3
"""
GPU Memory Analysis for Cohere Model
Shows what GPU memory is needed vs what you have
"""

import torch
import os

def analyze_gpu_requirements():
    print("🔍 GPU Memory Analysis for Cohere Model")
    print("=" * 50)
    
    # Your current GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ Your GPU: {gpu_name}")
        print(f"✅ Your GPU Memory: {gpu_memory:.1f} GB")
    else:
        print("❌ CUDA not available")
        return
    
    # Model requirements
    print(f"\n📊 Cohere Model Requirements:")
    print(f"   Model size: 194 GB (downloaded)")
    print(f"   Model parameters: ~104B parameters")
    print(f"   Precision: float16")
    
    # Memory calculations
    print(f"\n🧮 Memory Calculations:")
    
    # For full model in float16
    full_model_memory = 194  # GB
    print(f"   Full model (float16): {full_model_memory} GB")
    
    # For inference (model + activations + cache)
    inference_memory = full_model_memory + 8  # Model + activations
    print(f"   Inference memory needed: {inference_memory} GB")
    
    # Your GPU vs requirements
    print(f"\n📈 Your GPU vs Requirements:")
    print(f"   Your GPU memory: {gpu_memory:.1f} GB")
    print(f"   Required memory: {inference_memory} GB")
    print(f"   Shortage: {inference_memory - gpu_memory:.1f} GB")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if gpu_memory >= inference_memory:
        print("   ✅ Your GPU is sufficient!")
        print("   🚀 You can run the full model on GPU")
    elif gpu_memory >= full_model_memory * 0.5:
        print("   ⚠️  Your GPU can handle partial model")
        print("   🔧 Use CPU + GPU hybrid approach")
        print("   💡 This is what we're doing with memory optimization")
    else:
        print("   ❌ Your GPU is too small for this model")
        print("   🖥️  Use CPU-only approach")
    
    # GPU recommendations for this model
    print(f"\n🎯 Recommended GPU Sizes for Cohere:")
    print(f"   Minimum: 24GB VRAM (RTX 4090, A6000)")
    print(f"   Recommended: 32GB VRAM (RTX 6000 Ada, A100)")
    print(f"   Optimal: 40GB+ VRAM (A100 40GB, H100)")
    
    # Your options
    print(f"\n🚀 Your Options:")
    print(f"   1. Use CPU + GPU hybrid (current approach)")
    print(f"   2. Use CPU-only (reliable but slower)")
    print(f"   3. Upgrade to 24GB+ GPU (RTX 4090, A6000)")
    print(f"   4. Use cloud GPU (A100, H100)")
    
    # Performance comparison
    print(f"\n⚡ Performance Comparison:")
    print(f"   Your RTX A4000 (16GB): Hybrid CPU+GPU")
    print(f"   RTX 4090 (24GB): Full GPU (3-5x faster)")
    print(f"   A100 (40GB): Full GPU (5-10x faster)")
    print(f"   CPU-only: Reliable but 10x slower")
    
    # Cost analysis
    print(f"\n💰 Cost Analysis:")
    print(f"   Your RTX A4000: $1000+ (professional GPU)")
    print(f"   RTX 4090: $1600+ (gaming GPU, 24GB)")
    print(f"   RTX 6000 Ada: $4000+ (professional, 48GB)")
    print(f"   A100: $10000+ (data center GPU)")
    
    return gpu_memory, inference_memory

def main():
    gpu_memory, required_memory = analyze_gpu_requirements()
    
    print(f"\n🎯 Summary:")
    print(f"   Your GPU: {gpu_memory:.1f} GB")
    print(f"   Required: {required_memory} GB")
    print(f"   Gap: {required_memory - gpu_memory:.1f} GB")
    
    if gpu_memory < required_memory:
        print(f"\n💡 The model is {required_memory - gpu_memory:.1f} GB too large for your GPU")
        print(f"   This is why we need CPU + GPU hybrid approach")
        print(f"   Your 31GB RAM helps compensate for the GPU shortage")

if __name__ == "__main__":
    main()
