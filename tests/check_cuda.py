#!/usr/bin/env python3
"""
Check CUDA availability and GPU information on this VPS
"""

import torch
import subprocess
import os

def check_cuda():
    print("🔍 CUDA and GPU Check")
    print("=" * 40)
    
    # Check PyTorch CUDA availability
    print("1. PyTorch CUDA Support:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("   ❌ CUDA not available in PyTorch")
    
    print("\n2. System GPU Information:")
    try:
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("   ✅ nvidia-smi found:")
            print("   " + result.stdout.split('\n')[1])  # GPU info line
        else:
            print("   ❌ nvidia-smi not found or no GPU")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   ❌ nvidia-smi not available")
    
    print("\n3. CUDA Installation Check:")
    try:
        # Check if CUDA is installed
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   ✅ CUDA compiler found:")
            print("   " + result.stdout.split('\n')[0])
        else:
            print("   ❌ CUDA compiler not found")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   ❌ CUDA compiler not available")
    
    print("\n4. Memory Information:")
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    mem_gb = int(line.split()[1]) / 1024 / 1024
                    print(f"   Total RAM: {mem_gb:.1f} GB")
                    break
    except:
        print("   ❌ Could not read memory info")
    
    print("\n5. Recommendations:")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 8:
            print("   ✅ GPU has enough memory for Cohere model")
            print("   💡 You can use GPU acceleration")
        else:
            print("   ⚠️  GPU memory might be insufficient for large models")
            print("   💡 Consider using CPU with memory optimization")
    else:
        print("   ❌ No CUDA support detected")
        print("   💡 Use CPU-only configuration for Cohere")
        print("   💡 Consider upgrading to a GPU-enabled VPS")

def main():
    check_cuda()

if __name__ == "__main__":
    main()
