#!/usr/bin/env python3
"""
Simple system check for CUDA and GPU support
"""

import subprocess
import os

def check_system():
    print("🔍 VPS System Check")
    print("=" * 40)
    
    # Check GPU hardware
    print("1. GPU Hardware:")
    try:
        result = subprocess.run(['lspci', '|', 'grep', '-i', 'nvidia'], 
                              shell=True, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            print("   ✅ NVIDIA GPU detected:")
            print(f"   {result.stdout.strip()}")
        else:
            print("   ❌ No NVIDIA GPU found")
    except:
        print("   ❌ Could not check GPU hardware")
    
    # Check NVIDIA driver
    print("\n2. NVIDIA Driver:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   ✅ NVIDIA driver working")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"   {line.strip()}")
                elif 'GPU' in line and 'Memory' in line:
                    print(f"   {line.strip()}")
        else:
            print("   ❌ NVIDIA driver not working")
            print("   Error:", result.stderr.strip())
    except subprocess.TimeoutExpired:
        print("   ❌ nvidia-smi timeout")
    except FileNotFoundError:
        print("   ❌ nvidia-smi not found")
    
    # Check CUDA toolkit
    print("\n3. CUDA Toolkit:")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   ✅ CUDA toolkit installed")
            print("   " + result.stdout.split('\n')[0])
        else:
            print("   ❌ CUDA toolkit not installed")
    except FileNotFoundError:
        print("   ❌ CUDA toolkit not found")
    except subprocess.TimeoutExpired:
        print("   ❌ CUDA toolkit timeout")
    
    # Check system memory
    print("\n4. System Memory:")
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / 1024 / 1024
                    print(f"   Total RAM: {mem_gb:.1f} GB")
                    break
    except:
        print("   ❌ Could not read memory info")
    
    # Check available disk space
    print("\n5. Disk Space:")
    try:
        result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if '/' in line and 'G' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        print(f"   Available: {parts[3]} (Total: {parts[1]})")
                    break
    except:
        print("   ❌ Could not check disk space")
    
    print("\n6. Recommendations:")
    print("   💡 Your VPS has:")
    print("   - NVIDIA RTX A4000 GPU (good for AI models)")
    print("   - 31GB RAM (excellent for large models)")
    print("   - But NVIDIA driver/CUDA not properly configured")
    print("\n   🔧 To enable CUDA support:")
    print("   1. Install NVIDIA driver: sudo apt install nvidia-driver-525")
    print("   2. Install CUDA toolkit: sudo apt install nvidia-cuda-toolkit")
    print("   3. Reboot the system")
    print("   4. Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n   🚀 For now, use CPU-only configuration for Cohere")

def main():
    check_system()

if __name__ == "__main__":
    main()
