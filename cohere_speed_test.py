#!/usr/bin/env python3
"""
Speed test comparing CPU vs GPU performance
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_speed():
    print("🏃‍♂️ Cohere Speed Test: CPU vs GPU")
    print("=" * 40)
    
    model_name = "CohereLabs/c4ai-command-r-plus-08-2024"
    test_prompt = "Hello, how are you?"
    messages = [{"role": "user", "content": test_prompt}]
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    
    print("\n🖥️  Testing CPU performance...")
    try:
        model_cpu = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=True
        )
        
        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model_cpu.generate(
                inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
        cpu_time = time.time() - start_time
        
        print(f"✅ CPU generation time: {cpu_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ CPU test failed: {e}")
        cpu_time = None
    
    
    print("\n🚀 Testing GPU performance...")
    try:
        if torch.cuda.is_available():
            model_gpu = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True
            )
            
            inputs = tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(model_gpu.device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model_gpu.generate(
                    inputs,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id
                )
            gpu_time = time.time() - start_time
            
            print(f"✅ GPU generation time: {gpu_time:.2f} seconds")
            
            if cpu_time:
                speedup = cpu_time / gpu_time
                print(f"🚀 GPU is {speedup:.1f}x faster than CPU!")
        else:
            print("❌ CUDA not available")
            gpu_time = None
            
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        gpu_time = None
    
    
    print("\n📊 Performance Summary:")
    print("-" * 30)
    if cpu_time:
        print(f"CPU time: {cpu_time:.2f}s")
    if gpu_time:
        print(f"GPU time: {gpu_time:.2f}s")
    if cpu_time and gpu_time:
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
    
    print("\n💡 Recommendation:")
    if gpu_time and cpu_time:
        if gpu_time < cpu_time:
            print("🚀 Use GPU version for maximum speed!")
        else:
            print("🖥️  CPU version is sufficient for your use case")
    elif gpu_time:
        print("🚀 GPU version is ready to use!")
    else:
        print("🖥️  Use CPU version")

if __name__ == "__main__":
    test_speed()
