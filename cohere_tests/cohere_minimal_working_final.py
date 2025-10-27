#!/usr/bin/env python3
"""
Minimal working Cohere implementation that should not hang
Uses the smallest possible configuration
"""

import torch
import os
import signal
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def test_minimal():
    print("🚀 Cohere Minimal Working Test")
    print("=" * 40)
    
    try:
        model_name = "CohereLabs/c4ai-command-r-plus-08-2024"
        print(f"🔄 Loading {model_name}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with minimal memory usage
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
            max_memory={0: "8GB"}  # Use only 8GB of 16GB
        )
        
        print("✅ Model loaded")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Simple test with timeout
        print("\n🧪 Simple generation test...")
        messages = [{"role": "user", "content": "Hi"}]
        
        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        # Move to same device as model
        inputs = inputs.to(next(model.parameters()).device)
        print(f"Input device: {inputs.device}")
        print(f"Input shape: {inputs.shape}")
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        print("🔄 Generating (30 second timeout)...")
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=5,  # Very short
                    do_sample=False,   # No sampling
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            signal.alarm(0)  # Cancel timeout
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("✅ Generation successful!")
            print(f"Response: {response}")
            
            return True
            
        except TimeoutError:
            print("❌ Generation timed out after 30 seconds")
            print("💡 The model is too large for your GPU")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cpu_fallback():
    print("\n🖥️  Testing CPU fallback...")
    
    try:
        model_name = "CohereLabs/c4ai-command-r-plus-08-2024"
        print(f"🔄 Loading {model_name} on CPU...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model on CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        
        print("✅ Model loaded on CPU")
        
        # Simple test
        print("\n🧪 Simple generation test...")
        messages = [{"role": "user", "content": "Hello"}]
        
        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        print(f"Input shape: {inputs.shape}")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("✅ CPU generation works!")
        print(f"Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ CPU test failed: {e}")
        return False

def main():
    print("🔧 Starting minimal Cohere test...")
    print("This version has timeout protection and memory limits")
    
    # Try GPU first
    if test_minimal():
        print("\n🎯 SUCCESS!")
        print("✅ GPU version works!")
    else:
        print("\n🔄 GPU failed, trying CPU...")
        if test_cpu_fallback():
            print("\n🎯 SUCCESS!")
            print("✅ CPU version works!")
        else:
            print("\n❌ Both GPU and CPU failed")

if __name__ == "__main__":
    main()
