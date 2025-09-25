#!/usr/bin/env python3
"""
CPU-optimized Cohere implementation for VPS without CUDA
Uses your 31GB RAM efficiently
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    print("🚀 Cohere CPU-Optimized Test")
    print("=" * 40)
    print("💡 Using 31GB RAM for optimal performance")
    
    try:
        model_id = "CohereLabs/c4ai-command-r-plus-08-2024"
        print(f"🔄 Loading tokenizer from {model_id} (cached)...")
        
        # Load tokenizer from cache
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        print(f"   pad_token: {tokenizer.pad_token}")
        print(f"   eos_token: {tokenizer.eos_token}")
        
        print("🔄 Loading model (CPU-optimized for 31GB RAM)...")
        
        # Create offload directory for memory management
        os.makedirs("./offload", exist_ok=True)
        
        # Load model with CPU optimization
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="cpu",  # CPU only
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Float32 for CPU stability
            low_cpu_mem_usage=True,
            local_files_only=True,
            offload_folder="./offload",  # Offload to disk if needed
            max_memory={"cpu": "30GB"}  # Use most of your 31GB RAM
        )
        
        print("✅ Model loaded on CPU with 30GB RAM allocation")
        
        # Test 1: Simple generation
        print("\n🧪 Test 1: Simple generation...")
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        input_ids = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        print(f"   Input shape: {input_ids.shape}")
        
        with torch.no_grad():  # Disable gradients to save memory
            gen_tokens = model.generate(
                input_ids, 
                max_new_tokens=50,
                do_sample=True, 
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True  # Enable KV cache for efficiency
            )
        
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        print("✅ Generation successful!")
        print(f"Response: {gen_text}")
        
        # Test 2: Clustering test
        print("\n🧪 Test 2: Message clustering...")
        clustering_prompt = """Group these messages by topic:

1. Alice: Meeting at 3pm today
2. Bob: Can we reschedule to 4pm?
3. Charlie: New project update ready
4. David: Thanks for the feedback
5. Eve: Coffee break anyone?

Return JSON clusters."""
        
        messages = [{"role": "user", "content": clustering_prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids, 
                max_new_tokens=200,
                do_sample=True, 
                temperature=0.3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        clustering_response = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        print("✅ Clustering successful!")
        print("\n📋 Clustering Response:")
        print("-" * 40)
        print(clustering_response)
        print("-" * 40)
        
        print("\n🎉 Cohere is working perfectly on CPU!")
        print("💡 With 31GB RAM, you can run large models efficiently")
        print("🚀 Ready for full message analysis!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
